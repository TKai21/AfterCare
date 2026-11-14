from datetime import datetime

from sqlalchemy import select

from conftest import auth_headers, doctor_headers
from models import CareTeamQuestion, User
from routes_care_team import add_question
from test_routes_documents import upload_text


async def make_question(client, headers, fake_llm):
    doc = (await upload_text(client, headers, fake_llm)).json()
    await client.post(f"/api/documents/active/items/{doc['items'][0]['id']}/reject", headers=headers)
    return doc


async def test_resolve_question(client, fake_llm):
    h = await auth_headers(client)
    await make_question(client, h, fake_llm)
    qid = (await client.get("/api/care-team", headers=h)).json()["questions"][0]["id"]
    assert (await client.post(f"/api/care-team/{qid}/resolve", headers=h)).status_code == 200
    assert (await client.get("/api/care-team", headers=h)).json()["questions"][0]["status"] == "resolved"


async def test_care_team_newest_first(client, fake_llm):
    h = await auth_headers(client)
    doc = (await upload_text(client, h, fake_llm)).json()
    for item in doc["items"]:
        await client.post(f"/api/documents/active/items/{item['id']}/reject", headers=h)
    qs = (await client.get("/api/care-team", headers=h)).json()["questions"]
    assert [q["question"] for q in qs] == [
        "Please confirm this instruction: " + i["text"] for i in reversed(doc["items"])
    ]


async def test_other_users_question_404(client, fake_llm):
    ha = await auth_headers(client, "alice")
    hb = await auth_headers(client, "bob")
    await make_question(client, ha, fake_llm)
    qid = (await client.get("/api/care-team", headers=ha)).json()["questions"][0]["id"]
    assert (await client.post(f"/api/care-team/{qid}/resolve", headers=hb)).status_code == 404
    assert (await client.get("/api/care-team", headers=hb)).json() == {"questions": []}
    assert (await client.get("/api/care-team", headers=ha)).json()["questions"][0]["status"] == "open"
    assert (await client.post("/api/care-team/99999/resolve", headers=ha)).status_code == 404


async def test_care_team_list_survives_reupload(client, fake_llm):
    h = await auth_headers(client)
    first = await make_question(client, h, fake_llm)
    second = (await upload_text(client, h, fake_llm, text="Second document body text.")).json()
    qs = (await client.get("/api/care-team", headers=h)).json()["questions"]
    assert len(qs) == 1 and qs[0]["document_id"] == first["id"] != second["id"]


async def answer_question(client, qid, doctor_username):
    async with client.app.state.session_factory() as s:
        doctor = await s.scalar(select(User).where(User.username == doctor_username))
        q = await s.get(CareTeamQuestion, qid)
        q.status = "answered"
        q.answer = "Take it with food."
        q.answered_by = doctor.id
        q.answered_at = datetime(2026, 1, 2, 3, 4, 5)
        await s.commit()


async def test_resolve_keeps_answer(client, fake_llm):
    h = await auth_headers(client)
    await doctor_headers(client)
    await make_question(client, h, fake_llm)
    qid = (await client.get("/api/care-team", headers=h)).json()["questions"][0]["id"]
    await answer_question(client, qid, "dr.ting")
    q = (await client.get("/api/care-team", headers=h)).json()["questions"][0]
    assert q["status"] == "answered" and q["answered_by_name"] == "Dr. Ting Kai"
    resp = await client.post(f"/api/care-team/{qid}/resolve", headers=h)
    assert resp.status_code == 200
    q = (await client.get("/api/care-team", headers=h)).json()["questions"][0]
    assert q["status"] == "resolved"
    assert q["answer"] == "Take it with food."
    assert q["answered_by_name"] == "Dr. Ting Kai"
    assert q["answered_at"].startswith("2026-01-02T03:04:05")


async def test_add_question_dedupe(client):
    h = await auth_headers(client)
    async with client.app.state.session_factory() as s:
        uid = (await s.scalar(select(User).where(User.username == "alice"))).id
        args = dict(document_id=None, question="Q?", reason="R", source="agent")
        first = await add_question(s, uid, dedupe=True, **args)
        assert first is not None
        assert await add_question(s, uid, dedupe=True, **args) is None
        first.status = "resolved"
        await s.flush()
        assert await add_question(s, uid, dedupe=True, **args) is not None
