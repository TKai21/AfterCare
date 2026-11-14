from sqlalchemy import select

from conftest import auth_headers, doctor_headers
from models import CareTeamQuestion, User


async def ask(client, headers, doctor_id, question="Can I drive?"):
    return await client.post(
        "/api/care-team", headers=headers, json={"doctor_id": doctor_id, "question": question}
    )


async def two_doctors(client):
    ha = await doctor_headers(client, "dr.a", "Dr. A")
    hb = await doctor_headers(client, "dr.b", "  ")
    ids = {}
    async with client.app.state.session_factory() as s:
        for u in await s.scalars(select(User).where(User.role == "doctor")):
            ids[u.username] = u.id
    return ha, hb, ids["dr.a"], ids["dr.b"]


async def test_doctors_endpoint_lists_doctors_in_id_order(client):
    ha, hb, a, b = await two_doctors(client)
    h = await auth_headers(client)
    resp = await client.get("/api/doctors", headers=h)
    assert resp.status_code == 200
    assert resp.json() == {"doctors": [
        {"id": a, "name": "Dr. A", "specialty": None},
        {"id": b, "name": "dr.b", "specialty": None},
    ]}
    assert (await client.get("/api/doctors", headers=ha)).status_code == 403


async def test_doctors_endpoint_lists_specialties_and_hides_blank_ones(client):
    ha, hb, a, b = await two_doctors(client)
    async with client.app.state.session_factory() as s:
        (await s.get(User, a)).specialty = "Rheumatology"
        (await s.get(User, b)).specialty = "   "
        await s.commit()
    h = await auth_headers(client)
    doctors = (await client.get("/api/doctors", headers=h)).json()["doctors"]
    assert [(d["id"], d["specialty"]) for d in doctors] == [(a, "Rheumatology"), (b, None)]


async def test_direct_question_assigned_and_labelled(client):
    ha, hb, a, b = await two_doctors(client)
    h = await auth_headers(client)
    resp = await ask(client, h, b, "  Can I drive?  ")
    assert resp.status_code == 200
    q = resp.json()["question"]
    assert q["source"] == "patient" and q["reason"] == "Asked by the patient"
    assert q["question"] == "Can I drive?"
    assert q["doctor_id"] == b and q["doctor_name"] == "dr.b"
    assert q["document_id"] is None
    assert resp.json()["emergency"] == {"terms": [], "banner": False}
    listed = (await client.get("/api/care-team", headers=h)).json()["questions"]
    assert [x["doctor_name"] for x in listed] == ["dr.b"]


async def test_direct_question_rejects_non_doctor_id(client):
    await two_doctors(client)
    h = await auth_headers(client)
    async with client.app.state.session_factory() as s:
        me = (await s.scalar(select(User).where(User.username == "alice"))).id
    for bad in (me, 9999):
        resp = await ask(client, h, bad)
        assert resp.status_code == 422
        assert resp.json()["detail"] == "Choose one of the doctors."
    async with client.app.state.session_factory() as s:
        assert list(await s.scalars(select(CareTeamQuestion))) == []


async def test_direct_question_emergency_still_saved(client):
    ha, hb, a, b = await two_doctors(client)
    h = await auth_headers(client)
    resp = await ask(client, h, a, "I have chest pain")
    assert resp.status_code == 200
    assert resp.json()["emergency"]["banner"] is True
    assert resp.json()["emergency"]["terms"]
    async with client.app.state.session_factory() as s:
        assert len(list(await s.scalars(select(CareTeamQuestion)))) == 1


async def test_dedupe_is_per_doctor(client):
    from routes_care_team import add_question
    ha, hb, a, b = await two_doctors(client)
    await auth_headers(client)
    async with client.app.state.session_factory() as s:
        uid = (await s.scalar(select(User).where(User.username == "alice"))).id
        args = dict(document_id=None, question="Q?", reason="R", source="agent", dedupe=True)
        assert await add_question(s, uid, doctor_id=a, **args) is not None
        assert await add_question(s, uid, doctor_id=a, **args) is None
        assert await add_question(s, uid, doctor_id=b, **args) is not None
        assert await add_question(s, uid, **args) is not None
        assert await add_question(s, uid, **args) is None
