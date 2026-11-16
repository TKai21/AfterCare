import asyncio
import json

from sqlalchemy import select

from conftest import auth_headers, doctor_headers
from llm import LLMError
from models import ChatMessage, DischargeDocument, DoctorGuidance, User
from test_routes_documents import upload_text


async def asked(client, fake_llm, username="alice", text=None):
    """A patient with a document and one open question (a rejected item)."""
    h = await auth_headers(client, username)
    kw = {"text": text} if text else {}
    doc = (await upload_text(client, h, fake_llm, **kw)).json()
    await client.post(f"/api/documents/active/items/{doc['items'][0]['id']}/reject", headers=h)
    qid = (await client.get("/api/care-team", headers=h)).json()["questions"][0]["id"]
    return h, doc, qid


def answer_body(**kw):
    return {"answer": "Take it with food.", "general": False, "saved_question": None,
            "drug_tags": [], "condition_tags": [], **kw}


async def entries(client):
    async with client.app.state.session_factory() as s:
        return list(await s.scalars(select(DoctorGuidance)))


class FailingEmbed:
    """Wraps a FakeLLM but fails embedding."""
    def __init__(self, inner):
        self.inner = inner
        self.embedding_model = inner.embedding_model

    async def chat(self, *a, **k):
        return await self.inner.chat(*a, **k)

    async def embed(self, texts):
        raise LLMError("down")


async def test_patient_gets_403_on_every_doctor_route(client):
    h = await auth_headers(client)
    calls = [
        ("get", "/api/doctor/questions", None),
        ("post", "/api/doctor/questions/1/answer", answer_body()),
        ("post", "/api/doctor/suggest-tags", {"question": "q", "answer": "a"}),
        ("get", "/api/doctor/guidance", None),
        ("patch", "/api/doctor/guidance/1", {"active": False}),
        ("post", "/api/doctor/guidance/1/embed", None),
    ]
    for method, url, body in calls:
        resp = await client.request(method, url, headers=h, json=body)
        assert resp.status_code == 403, url


async def test_queue_lists_open_with_context(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    await client.post("/api/documents/active/confirm", headers=h)
    dh = await doctor_headers(client)
    data = (await client.get("/api/doctor/questions", headers=dh)).json()
    assert data["answered"] == []
    (item,) = data["open"]
    assert item["id"] == qid
    assert item["patient"]["username"] == "alice"
    ctx = item["context"]
    assert any("amoxicillin" in p for p in ctx["document_paragraphs"])
    assert {"category", "text"} == set(ctx["confirmed_items"][0])
    assert ctx["conversation"] == []


async def test_queue_question_without_document_has_no_document_context(client):
    await auth_headers(client)
    async with client.app.state.session_factory() as s:
        from routes_care_team import add_question
        uid = (await s.scalar(select(User).where(User.username == "alice"))).id
        await add_question(s, uid, None, "Q?", "R", "agent")
        await s.commit()
    dh = await doctor_headers(client)
    (item,) = (await client.get("/api/doctor/questions", headers=dh)).json()["open"]
    assert item["context"]["document_paragraphs"] is None
    assert item["context"]["confirmed_items"] == []


async def test_queue_open_oldest_first_answered_newest_first(client, fake_llm):
    h = await auth_headers(client)
    doc = (await upload_text(client, h, fake_llm)).json()
    for item in doc["items"]:
        await client.post(f"/api/documents/active/items/{item['id']}/reject", headers=h)
    ids = sorted(q["id"] for q in (await client.get("/api/care-team", headers=h)).json()["questions"])
    dh = await doctor_headers(client)
    data = (await client.get("/api/doctor/questions", headers=dh)).json()
    assert [q["id"] for q in data["open"]] == ids
    for qid in ids[:2]:
        assert (await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh,
                                  json=answer_body())).status_code == 200
    data = (await client.get("/api/doctor/questions", headers=dh)).json()
    assert [q["id"] for q in data["open"]] == ids[2:]
    assert [q["id"] for q in data["answered"]] == [ids[1], ids[0]]
    assert data["answered"][0]["answered_by_name"] == "Dr. Ting Kai"


async def test_answer_creates_patient_entry_tied_to_document(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    resp = await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["question"]["status"] == "answered"
    assert body["question"]["answer"] == "Take it with food."
    g = body["guidance"]
    assert g["scope"] == "patient" and g["patient"] == "alice"
    assert g["drug_tags"] == [] and g["searchable"] is True and g["stale"] is False
    (row,) = await entries(client)
    assert row.document_id == doc["id"] and row.source_question_id == qid
    q = (await client.get("/api/care-team", headers=h)).json()["questions"][0]
    assert q["answer"] == "Take it with food." and q["answered_by_name"] == "Dr. Ting Kai"


async def test_general_answer_uses_saved_question_and_tags(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    resp = await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body(
        general=True, saved_question="Can I drink with amoxicillin?",
        drug_tags=[" amoxicillin ", ""], condition_tags=[]))
    g = resp.json()["guidance"]
    assert g["scope"] == "general" and g["patient"] is None
    assert g["question"] == "Can I drink with amoxicillin?" and g["drug_tags"] == ["amoxicillin"]
    (row,) = await entries(client)
    assert row.patient_id is None and row.document_id is None


async def test_general_answer_without_tags_is_saved(client, fake_llm):
    from guidance import retrieve
    from models import GuidanceRule
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    resp = await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh,
                             json=answer_body(general=True, drug_tags=["  "]))
    assert resp.status_code == 200
    g = resp.json()["guidance"]
    assert g["scope"] == "general" and g["drug_tags"] == [] and g["condition_tags"] == []
    (row,) = await entries(client)
    assert row.answer == "Take it with food."
    # Its advice never reaches a patient: the gate refuses an untagged general entry.
    async with client.app.state.session_factory() as s:
        patient = await s.scalar(select(User).where(User.username == "alice"))
        document = await s.get(DischargeDocument, doc["id"])
        assert await s.scalar(select(GuidanceRule).where(GuidanceRule.guidance_id == row.id))
        hits = await retrieve(s, fake_llm, patient, document, [], "Take it with food.",
                              doctor_id=row.doctor_id, floor=0.0, margin=1.0)
    assert hits == []


async def add_rule(client, guidance_id, doctor_id, kind="habit", habit_type="ask",
                   statement="Asks about fever.", quote="Any fever?", suppressed=False, position=9):
    from models import GuidanceRule
    async with client.app.state.session_factory() as s:
        rule = GuidanceRule(guidance_id=guidance_id, doctor_id=doctor_id, position=position,
                            statement=statement, quote=quote, kind=kind,
                            habit_type=habit_type if kind == "habit" else None,
                            warning_terms=["fever"] if habit_type == "warning" else [],
                            suppressed=suppressed)
        s.add(rule)
        await s.commit()
        return rule.id


async def answered_entry(client, fake_llm, username="alice"):
    h, doc, qid = await asked(client, fake_llm, username)
    dh = await doctor_headers(client)
    g = (await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh,
                           json=answer_body())).json()["guidance"]
    (row,) = [e for e in await entries(client) if e.id == g["id"]]
    return dh, g, row


async def test_doctor_suppresses_own_habit(client, fake_llm):
    from models import GuidanceRule
    dh, g, row = await answered_entry(client, fake_llm)
    hid = await add_rule(client, g["id"], row.doctor_id)
    other = await doctor_headers(client, "dr.renny", "Dr. Renny")
    assert (await client.delete(f"/api/doctor/guidance/rules/{hid}", headers=other)).status_code == 404
    assert (await client.delete("/api/doctor/guidance/rules/99999", headers=dh)).status_code == 404
    resp = await client.delete(f"/api/doctor/guidance/rules/{hid}", headers=dh)
    assert resp.status_code == 204 and resp.content == b""
    async with client.app.state.session_factory() as s:
        rule = await s.get(GuidanceRule, hid)
        assert rule is not None and rule.suppressed is True
    data = (await client.get("/api/doctor/guidance", headers=dh)).json()
    assert data["habits_by_type"] == {"ask": [], "close": [], "warning": []}
    assert all(h["id"] != hid for e in data["entries"] for h in e["habits"])


async def test_deleting_a_habit_also_removes_its_duplicates(client, fake_llm):
    dh, g, row = await answered_entry(client, fake_llm)
    a = await add_rule(client, g["id"], row.doctor_id, statement="Asks about fever.")
    b = await add_rule(client, g["id"], row.doctor_id, statement="asks about  Fever.", position=10)
    assert (await client.delete(f"/api/doctor/guidance/rules/{a}", headers=dh)).status_code == 204
    data = (await client.get("/api/doctor/guidance", headers=dh)).json()
    assert data["habits_by_type"]["ask"] == []
    assert all(h["id"] != b for e in data["entries"] for h in e["habits"])


async def test_deleting_an_advice_rule_is_422(client, fake_llm):
    from models import GuidanceRule
    dh, g, row = await answered_entry(client, fake_llm)
    rid = g["rules"][0]["id"]
    resp = await client.delete(f"/api/doctor/guidance/rules/{rid}", headers=dh)
    assert resp.status_code == 422
    async with client.app.state.session_factory() as s:
        assert (await s.get(GuidanceRule, rid)).suppressed is False


async def test_guidance_list_groups_habits_by_type(client, fake_llm):
    dh, g, row = await answered_entry(client, fake_llm)
    ask = await add_rule(client, g["id"], row.doctor_id, "habit", "ask", "Asks about fever.", "Any fever?")
    close = await add_rule(client, g["id"], row.doctor_id, "habit", "close", "Closes with a call offer.", "Call me.")
    warn = await add_rule(client, g["id"], row.doctor_id, "habit", "warning", "Wants to hear about fever.", "Tell me about fever.")
    await add_rule(client, g["id"], row.doctor_id, "habit", "ask", "Asks about pain.", "Any pain?", suppressed=True)
    data = (await client.get("/api/doctor/guidance", headers=dh)).json()
    by = data["habits_by_type"]
    assert [h["id"] for h in by["ask"]] == [ask]
    assert [h["id"] for h in by["close"]] == [close]
    assert [h["id"] for h in by["warning"]] == [warn]
    (e,) = data["entries"]
    # Each habit names the entry it came from, so My guidance can open it.
    assert by["warning"][0] == {"id": warn, "habit_type": "warning",
                                "statement": "Wants to hear about fever.",
                                "quote": "Tell me about fever.", "warning_terms": ["fever"],
                                "guidance_id": g["id"], "question": e["question"]}
    assert all(h["guidance_id"] == g["id"] and h["question"] == e["question"]
               for h in e["habits"])
    # Advice rules are listed apart from habits; suppressed rows are hidden.
    assert {r["id"] for r in e["rules"]} == {r["id"] for r in g["rules"]}
    assert {h["id"] for h in e["habits"]} == {ask, close, warn}


async def test_conflicts_on_a_suppressed_rule_are_hidden(client, fake_llm):
    from models import GuidanceConflict, GuidanceRule
    dh, g, row = await answered_entry(client, fake_llm)
    hid = await add_rule(client, g["id"], row.doctor_id, suppressed=True)
    rid = g["rules"][0]["id"]
    async with client.app.state.session_factory() as s:
        s.add(GuidanceConflict(doctor_id=row.doctor_id, rule_id=hid,
                               other_rule_id=rid, explanation="x", status="open"))
        await s.commit()
    (e,) = (await client.get("/api/doctor/guidance", headers=dh)).json()["entries"]
    assert e["conflicts"] == []


async def test_second_answer_is_409(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body())
    resp = await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh,
                             json=answer_body(answer="Other."))
    assert resp.status_code == 409
    assert resp.json()["detail"] == "This question already has an answer."
    assert len(await entries(client)) == 1


async def test_answer_unknown_question_404(client):
    dh = await doctor_headers(client)
    resp = await client.post("/api/doctor/questions/999/answer", headers=dh, json=answer_body())
    assert resp.status_code == 404


async def test_concurrent_answers_one_wins(file_client, fake_llm):
    client = file_client
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    results = await asyncio.gather(*[
        client.post(f"/api/doctor/questions/{qid}/answer", headers=dh,
                    json=answer_body(answer=f"Answer {n}"))
        for n in range(2)
    ])
    assert sorted(r.status_code for r in results) == [200, 409]
    assert len(await entries(client)) == 1


async def test_embedding_failure_keeps_answer_then_embed_retry(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    client.app.state.llm = FailingEmbed(fake_llm)
    resp = await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body())
    assert resp.status_code == 200
    gid = resp.json()["guidance"]["id"]
    assert resp.json()["guidance"]["searchable"] is False
    q = (await client.get("/api/care-team", headers=h)).json()["questions"][0]
    assert q["answer"] == "Take it with food."
    client.app.state.llm = fake_llm
    resp = await client.post(f"/api/doctor/guidance/{gid}/embed", headers=dh)
    assert resp.status_code == 200 and resp.json()["searchable"] is True


async def test_answering_resolved_question_stays_resolved(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    await client.post(f"/api/care-team/{qid}/resolve", headers=h)
    dh = await doctor_headers(client)
    resp = await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body())
    assert resp.status_code == 200
    assert resp.json()["question"]["status"] == "resolved"
    assert resp.json()["question"]["answer"] == "Take it with food."


async def test_suggest_tags_drops_names_absent_from_text(client, fake_llm):
    dh = await doctor_headers(client)
    fake_llm.script({"content": json.dumps({
        "drug_tags": ["Metoprolol", "metoprolol", "blocker", "warfarin"],
        "condition_tags": ["atrial fibrillation", "AFib"],
    })})
    resp = await client.post("/api/doctor/suggest-tags", headers=dh, json={
        "question": "Can I take metoprolol?", "answer": "Yes, for your Atrial  Fibrillation, it is fine."})
    assert resp.status_code == 200
    assert resp.json() == {"drug_tags": ["Metoprolol"], "condition_tags": ["atrial fibrillation"]}
    assert fake_llm.calls[-1][2] is True


async def test_suggest_tags_without_model_503_and_llm_error_502(client, fake_llm):
    dh = await doctor_headers(client)
    body = {"question": "q", "answer": "a"}
    fake_llm.script({"content": "not json"})
    assert (await client.post("/api/doctor/suggest-tags", headers=dh, json=body)).status_code == 502

    async def boom(*a, **k):
        raise LLMError("x")
    fake_llm.chat = boom
    assert (await client.post("/api/doctor/suggest-tags", headers=dh, json=body)).status_code == 502
    client.app.state.llm = None
    assert (await client.post("/api/doctor/suggest-tags", headers=dh, json=body)).status_code == 503


async def test_guidance_list_newest_first_with_cited_count(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    first = (await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh,
                               json=answer_body(general=True, drug_tags=["amoxicillin"]))).json()
    gid = first["guidance"]["id"]
    async with client.app.state.session_factory() as s:
        me = await s.scalar(select(User.id).where(User.username == "dr.ting"))
        s.add(ChatMessage(document_id=doc["id"], doctor_id=me, role="assistant", content="x",
                          payload={"guidance_ids": [gid]}))
        # Only this doctor's clone conversations count: not another doctor's, not
        # a message with no doctor.
        other = User(username="dr.other", password_hash="x", role="doctor")
        s.add(other)
        await s.flush()
        s.add(ChatMessage(document_id=doc["id"], doctor_id=other.id, role="assistant",
                          content="w", payload={"guidance_ids": [gid]}))
        s.add(ChatMessage(document_id=doc["id"], role="assistant", content="v",
                          payload={"guidance_ids": [gid]}))
        s.add(ChatMessage(document_id=doc["id"], role="assistant", content="y", payload=None))
        s.add(ChatMessage(document_id=doc["id"], role="user", content="z",
                          payload={"guidance_ids": [gid]}))
        await s.commit()
    got = (await client.get("/api/doctor/guidance", headers=dh)).json()["entries"]
    assert [e["id"] for e in got] == [gid]
    assert got[0]["cited_count"] == 1 and got[0]["active"] is True and got[0]["stale"] is False


async def test_searchable_false_without_model(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body())
    client.app.state.llm = None
    (e,) = (await client.get("/api/doctor/guidance", headers=dh)).json()["entries"]
    assert e["searchable"] is False
    assert (await client.post(f"/api/doctor/guidance/{e['id']}/embed", headers=dh)).status_code == 503


async def test_patch_to_general_without_tags_is_saved(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    gid = (await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh,
                             json=answer_body())).json()["guidance"]["id"]
    resp = await client.patch(f"/api/doctor/guidance/{gid}", headers=dh, json={"scope": "general"})
    assert resp.status_code == 200 and resp.json()["scope"] == "general"
    resp = await client.patch(f"/api/doctor/guidance/{gid}", headers=dh,
                              json={"scope": "general", "drug_tags": ["amoxicillin"]})
    assert resp.status_code == 200 and resp.json()["scope"] == "general"
    resp = await client.patch(f"/api/doctor/guidance/{gid}", headers=dh, json={"scope": "patient"})
    assert resp.status_code == 200 and resp.json()["patient"] == "alice"


async def test_patch_to_patient_needs_patient_id(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    gid = (await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh,
                             json=answer_body(general=True, drug_tags=["amoxicillin"]))).json()["guidance"]["id"]
    resp = await client.patch(f"/api/doctor/guidance/{gid}", headers=dh, json={"scope": "patient"})
    assert resp.status_code == 422
    assert (await client.patch("/api/doctor/guidance/999", headers=dh, json={})).status_code == 404


async def test_patch_text_reembeds_and_failure_keeps_edit(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    gid = (await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh,
                             json=answer_body())).json()["guidance"]["id"]
    async with client.app.state.session_factory() as s:
        before = (await s.get(DoctorGuidance, gid)).embedding
    resp = await client.patch(f"/api/doctor/guidance/{gid}", headers=dh,
                              json={"answer": "Completely different words here.", "active": False})
    assert resp.status_code == 200 and resp.json()["searchable"] is True
    assert resp.json()["active"] is False
    async with client.app.state.session_factory() as s:
        assert (await s.get(DoctorGuidance, gid)).embedding != before
    client.app.state.llm = FailingEmbed(fake_llm)
    resp = await client.patch(f"/api/doctor/guidance/{gid}", headers=dh, json={"answer": "Edited again."})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Edited again." and resp.json()["searchable"] is False


async def test_stale_after_patient_uploads_new_document(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body())
    (e,) = (await client.get("/api/doctor/guidance", headers=dh)).json()["entries"]
    assert e["stale"] is False
    await upload_text(client, h, fake_llm, text="A brand new discharge document.")
    (e,) = (await client.get("/api/doctor/guidance", headers=dh)).json()["entries"]
    assert e["stale"] is True


async def test_answer_lost_race_hits_conditional_update_409(client, fake_llm):
    """The answer lands between the handler's read and its UPDATE: only the
    UPDATE's rowcount can catch it."""
    from datetime import datetime
    from sqlalchemy import update
    from models import CareTeamQuestion
    import routes_doctor

    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    original_get = routes_doctor.AsyncSession.get

    async def get_then_answer(self, entity, ident, *a, **k):
        row = await original_get(self, entity, ident, *a, **k)
        if entity is CareTeamQuestion:
            async with client.app.state.session_factory() as other:
                await other.execute(update(CareTeamQuestion).where(CareTeamQuestion.id == ident).values(
                    answer="Rival answer", answered_at=datetime(2026, 1, 1), status="answered"))
                await other.commit()
        return row

    routes_doctor.AsyncSession.get = get_then_answer
    try:
        resp = await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body())
    finally:
        routes_doctor.AsyncSession.get = original_get
    assert resp.status_code == 409
    assert resp.json()["detail"] == "This question already has an answer."
    assert await entries(client) == []


async def test_queue_loads_each_document_once(client, fake_llm):
    h = await auth_headers(client)
    doc = (await upload_text(client, h, fake_llm)).json()
    for item in doc["items"]:
        await client.post(f"/api/documents/active/items/{item['id']}/reject", headers=h)
    dh = await doctor_headers(client)
    import routes_doctor
    calls = []
    real = routes_doctor.load_items

    async def counting(session, d):
        calls.append(d.id)
        return await real(session, d)

    routes_doctor.load_items = counting
    try:
        data = (await client.get("/api/doctor/questions", headers=dh)).json()
    finally:
        routes_doctor.load_items = real
    assert len(data["open"]) >= 2
    assert calls == [doc["id"]]


async def test_doctor_page_served(client):
    resp = await client.get("/doctor")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert resp.headers["cache-control"] == "no-cache"
    assert "doctor.js" in resp.text


async def test_patient_entry_without_a_document_is_stale(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body())
    async with client.app.state.session_factory() as s:
        (entry,) = list(await s.scalars(select(DoctorGuidance)))
        entry.document_id = None
        # No active document either: None == None must not read as "current".
        (document,) = list(await s.scalars(select(DischargeDocument)))
        document.active_user_id = None
        await s.commit()
    (e,) = (await client.get("/api/doctor/guidance", headers=dh)).json()["entries"]
    assert e["scope"] == "patient" and e["stale"] is True


async def test_queue_shows_own_and_unassigned_only(client, fake_llm):
    ha = await doctor_headers(client, "dr.a", "Dr. A")
    hb = await doctor_headers(client, "dr.b", "Dr. B")
    async with client.app.state.session_factory() as s:
        ids = {u.username: u.id for u in await s.scalars(select(User).where(User.role == "doctor"))}
    h = await auth_headers(client)
    for doctor_id, text in ((ids["dr.a"], "For A?"), (ids["dr.b"], "For B?")):
        r = await client.post("/api/care-team", headers=h, json={"doctor_id": doctor_id, "question": text})
        assert r.status_code == 200
    async with client.app.state.session_factory() as s:
        from models import CareTeamQuestion
        alice_id = (await s.scalar(select(User).where(User.username == "alice"))).id
        s.add(CareTeamQuestion(user_id=alice_id, question="For anyone?", reason="r", source="agent"))
        await s.commit()
    data = (await client.get("/api/doctor/questions", headers=ha)).json()
    assert [q["question"] for q in data["open"]] == ["For A?", "For anyone?"]
    assert data["open"][0]["doctor_name"] == "Dr. A"
    data = (await client.get("/api/doctor/questions", headers=hb)).json()
    assert [q["question"] for q in data["open"]] == ["For B?", "For anyone?"]


async def test_answering_unassigned_assigns_it(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    resp = await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body())
    assert resp.status_code == 200
    q = resp.json()["question"]
    assert q["doctor_id"] is not None and q["doctor_name"] == "Dr. Ting Kai"
    other = await doctor_headers(client, "dr.b", "Dr. B")
    data = (await client.get("/api/doctor/questions", headers=other)).json()
    assert data["answered"] == [] and data["open"] == []


async def test_answer_to_another_doctors_question_is_404(client, fake_llm):
    ha = await doctor_headers(client, "dr.a", "Dr. A")
    hb = await doctor_headers(client, "dr.b", "Dr. B")
    async with client.app.state.session_factory() as s:
        b_id = (await s.scalar(select(User).where(User.username == "dr.b"))).id
    h = await auth_headers(client)
    qid = (await client.post("/api/care-team", headers=h,
                             json={"doctor_id": b_id, "question": "For B?"})).json()["question"]["id"]
    resp = await client.post(f"/api/doctor/questions/{qid}/answer", headers=ha, json=answer_body())
    assert resp.status_code == 404
    assert (await client.post(f"/api/doctor/questions/{qid}/answer", headers=hb,
                              json=answer_body())).status_code == 200


async def test_guidance_of_another_doctor_is_404(client, fake_llm):
    h, doc, qid = await asked(client, fake_llm)
    ha = await doctor_headers(client, "dr.a", "Dr. A")
    hb = await doctor_headers(client, "dr.b", "Dr. B")
    gid = (await client.post(f"/api/doctor/questions/{qid}/answer", headers=ha,
                             json=answer_body())).json()["guidance"]["id"]
    assert [e["id"] for e in (await client.get("/api/doctor/guidance", headers=ha)).json()["entries"]] == [gid]
    assert (await client.get("/api/doctor/guidance", headers=hb)).json()["entries"] == []
    resp = await client.patch(f"/api/doctor/guidance/{gid}", headers=hb, json={"answer": "Hijacked."})
    assert resp.status_code == 404
    assert (await client.post(f"/api/doctor/guidance/{gid}/embed", headers=hb)).status_code == 404
    async with client.app.state.session_factory() as s:
        assert (await s.get(DoctorGuidance, gid)).answer == "Take it with food."


async def test_reread_rules_reads_every_active_entry(client, fake_llm):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import reread_rules
    from models import GuidanceRule
    from rules import RULES_PROMPT
    dh, g, row = await answered_entry(client, fake_llm)
    h2, doc2, qid2 = await asked(client, fake_llm, "bob")
    second = (await client.post(f"/api/doctor/questions/{qid2}/answer", headers=dh,
                                json=answer_body(answer="Any fever? Rest."))).json()["guidance"]["id"]
    retired = (await client.patch(f"/api/doctor/guidance/{second}", headers=dh,
                                  json={"active": False})).json()["id"]
    h3, doc3, qid3 = await asked(client, fake_llm, "carol")
    third = (await client.post(f"/api/doctor/questions/{qid3}/answer", headers=dh,
                               json=answer_body(answer="Any chills? Drink water."))).json()["guidance"]["id"]

    def reader(messages):
        text = messages[-1]["content"]
        found = {"kind": "habit", "habit_type": "ask", "statement": "Asks about chills.",
                 "quote": "Any chills?", "warning_terms": []}
        return {"content": json.dumps({"rules": [found] if "Any chills?" in text else []})}

    fake_llm.route(RULES_PROMPT, reader)
    before = len(fake_llm.routed_calls)
    async with client.app.state.session_factory() as s:
        retired_rules = list(await s.scalars(select(GuidanceRule).where(GuidanceRule.guidance_id == retired)))
        snapshot = [(r.id, r.statement) for r in retired_rules]
    lines = await reread_rules.reread(client.app.state.session_factory, fake_llm,
                                      client.app.state.store, "dr.ting")
    reads = [c for c in fake_llm.routed_calls[before:] if c[0][0]["content"] == RULES_PROMPT]
    assert len(reads) == 2
    assert lines[-1] == "Read 2 active entries."
    assert any("[habit ask] Asks about chills." in line for line in lines)
    async with client.app.state.session_factory() as s:
        assert [(r.id, r.statement) for r in await s.scalars(
            select(GuidanceRule).where(GuidanceRule.guidance_id == retired))] == snapshot
        habits = list(await s.scalars(select(GuidanceRule).where(
            GuidanceRule.kind == "habit", GuidanceRule.guidance_id == third)))
        assert [h.statement for h in habits] == ["Asks about chills."]
    with __import__("pytest").raises(ValueError):
        await reread_rules.reread(client.app.state.session_factory, fake_llm,
                                  client.app.state.store, "alice")


async def add_clarify(client, doctor_username="dr.ting", created_at=None, **kw):
    """A waiting_on_patient question the clone asked, assigned to the doctor."""
    import routes_doctor
    from models import CareTeamQuestion
    async with client.app.state.session_factory() as s:
        doctor_id = await s.scalar(select(User.id).where(User.username == doctor_username))
        alice_id = (await s.scalar(select(User).where(User.username == "alice"))).id
        kw.setdefault("question", "Can I drive?")
        q = CareTeamQuestion(
            user_id=alice_id, doctor_id=doctor_id, reason="Clone asked",
            source="agent", status="waiting_on_patient", asked="Are you dizzy? Any new pills?",
            created_at=created_at or routes_doctor.utcnow(), **kw)
        s.add(q)
        await s.commit()
        return q.id


async def test_waiting_questions_are_listed_apart(client, fake_llm):
    await auth_headers(client)
    dh = await doctor_headers(client)
    await doctor_headers(client, "dr.b", "Dr. B")
    mine = await add_clarify(client)
    await add_clarify(client, "dr.b")
    data = (await client.get("/api/doctor/questions", headers=dh)).json()
    assert [q["id"] for q in data["waiting"]] == [mine]
    assert data["waiting"][0]["asked"] == "Are you dizzy? Any new pills?"
    assert data["open"] == [] and data["answered"] == []


async def test_waiting_question_times_out_on_read(client, fake_llm, monkeypatch):
    from datetime import datetime, timedelta
    import routes_doctor
    from models import CareTeamQuestion
    await auth_headers(client)
    dh = await doctor_headers(client)
    now = datetime(2026, 1, 10, 12, 0)
    monkeypatch.setattr(routes_doctor, "utcnow", lambda: now)
    old = await add_clarify(client, created_at=now - timedelta(hours=25))
    fresh = await add_clarify(client, created_at=now - timedelta(hours=23))
    data = (await client.get("/api/doctor/questions", headers=dh)).json()
    assert [q["id"] for q in data["waiting"]] == [fresh]
    (item,) = data["open"]
    assert item["id"] == old
    assert item["status"] == "open"
    assert item["reason"] == "Patient did not answer the clone's questions"
    assert item["stopped_by"] == "Patient did not answer the clone's questions"
    async with client.app.state.session_factory() as s:
        assert (await s.get(CareTeamQuestion, old)).status == "open"
        assert (await s.get(CareTeamQuestion, fresh)).status == "waiting_on_patient"


async def test_open_count_excludes_waiting(client, fake_llm):
    h, _doc, _qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    await add_clarify(client)
    data = (await client.get("/api/doctor/questions", headers=dh)).json()
    assert len(data["open"]) == 1 and len(data["waiting"]) == 1


async def test_clone_closed_question_is_answered_and_labelled(client, fake_llm):
    import routes_doctor
    from models import CareTeamQuestion
    await auth_headers(client)
    dh = await doctor_headers(client)
    qid = await add_clarify(client)
    async with client.app.state.session_factory() as s:
        q = await s.get(CareTeamQuestion, qid)
        q.status, q.answer, q.answered_at = "resolved", "Rest and call if worse.", routes_doctor.utcnow()
        await s.commit()
    data = (await client.get("/api/doctor/questions", headers=dh)).json()
    (item,) = data["answered"]
    assert item["id"] == qid and item["closed_by_clone"] is True
    assert data["waiting"] == [] and data["open"] == []


async def test_deleting_a_habit_removes_twins_of_the_same_type_only(client, fake_llm):
    dh, g, row = await answered_entry(client, fake_llm)
    ask = await add_rule(client, g["id"], row.doctor_id, statement="Wants to hear about fever.",
                         quote="Tell me about any fever.")
    same_quote = await add_rule(client, g["id"], row.doctor_id, statement="Asks if there is fever.",
                                quote="tell me about  any FEVER.", position=10)
    warning = await add_rule(client, g["id"], row.doctor_id, habit_type="warning",
                             statement="Wants to hear about fever.",
                             quote="Tell me about any fever.", position=11)
    assert (await client.delete(f"/api/doctor/guidance/rules/{ask}", headers=dh)).status_code == 204
    data = (await client.get("/api/doctor/guidance", headers=dh)).json()
    assert data["habits_by_type"]["ask"] == []
    assert [h["id"] for h in data["habits_by_type"]["warning"]] == [warning]
    from models import GuidanceRule
    async with client.app.state.session_factory() as s:
        assert (await s.get(GuidanceRule, same_quote)).suppressed is True
        assert (await s.get(GuidanceRule, warning)).suppressed is False


async def test_the_timeout_leaves_another_doctors_waiting_row_alone(client, fake_llm, monkeypatch):
    from datetime import datetime, timedelta
    import routes_doctor
    from models import CareTeamQuestion
    await auth_headers(client)
    dh = await doctor_headers(client)
    await doctor_headers(client, "dr.b", "Dr. B")
    now = datetime(2026, 1, 10, 12, 0)
    monkeypatch.setattr(routes_doctor, "utcnow", lambda: now)
    theirs = await add_clarify(client, "dr.b", created_at=now - timedelta(hours=48))
    data = (await client.get("/api/doctor/questions", headers=dh)).json()
    assert data["open"] == [] and data["waiting"] == []
    async with client.app.state.session_factory() as s:
        row = await s.get(CareTeamQuestion, theirs)
        assert (row.status, row.reason, row.stopped_by) == ("waiting_on_patient", "Clone asked", None)


async def test_waiting_question_flags_a_warning_sign_in_the_first_message(client, fake_llm):
    dh, g, row = await answered_entry(client, fake_llm)
    await add_rule(client, g["id"], row.doctor_id, habit_type="warning",
                   statement="Wants to hear about fever.", quote="Tell me about any fever.")
    flagged = await add_clarify(client, question="I have a fever and my incision is red.")
    plain = await add_clarify(client, question="My incision is a bit pink.")
    data = (await client.get("/api/doctor/questions", headers=dh)).json()
    by_id = {q["id"]: q for q in data["waiting"]}
    assert by_id[flagged]["first_warning"] == ["fever"]
    assert by_id[plain]["first_warning"] == []
    # Computed on read: a habit deleted later no longer flags the message.
    from models import GuidanceRule
    async with client.app.state.session_factory() as s:
        for r in await s.scalars(select(GuidanceRule).where(GuidanceRule.habit_type == "warning")):
            r.suppressed = True
        await s.commit()
    data = (await client.get("/api/doctor/questions", headers=dh)).json()
    assert all(q["first_warning"] == [] for q in data["waiting"])
