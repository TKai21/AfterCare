import json

import pytest_asyncio
from sqlalchemy import func, select

import agent
import rules
from conftest import doctor_headers
from db import init_db, make_engine, make_session_factory
from fake_llm import FakeLLM, sentence_rules
from guidance import RuleHit, embed_entry, retrieve
from llm import LLMError
from models import (
    DischargeDocument, DoctorGuidance, DrugLabel, GuidanceConflict, GuidanceRule, User,
)
from test_routes_doctor import answer_body, asked

COLCHICINE = (
    "Take one extra 0.6 mg tablet after 1 hour, then no more that day. "
    "Stop and call the clinic if you get diarrhea."
)


def rules_reply(*pairs):
    return {"content": json.dumps({"rules": [{"statement": s, "quote": q} for s, q in pairs]})}


def conflicts_reply(*items):
    return {"content": json.dumps({"conflicts": [
        {"rule": r, "candidate": c, "explanation": e} for r, c, e in items]})}


# ---------- quotes and validation ----------

def test_find_quote_forgives_whitespace_case_and_quote_marks_only():
    answer = "Take one extra 0.6 mg tablet\nafter 1 hour.  Then rest."
    assert rules.find_quote(answer, '"take one extra 0.6 mg tablet after 1 hour"') == (
        "Take one extra 0.6 mg tablet\nafter 1 hour.")
    assert rules.find_quote(answer, "Take an extra 0.6 mg tablet") is None
    assert rules.find_quote(answer, "..") is None


def test_validate_rules_drops_bad_rules_and_keeps_order():
    data = {"rules": [
        {"statement": "Colchicine: one extra 0.6 mg tablet after 1 hour in a flare.",
         "quote": "Take one extra 0.6 mg tablet after 1 hour"},
        {"statement": "Colchicine: an extra 1.2 mg tablet is fine.",  # number not in quote
         "quote": "Take one extra 0.6 mg tablet after 1 hour"},
        {"statement": "Colchicine is safe with grapefruit.", "quote": "grapefruit is fine"},  # not in answer
        {"statement": "x" * 401, "quote": "then no more that day"},
        {"statement": "Colchicine: one extra 0.6 mg tablet after 1 hour in a flare.",  # duplicate
         "quote": "Take one extra 0.6 mg tablet after 1 hour"},
        "not a rule",
        {"statement": "Stop colchicine and call the clinic if you get diarrhea.",
         "quote": "Stop and call the clinic if you get diarrhea"},
    ]}
    got = rules.validate_rules(COLCHICINE, data)
    assert [r.statement for r in got] == [
        "Colchicine: one extra 0.6 mg tablet after 1 hour in a flare.",
        "Stop colchicine and call the clinic if you get diarrhea.",
    ]
    # The stored quote is the answer's own text, with its closing period.
    assert got[1].quote == "Stop and call the clinic if you get diarrhea."
    assert rules.validate_rules(COLCHICINE, {"rules": "nope"}) == []
    assert rules.validate_rules(COLCHICINE, []) == []


def test_validate_rules_caps_the_count():
    answer = " ".join(f"Rule {i} applies." for i in range(12))
    data = {"rules": [{"statement": f"Rule {i} applies.", "quote": f"Rule {i} applies."}
                      for i in range(12)]}
    assert len(rules.validate_rules(answer, data)) == rules.MAX_RULES


# ---------- building rules ----------

@pytest_asyncio.fixture
async def session():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    async with make_session_factory(engine)() as s:
        yield s
    await engine.dispose()


async def seed(session):
    users = [User(username=n, password_hash="x") for n in ("doc", "a", "b", "doc2")]
    session.add_all(users)
    await session.flush()
    doc = DischargeDocument(user_id=users[1].id, text="gout colchicine naproxen")
    session.add(doc)
    await session.flush()
    return users, doc


async def general(session, llm, doctor, answer, question="q", drugs=("colchicine",), **kw):
    e = DoctorGuidance(doctor_id=doctor.id, question=question, answer=answer, scope="general",
                       drug_tags=list(drugs), condition_tags=["gout"], active=True, **kw)
    session.add(e)
    await session.flush()
    assert await embed_entry(llm, e)
    return e


async def test_build_rules_saves_rules_with_embeddings(session):
    llm = FakeLLM()
    (doc, *_), _ = await seed(session)
    e = await general(session, llm, doc, COLCHICINE)
    built = await rules.build_rules(session, llm, e)
    assert e.rules_status == "ready"
    assert [r.quote for r in built] == [
        "Take one extra 0.6 mg tablet after 1 hour, then no more that day.",
        "Stop and call the clinic if you get diarrhea.",
    ]
    assert all(r.embedding and r.embedding_model == llm.embedding_model for r in built)
    assert [r.position for r in built] == [0, 1]
    # Rebuilding replaces, never appends.
    await rules.build_rules(session, llm, e)
    assert await session.scalar(select(func.count()).select_from(GuidanceRule)) == 2


async def test_build_rules_failure_marks_failed_and_clears(session):
    llm = FakeLLM()
    (doc, *_), _ = await seed(session)
    e = await general(session, llm, doc, COLCHICINE)
    await rules.build_rules(session, llm, e)
    for reply in ({"content": "not json"}, rules_reply(("Made up.", "not in the answer"))):
        llm.route(rules.RULES_PROMPT, lambda m, r=reply: r)
        assert await rules.build_rules(session, llm, e) == []
        assert e.rules_status == "failed"
        assert await session.scalar(select(func.count()).select_from(GuidanceRule)) == 0

    def down(messages):
        raise LLMError("down")
    llm.route(rules.RULES_PROMPT, down)
    assert await rules.build_rules(session, llm, e) == []
    assert e.rules_status == "failed"


# ---------- retrieval ----------

async def test_retrieve_scores_rules_and_falls_back_to_whole_entries(session):
    llm = FakeLLM()
    (doc, a, *_), d = await seed(session)
    long = await general(session, llm, doc,
                         "Drink 2 liters of water every day. Take naproxen with food. "
                         "Walk for 20 minutes after dinner.", drugs=("naproxen",))
    await rules.build_rules(session, llm, long)
    whole = await general(session, llm, doc, COLCHICINE, rules_status="failed")
    hits = await retrieve(session, llm, a, d, [], "naproxen food", floor=0.1, margin=1.0, doctor_id=doc.id)
    top, _ = hits[0]
    assert isinstance(top, RuleHit) and top.rule.quote == "Take naproxen with food."
    assert top.entry.id == long.id
    hits = await retrieve(session, llm, a, d, [], "colchicine diarrhea", floor=0.1, margin=1.0, doctor_id=doc.id)
    assert any(unit is whole for unit, _ in hits)


async def test_retrieve_caps_rules_per_entry(session):
    llm = FakeLLM()
    (doc, a, *_), d = await seed(session)
    e = await general(session, llm, doc, " ".join(
        f"Colchicine note {i} for gout." for i in range(6)))
    await rules.build_rules(session, llm, e)
    hits = await retrieve(session, llm, a, d, [], "colchicine note gout", floor=0.0, margin=1.0, doctor_id=doc.id)
    assert len(hits) == 3 and all(isinstance(u, RuleHit) for u, _ in hits)


async def test_rules_never_bypass_the_gate(session):
    llm = FakeLLM()
    (doc, a, *_), d = await seed(session)
    e = await general(session, llm, doc, "Warfarin: avoid kale.", drugs=("warfarin",))
    await rules.build_rules(session, llm, e)
    assert await retrieve(session, llm, a, d, [], "warfarin kale", floor=0.0, margin=1.0, doctor_id=doc.id) == []


async def test_stale_rule_embeddings_fall_back_to_the_entry(session):
    llm = FakeLLM()
    (doc, a, *_), d = await seed(session)
    e = await general(session, llm, doc, COLCHICINE)
    await rules.build_rules(session, llm, e)
    for r in await session.scalars(select(GuidanceRule)):
        r.embedding_model = "an-older-model"
    hits = await retrieve(session, llm, a, d, [], "colchicine", floor=0.0, margin=1.0, doctor_id=doc.id)
    assert [unit for unit, _ in hits] == [e]


# ---------- the agent ----------

def test_agent_numbers_each_rule_and_grounds_on_the_whole_advice():
    document = DischargeDocument(user_id=1, text="Paragraph one.")
    e = DoctorGuidance(id=7, doctor_id=1, question="Flare?", answer=COLCHICINE, scope="general",
                       drug_tags=[], condition_tags=[])
    first = GuidanceRule(id=1, guidance_id=7, doctor_id=1, position=0,
                         statement="In a gout flare, one extra 0.6 mg colchicine after 1 hour.",
                         quote="Take one extra 0.6 mg tablet after 1 hour, then no more that day.")
    second = GuidanceRule(id=2, guidance_id=7, doctor_id=1, position=1,
                          statement="Stop colchicine if you get diarrhea.",
                          quote="Stop and call the clinic if you get diarrhea.")
    turn = agent._Turn(None, None, document, doctor_name="Dr. Ting Kai")
    a = turn.doctor_entry(RuleHit(e, first))
    b = turn.doctor_entry(RuleHit(e, second))
    assert a["evidence_number"] != b["evidence_number"]
    assert turn.doctor_entry(RuleHit(e, first))["evidence_number"] == a["evidence_number"]
    assert a["rule"] == first.statement and a["my_words"] == first.quote and "advice" not in a
    assert a["full_advice"] == COLCHICINE
    ev = turn.evidence[a["evidence_number"]]
    # The whole advice is the evidence: its warnings can be given with the rule.
    assert ev.kind == "doctor" and ev.guidance_id == 7 and ev.content == COLCHICINE
    assert ev.title.startswith("Dr. Ting Kai's advice")
    # Numbers come from the doctor's own words only: 1.2 is not grounded, 0.6 is.
    answer = json.dumps({"kind": "answer", "answer": f"Take 0.6 mg after 1 hour [{a['evidence_number']}].",
                         "citations": [a["evidence_number"]], "care_team_question": None})
    assert agent.validate(answer, turn.evidence)[0] == "answer"
    bad = answer.replace("0.6 mg", "1.2 mg")
    try:
        agent.validate(bad, turn.evidence)
        raise AssertionError("expected a grounding error")
    except ValueError as exc:
        assert "1.2" in str(exc)
    context = agent._prefetch_context([a])
    assert "Rule: In a gout flare" in context and "My words: Take one extra" in context and f"Full advice it comes from: {COLCHICINE}" in context


# ---------- conflicts ----------

async def test_conflict_with_the_doctors_earlier_rule(session):
    llm = FakeLLM()
    (doc, *_), _ = await seed(session)
    old = await general(session, llm, doc, "Colchicine in a flare: one extra 0.6 mg tablet after 1 hour.")
    await rules.build_rules(session, llm, old)
    new = await general(session, llm, doc, "Colchicine in a flare: repeat 0.6 mg every 2 hours.")
    built = await rules.build_rules(session, llm, new)
    seen = []

    def judge(messages):
        seen.append(messages[1]["content"])
        return conflicts_reply(("N1", "C1", "Earlier advice allows one extra tablet, this repeats it."),
                               ("N9", "C1", "out of range"), ("N1", "C1", "duplicate"))
    llm.route(rules.CONFLICT_PROMPT, judge)
    saved = await rules.find_conflicts(session, llm, None, new, built)
    assert len(saved) == 1
    assert saved[0].other_rule_id is not None and saved[0].status == "open"
    assert "N1. Colchicine in a flare: repeat" in seen[0] and "C1. (the doctor's earlier advice)" in seen[0]


async def test_no_candidates_means_no_model_call(session):
    llm = FakeLLM()
    (doc, *_), _ = await seed(session)
    e = await general(session, llm, doc, COLCHICINE)
    built = await rules.build_rules(session, llm, e)
    before = len(llm.routed_calls)
    assert await rules.find_conflicts(session, llm, None, e, built) == []
    assert len(llm.routed_calls) == before


async def test_other_doctors_and_unmeetable_entries_are_never_candidates(session):
    llm = FakeLLM()
    (doc, a, b, doc2), d = await seed(session)
    text = "Colchicine in a flare: one extra 0.6 mg tablet after 1 hour."
    theirs = await general(session, llm, doc2, text)
    await rules.build_rules(session, llm, theirs)
    for_b = DoctorGuidance(doctor_id=doc.id, question="q", answer=text, scope="patient",
                           patient_id=b.id, drug_tags=[], condition_tags=[], active=True)
    session.add(for_b)
    await session.flush()
    await embed_entry(llm, for_b)
    await rules.build_rules(session, llm, for_b)
    for_a = DoctorGuidance(doctor_id=doc.id, question="q", answer=text, scope="patient",
                           patient_id=a.id, drug_tags=[], condition_tags=[], active=True)
    session.add(for_a)
    await session.flush()
    await embed_entry(llm, for_a)
    built = await rules.build_rules(session, llm, for_a)
    assert await rules.find_conflicts(session, llm, None, for_a, built) == []


class LabelStore:
    def __init__(self, passages):
        self.passages = passages
        self.calls = []

    async def retrieve(self, query, limit=5, rxcuis=()):
        self.calls.append(set(rxcuis))
        return self.passages


async def test_conflict_with_an_approved_drug_label(session):
    llm = FakeLLM()
    (doc, *_), _ = await seed(session)
    session.add_all([
        DrugLabel(rxcui="2683", ingredient="colchicine", status="approved", candidates=[],
                  title="COLCRYS label", url="https://dailymed.example/colcrys"),
        DrugLabel(rxcui="7258", ingredient="naproxen", status="pending", candidates=[]),
    ])
    await session.flush()
    e = await general(session, llm, doc, "Colchicine: you can take 3 tablets at once in a flare.",
                      drugs=("colchicine / Colcrys",))
    built = await rules.build_rules(session, llm, e)
    store = LabelStore([
        {"source_id": "dailymed-rxcui-2683", "title": "COLCRYS label", "url": None,
         "content": "The maximum dose is 1.8 mg over a one-hour period."},
        {"source_id": "medlineplus-gout", "title": "Gout", "url": None, "content": "Not a label."},
    ])
    llm.route(rules.CONFLICT_PROMPT, lambda m: conflicts_reply(("N1", "C1", "Label caps the dose.")))
    (saved,) = await rules.find_conflicts(session, llm, store, e, built)
    assert store.calls == [{"2683"}]
    assert saved.other_rule_id is None and saved.label_rxcui == "2683"
    assert saved.label_quote.startswith("The maximum dose") and saved.label_title == "COLCRYS label"
    assert saved.label_url == "https://dailymed.example/colcrys"


async def test_rebuilding_drops_conflicts_on_both_sides(session):
    llm = FakeLLM()
    (doc, *_), _ = await seed(session)
    old = await general(session, llm, doc, "Colchicine in a flare: one extra 0.6 mg tablet after 1 hour.")
    await rules.build_rules(session, llm, old)
    new = await general(session, llm, doc, "Colchicine in a flare: repeat 0.6 mg every 2 hours.")
    built = await rules.build_rules(session, llm, new)
    llm.route(rules.CONFLICT_PROMPT, lambda m: conflicts_reply(("N1", "C1", "Differs.")))
    await rules.find_conflicts(session, llm, None, new, built)
    await rules.build_rules(session, llm, old)  # the other side changes
    assert await session.scalar(select(func.count()).select_from(GuidanceConflict)) == 0


# ---------- routes ----------

async def test_general_answer_shows_rules_and_a_conflict_on_both_entries(client, fake_llm):
    h, _doc, q1 = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    first = await client.post(f"/api/doctor/questions/{q1}/answer", headers=dh, json=answer_body(
        general=True, answer="Colchicine in a flare: one extra 0.6 mg tablet after 1 hour.",
        drug_tags=["colchicine"]))
    g1 = first.json()["guidance"]
    assert g1["rules_status"] == "ready" and len(g1["rules"]) == 1 and g1["conflicts"] == []
    _h, _d, q2 = await asked(client, fake_llm, username="bob")
    fake_llm.route(rules.CONFLICT_PROMPT, lambda m: conflicts_reply(
        ("N1", "C1", "One says one extra tablet, the other says repeat every 2 hours.")))
    second = await client.post(f"/api/doctor/questions/{q2}/answer", headers=dh, json=answer_body(
        general=True, answer="Colchicine in a flare: repeat 0.6 mg every 2 hours.",
        drug_tags=["colchicine"]))
    g2 = second.json()["guidance"]
    (c,) = g2["conflicts"]
    assert c["other"]["kind"] == "advice" and c["other"]["guidance_id"] == g1["id"]
    assert c["rule"]["quote"].startswith("Colchicine in a flare: repeat")
    listed = {e["id"]: e for e in (await client.get("/api/doctor/guidance", headers=dh)).json()["entries"]}
    (back,) = listed[g1["id"]]["conflicts"]
    assert back["id"] == c["id"] and back["other"]["guidance_id"] == g2["id"]

    # Retiring one side hides the conflict on the other.
    await client.patch(f"/api/doctor/guidance/{g1['id']}", headers=dh, json={"active": False})
    listed = {e["id"]: e for e in (await client.get("/api/doctor/guidance", headers=dh)).json()["entries"]}
    assert listed[g2["id"]]["conflicts"] == []
    await client.patch(f"/api/doctor/guidance/{g1['id']}", headers=dh, json={"active": True})

    other = await doctor_headers(client, "dr.renny", "Dr. Renny")
    assert (await client.post(f"/api/doctor/conflicts/{c['id']}/dismiss", headers=other)).status_code == 404
    assert (await client.post(f"/api/doctor/conflicts/{c['id']}/dismiss", headers=h)).status_code == 403
    resp = await client.post(f"/api/doctor/conflicts/{c['id']}/dismiss", headers=dh)
    assert resp.json() == {"id": c["id"], "status": "dismissed"}
    listed = {e["id"]: e for e in (await client.get("/api/doctor/guidance", headers=dh)).json()["entries"]}
    assert listed[g1["id"]]["conflicts"] == [] and listed[g2["id"]]["conflicts"] == []


async def test_editing_the_answer_rebuilds_rules_and_retry_reads_them(client, fake_llm):
    _h, _doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    gid = (await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh,
                             json=answer_body())).json()["guidance"]["id"]
    fake_llm.route(rules.RULES_PROMPT, lambda m: {"content": "garbled"})
    resp = await client.patch(f"/api/doctor/guidance/{gid}", headers=dh,
                              json={"answer": "Eat a snack first. Then take it."})
    body = resp.json()
    assert body["rules_status"] == "failed" and body["rules"] == [] and body["searchable"]
    fake_llm.route(rules.RULES_PROMPT, sentence_rules)
    body = (await client.post(f"/api/doctor/guidance/{gid}/embed", headers=dh)).json()
    assert body["rules_status"] == "ready"
    assert [r["quote"] for r in body["rules"]] == ["Eat a snack first.", "Then take it."]


async def test_without_a_model_edits_clear_rules(client, fake_llm):
    _h, _doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    gid = (await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh,
                             json=answer_body())).json()["guidance"]["id"]
    client.app.state.llm = None
    body = (await client.patch(f"/api/doctor/guidance/{gid}", headers=dh,
                               json={"answer": "New words."})).json()
    assert body["rules_status"] is None and body["rules"] == []


def evidence_number_of(first_message: str, words: str) -> int:
    """The evidence number of the prefetched rule whose "My words" are `words`."""
    for block in first_message.split("\n\n"):
        if f"My words: {words}" in block:
            head = block[block.index("["):]
            return int(head[1:head.index("]")])
    raise AssertionError(f"no prefetched rule with {words!r}")


async def test_ask_cites_a_rule_and_shows_the_doctors_words(client, fake_llm):
    h, _doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body(
        answer="Drink 2 liters of water every day. Take your pill with food."))
    assert (await client.post("/api/documents/active/confirm", headers=h)).status_code == 200
    async with client.app.state.session_factory() as s:
        doctor_id = await s.scalar(select(User.id).where(User.username == "dr.ting"))
    seen = {}

    def clone(messages):
        first = messages[-1]["content"]
        seen["first"] = first
        n = evidence_number_of(first, "Take your pill with food.")
        return {"content": json.dumps({"kind": "answer", "answer": f"Take it with food [{n}].",
                                       "citations": [n], "care_team_question": None})}
    fake_llm.route(agent.system_prompt("Dr. Ting Kai"), clone)
    resp = await client.post("/api/ask", headers=h, json={
        "question": "Should I take my pill with food?", "doctor_id": doctor_id})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["kind"] == "answer"
    assert "Rule: Take your pill with food." in seen["first"]
    (source,) = data["sources"]
    assert source["kind"] == "doctor"
    assert source["content"] == "Drink 2 liters of water every day. Take your pill with food."
    assert data["guidance_ids"] == [source["guidance_id"]]
