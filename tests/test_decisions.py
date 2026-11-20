"""Decision memory: cases kept as (situation, decision) and shown as similar past cases."""
import asyncio
import json

import pytest
from sqlalchemy import select

import decisions
from conftest import doctor_headers
from decisions import BOOK_VISIT, SEEN_SOON, WATCH, Neighbour
from fake_llm import FakeLLM
from llm import LLMError
from models import CaseDecision, DoctorGuidance, User
from test_routes_doctor import answer_body, asked


def reads(kind=BOOK_VISIT, asks=False, medication=False):
    return lambda messages: {"content": json.dumps(
        {"kind": kind, "asks": asks, "medication": medication})}


async def rows(client):
    async with client.app.state.session_factory() as s:
        return list(await s.scalars(select(CaseDecision).order_by(CaseDecision.id)))


# ---------- reading the decision ----------

@pytest.mark.parametrize("data, expected", [
    ({"kind": "seen soon", "asks": True, "medication": False},
     decisions.Decision("seen soon", True, False)),
    ({"kind": "come in", "asks": True, "medication": False}, None),  # not a kind
    ({"kind": "watch", "asks": "yes", "medication": False}, None),
    ({"kind": "watch", "asks": True}, None),
    (["watch"], None),
])
def test_parse_decision(data, expected):
    assert decisions.parse_decision(data) == expected


def test_a_failed_read_is_no_decision():
    llm = FakeLLM()

    def fail(messages):
        raise LLMError("down")
    llm.route(decisions.KIND_PROMPT, fail)
    assert asyncio.run(decisions.read_decision(llm, "q", "a")) is None
    llm.route(decisions.KIND_PROMPT, lambda m: {"content": "not json"})
    assert asyncio.run(decisions.read_decision(llm, "q", "a")) is None


def test_the_reader_is_given_both_texts():
    llm = FakeLLM()
    llm.route(decisions.KIND_PROMPT, reads(WATCH))
    got = asyncio.run(decisions.read_decision(llm, "My toe hurts.", "Rest it."))
    assert got == decisions.Decision(WATCH, False, False)
    sent = llm.routed_calls[0][0][1]["content"]
    assert "My toe hurts." in sent and "Rest it." in sent


# ---------- nearest cases ----------

def test_nearest_orders_by_similarity_and_applies_the_floor_and_the_limit():
    cases = [(1, WATCH, [1.0, 0.0]), (2, BOOK_VISIT, [0.9, 0.1]), (3, SEEN_SOON, [0.0, 1.0]),
             (4, WATCH, [0.0, 0.0]), (5, WATCH, [1.0, 0.0, 0.0])]  # zero and wrong-size: skipped
    found = decisions.nearest([1.0, 0.0], cases, limit=5, floor=0.5)
    assert [n.id for n in found] == [1, 2]
    assert [n.id for n in decisions.nearest([1.0, 0.0], cases, limit=1, floor=0.0)] == [1]
    assert decisions.nearest([0.0, 0.0], cases) == []


def test_proposed_kind_is_weighted_and_a_tie_goes_to_the_nearest():
    near = [Neighbour(1, WATCH, 0.9), Neighbour(2, BOOK_VISIT, 0.5), Neighbour(3, BOOK_VISIT, 0.5)]
    assert decisions.proposed_kind(near) == BOOK_VISIT
    assert decisions.proposed_kind([Neighbour(1, WATCH, 0.5), Neighbour(2, BOOK_VISIT, 0.5)]) == WATCH
    assert decisions.proposed_kind([]) is None


def test_only_a_known_kind_with_a_confirmed_summary_is_evidence():
    def row(**kw):
        return CaseDecision(**{"kind": WATCH, "facts": "", "confirmed": None, **kw})
    assert decisions.is_evidence(row())
    assert decisions.is_evidence(row(facts="Known: x", confirmed=True))
    assert not decisions.is_evidence(row(facts="Known: x", confirmed=None))
    assert not decisions.is_evidence(row(facts="Known: x", confirmed=False))
    assert not decisions.is_evidence(row(kind=None))


def test_replay_counts_what_the_nearest_cases_get_right_next_to_the_baseline():
    a, b = [1.0, 0.0], [0.0, 1.0]
    cases = [(1, WATCH, a), (2, WATCH, a), (3, WATCH, a), (4, BOOK_VISIT, b), (5, BOOK_VISIT, b),
             (6, SEEN_SOON, [0.7, 0.7])]
    got = decisions.replay(cases, floor=0.9)
    assert got == {"cases": 6, "proposed": 5, "right": 5, "baseline": 3,
                   "per_kind": {WATCH: {"right": 3, "cases": 3},
                                BOOK_VISIT: {"right": 2, "cases": 2},
                                SEEN_SOON: {"right": 0, "cases": 1}}}
    assert decisions.replay([]) == {"cases": 0, "proposed": 0, "right": 0, "baseline": 0,
                                    "per_kind": {}}


@pytest.mark.parametrize("right, expected", [(28, False), (29, False), (35, False), (36, True)])
def test_a_replay_counts_as_learned_only_clearly_above_the_baseline(right, expected):
    assert decisions.learned({"cases": 60, "right": right, "baseline": 28}) is expected
    assert decisions.learned({"cases": 0, "right": 0, "baseline": 0}) is False


# ---------- through the app ----------

async def test_an_answered_case_is_kept_as_a_pair(client, fake_llm):
    _h, _doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    fake_llm.route(decisions.KIND_PROMPT, reads(SEEN_SOON, asks=True))
    resp = await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh,
                             json=answer_body(answer="Any fever? Please be seen today."))
    assert resp.status_code == 200, resp.text
    assert resp.json()["question"]["decision"]["kind"] == SEEN_SOON
    assert resp.json()["question"]["decision"]["label"] == "Be seen as soon as possible"
    (row,) = await rows(client)
    assert (row.question_id, row.kind, row.asks, row.medication, row.made) == (
        qid, SEEN_SOON, True, False, "written")
    assert row.reply == "Any fever? Please be seen today." and row.complaint
    assert row.embedding and row.embedding_model == fake_llm.embedding_model
    # The answered list shows the decision.
    answered = (await client.get("/api/doctor/questions", headers=dh)).json()["answered"]
    assert answered[0]["decision"]["kind"] == SEEN_SOON


async def test_a_case_that_cannot_be_read_is_kept_without_a_kind_and_the_doctor_names_it(
        client, fake_llm):
    _h, _doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    fake_llm.route(decisions.KIND_PROMPT, lambda m: {"content": "{}"})
    resp = await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body())
    assert resp.status_code == 200 and resp.json()["question"]["decision"]["kind"] is None
    (row,) = await rows(client)
    assert row.kind is None and not decisions.is_evidence(row)
    resp = await client.patch(f"/api/doctor/decisions/{row.id}", headers=dh, json={"kind": WATCH})
    assert resp.status_code == 200 and resp.json() == {
        "id": row.id, "kind": WATCH, "label": "Self-care, come back if not better",
        "asks": False, "medication": False, "made": "written", "by_doctor": True}
    assert (await client.patch(f"/api/doctor/decisions/{row.id}", headers=dh,
                               json={"kind": "come in"})).status_code == 422
    other = await doctor_headers(client, "dr.other", "Dr. Other")
    assert (await client.patch(f"/api/doctor/decisions/{row.id}", headers=other,
                               json={"kind": WATCH})).status_code == 404


async def test_an_open_case_shows_the_doctors_similar_past_cases(client, fake_llm):
    _h, _doc, first = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    fake_llm.route(decisions.KIND_PROMPT, reads(BOOK_VISIT))
    await client.post(f"/api/doctor/questions/{first}/answer", headers=dh,
                      json=answer_body(answer="Please call the office to schedule a visit."))
    # A second patient with the same open question.
    _h2, _doc2, second = await asked(client, fake_llm, username="bob")
    opened = (await client.get("/api/doctor/questions", headers=dh)).json()["open"]
    (card,) = [q for q in opened if q["id"] == second]
    similar = card["similar"]
    assert similar["counts"] == [{"kind": BOOK_VISIT, "label": "Book a visit", "count": 1}]
    (case,) = similar["cases"]
    assert case["reply"] == "Please call the office to schedule a visit."
    assert case["kind"] == BOOK_VISIT and case["history"] is False and case["similarity"] > 0.9
    # Another doctor's cases are never shown.
    other = await doctor_headers(client, "dr.other", "Dr. Other")
    opened = (await client.get("/api/doctor/questions", headers=other)).json()["open"]
    assert all(q.get("similar") is None for q in opened)


async def test_a_failing_embedding_shows_no_cases_and_does_not_break_the_inbox(
        client, fake_llm, monkeypatch):
    _h, _doc, first = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    await client.post(f"/api/doctor/questions/{first}/answer", headers=dh, json=answer_body())
    async with client.app.state.session_factory() as s:
        row = await s.scalar(select(CaseDecision))
        row.kind = WATCH
        await s.commit()
    await asked(client, fake_llm, username="bob")

    async def fail(texts):
        raise LLMError("down")
    monkeypatch.setattr(client.app.state.llm, "embed", fail)
    resp = await client.get("/api/doctor/questions", headers=dh)
    assert resp.status_code == 200 and resp.json()["open"][0]["similar"] is None


async def test_history_is_read_once_into_case_rows(client, fake_llm):
    _h, _doc, qid = await asked(client, fake_llm)
    await doctor_headers(client)
    async with client.app.state.session_factory() as s:
        doctor_id = await s.scalar(select(User.id).where(User.username == "dr.ting"))
        s.add_all([DoctorGuidance(doctor_id=doctor_id, question=f"My knee hurts {n}.",
                                  answer="Please call the office to schedule an appointment.",
                                  scope="general") for n in range(3)])
        s.add(DoctorGuidance(doctor_id=doctor_id, question="Old.", answer="x", scope="general",
                             active=False))
        s.add(DoctorGuidance(doctor_id=doctor_id, question="From a case.", answer="Rest.",
                             scope="patient", source_question_id=qid))
        await s.commit()
        llm = client.app.state.llm
        fake_llm.route(decisions.KIND_PROMPT, reads(BOOK_VISIT, asks=True))
        assert await decisions.backfill_history(s, llm, doctor_id, limit=2) == 2
        assert await decisions.backfill_history(s, llm, doctor_id) == 2
        assert await decisions.backfill_history(s, llm, doctor_id) == 0
    got = await rows(client)
    assert [(r.made, r.kind, r.asks, r.facts, r.confirmed) for r in got] == [
        ("history", BOOK_VISIT, True, "", None)] * 4
    assert all(r.guidance_id and r.embedding for r in got)
    # The row of advice that answered a case keeps that case.
    assert [r.question_id for r in got] == [None, None, None, qid]
