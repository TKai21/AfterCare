import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

import drafts
from conftest import doctor_headers
from llm import LLMError
from models import AnswerDraft
from test_routes_doctor import answer_body, asked


def draft_reply(*sentences):
    return {"content": json.dumps({"sentences": [{"text": t, "sources": s} for t, s in sentences]})}


def ctx_of(*quotes):
    ctx = drafts.DraftContext()
    for q in quotes:
        ctx.add("document", "Discharge document", q)
    return ctx


def test_validate_flags_missing_sources_and_ungrounded_numbers():
    ctx = ctx_of("Take naproxen 500 mg with food.", "Drink 2 liters of water a day.")
    got = drafts.validate_sentences({"sentences": [
        {"text": "Take naproxen  500 mg with food.", "sources": [1]},
        {"text": "Take 750 mg instead.", "sources": [1]},
        {"text": "Rest your foot.", "sources": []},
        {"text": "Drink 2 liters.", "sources": [2, 99, True, "1"]},
        {"text": "   ", "sources": [1]},
        "junk",
    ]}, ctx)
    assert [(s["text"], s["supported"], s["problem"], s["sources"]) for s in got] == [
        ("Take naproxen 500 mg with food.", True, None, [1]),
        ("Take 750 mg instead.", False, "number", [1]),
        ("Rest your foot.", False, "no_source", []),
        ("Drink 2 liters.", True, None, [2]),
    ]
    with pytest.raises(ValueError):
        drafts.validate_sentences({"sentences": []}, ctx)
    with pytest.raises(ValueError):
        drafts.validate_sentences({"nope": 1}, ctx)


def test_kept_ratio_and_same_text():
    assert drafts.kept_ratio("Take it with food.", "Take it with food.") == 1.0
    assert drafts.kept_ratio("Take it with food.", "Never.") == 0.0
    assert 0 < drafts.kept_ratio("Take it with food every time.", "Take it with food.") < 1
    assert drafts.same_text("a  b\nc", "a b c") and not drafts.same_text("a", "b")


async def doctor_id_of(client, username="dr.ting"):
    from models import User
    async with client.app.state.session_factory() as s:
        return await s.scalar(select(User.id).where(User.username == username))


async def test_draft_lifecycle_send_and_stats(client, fake_llm):
    _h, _doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    seen = []

    def drafting(messages):
        seen.append(messages)
        return draft_reply(("Take it with breakfast.", [1]), ("Call me if it hurts.", []))
    fake_llm.route(drafts.DRAFT_PROMPT.replace("{DOCTOR}", "Dr. Ting Kai"), drafting)
    resp = await client.post(f"/api/doctor/questions/{qid}/draft", headers=dh)
    assert resp.status_code == 200, resp.text
    first = resp.json()
    assert first["text"] == "Take it with breakfast. Call me if it hurts."
    assert [s["supported"] for s in first["sentences"]] == [True, False]
    assert [e["number"] for e in first["evidence"]] == [1]  # only cited evidence is kept
    assert "Discharge document" in first["evidence"][0]["title"]
    assert "Patient question:" in seen[0][1]["content"]

    # Redrafting replaces the unsent draft; the queue shows the newest.
    second = (await client.post(f"/api/doctor/questions/{qid}/draft", headers=dh)).json()
    assert second["id"] != first["id"]
    (open_q,) = (await client.get("/api/doctor/questions", headers=dh)).json()["open"]
    assert open_q["draft"]["id"] == second["id"]
    async with client.app.state.session_factory() as s:
        assert len(list(await s.scalars(select(AnswerDraft)))) == 1

    # A draft id of another question is ignored; the right one is recorded.
    resp = await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body(
        answer="Take it with breakfast. Call the clinic if it hurts.", draft_id=second["id"]))
    assert resp.status_code == 200
    async with client.app.state.session_factory() as s:
        d = await s.get(AnswerDraft, second["id"])
        assert d.sent_answer.startswith("Take it with breakfast.") and d.sent_at is not None
        assert 0.5 < d.kept < 1 and d.sent_unchanged is False

    st = (await client.get("/api/doctor/draft-stats", headers=dh)).json()
    assert st["sent"] == 1 and st["unchanged"] == 0 and st["mean_kept"] == d.kept
    assert len(st["weeks"]) == drafts.STATS_WEEKS and st["weeks"][-1]["sent"] == 1

    # Answered: 409. The corrected draft becomes an example for the next draft.
    assert (await client.post(f"/api/doctor/questions/{qid}/draft", headers=dh)).status_code == 409
    _h2, _d2, q2 = await asked(client, fake_llm, username="bob")
    seen.clear()
    await client.post(f"/api/doctor/questions/{q2}/draft", headers=dh)
    assert "What I sent instead: Take it with breakfast. Call the clinic" in seen[0][1]["content"]


async def test_draft_errors(client, fake_llm):
    _h, _doc, qid = await asked(client, fake_llm)
    _h2, _d2, q2 = await asked(client, fake_llm, username="bob")
    dh = await doctor_headers(client)
    prompt = drafts.DRAFT_PROMPT.replace("{DOCTOR}", "Dr. Ting Kai")
    fake_llm.route(prompt, lambda m: {"content": "not json"})
    assert (await client.post(f"/api/doctor/questions/{qid}/draft", headers=dh)).status_code == 502
    fake_llm.route(prompt, lambda m: draft_reply())
    assert (await client.post(f"/api/doctor/questions/{qid}/draft", headers=dh)).status_code == 502

    def down(m):
        raise LLMError("down")
    fake_llm.route(prompt, down)
    assert (await client.post(f"/api/doctor/questions/{qid}/draft", headers=dh)).status_code == 502
    assert (await client.post("/api/doctor/questions/999/draft", headers=dh)).status_code == 404
    other = await doctor_headers(client, "dr.renny", "Dr. Renny")
    # Unassigned questions may be drafted by any doctor; assign it first.
    fake_llm.route(prompt, lambda m: draft_reply(("Fine.", [])))
    await client.post(f"/api/doctor/questions/{qid}/answer", headers=dh, json=answer_body())
    assert (await client.post(f"/api/doctor/questions/{qid}/draft", headers=other)).status_code in (404, 409)
    client.app.state.llm = None
    assert (await client.post(f"/api/doctor/questions/{q2}/draft", headers=dh)).status_code == 503
    async with client.app.state.session_factory() as s:
        assert list(await s.scalars(select(AnswerDraft))) == []


async def test_draft_uses_only_guidance_the_patient_may_see(client, fake_llm):
    # Dr. Ting's general advice about warfarin never reaches a draft for a
    # patient whose document does not mention warfarin.
    _h, _doc, q1 = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    await client.post(f"/api/doctor/questions/{q1}/answer", headers=dh, json=answer_body(
        general=True, answer="Warfarin: keep your kale steady.", drug_tags=["warfarin"]))
    _h2, _d2, q2 = await asked(client, fake_llm, username="bob")
    seen = []
    fake_llm.route(drafts.DRAFT_PROMPT.replace("{DOCTOR}", "Dr. Ting Kai"),
                   lambda m: seen.append(m) or draft_reply(("Fine.", [])))
    await client.post(f"/api/doctor/questions/{q2}/draft", headers=dh)
    assert "kale" not in seen[0][1]["content"]


async def test_stats_bucket_by_week(client, fake_llm):
    dh = await doctor_headers(client)
    doctor_id = await doctor_id_of(client)
    _h, _doc, qid = await asked(client, fake_llm)
    now = datetime(2026, 11, 14, 12, 0)  # a Saturday
    async with client.app.state.session_factory() as s:
        s.add_all([
            AnswerDraft(question_id=qid, doctor_id=doctor_id, sentences=[], evidence=[], text="a",
                        sent_answer="a", kept=1.0, sent_unchanged=True, sent_at=now - timedelta(days=1)),
            AnswerDraft(question_id=qid, doctor_id=doctor_id, sentences=[], evidence=[], text="a b",
                        sent_answer="a", kept=0.5, sent_unchanged=False, sent_at=now - timedelta(days=8)),
            AnswerDraft(question_id=qid, doctor_id=doctor_id, sentences=[], evidence=[], text="unsent"),
        ])
        await s.commit()
        st = await drafts.stats(s, doctor_id, now=now)
    assert st["sent"] == 2 and st["unchanged"] == 1 and st["mean_kept"] == 0.75
    assert st["weeks"][-1] == {"week_start": "2026-11-09", "sent": 1, "mean_kept": 1.0}
    assert st["weeks"][-2] == {"week_start": "2026-11-02", "sent": 1, "mean_kept": 0.5}
    assert (await client.get("/api/doctor/draft-stats", headers=dh)).status_code == 200


async def test_cited_evidence_is_renumbered_from_one(client, fake_llm):
    _h, _doc, qid = await asked(client, fake_llm)
    dh = await doctor_headers(client)
    fake_llm.route(drafts.DRAFT_PROMPT.replace("{DOCTOR}", "Dr. Ting Kai"),
                   lambda m: draft_reply(("A.", [3]), ("B.", [1, 3])))
    d = (await client.post(f"/api/doctor/questions/{qid}/draft", headers=dh)).json()
    assert [s["sources"] for s in d["sentences"]] == [[1], [1, 2]]
    assert [e["number"] for e in d["evidence"]] == [1, 2]
