"""The five-stage interview procedure (spec 2026-11-19)."""
import asyncio
import json
from dataclasses import replace

import pytest

import interview as iv_mod
from fake_llm import FakeLLM
from interview import FOLLOW_UP, MEDICATION, PROBLEM, SAFETY, Interview, Item
from test_routes_agent import (
    CLOSE_TEXT, DISCHARGE_Q, FEVER_Q, ORIGINAL, REDNESS_Q, Interviewer, absent, all_questions,
    confirmed, final, interview_doctor, judged, post_ask, start_interview,
)


class Habit:
    def __init__(self, id, quote, terms=()):
        self.id, self.quote, self.warning_terms = id, quote, list(terms)


def build_llm(reply):
    llm = FakeLLM()
    llm.route(iv_mod.BUILD_PROMPT, lambda m: {"content": json.dumps(reply)})
    return llm


def build(reply, asks, warns=(), medications=("amoxicillin",)):
    llm = build_llm(reply)
    got = asyncio.run(iv_mod.build_items(llm, "My incision is red.", list(asks), list(warns), [],
                                         [], max_items=5, max_safety=3,
                                         procedure=list(medications)))
    return got, llm


# ---------- items ----------

@pytest.mark.parametrize("names, expected", [
    ([], ""),
    (["colchicine"], "Have you been taking colchicine as your discharge instructions say?"),
    (["colchicine", "naproxen"],
     "Have you been taking colchicine and naproxen as your discharge instructions say?"),
    (["a", "b", "c"], "Have you been taking a, b and c as your discharge instructions say?"),
    (["a", "b", "c", "d"], "Have you been taking your medicines as your discharge instructions say?"),
    (["colchicine", " colchicine "], "Have you been taking colchicine as your discharge instructions say?"),
])
def test_medicines_question(names, expected):
    assert iv_mod._medicines_question(names) == expected


def test_core_items_and_their_stages():
    items = iv_mod.core_items({"duration": [7]}, ["colchicine"])
    assert [(i.core, i.stage) for i in items] == [
        ("duration", PROBLEM), ("severity", PROBLEM), ("course", PROBLEM),
        ("medicines", MEDICATION), ("other_medicines", MEDICATION)]
    assert items[0].habits == [7] and items[1].habits == []
    assert all(i.fixed.endswith("?") and not i.safety for i in items)
    # No medication in the document: nothing to ask about taking it.
    assert [i.core for i in iv_mod.core_items({}, [])][3:] == ["other_medicines"]
    # Medicines do not matter to this complaint: still whether they are taken as
    # directed, but not what else was taken.
    assert [i.core for i in iv_mod.core_items({}, ["colchicine"], ask_medicines=False)] == [
        "duration", "severity", "course", "medicines"]
    assert [i.core for i in iv_mod.core_items({}, [], ask_medicines=False)] == [
        "duration", "severity", "course"]


def test_build_maps_habits_to_core_items():
    asks = [Habit(1, REDNESS_Q), Habit(2, DISCHARGE_Q), Habit(3, "What have you tried?")]
    warns = [Habit(9, "Tell me about any fever.", ["fever"])]
    (symptom, items), llm = build({
        "reports_symptom": True,
        "items": [{"label": "how long the redness lasted", "habits": [1]},
                  {"label": "incision discharge", "habits": [2]}],
        "core": {"duration": [1], "other_medicines": [3, 99, "x"], "severity": "no"},
        "ask_medicines": True, "relevant_document_terms": []}, asks, warns)
    assert symptom is True
    by_core = {i.core: i for i in items if i.core}
    assert by_core["duration"].habits == [1] and by_core["other_medicines"].habits == [3]
    assert by_core["severity"].habits == []
    # The redness item only had the duration habit: it is the core item now.
    assert [i.label for i in items if not i.core and not i.safety] == ["incision discharge"]
    assert [(i.label, i.stage) for i in items if i.safety] == [("fever", SAFETY)]
    assert iv_mod.CORE_REQUEST in llm.routed_calls[0][0][1]["content"]


def test_build_without_the_procedure_is_unchanged():
    llm = build_llm({"reports_symptom": True,
                     "items": [{"label": "incision discharge", "habits": [2]}],
                     "core": {"duration": [2]}})
    _, items = asyncio.run(iv_mod.build_items(llm, "x", [Habit(2, DISCHARGE_Q)], [], [], [],
                                              max_items=5, max_safety=3))
    assert [(i.label, i.core, i.habits) for i in items] == [("incision discharge", "", [2])]
    assert iv_mod.CORE_REQUEST not in llm.routed_calls[0][0][1]["content"]


def test_a_doctor_with_no_habits_still_gets_the_core_items():
    (_, items), _ = build({"reports_symptom": True, "items": [], "ask_medicines": True}, [])
    assert [i.core for i in items] == list(iv_mod.CORE)
    # What else was taken is asked only when the build call says medicines matter here.
    for flag in ({}, {"ask_medicines": False}, {"ask_medicines": "yes"}):
        (_, items), _ = build({"reports_symptom": True, "items": [], **flag}, [])
        assert [i.core for i in items] == ["duration", "severity", "course", "medicines"]


def test_one_danger_sign_is_asked_once_and_worsening_is_left_to_its_own_question():
    warns = [Habit(9, "x", ["symptoms become acutely worse", "symptoms worsen acutely",
                            "neck swelling", "acute worsening of symptoms", "trouble breathing",
                            "shortness of breath", "chest pain", "abdominal pain"])]
    (_, items), _ = build({"reports_symptom": True, "items": []}, [], warns)
    assert [i.label for i in items if i.safety] == ["neck swelling", "trouble breathing",
                                                    "chest pain"]
    # Without the procedure there is no question about worsening: the sign stays, once.
    llm = build_llm({"reports_symptom": True, "items": []})
    _, items = asyncio.run(iv_mod.build_items(llm, "x", [], warns, [], [], max_items=5,
                                              max_safety=3))
    labels = [i.label for i in items]
    assert labels[:2] == ["symptoms become acutely worse", "neck swelling"]
    assert "symptoms worsen acutely" not in labels


@pytest.mark.parametrize("term, question", [
    ("fever", "Have you had any fever?"),
    ("unable to urinate", "Is this happening now: unable to urinate?"),
    ("symptoms are persisting", "Is this happening now: symptoms are persisting?"),
])
def test_safety_question_reads_as_a_sentence(term, question):
    assert iv_mod.safety_question(term, False) == question


@pytest.mark.parametrize("reply, unsure", [
    ("I'm not sure.", True), ("Not sure", True), ("I do not know", True), ("No idea", True),
    ("I dont remember", True), ("No.", False), ("Not sure, why?", False),
    ("Not really.", False), ("I am sure it is mild.", False),
])
def test_a_typed_not_sure(reply, unsure):
    assert bool(iv_mod.NOT_SURE_REPLY.match(reply)) is unsure


def procedure_interview():
    items = [Item("incision discharge", [2], False),
             Item("fever", [], True, stage=SAFETY),
             *iv_mod.core_items({"duration": [1]}, ["amoxicillin"])]
    return Interview(items=items)


def test_items_are_asked_stage_by_stage():
    iv = procedure_interview()
    order = []
    while (n := iv_mod.next_item(iv)) is not None:
        order.append((iv_mod.stage_of(iv.items[n]), iv.items[n].core or iv.items[n].label))
        iv.items[n].status = "answered"
    assert order == [(SAFETY, "fever"), (PROBLEM, "duration"), (PROBLEM, "severity"),
                     (PROBLEM, "course"), (FOLLOW_UP, "incision discharge"),
                     (MEDICATION, "medicines"), (MEDICATION, "other_medicines")]
    assert iv_mod.steps(iv) == [SAFETY, PROBLEM, FOLLOW_UP, MEDICATION]


def test_core_question_wording():
    iv = procedure_interview()
    core = {i.core: i for i in iv.items if i.core}
    ask = lambda item, texts, asked=(), rephrase=False: asyncio.run(iv_mod.write_question(
        None, item, texts, list(asked), [], "Dr. Lee", rephrase))
    # The doctor's own question when a habit asks it, else the fixed one. No model call.
    assert ask(core["duration"], {1: REDNESS_Q}) == (REDNESS_Q, True)
    assert ask(core["severity"], {}) == (iv_mod.CORE["severity"][2], False)
    # The habit was suppressed since: the fixed question.
    assert ask(core["duration"], {}) == (iv_mod.CORE["duration"][2], False)


def test_state_round_trips_and_old_rows_still_load():
    iv = procedure_interview()
    iv.side, iv.confirmed = 2, True
    iv.rounds.append({"asked": "x", "item": 1, "reply": None, "doctor": False})
    again = iv_mod.load_interview(iv_mod.dump_interview(iv))
    assert again == iv and iv_mod.is_procedure(again)
    old = iv_mod.load_interview({"items": [{"label": "fever", "habits": [], "safety": True},
                                           {"label": "cough", "habits": [3], "safety": False}],
                                 "rounds": []})
    assert [(i.stage, i.core) for i in old.items] == [(SAFETY, ""), (FOLLOW_UP, "")]
    assert (old.side, old.confirmed) == (0, None) and not iv_mod.is_procedure(old)


def test_so_far_is_by_stage_in_the_procedure():
    iv = procedure_interview()
    iv.items[0].status, iv.items[0].quote = "answered", "no discharge"
    iv.items[1].status = "answered"
    iv.items[2].status, iv.items[2].quote = "answered", "two days"
    assert iv_mod.so_far(iv) == [
        {"label": "fever", "status": "absent", "quote": "", "stage": SAFETY},
        {"label": "how long it has lasted", "status": "answered", "quote": "two days",
         "stage": PROBLEM},
        {"label": "incision discharge", "status": "answered", "quote": "no discharge",
         "stage": FOLLOW_UP}]


def test_summary_says_whether_the_patient_confirmed():
    iv = procedure_interview()
    assert "confirmed" not in iv_mod.summary_text(iv)
    iv.confirmed = True
    assert iv_mod.summary_text(iv).endswith("The patient confirmed this summary.")
    iv.confirmed = False
    assert "added to or corrected" in iv_mod.summary_text(iv)


# ---------- through the app ----------

MEDICINES_Q = "Have you been taking amoxicillin as your discharge instructions say?"
OTHER_Q = iv_mod.CORE["other_medicines"][2]
SEVERITY_Q, COURSE_Q = iv_mod.CORE["severity"][2], iv_mod.CORE["course"][2]
# (question, the doctor's words?, the reply, the step).
ROUNDS = [(FEVER_Q, False, "No fever at all.", 1),
          (REDNESS_Q, True, "About two days.", 2),
          (SEVERITY_Q, False, "Mild.", 2),
          (COURSE_Q, False, "No.", 2),
          (DISCHARGE_Q, True, "No", 3),
          (MEDICINES_Q, False, "Yes I am.", 4),
          (OTHER_Q, False, "I'm not sure.", 4)]


def procedure_on(client):
    client.app.state.settings = replace(client.app.state.settings, interview_procedure=True)


def procedure_build(ids, medicines=True):
    return {"reports_symptom": True,
            "items": [{"label": "how long the redness lasted", "habits": [ids[REDNESS_Q]]},
                      {"label": "incision discharge", "habits": [ids[DISCHARGE_Q]]}],
            "core": {"duration": [ids[REDNESS_Q]]}, "ask_medicines": medicines}


async def begin(client, fake_llm, judges=None, medicines=True):
    procedure_on(client)
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    talk = Interviewer(fake_llm, procedure_build(ids, medicines),
                       judges=judges or [judged(signs=absent("fever"))])
    body = await start_interview(client, fake_llm, h, doctor_id)
    return h, doctor_id, talk, body


async def to_summary(client, h, doctor_id, body, rounds=ROUNDS, stages=4):
    for n, (asked, doctor, reply, step) in enumerate(rounds):
        assert body["questions"] == [{"text": asked, "doctor": doctor}], (n, body)
        iv = body["interview"]
        assert (iv["stage"], iv["stages"], iv["round"]) == (step, stages, n + 1)
        body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    return body


async def confirm(client, h, doctor_id, round_number):
    return await client.post("/api/ask", headers=h, json={
        "doctor_id": doctor_id, "question": "", "confirm": True, "round": round_number})


async def test_the_procedure_runs_every_stage_then_asks_to_confirm_and_closes(client, fake_llm):
    h, doctor_id, talk, body = await begin(client, fake_llm)
    body = await to_summary(client, h, doctor_id, body)
    assert body["kind"] == "clarify" and body["pending"] is True and body["questions"] == []
    assert body["answer"] == iv_mod.CONFIRM_TEXT and body["interview"]["confirm"] is True
    # Each reply is its question's answer, word for word; a typed "not sure" is unsure.
    assert [(i["stage"], i["label"], i["status"], i["quote"])
            for i in body["interview"]["so_far"]] == [
        (1, "fever", "absent", ""), (2, "how long it has lasted", "answered", "About two days."),
        (2, "how bad it is", "answered", "Mild."), (2, "getting better or worse", "answered", "No."),
        (3, "incision discharge", "answered", "No"),
        (4, "taking prescribed medicines", "answered", "Yes I am."),
        (4, "anything else taken for it", "unsure", "")]
    (q,) = await all_questions(client)
    assert q.status == "waiting_on_patient" and q.intake["confirmed"] is None
    assert talk.calls["extract"] == 1  # only the first message is read for answers
    calls_before = len(fake_llm.calls)

    fake_llm.script(final("close", CLOSE_TEXT + " [5]", [5]))
    talk.lists["judge"] = [judged(signs=absent("fever"), close=5)]
    body = (await confirm(client, h, doctor_id, body["interview"]["round"])).json()
    assert body["kind"] == "close" and body["answer"] == CLOSE_TEXT + " [5]", body
    assert body["interview_end"] == iv_mod.END_TEXT
    assert len(fake_llm.calls) == calls_before + 1  # the agent ran once, at the end
    (q,) = await all_questions(client)
    assert (q.status, q.patient_reply) == ("resolved", iv_mod.CONFIRMED_TEXT)
    assert q.intake["confirmed"] is True
    assert q.intake["rounds"][-1] == {"asked": iv_mod.CONFIRM_TEXT, "item": None,
                                      "reply": iv_mod.CONFIRMED_TEXT, "confirm": True}


async def test_a_typed_correction_of_the_summary_goes_to_the_doctor(client, fake_llm):
    h, doctor_id, _talk, body = await begin(client, fake_llm)
    body = await to_summary(client, h, doctor_id, body)
    calls_before = len(fake_llm.calls)
    body = (await post_ask(client, h, "The redness started a week ago.", doctor_id=doctor_id)).json()
    assert body["kind"] == "not_covered" and body["interview_end"] == iv_mod.END_TEXT
    assert len(fake_llm.calls) == calls_before  # no agent call
    (q,) = await all_questions(client)
    assert q.status == "open" and q.intake["confirmed"] is False
    assert q.reason == "Needs doctor after questions"
    assert q.stopped_by == "The patient added to or corrected the summary: see their last reply."
    assert q.patient_reply == "The redness started a week ago."


async def test_confirm_needs_a_summary_of_that_round(client, fake_llm):
    h, doctor_id, _talk, body = await begin(client, fake_llm)
    # A question is waiting, not the summary: the current state comes back.
    resp = await confirm(client, h, doctor_id, 1)
    assert resp.status_code == 200 and resp.json()["answer"] == FEVER_Q
    body = await to_summary(client, h, doctor_id, body)
    stale = await confirm(client, h, doctor_id, body["interview"]["round"] - 1)
    assert stale.json()["interview"]["confirm"] is True
    unsure = await client.post("/api/ask", headers=h, json={
        "doctor_id": doctor_id, "question": "", "not_sure": True,
        "round": body["interview"]["round"]})
    assert unsure.json()["interview"]["confirm"] is True
    (q,) = await all_questions(client)
    assert q.status == "waiting_on_patient" and q.intake["confirmed"] is None


async def test_confirm_with_nothing_waiting_is_409(client, fake_llm):
    procedure_on(client)
    h = await confirmed(client, fake_llm)
    doctor_id, _ = await interview_doctor(client, fake_llm)
    assert (await confirm(client, h, doctor_id, 1)).status_code == 409


async def test_a_side_question_is_answered_and_the_interview_goes_on(client, fake_llm):
    replying, asking = judged(signs=absent("fever")), judged(replies=False)
    h, doctor_id, talk, body = await begin(client, fake_llm, judges=[replying, asking, replying])
    body = (await post_ask(client, h, ROUNDS[0][2], doctor_id=doctor_id)).json()
    assert body["answer"] == REDNESS_Q and body["interview"]["round"] == 2
    said = "Take amoxicillin 500 mg three times daily [2]."
    fake_llm.script(final("answer", said, [2]))
    body = (await post_ask(client, h, "How often do I take the amoxicillin?",
                           doctor_id=doctor_id)).json()
    # The answer, then the same question again; no round was used.
    assert body["aside"]["kind"] == "answer" and body["aside"]["answer"] == said
    assert body["kind"] == "clarify" and body["pending"] is True
    assert body["questions"] == [{"text": REDNESS_Q, "doctor": True}]
    assert body["interview"]["round"] == 2 and body["interview"]["stage"] == 2
    (q,) = await all_questions(client)
    assert q.status == "waiting_on_patient" and q.intake["side"] == 1
    assert q.intake["rounds"][-1]["reply"] is None
    # The conversation holds the answer and then the question.
    from test_routes_agent import get_conversation
    messages = (await get_conversation(client, h, doctor_id)).json()["messages"]
    assert [m["content"] for m in messages[-3:]] == [
        "How often do I take the amoxicillin?", said, REDNESS_Q]
    assert messages[-2]["payload"].get("pending") is not True
    assert messages[-1]["payload"]["pending"] is True
    # The interview goes on from the same item.
    body = (await post_ask(client, h, ROUNDS[1][2], doctor_id=doctor_id)).json()
    assert body["answer"] == SEVERITY_Q and body["interview"]["round"] == 3


async def test_a_fourth_side_question_ends_the_interview(client, fake_llm):
    asking = judged(replies=False)
    h, doctor_id, _talk, body = await begin(
        client, fake_llm, judges=[judged(signs=absent("fever")), asking])
    await post_ask(client, h, ROUNDS[0][2], doctor_id=doctor_id)
    said = "Take amoxicillin 500 mg three times daily [2]."
    for n in range(iv_mod.MAX_SIDE_QUESTIONS):
        fake_llm.script(final("answer", said, [2]))
        body = (await post_ask(client, h, f"How often do I take it, again {n}?",
                               doctor_id=doctor_id)).json()
        assert "aside" in body and body["pending"] is True
    fake_llm.script(final("answer", said, [2]))
    body = (await post_ask(client, h, "How often do I take it, once more?",
                           doctor_id=doctor_id)).json()
    assert "aside" not in body and body.get("pending") is not True
    (q,) = await all_questions(client)
    assert q.status == "open"  # the symptom goes to the doctor, as before the procedure


async def test_a_side_question_that_needs_questions_itself_goes_to_the_doctor(client, fake_llm):
    h, doctor_id, talk, body = await begin(
        client, fake_llm, judges=[judged(signs=absent("fever")), judged(replies=False)])
    await post_ask(client, h, ROUNDS[0][2], doctor_id=doctor_id)
    # The model hands it off and the ask-first picker calls it a symptom: on a
    # first message that would start an interview.
    fake_llm.script(final("not_covered", "x", []),
                    {"content": json.dumps({"reports_symptom": True, "ask": []})})
    body = (await post_ask(client, h, "And is my knee pain normal?", doctor_id=doctor_id)).json()
    assert body["aside"]["kind"] == "not_covered" and body["answer"] == REDNESS_Q
    assert talk.calls["build"] == 1  # no second interview was built
    questions = await all_questions(client)
    assert [q.status for q in questions] == ["waiting_on_patient", "open"]
    assert questions[1].question == "And is my knee pain normal?"


async def test_an_emergency_reply_shows_no_end_line(client, fake_llm):
    h, doctor_id, _talk, body = await begin(client, fake_llm)
    fake_llm.script(final("emergency", "Call 911 now.", []))
    body = (await post_ask(client, h, "I have chest pain now.", doctor_id=doctor_id)).json()
    assert body["emergency"]["banner"] is True and "interview_end" not in body


async def test_only_taking_them_as_directed_is_asked_when_medicines_do_not_matter(client, fake_llm):
    h, doctor_id, _talk, body = await begin(client, fake_llm, medicines=False)
    body = await to_summary(client, h, doctor_id, body, rounds=ROUNDS[:-1])
    assert body["interview"]["confirm"] is True
    assert [i["label"] for i in body["interview"]["so_far"] if i["stage"] == 4] == [
        "taking prescribed medicines"]


async def test_a_case_the_patient_cannot_describe_is_not_closed(client, fake_llm):
    h, doctor_id, talk, body = await begin(client, fake_llm, medicines=False)
    rounds = [r if r[0] != SEVERITY_Q else (r[0], r[1], "I'm not sure.", r[3]) for r in ROUNDS[:-1]]
    body = await to_summary(client, h, doctor_id, body, rounds=rounds)
    fake_llm.script(final("close", CLOSE_TEXT + " [5]", [5]))
    talk.lists["judge"] = [judged(signs=absent("fever"), close=5)]
    body = (await confirm(client, h, doctor_id, body["interview"]["round"])).json()
    assert body["kind"] == "not_covered", body
    (q,) = await all_questions(client)
    assert q.status == "open" and q.stopped_by == "Patient is not sure: how bad it is"


async def test_a_reply_that_is_no_answer_is_asked_again_once(client, fake_llm):
    replying, off_topic = judged(signs=absent("fever")), judged(replies=False)
    h, doctor_id, _talk, body = await begin(client, fake_llm,
                                            judges=[replying, off_topic, off_topic, replying])
    body = (await post_ask(client, h, ROUNDS[0][2], doctor_id=doctor_id)).json()
    assert body["answer"] == REDNESS_Q
    # Not an answer and not a question: the item stays open and is asked again...
    fake_llm.route(iv_mod.QUESTION_PROMPT.format(doctor="Dr. [DOCTOR_1]"), lambda m: {
        "content": json.dumps({"question": "When did the redness first appear?"})})
    body = (await post_ask(client, h, "My cat is called Momo.", doctor_id=doctor_id)).json()
    assert body["answer"] == "When did the redness first appear?"
    # ...once: then it is "not sure" and the interview moves on.
    body = (await post_ask(client, h, "She is a tabby.", doctor_id=doctor_id)).json()
    assert body["answer"] == SEVERITY_Q
    assert {"label": "how long it has lasted", "status": "unsure", "quote": "",
            "stage": 2} in body["interview"]["so_far"]


async def test_the_clone_does_not_compose_advice_at_the_end_of_an_interview(client, fake_llm):
    h, doctor_id, talk, body = await begin(client, fake_llm, medicines=False)
    body = await to_summary(client, h, doctor_id, body, rounds=ROUNDS[:-1])
    # The model writes an answer of its own instead of closing in the doctor's words.
    fake_llm.script(final("answer", "Take amoxicillin 500 mg three times daily [2].", [2]))
    body = (await confirm(client, h, doctor_id, body["interview"]["round"])).json()
    assert body["kind"] == "not_covered" and "amoxicillin" not in body["answer"]
    assert body["interview_end"] == iv_mod.END_TEXT
    (q,) = await all_questions(client)
    assert q.status == "open" and q.stopped_by == "Interview complete: the doctor decides"
    assert q.intake["confirmed"] is True
