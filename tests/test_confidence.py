import json
import random

import pytest

import confidence
from agent import Evidence
from app import clone_threshold
from config import Settings
from fake_llm import FakeLLM
from llm import LLMError
from test_routes_agent import answer, care_team, confirmed, post_ask


# ---------- conformal risk control ----------

def test_risk_counts_only_answered_wrong_at_or_above_threshold():
    scores = [0.9, 0.8, None, 0.4]
    wrong = [True, False, True, True]
    assert confidence.risk(0.0, scores, wrong) == 0.5
    assert confidence.risk(0.5, scores, wrong) == 0.25
    assert confidence.risk(0.95, scores, wrong) == 0.0


def test_crc_threshold_worked_example():
    # 19 questions: 15 right answers scored 0.9, 4 wrong answers scored 0.3.
    scores = [0.9] * 15 + [0.3] * 4
    wrong = [False] * 15 + [True] * 4
    # At 0.3: risk 4/19, bound 19/20 * 4/19 + 1/20 = 0.25 > 0.1. At 0.9: risk 0, bound 0.05.
    assert confidence.crc_threshold(scores, wrong, 0.1) == 0.9
    assert confidence.crc_threshold(scores, wrong, 0.25) == 0.3


def test_crc_threshold_answers_nothing_or_refuses():
    scores, wrong = [0.9] * 19, [True] * 19
    assert confidence.crc_threshold(scores, wrong, 0.1) == confidence.ANSWER_NOTHING
    # 5 questions cannot certify 0.1: even answering nothing gives 1/6 > 0.1.
    assert confidence.crc_threshold([0.9] * 5, [False] * 5, 0.1) is None
    with pytest.raises(ValueError):
        confidence.crc_threshold([0.5], [True, False], 0.1)
    with pytest.raises(ValueError):
        confidence.crc_threshold([0.5], [True], 1.0)


def test_crc_threshold_is_the_smallest_valid_and_monotone_in_alpha():
    rng = random.Random(7)
    scores = [rng.random() if rng.random() > 0.1 else None for _ in range(80)]
    wrong = [rng.random() < (0.6 if (s or 0) < 0.5 else 0.1) for s in scores]
    previous = None
    for alpha in (0.05, 0.1, 0.2, 0.3):
        t = confidence.crc_threshold(scores, wrong, alpha)
        n = len(scores)
        assert n / (n + 1) * confidence.risk(t, scores, wrong) + 1 / (n + 1) <= alpha
        lower = [s for s in scores if s is not None and s < t]
        if lower:  # the next lower candidate breaks the bound
            below = max(lower)
            assert n / (n + 1) * confidence.risk(below, scores, wrong) + 1 / (n + 1) > alpha
        if previous is not None:
            assert t <= previous  # a looser alpha never needs a higher threshold
        previous = t


# ---------- the verifier ----------

def ev(n, content):
    return Evidence(n, "document", "Your discharge document", None, content, check_text=content)


async def test_score_answer_parses_and_clamps():
    llm = FakeLLM()
    seen = []
    llm.route(confidence.VERIFY_PROMPT,
              lambda m: seen.append(m) or {"content": json.dumps({"score": 87, "unsupported": []})})
    cited = ev(1, "Take it with food.")
    got = await confidence.score_answer(llm, "Q?", "Take it [1].", [cited],
                                        [cited, ev(2, "Call 911 if you faint.")])
    assert got == 0.87
    user = seen[0][1]["content"]
    cited_part, other_part = user.split("Other evidence the assistant had:")
    assert "[1] Your discharge document\nTake it with food." in cited_part
    assert "[2] Your discharge document\nCall 911 if you faint." in other_part
    assert "Take it with food." not in other_part  # cited evidence is not repeated
    for reply, expected in (({"score": 140}, 1.0), ({"score": -3}, 0.0)):
        llm.route(confidence.VERIFY_PROMPT, lambda m, r=reply: {"content": json.dumps(r)})
        assert await confidence.score_answer(llm, "Q?", "A", []) == expected
    for bad in ("nope", json.dumps({"score": "high"}), json.dumps({"score": True}), json.dumps([1])):
        llm.route(confidence.VERIFY_PROMPT, lambda m, b=bad: {"content": b})
        with pytest.raises(ValueError):
            await confidence.score_answer(llm, "Q?", "A", [])


def test_load_threshold_and_settings(tmp_path):
    path = tmp_path / "cal.json"
    assert confidence.load_threshold(path, "gpt-x") is None
    path.write_text(json.dumps({"chat_model": "gpt-x", "threshold": 0.72}))
    assert confidence.load_threshold(path, "gpt-x") == 0.72
    assert confidence.load_threshold(path, "other-model") is None
    assert confidence.load_threshold(path, "") is None
    path.write_text("{broken")
    assert confidence.load_threshold(path, "gpt-x") is None
    path.write_text(json.dumps({"chat_model": "gpt-x", "threshold": 0.72}))
    base = dict(database_url="sqlite+aiosqlite:///:memory:", jwt_secret="x" * 32,
                clone_calibration_path=str(path))
    assert clone_threshold(Settings(**base, llm_model="gpt-x")) == 0.72
    assert clone_threshold(Settings(**base, llm_model="gpt-x", clone_min_confidence=0.5)) == 0.5
    assert clone_threshold(Settings(**base, llm_model="other")) is None


# ---------- the gate in /api/ask ----------

ANSWER = "Take amoxicillin 500 mg three times daily [2]."


def verifier(score):
    return lambda m: {"content": json.dumps({"score": score, "unsupported": []})}


async def test_gate_off_by_default(client, fake_llm):
    assert client.app.state.clone_threshold is None
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer(ANSWER, [2]))
    body = (await post_ask(client, h, "How often?")).json()
    assert body["kind"] == "answer"
    assert not any(c[0][0]["content"] == confidence.VERIFY_PROMPT for c in fake_llm.routed_calls)


async def test_confident_answer_passes_the_gate(client, fake_llm):
    client.app.state.clone_threshold = 0.8
    h = await confirmed(client, fake_llm)
    fake_llm.route(confidence.VERIFY_PROMPT, verifier(92))
    fake_llm.script(answer(ANSWER, [2]))
    body = (await post_ask(client, h, "How often?")).json()
    assert body["kind"] == "answer" and body["answer"] == ANSWER and body["confidence"] == 0.92
    assert await care_team(client, h) == []
    # The verifier saw the document paragraphs the clone did not cite.
    (verify,) = [c for c in fake_llm.routed_calls if c[0][0]["content"] == confidence.VERIFY_PROMPT]
    other = verify[0][1]["content"].split("Other evidence the assistant had:")[1]
    assert "(none)" not in other


@pytest.mark.parametrize("route", ["low", "error", "garbled"])
async def test_low_confidence_or_failed_verifier_goes_to_the_doctor(client, fake_llm, route):
    client.app.state.clone_threshold = 0.8
    h = await confirmed(client, fake_llm)

    def down(m):
        raise LLMError("down")
    fake_llm.route(confidence.VERIFY_PROMPT, {
        "low": verifier(41), "error": down, "garbled": lambda m: {"content": "?"}}[route])
    fake_llm.script(answer(ANSWER, [2]))
    body = (await post_ask(client, h, "How often do I take it?")).json()
    assert body["kind"] == "not_covered" and body["sources"] == [] and body["flagged"]
    assert body["confidence"] == (0.41 if route == "low" else None)
    assert "sent it to Dr. Default" in body["answer"]
    (q,) = await care_team(client, h)
    assert q["question"] == "How often do I take it?"
    assert q["reason"] == confidence.LOW_CONFIDENCE_REASON


async def test_gate_never_scores_an_emergency(client, fake_llm):
    client.app.state.clone_threshold = 0.99
    h = await confirmed(client, fake_llm)
    fake_llm.route(confidence.VERIFY_PROMPT, verifier(0))
    fake_llm.script(answer("Call 911 now.", [], kind="emergency"))
    body = (await post_ask(client, h, "I have chest pain right now")).json()
    assert body["kind"] == "emergency" and body["emergency"]["banner"]
    assert not any(c[0][0]["content"] == confidence.VERIFY_PROMPT for c in fake_llm.routed_calls)


async def test_uncited_habits_are_not_other_evidence():
    # Every turn sees up to 30 habits; an uncited warning or close habit is not
    # a when-to-call instruction the answer left out. A cited habit stays.
    llm = FakeLLM()
    llm.script({"content": json.dumps({"score": 90})})
    doc = Evidence(1, "document", "Your discharge document", None, "Drink water.", "Drink water.")
    other_doc = Evidence(2, "document", "Your discharge document", None,
                         "Call the clinic if you have a fever.", "Call the clinic if you have a fever.")
    cited_habit = Evidence(3, "habit", "Dr. A's usual practice", None,
                           "Keep your follow-up visit.", "Keep your follow-up visit.",
                           habit_type="close")
    warning = Evidence(4, "habit", "Dr. A's usual practice", None,
                       "Go to the ER if you vomit blood.", "Go to the ER if you vomit blood.",
                       habit_type="warning")
    ask_habit = Evidence(5, "habit", "Dr. A's usual practice", None, "Any fever?", "Any fever?",
                         habit_type="ask")
    await confidence.score_answer(llm, "Can I drink water?", "Drink water [1]. Keep your "
                                  "follow-up visit [3].", [doc, cited_habit],
                                  [doc, other_doc, cited_habit, warning, ask_habit])
    prompt = llm.calls[0][0][1]["content"]
    sources = prompt.split("Cited sources:")[1].split("Other evidence")[0]
    other = prompt.split("Other evidence the assistant had:")[1]
    assert "Keep your follow-up visit." in sources
    assert "Call the clinic if you have a fever." in other
    assert "vomit blood" not in prompt and "Any fever?" not in prompt
