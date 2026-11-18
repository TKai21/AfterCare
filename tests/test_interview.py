import interview as iv_mod
from interview import Interview, Item


def make(*specs):
    """specs: (label, safety, status)"""
    return Interview(items=[Item(label=l, habits=[], safety=s, status=st) for l, s, st in specs],
                     rounds=[])


def test_round_trip_dump_load():
    iv = Interview(
        items=[Item("fever", [1, 2], False, "answered", "I have a fever", 1),
               Item("chest pain", [], True)],
        rounds=[{"asked": "Do you have a fever?", "item": 0, "reply": "I have a fever"}])
    data = iv_mod.dump_interview(iv)
    assert iv_mod.load_interview(data) == iv


def test_old_list_intake_is_not_interview():
    old = [{"asked": "q", "reply": "r"}]
    assert not iv_mod.is_interview(old)
    assert iv_mod.load_interview(old) is None
    assert not iv_mod.is_interview(None)
    assert iv_mod.is_interview({"items": [], "rounds": []})


def test_next_item_safety_first():
    iv = make(("fever", False, "unknown"), ("chest pain", True, "unknown"),
              ("cough", False, "unknown"))
    assert iv_mod.next_item(iv) == 1
    iv.items[1].status = "answered"
    assert iv_mod.next_item(iv) == 0


def test_next_item_none_when_all_skipped():
    iv = make(("fever", False, "skipped"), ("chest pain", True, "unsure"),
              ("cough", False, "answered"))
    assert iv_mod.next_item(iv) is None
    assert iv_mod.done(iv, 6)


def test_done_at_max_rounds():
    iv = make(("fever", False, "unknown"))
    assert not iv_mod.done(iv, 2)
    iv.rounds = [{"asked": "a", "item": 0, "reply": "b"}] * 2
    assert iv_mod.done(iv, 2)


def test_fill_accepts_substring_case_and_space_insensitive():
    iv = make(("fever", False, "unknown"))
    assert iv_mod.fill(iv, 0, "  HIGH   fever ", ["I had a high\nfever since Monday"])
    assert iv.items[0].status == "answered"
    assert iv.items[0].quote == "HIGH fever"


def test_fill_rejects_quote_not_in_sources():
    iv = make(("fever", False, "unknown"))
    assert not iv_mod.fill(iv, 0, "no fever at all", ["my knee hurts"])
    assert not iv_mod.fill(iv, 0, "   ", ["my knee hurts"])
    assert iv.items[0].status == "unknown"


def test_so_far_shortens_long_quote_at_word_boundary():
    quote = "I have had a bad pain in my left knee for about three days now since"
    assert len(quote) >= 65
    iv = make(("knee", False, "unknown"), ("fever", False, "unknown"), ("rash", False, "skipped"))
    iv.items[0].status, iv.items[0].quote = "answered", quote
    iv.items[1].status = "unsure"
    got = iv_mod.so_far(iv)
    assert [g["label"] for g in got] == ["knee", "fever"]
    shown = got[0]["quote"]
    assert shown.endswith("...") and len(shown) <= 63
    assert quote.startswith(shown[:-3].rstrip())
    assert quote[len(shown[:-3].rstrip())] == " "
    assert iv.items[0].quote == quote
    assert got[1]["quote"] == ""


def test_summary_text_lists_three_groups():
    iv = make(("knee", False, "answered"), ("fever", False, "unsure"),
              ("rash", False, "skipped"), ("cough", False, "unknown"))
    iv.items[0].quote = "my knee hurts"
    assert iv_mod.summary_text(iv) == (
        'Known: knee ("my knee hurts")\nNot sure: fever\nNot answered: rash, cough')
    assert iv_mod.summary_text(make(("a", False, "unsure"))) == "Not sure: a"


# ---------- model calls ----------

import json

import pytest
import pytest_asyncio

from db import init_db, make_engine, make_session_factory
from fake_llm import FakeLLM
from guidance import embed_entry
from models import DoctorGuidance, GuidanceRule, User


def reply(data):
    return {"content": json.dumps(data)}


def habit(rule_id, habit_type="ask", statement="", quote="", terms=()):
    return GuidanceRule(id=rule_id, guidance_id=1, doctor_id=1, position=rule_id,
                        statement=statement or quote, quote=quote or statement, kind="habit",
                        habit_type=habit_type, warning_terms=list(terms))


ASKS = [habit(1, quote="Is the pain worse when you walk?"),
        habit(2, quote="Have you had a fever?")]


async def build(llm, ask=ASKS, warn=(), doc=(), **kw):
    return await iv_mod.build_items(
        llm, "my knee hurts", list(ask), list(warn), list(doc), frozenset(),
        max_items=kw.get("max_items", 5), max_safety=kw.get("max_safety", 3))


async def test_build_items_drops_unknown_habit_ids():
    llm = FakeLLM()
    llm.script(reply({"reports_symptom": True, "items": [
        {"label": "knee pain on walking", "habits": [1]},
        {"label": "fever", "habits": [99]},
        {"label": "swelling", "habits": [2, 77]},
        {"label": "chills", "habits": [2]}]}))
    reports, items = await build(llm)
    assert reports is True
    assert [(i.label, i.habits, i.safety) for i in items] == [
        ("knee pain on walking", [1], False), ("chills", [2], False)]


async def test_build_items_rejects_label_with_digit():
    llm = FakeLLM()
    llm.script(reply({"reports_symptom": True, "items": [
        {"label": "fever above 101", "habits": [2]},
        {"label": "one two three four five six seven", "habits": [2]},
        {"label": "you should rest", "habits": [1]},
        {"label": "fever", "habits": [2]}]}))
    _, items = await build(llm)
    assert [i.label for i in items] == ["fever"]


async def test_build_items_adds_safety_items_from_warning_terms_capped_at_3():
    llm = FakeLLM()
    llm.script(reply({"reports_symptom": True, "items": [
        {"label": "fever", "habits": [2]}]}))
    warn = [habit(3, "warning", quote="Call if chest pain or fever.",
                  terms=["chest pain", "fever", "black stool"])]
    _, items = await build(llm, warn=warn, doc=["Chest Pain", "vomiting blood", "dizziness"])
    assert [(i.label, i.safety) for i in items] == [
        ("fever", False), ("chest pain", True), ("black stool", True), ("vomiting blood", True)]


async def test_build_items_none_on_bad_json():
    llm = FakeLLM()
    llm.script({"content": "not json"})
    assert await build(llm) is None
    llm.script(reply({"items": "nope"}))
    assert await build(llm) is None


def three_items():
    return Interview(items=[Item("knee pain", [1], False), Item("fever", [2], False),
                            Item("chest pain", [], True)], rounds=[])


async def test_extract_rejects_invented_quote():
    llm = FakeLLM()
    llm.script(reply({"items": [
        {"index": 0, "status": "answered", "quote": "it hurts when I walk"},
        {"index": 1, "status": "answered", "quote": "no fever since Monday"}]}))
    iv = three_items()
    got = await iv_mod.extract(llm, iv, "my knee hurts", "It hurts when I walk, a lot")
    assert got == {0: "it hurts when I walk"}
    assert iv.items[0].status == "answered" and iv.items[1].status == "unknown"


async def test_extract_fills_from_original_question():
    llm = FakeLLM()
    llm.script(reply({"items": [{"index": 0, "status": "answered", "quote": "my knee hurts"},
                                {"index": 2, "status": "answered", "quote": "my knee hurts"}]}))
    iv = three_items()
    got = await iv_mod.extract(llm, iv, "my knee hurts", "dunno")
    assert list(got) == [0]
    assert iv.items[2].status == "unknown"  # a safety item is never filled by the extractor


async def test_extract_empty_on_failure():
    llm = FakeLLM()
    llm.script({"content": "oops"})
    assert await iv_mod.extract(llm, three_items(), "q", "r") == {}


async def test_write_question_prefers_habit_verbatim():
    llm = FakeLLM()
    item = Item("knee pain", [1], False)
    got = await iv_mod.write_question(
        llm, item, {1: "Is the pain worse when you walk?"}, [], frozenset(), "Dr. Lee", False)
    assert got == ("Is the pain worse when you walk?", True)
    assert llm.calls == []


async def test_write_question_writes_one_when_no_habit_question():
    llm = FakeLLM()
    llm.script(reply({"question": "When did the knee pain start?"}))
    item = Item("knee pain", [1], False)
    got = await iv_mod.write_question(
        llm, item, {1: "Check the knee."}, [], frozenset(), "Dr. Lee", False)
    assert got == ("When did the knee pain start?", False)


async def test_write_question_rejects_advice_then_falls_back_to_habit():
    llm = FakeLLM()
    llm.script(reply({"question": "Should you rest the knee?"}),
               reply({"question": "Have you tried taking more ibuprofen?"}))
    item = Item("knee pain", [1], False)
    got = await iv_mod.write_question(
        llm, item, {1: "Is the pain worse when you walk?"}, [], frozenset(), "Dr. Lee", True)
    assert got == ("Is the pain worse when you walk?", True)
    assert len(llm.calls) == 2


async def test_write_question_none_without_habit_question():
    llm = FakeLLM()
    llm.script(reply({"question": "Call the clinic now?"}),
               reply({"question": "Do you take 400 mg?"}))
    item = Item("knee pain", [1], False)
    assert await iv_mod.write_question(
        llm, item, {1: "Check the knee."}, [], frozenset(), "Dr. Lee", False) is None


async def test_write_question_repair_call_can_succeed():
    llm = FakeLLM()
    llm.script(reply({"question": "Should you rest?"}),
               reply({"question": "Does the knee feel swollen?"}))
    got = await iv_mod.write_question(
        llm, Item("knee", [], False), {}, [], frozenset(), "Dr. Lee", False)
    assert got == ("Does the knee feel swollen?", False)


@pytest_asyncio.fixture
async def session():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    async with make_session_factory(engine)() as s:
        yield s
    await engine.dispose()


async def test_similar_case_habits_skips_suppressed(session):
    llm = FakeLLM()
    doc = User(username="doc", password_hash="x")
    other = User(username="doc2", password_hash="x")
    session.add_all([doc, other])
    await session.flush()
    entries = {}
    for name, doctor, question in (("knee", doc, "my knee hurts when I walk"),
                                   ("rash", doc, "itchy red rash on my arm"),
                                   ("foreign", other, "my knee hurts when I walk")):
        e = DoctorGuidance(doctor_id=doctor.id, question=question, answer="ok", scope="general",
                           active=True)
        session.add(e)
        await session.flush()
        assert await embed_entry(llm, e)
        entries[name] = e

    def rule(entry, position, habit_type, text, **kw):
        session.add(GuidanceRule(guidance_id=entry.id, doctor_id=entry.doctor_id,
                                 position=position, statement=text, quote=text, kind="habit",
                                 habit_type=habit_type, **kw))

    rule(entries["knee"], 0, "ask", "Is it swollen?")
    rule(entries["knee"], 1, "ask", "Is it red?", suppressed=True)
    rule(entries["knee"], 2, "warning", "Call if fever.", warning_terms=["fever"])
    rule(entries["knee"], 3, "close", "Take care.")
    rule(entries["rash"], 0, "ask", "Is it itchy?")
    rule(entries["foreign"], 0, "ask", "Foreign question?")
    await session.commit()

    asks, warns = await iv_mod.similar_case_habits(
        session, llm, doc.id, "my knee hurts when I walk", 1)
    assert [r.quote for r in asks] == ["Is it swollen?"]
    assert [r.quote for r in warns] == ["Call if fever."]
    asks, _ = await iv_mod.similar_case_habits(session, llm, doc.id, "my knee hurts", 4)
    assert [r.quote for r in asks] == ["Is it swollen?", "Is it itchy?"]


async def test_build_items_skips_safety_term_failing_label_check():
    llm = FakeLLM()
    llm.script(reply({"reports_symptom": True, "items": []}))
    warn = [habit(3, "warning", quote="x", terms=["temperature over 101", "chest pain",
                                                  "take more naproxen", "black stool"])]
    _, items = await build(llm, warn=warn, doc=["one two three four five six seven"])
    assert [(i.label, i.safety) for i in items] == [("chest pain", True), ("black stool", True)]


async def test_write_question_rejects_off_topic_question():
    llm = FakeLLM()
    llm.script(reply({"question": "Do you drink coffee?"}),
               reply({"question": "Do you like walking outdoors?"}))
    item = Item("knee pain", [1], False)
    assert await iv_mod.write_question(
        llm, item, {1: "Check the knee."}, [], frozenset(), "Dr. Lee", False) is None
    assert len(llm.calls) == 2


async def test_write_question_accepts_label_word_overlap():
    llm = FakeLLM()
    llm.script(reply({"question": "Is your knee swollen?"}))
    item = Item("knee pain", [1], False)
    got = await iv_mod.write_question(
        llm, item, {1: "Check the knee."}, [], frozenset(), "Dr. Lee", False)
    assert got == ("Is your knee swollen?", False)


def test_safety_item_question_is_yes_no_template():
    import asyncio
    iv = make(("chest pain", True, "unknown"))
    got = asyncio.run(iv_mod.write_question(None, iv.items[0], {}, [], [], "Dr. Ting Kai", False))
    assert got == ("Have you had any chest pain?", False)
    assert iv_mod.safety_question("you are confused", False) == "Have you noticed this: you are confused?"
    assert iv_mod.safety_question("fever is rising", False) == "Have you noticed this: fever is rising?"


def test_safety_item_rephrase_template():
    import asyncio
    iv = make(("chest pain", True, "unknown"))
    got = asyncio.run(iv_mod.write_question(None, iv.items[0], {}, [], [], "Dr. Ting Kai", True))
    assert got == ("Just to be sure, do you have any chest pain right now?", False)


def test_so_far_shows_denied_safety_item_as_absent():
    iv = make(("chest pain", True, "answered"), ("fever", True, "unknown"))
    assert iv_mod.so_far(iv) == [{"label": "chest pain", "status": "absent", "quote": ""}]
    assert iv_mod.summary_text(iv).startswith("Known: no chest pain")


def test_so_far_whole_first_message_is_mentioned_not_quoted():
    original = "I just started getting a really runny nose and I have been sneezing nonstop."
    iv = make(("additional symptoms", False, "answered"), ("duration", False, "answered"))
    iv.items[0].quote = original
    iv.items[1].quote = "started two days ago"
    got = iv_mod.so_far(iv, original.upper())
    assert got == [{"label": "additional symptoms", "status": "mentioned", "quote": ""},
                   {"label": "duration", "status": "answered", "quote": "started two days ago"}]
    # A long stretch of the first message is "mentioned" too; a long reply quote is cut.
    iv.items[1].quote = "I just started getting a really runny nose and I have been sneezing"
    assert iv_mod.so_far(iv, original)[1]["status"] == "mentioned"
    iv.items[1].quote = ("word " * 30).strip()
    shown = iv_mod.so_far(iv, original)[1]
    assert shown["status"] == "answered" and len(shown["quote"]) <= 63
