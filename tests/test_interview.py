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
