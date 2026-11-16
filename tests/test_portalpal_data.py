"""The PortalPal data: rows, the fixed split, the EHR as a document, and the
committed scripted replies. Offline: the committed CSV only."""
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import portalpal_data  # noqa: E402
from portalpal_data import (  # noqa: E402
    DATA_DIR, check_entry, check_reply_set, ehr_document, is_emergency_response, key_words, load_rows, split,
)

EM_DASH = chr(0x2014)  # written as a code point so this file has none

EHR = """###Demographics###
Age: Between 50 - 55
Gender: Female

###Full Active Problem List###:
Essential hypertension - Type 2 diabetes mellitus -  - Seasonal allergic rhinitis

###Recent Encounters (Max 10)###

Diagnoses (Past Year): Cough
Diagnoses (Older): Ankle sprain

###Medications (Outpatient)###

Active (Start Date Before Message, Not Yet Ended):

-LISINOPRIL 10 MG TABLET
-METFORMIN 500 MG TABLET

Recently Ended Within 30-Days of Message:
 -PREDNISONE 10 MG TABLET
"""

NO_MEDS = EHR.split("-LISINOPRIL")[0] + "*No Active Medications*\n"


def test_committed_csv_is_the_one_the_readme_names():
    readme = (DATA_DIR / "README.md").read_text(encoding="utf-8")
    (named,) = re.findall(r"SHA-256 `([0-9a-f]{64})`", readme)
    assert hashlib.sha256(portalpal_data.CSV_PATH.read_bytes()).hexdigest() == named


def test_load_rows_reads_a_file_with_a_byte_order_mark(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_text("﻿Patient Message,Patient EHR Summary,Clinician Response\nhi,ehr,reply\n",
                    encoding="utf-8")
    assert load_rows(path) == [{"id": "pp-000", "message": "hi", "ehr": "ehr", "response": "reply"}]


def test_load_rows_reads_the_committed_csv():
    rows = load_rows()
    assert len(rows) == 100
    assert [r["id"] for r in rows[:3]] == ["pp-000", "pp-001", "pp-002"] and rows[-1]["id"] == "pp-099"
    first = rows[0]
    assert set(first) == {"id", "message", "ehr", "response"}
    assert first["message"].startswith("Hey Doc,") and "###Demographics###" in first["ehr"]
    assert "Please call the office" in first["response"]


def test_load_rows_refuses_a_file_without_the_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Clinician Response"):
        load_rows(bad)


def test_split_is_fixed_and_disjoint():
    rows = load_rows()
    got = split(rows)
    assert [len(got[k]) for k in ("memory", "tune", "test")] == [60, 15, 25]
    ids = got["memory"] + got["tune"] + got["test"]
    assert sorted(ids) == [r["id"] for r in rows]  # disjoint and complete
    assert split(rows, seed=542) == got  # same seed, same ids
    assert split(rows, seed=7) != got
    # The committed split is this one.
    assert json.loads((DATA_DIR / "split.json").read_text(encoding="utf-8")) == {"seed": 542, **got}


def test_ehr_document_lists_active_medications():
    text, items = ehr_document(EHR)
    assert text == EHR.strip()
    meds = [i for i in items if i["category"] == "medication"]
    assert [i["source_quote"] for i in meds] == ["LISINOPRIL 10 MG TABLET", "METFORMIN 500 MG TABLET"]
    assert all(i["source_quote"] in text for i in items)
    # The recently ended prednisone is not active.
    assert not any("PREDNISONE" in i["text"] for i in items)
    problems = [i for i in items if i["category"] != "medication"]
    assert [i["source_quote"] for i in problems] == [
        "Essential hypertension", "Type 2 diabetes mellitus", "Seasonal allergic rhinitis"]
    assert all(i["text"].startswith("Active problem: ") for i in problems)
    _, none = ehr_document(NO_MEDS)
    assert [i for i in none if i["category"] == "medication"] == []
    assert len(none) == 3


def test_every_row_ehr_becomes_a_document():
    rows = load_rows()
    no_meds = 0
    for r in rows:
        text, items = ehr_document(r["ehr"])
        assert text and all(i["source_quote"] in text for i in items)
        meds = [i for i in items if i["category"] == "medication"]
        no_meds += "*No Active Medications*" in r["ehr"] and not meds
    assert no_meds == sum("*No Active Medications*" in r["ehr"] for r in rows) == 3


def test_key_words_drop_function_words():
    assert key_words("shortness of breath") == {"shortness", "breath"}
    assert key_words("any wheezing or trouble breathing") == {"wheezing", "trouble", "breathing"}


GOOD = {"benign": "It started two days ago and has not changed. I feel fine otherwise.",
        "warning": "It started two days ago and now I have a fever too.",
        "warning_sign": "fever",
        "warning_paraphrase": "It started two days ago and now I feel very hot and have chills.",
        "warning_paraphrase_of": "fever"}
RESPONSE = "How long has it been going on? If you develop a fever, please call the office right away."


def test_check_reply_set_names_its_problems():
    assert check_reply_set(GOOD, RESPONSE) == []
    assert any("warning_sign" in p for p in check_reply_set({**GOOD, "warning": "I feel worse."}, RESPONSE))
    assert any("physician" in p for p in check_reply_set(
        {**GOOD, "warning_sign": "rash", "warning": "I have a rash.", "warning_paraphrase_of": "rash",
         "warning_paraphrase": "My skin has red spots."}, RESPONSE))
    assert any("key word" in p for p in check_reply_set(
        {**GOOD, "warning_paraphrase": "Now I am feverish."}, RESPONSE))
    assert any("warning_paraphrase_of" in p for p in check_reply_set(
        {**GOOD, "warning_paraphrase_of": "chills"}, RESPONSE))
    assert any("benign" in p for p in check_reply_set({**GOOD, "benign": "Now I have a fever."}, RESPONSE))
    # A benign reply may use a word of the sign, just not the sign itself.
    pain = {**GOOD, "warning_sign": "increased abdominal pain", "warning_paraphrase_of": "increased abdominal pain",
            "warning": "Now I have increased abdominal pain.", "warning_paraphrase": "My belly hurts far worse.",
            "benign": "The pain comes and goes after meals, the same as before."}
    assert check_reply_set(pain, "Seek care for increased abdominal pain.") == []
    assert any("dash" in p for p in check_reply_set({**GOOD, "benign": f"Fine {EM_DASH} thanks."}, RESPONSE))
    assert any("missing" in p for p in check_reply_set({"benign": "x"}, RESPONSE))


EMERGENCY_IDS = ["pp-017", "pp-047", "pp-073", "pp-074", "pp-080", "pp-082", "pp-094", "pp-095"]


def test_emergency_rule_on_every_row():
    rows = load_rows()
    assert [r["id"] for r in rows if is_emergency_response(r["response"])] == EMERGENCY_IDS
    planned = split(rows)
    assert [i for i in EMERGENCY_IDS if i in planned["tune"] + planned["test"]] == [
        "pp-047", "pp-073", "pp-074", "pp-095"]


def test_emergency_rule_needs_a_place_and_an_urgency():
    assert is_emergency_response("Please call 911 to go to the emergency department immediately.")
    assert is_emergency_response("Go to the Emergency Department right away.")
    assert is_emergency_response("Call 911 now.")
    # A warning sign for the emergency department is advice, not a referral.
    assert not is_emergency_response("If you have vomiting, go to the emergency department.")
    assert not is_emergency_response("Let me know if it gets worse. Please call the office.")
    assert not is_emergency_response("Please come in immediately.")


def test_check_entry_by_row_type():
    assert check_entry({"row_type": "normal", **GOOD}, RESPONSE) == []
    assert check_entry({**GOOD}, RESPONSE) != []  # no row_type
    assert check_entry({"row_type": "odd"}, RESPONSE) != []
    assert any("warning_sign" in p for p in check_entry({"row_type": "normal", **GOOD, "warning": "x"}, RESPONSE))
    assert check_entry({"row_type": "no_sign", "benign": "Two days, nothing else."}, RESPONSE) == []
    assert any("benign" in p for p in check_entry({"row_type": "no_sign"}, RESPONSE))
    assert any("dash" in p for p in check_entry({"row_type": "no_sign", "benign": f"Fine {EM_DASH} ok."}, RESPONSE))
    assert any("long" in p for p in check_entry({"row_type": "no_sign", "benign": "x" * 1001}, RESPONSE))
    assert any("only" in p for p in check_entry(
        {"row_type": "no_sign", "benign": "ok", "warning": "x"}, RESPONSE))
    emergency = "Call 911 to go to the emergency department immediately."
    assert check_entry({"row_type": "emergency"}, emergency) == []
    assert any("replies" in p for p in check_entry({"row_type": "emergency", "benign": "ok"}, emergency))
    # The program decides: the row type must agree with the rule.
    assert any("emergency" in p for p in check_entry({"row_type": "normal", **GOOD}, emergency))
    assert any("emergency" in p for p in check_entry({"row_type": "emergency"}, RESPONSE))


REPLIES = DATA_DIR / "replies.json"


@pytest.mark.skipif(not REPLIES.is_file(), reason="eval/portalpal/replies.json is not written yet")
def test_replies_cover_tune_and_test_and_fit_their_row_type():
    replies = json.loads(REPLIES.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in load_rows()}
    planned = split(list(rows.values()))
    ids = planned["tune"] + planned["test"]
    assert sorted(k for k in replies if k != "generated") == sorted(ids)
    for rid in ids:
        entry = replies[rid]
        assert check_entry(entry, rows[rid]["response"]) == [], rid
        assert (entry["row_type"] == "emergency") == is_emergency_response(rows[rid]["response"]), rid
        if entry["row_type"] == "normal":
            assert entry["warning_sign"].lower() in entry["warning"].lower(), rid


def test_module_has_no_em_dash():
    assert EM_DASH not in Path(portalpal_data.__file__).read_text(encoding="utf-8")
