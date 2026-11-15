"""The clone preference evaluation's committed data and its expected-behaviour
rule. Offline: no model, no network."""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from clone_eval_data import (  # noqa: E402
    ALLOWED_DRUGS,
    DATA_DIR,
    DIMENSIONS,
    check_entries,
    doc_governs,
    DOCTORS,
    expected_items,
    load,
    near_duplicate,
    no_memory_items,
    prescription_names,
)
from guidance import eligible, patient_text  # noqa: E402
from models import DischargeDocument, DoctorGuidance, ExtractedItem  # noqa: E402

PROFILES = load("profiles")
PATIENTS = load("patients")
by_id = {p["id"]: p for p in PATIENTS}
EM_DASH = chr(0x2014)  # written as a code point so this file has none


def patient(prescribed=("colchicine",), ckd=False, past_effective=None, flare_plan=False):
    return {"id": "t", "text": "", "prescribed": list(prescribed), "ckd": ckd,
            "past_effective": past_effective, "flare_plan": flare_plan, "items": []}


def test_data_dir_and_doctors():
    assert DATA_DIR == ROOT / "eval" / "clone"
    assert DOCTORS == (
        ("dr.ting", "Dr. Ting Kai", "Rheumatology"),
        ("dr.patel", "Dr. Parin Patel", "Internal Medicine"),
        ("dr.renny", "Dr. Aaron Renny", "Nephrology"),
    )
    assert ALLOWED_DRUGS == ("colchicine", "naproxen", "prednisone")


def test_profiles_follow_the_spec_table():
    assert set(PROFILES) == {username for username, *_ in DOCTORS}
    assert {u: PROFILES[u]["D1"]["preferred"] for u in PROFILES} == {
        "dr.ting": "colchicine", "dr.patel": "naproxen", "dr.renny": "past_effective"}
    assert {u: (PROFILES[u]["D3"]["first_ask"], PROFILES[u]["D4"]["first_ask"]) for u in PROFILES} == {
        "dr.ting": ("timing", "past_experience"),
        "dr.patel": ("medication_list", "other_conditions"),
        "dr.renny": ("past_reaction", "current_plan"),
    }
    for profile in PROFILES.values():
        assert set(profile) == {"D1", "D2", "D3", "D4"}
        assert all(profile[d]["text"].strip() for d in profile)


def test_rule_d1_preferred_prescribed():
    p = patient(prescribed=["colchicine", "naproxen"], ckd=False)
    assert expected_items("dr.ting", p, "D1", PROFILES) == {"first_choice": "colchicine", "must_not_recommend": ["prednisone"]}
    assert expected_items("dr.patel", p, "D1", PROFILES)["first_choice"] == "naproxen"


def test_rule_d1_nsaid_not_allowed_with_ckd_refers():
    p = patient(prescribed=["colchicine", "naproxen"], ckd=True)
    assert expected_items("dr.patel", p, "D1", PROFILES)["refer_to_clinic"] is True
    assert "first_choice" not in expected_items("dr.patel", p, "D1", PROFILES)
    # Colchicine is allowed for every patient, CKD or not.
    assert expected_items("dr.ting", p, "D1", PROFILES)["first_choice"] == "colchicine"


def test_rule_d1_preferred_missing_refers_and_prescribed_never_forbidden():
    p = patient(prescribed=["naproxen"])
    items = expected_items("dr.ting", p, "D1", PROFILES)
    assert items["refer_to_clinic"] is True and "naproxen" not in items["must_not_recommend"]
    assert items == {"refer_to_clinic": True, "must_not_recommend": ["colchicine", "prednisone"]}


def test_rule_renny_without_history_confirms_plan_and_refers():
    items = expected_items("dr.renny", patient(past_effective=None), "D1", PROFILES)
    assert items["refer_to_clinic"] is True and items["first_ask"] == "current_plan"


def test_rule_renny_with_history_names_the_drug_that_worked():
    p = patient(prescribed=["prednisone"], past_effective="prednisone")
    assert expected_items("dr.renny", p, "D1", PROFILES) == {
        "first_choice": "prednisone", "must_not_recommend": ["colchicine", "naproxen"]}
    # A drug that worked before but is no longer prescribed is referred, not named.
    p = patient(prescribed=["colchicine"], past_effective="prednisone")
    assert expected_items("dr.renny", p, "D1", PROFILES)["refer_to_clinic"] is True


def test_rule_d2_values():
    with_plan, without_plan = patient(flare_plan=True), patient(flare_plan=False)
    # A written plan in the document governs for every doctor (the clone's rule 1).
    for username, *_ in DOCTORS:
        assert expected_items(username, with_plan, "D2", PROFILES)["plan_behavior"] == "follow_plan"
    assert expected_items("dr.ting", without_plan, "D2", PROFILES)["plan_behavior"] == "set_up_plan"
    assert expected_items("dr.patel", without_plan, "D2", PROFILES)["plan_behavior"] == "call_first"
    assert expected_items("dr.renny", without_plan, "D2", PROFILES)["plan_behavior"] == "contact_clinic"


def test_doc_governs_only_d2_with_a_document_plan():
    assert doc_governs(patient(flare_plan=True), "D2") is True
    assert doc_governs(patient(flare_plan=False), "D2") is False
    for dimension in ("D1", "D3", "D4", "none"):
        assert doc_governs(patient(flare_plan=True), dimension) is False


def test_rule_d3_d4_first_ask_from_profile():
    p = patient()
    assert expected_items("dr.ting", p, "D3", PROFILES) == {"first_ask": "timing", "must_not_recommend": ["naproxen", "prednisone"]}
    assert expected_items("dr.patel", p, "D3", PROFILES)["first_ask"] == "medication_list"
    assert expected_items("dr.renny", p, "D3", PROFILES)["first_ask"] == "past_reaction"
    assert expected_items("dr.ting", p, "D4", PROFILES)["first_ask"] == "past_experience"
    assert expected_items("dr.patel", p, "D4", PROFILES)["first_ask"] == "other_conditions"
    assert expected_items("dr.renny", p, "D4", PROFILES)["first_ask"] == "current_plan"


def test_no_memory_items():
    assert no_memory_items(patient(prescribed=["naproxen"])) == {
        "must_not_recommend": ["colchicine", "prednisone"], "no_memory_use": True}


def test_cases_file_matches_the_rule():
    cases = load("cases")
    assert cases
    for case in cases:
        assert set(case["expected"]) == {username for username, *_ in DOCTORS}
        for username, *_ in DOCTORS:
            want = no_memory_items(by_id[case["patient"]]) if case["dimension"] == "none" else expected_items(username, by_id[case["patient"]], case["dimension"], PROFILES)
            assert case["expected"][username] == want
        assert case["doc_governs"] is doc_governs(by_id[case["patient"]], case["dimension"]), case["id"]


def test_every_patient_passes_the_gout_gate():
    entry = DoctorGuidance(scope="general", condition_tags=["gout"], drug_tags=[])
    assert 4 <= len(PATIENTS) <= 5
    assert len(by_id) == len(PATIENTS)
    for p in load("patients"):
        assert p["text"].startswith("FICTIONAL - FOR EVALUATION")
        assert eligible(entry, patient_text(DischargeDocument(text=p["text"]), [ExtractedItem(**i, status="active") for i in p["items"]]))
        # Also from the document text alone, so the gate does not depend on items.
        assert eligible(entry, patient_text(DischargeDocument(text=p["text"]), []))
        assert all(i["source_quote"] in p["text"] for i in p["items"])
        assert prescription_names(p) <= set(ALLOWED_DRUGS)
        assert prescription_names(p) == set(p["prescribed"])
        assert p["past_effective"] is None or p["past_effective"] in p["prescribed"]
        # "The case allows" an NSAID only when the document notes no CKD.
        assert ("chronic kidney disease" in p["text"].lower()) == p["ckd"]


def test_patients_vary_what_the_profiles_condition_on():
    kinds = {(tuple(sorted(p["prescribed"])), p["ckd"]) for p in PATIENTS}
    assert (("colchicine", "naproxen"), False) in kinds
    assert (("colchicine",), True) in kinds
    assert (("naproxen",), False) in kinds
    assert any(p["past_effective"] == "prednisone" and p["flare_plan"] for p in PATIENTS)
    assert any(p["past_effective"] is None and not p["flare_plan"] for p in PATIENTS)


def test_no_drug_outside_the_allowed_set_is_named():
    other = ("ibuprofen", "indomethacin", "allopurinol", "celecoxib", "febuxostat", "aspirin", "probenecid")
    for p in PATIENTS:
        assert not any(drug in p["text"].lower() for drug in other), p["id"]
    for case in load("cases"):
        assert not any(drug in case["question"].lower() for drug in other), case["id"]


def test_cases_cover_every_dimension_for_every_patient_kind():
    cases = load("cases")
    counts = Counter(case["dimension"] for case in cases)
    assert set(counts) == {"D1", "D2", "D3", "D4", "none"}
    assert all(counts[d] >= 5 for d in ("D1", "D2", "D3", "D4"))
    assert 3 <= counts["none"] <= 5
    assert 22 <= len(cases) <= 28
    assert len({case["id"] for case in cases}) == len(cases)
    for d in ("D1", "D2", "D3", "D4"):
        assert {case["patient"] for case in cases if case["dimension"] == d} == set(by_id), d

    branches = set()
    for case in cases:
        if case["dimension"] != "D1":
            continue
        p = by_id[case["patient"]]
        for username, *_ in DOCTORS:
            items = case["expected"][username]
            preferred = PROFILES[username]["D1"]["preferred"]
            if "first_choice" in items:
                branches.add("first_choice")
            elif username == "dr.renny" and p["past_effective"] is None:
                branches.add("renny_without_history")
            elif preferred not in p["prescribed"]:
                branches.add("refer_missing")
            elif preferred == "naproxen" and p["ckd"]:
                branches.add("refer_ckd")
    assert branches == {"first_choice", "refer_missing", "refer_ckd", "renny_without_history"}

    plan_values = {case["expected"][u]["plan_behavior"] for case in cases if case["dimension"] == "D2" for u, *_ in DOCTORS}
    assert plan_values == {"set_up_plan", "call_first", "follow_plan", "contact_clinic"}
    # Patel's call-first habit is measured on patients without a plan.
    assert any(case["expected"]["dr.patel"]["plan_behavior"] == "call_first"
               for case in cases if case["dimension"] == "D2")
    assert any(case["doc_governs"] for case in cases)


def test_case_questions_are_distinct():
    questions = [case["question"] for case in load("cases")]
    for i, a in enumerate(questions):
        assert a.strip() and EM_DASH not in a
        for b in questions[i + 1:]:
            assert not near_duplicate(a, b), (a, b)


def test_side_effect_questions_leave_the_timing_open():
    # A D3 question that states when the symptom started relative to the doses
    # pre-answers Dr. Ting's first ask (timing).
    timing = re.compile(r"\b(since|after|before|yesterday|today|tonight|morning|started|when i take|hours?|days?)\b")
    for case in load("cases"):
        if case["dimension"] == "D3":
            assert not timing.search(case["question"].lower()), case["id"]


def test_no_em_dash_in_data_files():
    for name in ("profiles", "patients", "cases"):
        assert EM_DASH not in (DATA_DIR / f"{name}.json").read_text(encoding="utf-8")


def test_near_duplicate():
    assert near_duplicate("Can I take more colchicine?", "can i take more colchicine")
    assert not near_duplicate("Can I take more colchicine?", "What should I eat during a flare?")
    # 8 shared tokens of 10: Jaccard exactly 0.8.
    assert near_duplicate("Should I take my colchicine now for this flare?", "Should I take my colchicine now for the flare?")
    # 7 shared tokens of 11: below 0.8.
    assert not near_duplicate("Should I take my colchicine now for this flare?", "Should I take my colchicine for a bad flare?")
    assert not near_duplicate("", "Can I take more colchicine?")


TOPIC = {"D1": "flare medicine", "D2": "the flare plan", "D3": "side effects", "D4": "missing details"}


def good_entries():
    entries = []
    for d in DIMENSIONS:
        for n in ("first", "second"):
            answer = ("I usually prefer colchicine when the case allows. If it is not on your prescription, "
                      "please contact the clinic before taking anything new.") if d in ("D1", "D2") else \
                     "When you tell me about this, I first ask: when did it start compared with your doses?"
            entries.append({"dimension": d, "question": f"A general gout question about {TOPIC[d]}, the {n} one",
                            "answer": answer})
    return entries


def test_check_entries_accepts_a_well_formed_set():
    assert check_entries(good_entries(), load("cases")) == []


def test_check_entries_rejects_other_drugs_and_case_copies():
    cases = load("cases")
    entries = good_entries()
    entries[0]["answer"] += " Ibuprofen is another option."
    entries[1]["question"] = "Which medication do you use?"
    entries[1]["answer"] = "Celebrex works too. " + entries[1]["answer"]
    entries[2]["question"] = cases[0]["question"].upper()  # a copy of a test question
    entries[3]["question"] = cases[6]["question"].replace("these", "the")  # a near copy
    problems = check_entries(entries, cases)
    assert any("ibuprofen" in p and "entry 1" in p for p in problems), problems
    assert any("celebrex" in p and "entry 2" in p for p in problems), problems
    assert any("entry 3" in p and "test question" in p for p in problems), problems
    assert any("entry 4" in p and "test question" in p for p in problems), problems
    # A problem names no case question: the retry message goes back to the generator.
    assert not any(case["question"].lower() in p.lower() for p in problems for case in cases)


def test_check_entries_rejects_wrong_dimension_counts():
    entries = good_entries()
    entries[1]["dimension"] = "D3"
    problems = check_entries(entries, load("cases"))
    assert any("D1" in p and "1 entries" in p for p in problems), problems
    assert any("D3" in p and "3 entries" in p for p in problems), problems
    assert any("D9" in p for p in check_entries(good_entries() + [dict(good_entries()[0], dimension="D9")], []))


def test_check_entries_rejects_numbers_missing_referral_or_first_ask_and_em_dash():
    entries = good_entries()
    entries[0]["answer"] = "I prefer colchicine 0.6 mg first. If it is not on your prescription, contact the clinic."
    entries[2]["answer"] = "I like a written flare plan set up in advance."  # D2 without the clinic
    entries[4]["answer"] = "Side effects can happen with any medicine."  # D3 without a first ask
    entries[6]["question"] = "Missing details " + EM_DASH + " what then?"
    entries[7]["question"] = " "
    problems = check_entries(entries, [])
    assert any("entry 1" in p and "number" in p for p in problems), problems
    assert any("entry 3" in p and "clinic" in p for p in problems), problems
    assert any("entry 5" in p and "ask" in p for p in problems), problems
    assert any("entry 7" in p and "dash" in p for p in problems), problems
    assert any("entry 8" in p and "empty" in p for p in problems), problems


def test_committed_memory_is_valid():
    memory = load("memory")
    cases = load("cases")
    assert set(memory) == {username for username, *_ in DOCTORS} | {"generated"}
    assert memory["generated"]["model"] and re.fullmatch(r"\d{4}-\d{2}-\d{2}", memory["generated"]["date"])
    assert EM_DASH not in (DATA_DIR / "memory.json").read_text(encoding="utf-8")
    for username, *_ in DOCTORS:
        entries = memory[username]
        assert check_entries(entries, cases) == [], username
        assert all(e["drug_tags"] == [] and e["condition_tags"] == ["gout"] and e["synthetic"] is True
                   and e["scope"] == "general" for e in entries)
        assert all(set(e) == {"dimension", "question", "answer", "scope", "condition_tags", "drug_tags",
                              "synthetic"} for e in entries)


def test_committed_memory_leaves_the_no_memory_topics_uncovered():
    # The "none" cases (coffee, walks, seafood, water) must find no entry about them.
    topics = re.compile(r"\b(coffee|caffeine|walks?|walking|exercise|seafood|shrimp|fish|water|fluids?|drinks?|"
                        r"diet|foods?|meat|beer|alcohol)\b")
    memory = load("memory")
    for username, *_ in DOCTORS:
        for e in memory[username]:
            assert not topics.search((e["question"] + " " + e["answer"]).lower()), (username, e)
