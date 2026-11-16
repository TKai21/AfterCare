from closing import (
    DRUG_REASON, EMERGENCY_REASON, MEDICATION_REASON, WARNING_REASON, close_blocker,
)
from drugnames import get_lexicon

ORIGINAL = "My incision looks a bit pink."
DOCTOR_TERMS = ["fever", "vomiting", "threw up"]


def absent(*terms):
    return [{"term": t, "status": "absent"} for t in terms]


def blocker(reply, original=ORIGINAL, emergency=(), doctor=DOCTOR_TERMS, document=(),
            signs=(), prescribed=("amoxicillin",)):
    return close_blocker(original, reply, list(emergency), list(doctor), list(document),
                         list(signs), set(prescribed), get_lexicon())


def test_reason_strings():
    assert EMERGENCY_REASON == "Emergency terms"
    assert WARNING_REASON == "Warning sign: {term}"
    assert MEDICATION_REASON == "Medication change reported"
    assert DRUG_REASON == "Drug not on the prescription: {name}"


def test_benign_reply_closes():
    assert blocker("Two days, no fever, no medication changes",
                   signs=absent("fever", "vomiting")) is None


def test_emergency_terms_block():
    assert blocker("I have chest pain", emergency=["chest pain"]) == "Emergency terms"


def test_review_focus_1_threw_up_after_a_denied_fever_blocks():
    # "fever" is denied by both checks, but "threw up" is present.
    got = blocker("no fever, but I threw up twice",
                  signs=[{"term": "fever", "status": "absent"},
                         {"term": "threw up", "status": "present"}])
    assert got == "Warning sign: threw up"


def test_model_and_program_must_agree():
    # The program negates "fever"; the model says unclear: blocked.
    assert blocker("No fever", signs=[{"term": "fever", "status": "unclear"}]) == (
        "Warning sign: fever")
    # The model says absent, but the program does not see a plain denial: blocked.
    assert blocker("I don't think I have a fever", signs=absent("fever")) == (
        "Warning sign: fever")


def test_missing_model_status_blocks():
    assert blocker("No fever", signs=[]) == "Warning sign: fever"
    assert blocker("No fever", signs=absent("vomiting")) == "Warning sign: fever"


def test_model_term_matches_case_insensitively():
    assert blocker("No fever", signs=absent("  FEVER ")) is None


def test_warning_in_first_message_blocks():
    # The reply denies it, but the original question reported it.
    assert blocker("No fever now", original="I had a fever last night.",
                   signs=absent("fever")) == "Warning sign: fever"


def test_document_warning_terms_count():
    assert blocker("Some redness around it", doctor=[], document=["redness"]) == (
        "Warning sign: redness")
    assert blocker("No redness", doctor=[], document=["redness"],
                   signs=absent("redness")) is None


def test_medication_change_blocks():
    assert blocker("I started taking my pills twice a day") == "Medication change reported"
    assert blocker("I stopped the antibiotic yesterday") == "Medication change reported"
    assert blocker("I stopped amoxicillin on Monday") == "Medication change reported"
    assert blocker("I skipped my morning dose") == "Medication change reported"
    assert blocker("I took an extra one last night") == "Medication change reported"
    assert blocker("My other doctor gave me a new prescription") == "Medication change reported"
    assert blocker("Do not take more, ok") == "Medication change reported"  # MEDICATION_CHANGE


def test_ordinary_reply_is_not_a_medication_change():
    for reply in ("The pain started yesterday and it is better now",
                  "No medication changes, I take everything as written",
                  "It stopped hurting this morning"):
        assert blocker(reply) is None, reply


def test_prescribed_drug_in_reply_does_not_block():
    assert blocker("Still on amoxicillin, all fine") is None


def test_unprescribed_drug_in_reply_blocks():
    assert blocker("I had some ibuprofen for the pain") == (
        "Drug not on the prescription: ibuprofen")


def test_checks_run_in_order():
    reply = "I have chest pain and a fever, and I started taking ibuprofen"
    assert blocker(reply, emergency=["chest pain"]) == "Emergency terms"
    assert blocker(reply) == "Warning sign: fever"
    assert blocker("I started taking ibuprofen") == "Medication change reported"


def test_original_question_gets_the_medication_and_drug_checks():
    signs = absent("fever", "threw up")
    assert blocker("No fever, never threw up.",
                   original="I stopped my antibiotic and the incision is pink.",
                   signs=signs) == "Medication change reported"
    assert blocker("No fever, never threw up.", original="I took ibuprofen, incision is pink.",
                   signs=signs) == "Drug not on the prescription: ibuprofen"


def test_timing_changes_are_medication_changes():
    signs = absent("fever", "threw up")
    for reply in ("No fever, never threw up. I took my amoxicillin late yesterday.",
                  "I took it early this morning", "I took the pill at the wrong time",
                  "I take it at night instead", "I missed one", "I doubled up yesterday"):
        assert blocker(reply, signs=signs) == "Medication change reported", reply
