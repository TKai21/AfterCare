from closing import (
    DRUG_REASON, EMERGENCY_REASON, MEDICATION_REASON, NO_TERMS_REASON, OTHER_CONCERNS_REASON,
    WARNING_REASON, close_blocker,
)
from drugnames import get_lexicon

ORIGINAL = "My incision looks a bit pink."
DOCTOR_TERMS = ["fever", "vomiting", "threw up"]


def absent(*terms):
    return [{"term": t, "status": "absent"} for t in terms]


def blocker(reply, original=ORIGINAL, emergency=(), doctor=DOCTOR_TERMS, document=(),
            signs=None, prescribed=("amoxicillin",), concerns=False):
    """signs=None: the model lists every given term as absent."""
    if signs is None:
        signs = absent(*doctor, *document)
    return close_blocker(original, reply, list(emergency), list(doctor), list(document),
                         list(signs), set(prescribed), get_lexicon(), other_concerns=concerns)


def test_reason_strings():
    assert EMERGENCY_REASON == "Emergency terms"
    assert WARNING_REASON == "Warning sign: {term}"
    assert MEDICATION_REASON == "Medication change reported"
    assert DRUG_REASON == "Drug not on the prescription: {name}"
    assert NO_TERMS_REASON == "No warning signs taught"
    assert OTHER_CONCERNS_REASON == "Other concerns"


def test_benign_reply_closes():
    assert blocker("Two days, no fever, no medication changes",
                   signs=absent(*DOCTOR_TERMS)) is None


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
    assert blocker("No fever", signs=absent("  FEVER ", "VOMITING", "Threw Up")) is None


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
    signs = absent(*DOCTOR_TERMS)
    assert blocker("No fever, never threw up.",
                   original="I stopped my antibiotic and the incision is pink.",
                   signs=signs) == "Medication change reported"
    assert blocker("No fever, never threw up.", original="I took ibuprofen, incision is pink.",
                   signs=signs) == "Drug not on the prescription: ibuprofen"


def test_timing_changes_are_medication_changes():
    signs = absent(*DOCTOR_TERMS)
    for reply in ("No fever, never threw up. I took my amoxicillin late yesterday.",
                  "I took it early this morning", "I took the pill at the wrong time",
                  "I take it at night instead", "I missed one", "I doubled up yesterday"):
        assert blocker(reply, signs=signs) == "Medication change reported", reply


def test_model_present_blocks_a_term_the_program_did_not_match():
    reply = "No fever but I vomited twice"
    doctor = ["fever", "threw up"]
    assert blocker(reply, doctor=doctor,
                   signs=[{"term": "fever", "status": "absent"},
                          {"term": "threw up", "status": "present"}]) == "Warning sign: threw up"
    assert blocker(reply, doctor=doctor,
                   signs=[{"term": "fever", "status": "absent"},
                          {"term": "threw up", "status": "unclear"}]) == "Warning sign: threw up"


def test_model_status_counts_for_document_terms_too():
    assert blocker("It looks fine", doctor=[], document=["redness"],
                   signs=[{"term": "Redness", "status": "unclear"}]) == "Warning sign: redness"


def test_model_status_for_a_term_it_was_not_given_is_ignored():
    assert blocker("It looks fine", signs=absent(*DOCTOR_TERMS) + [
        {"term": "made up", "status": "present"}]) is None


def test_benign_reply_with_every_term_absent_and_none_matched_closes():
    assert blocker("It looks fine and I feel well", signs=absent(*DOCTOR_TERMS)) is None


def test_a_given_term_left_out_of_warning_signs_blocks():
    # The matcher misses "vomited" for "threw up", and the model left it out.
    assert blocker("No fever. I vomited twice though.", doctor=["fever", "threw up"],
                   signs=absent("fever")) == "Warning sign: threw up"
    assert close_blocker(ORIGINAL, "It looks fine", [], DOCTOR_TERMS, ["redness"], [],
                         {"amoxicillin"}, get_lexicon(),
                         other_concerns=False) == "Warning sign: fever"


def test_every_given_term_absent_still_closes():
    assert blocker("It looks fine", document=["redness"],
                   signs=absent("fever", "vomiting", "threw up", "redness")) is None


def test_conflicting_entries_for_a_term_block():
    terms = ["fever", "threw up"]
    for signs in ([*absent("fever", "threw up"), {"term": "threw up", "status": "present"}],
                  [{"term": "Threw up.", "status": "present"}, *absent("fever", "threw up")],
                  [*absent("fever", "threw up"), {"term": "THREW UP", "status": "unclear"}],
                  [*absent("fever", "threw up"), {"term": "threw up", "status": "gone"}]):
        assert blocker("I vomited", doctor=terms, signs=signs) == "Warning sign: threw up", signs


def test_a_term_absent_twice_is_absent():
    assert blocker("It looks fine", doctor=["fever", "threw up"],
                   signs=absent("fever", "threw up", " Threw Up ")) is None


def test_an_unhashable_status_blocks_without_crashing():
    signs = [*absent("fever", "threw up"), {"term": "threw up", "status": ["absent"]}]
    assert blocker("It looks fine", doctor=["fever", "threw up"], signs=signs) == (
        "Warning sign: threw up")


def test_no_warning_terms_at_all_never_closes():
    # Nothing to check a reply against: the clone cannot know it is benign.
    reply = "Yes I have a fever and pus is coming out of the wound"
    assert blocker(reply, doctor=[], document=[], signs=[]) == "No warning signs taught"
    assert blocker("It looks fine", doctor=[], document=[], signs=[]) == (
        "No warning signs taught")


def test_other_concerns_must_be_an_explicit_false():
    assert blocker("It looks fine") is None
    for concerns in (True, None, "false", 0):
        assert blocker("It looks fine", concerns=concerns) == "Other concerns", concerns


def test_specific_reasons_come_before_the_general_ones():
    assert blocker("I stopped the antibiotic yesterday", doctor=[], signs=[],
                   concerns=True) == "Medication change reported"
    assert blocker("I had a fever", concerns=True,
                   signs=[{"term": "fever", "status": "present"}]) == "Warning sign: fever"


def test_the_check_button_stands_in_for_the_plain_denial_not_for_the_model():
    reply = "No, I have not had any more symptoms such as fever, since Monday"
    args = (ORIGINAL, reply, [], ["fever"], [])
    kwargs = dict(other_concerns=False)
    lexicon = get_lexicon()
    # Not a plain denial: blocked, as before the danger check.
    assert close_blocker(*args, absent("fever"), {"amoxicillin"}, lexicon, **kwargs) \
        == "Warning sign: fever"
    assert close_blocker(*args, absent("fever"), {"amoxicillin"}, lexicon, **kwargs,
                         button_denied={"fever"}) is None
    # The model must still read the sign as absent.
    assert close_blocker(*args, [{"term": "fever", "status": "present"}], {"amoxicillin"},
                         lexicon, **kwargs, button_denied={"fever"}) == "Warning sign: fever"
