"""When the clone may close a case after its questions (spec 4.3-4.4).

The server, not the model, decides. A close is blocked (fails closed into a
handoff to the doctor) by emergency terms, any warning sign in the original
question or the reply that is not denied by both the program and the model, a
reported medication change, a drug the patient is not prescribed, no warning
signs to check against, or any other concern the model reports."""
import re
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent import MEDICATION_CHANGE
from drugnames import DrugLexicon
from interview import load_interview, summary_text
from models import CareTeamQuestion, ChatMessage, DoctorGuidance, GuidanceRule
from safety import _canon, match_terms, negated_terms

EMERGENCY_REASON = "Emergency terms"
WARNING_REASON = "Warning sign: {term}"
MEDICATION_REASON = "Medication change reported"
DRUG_REASON = "Drug not on the prescription: {name}"
NO_TERMS_REASON = "No warning signs taught"
OTHER_CONCERNS_REASON = "Other concerns"

WAITING = "waiting_on_patient"

# MEDICATION_CHANGE is written for the clone's own (imperative) wording; a
# patient reports a change in the past tense ("I stopped my pills"). These
# patterns lean broad: a false match only sends the case to the doctor.
_CHANGE_VERBS = (
    r"start(?:ed|ing)?|began|begun|begin(?:ning)?|stop(?:ped|ping)?|quit(?:ting)?"
    r"|discontinu\w*|skip(?:ped|ping)?|miss(?:ed|ing)?|forg[eo]t(?:ten)?|doubl\w*|halv\w*"
    r"|switch\w*|chang\w*|increas\w*|decreas\w*|reduc\w*|lower\w*|rais\w*|adjust\w*|cut(?:ting)?"
)
_MEDICATION_WORDS = (
    r"medications?|medicines?|meds?|pills?|tablets?|capsules?|doses?|dosage|antibiotics?"
    r"|prescriptions?|taking|using|to take"
)
PATIENT_MEDICATION_CHANGE = re.compile(
    rf"\b(?:{_CHANGE_VERBS})\s+(?:\S+\s+){{0,3}}?(?:{_MEDICATION_WORDS})\b"
    r"|\b(?:took|taken|taking|take)\s+(?:an?\s+|one\s+|two\s+)?"
    r"(?:extra|more|less|fewer|another|additional|double|half|a second|twice)\b"
    r"|\bnew\s+(?:medications?|medicines?|meds?|pills?|prescriptions?)\b"
    # Timing: a dose taken late, early or at another time, missed or doubled.
    r"|\b(?:took|taken|taking|take|had)\s+(?:\S+\s+){0,3}?"
    r"(?:late|later|early|earlier|at the wrong time)\b"
    r"|\bat the wrong time\b|\b(?:at night|in the morning|in the evening) instead\b"
    r"|\bmissed\b|\bdoubled up\b",
    re.IGNORECASE,
)


def _reports_medication_change(reply: str, prescribed: set[str]) -> bool:
    """MEDICATION_CHANGE or the patient-wording pattern, plus a change verb
    governing one of the patient's own drug names ("I stopped amoxicillin")."""
    if MEDICATION_CHANGE.search(reply) or PATIENT_MEDICATION_CHANGE.search(reply):
        return True
    names = sorted((re.escape(n) for n in prescribed if n), key=len, reverse=True)
    if not names:
        return False
    own = re.compile(rf"\b(?:{_CHANGE_VERBS})\s+(?:\S+\s+){{0,3}}?(?:{'|'.join(names)})\b",
                     re.IGNORECASE)
    return bool(own.search(reply))


def _key(term: str) -> str:
    return _canon(term).casefold()


def close_blocker(original: str, reply: str, emergency_terms: list[str],
                  doctor_warning_terms: list[str], document_warning_terms: list[str],
                  warning_signs: list[dict], prescribed: set[str],
                  lexicon: DrugLexicon, *, other_concerns) -> Optional[str]:
    """None when a close may be delivered, else the reason shown to the doctor.

    Checks, in order: emergency terms in the reply; a warning term matched in
    the original question (always blocks: the case was reported with it) or
    in the reply, unless negated_terms finds a plain denial AND the model's
    warning_signs lists that term as "absent" (a term the model left out is
    present, and a term the model lists more than once is absent only when
    every one of its entries says "absent"); any given term the model does not list as "absent" (present,
    unclear or left out), whether or not the program matched it, so every
    given term must be explicitly absent; medication-change wording in the
    original question or the reply; a lexicon drug name in either that is not
    in `prescribed`; no warning term given at all (nothing to check the reply
    against); and `other_concerns` (the model's reply-turn field) anything
    but an explicit False."""
    if emergency_terms:
        return EMERGENCY_REASON
    terms = list(dict.fromkeys([*doctor_warning_terms, *document_warning_terms]))
    in_original = match_terms(original, terms)
    if in_original:
        return WARNING_REASON.format(term=in_original[0])
    # A term is absent only when every entry for it says "absent": one
    # present, unclear or unknown status among them blocks.
    statuses: dict[str, set] = {}
    for sign in warning_signs:
        if isinstance(sign, dict) and isinstance(sign.get("term"), str):
            status = sign.get("status")
            statuses.setdefault(_key(sign["term"]), set()).add(
                status if isinstance(status, str) else None)  # an odd status blocks
    model_absent = {key for key, found in statuses.items() if found == {"absent"}}
    denied = negated_terms(reply, terms)
    for term in match_terms(reply, terms):
        if not (term in denied and _key(term) in model_absent):
            return WARNING_REASON.format(term=term)
    # The model reads meaning the phrase matcher misses ("vomited" for "threw
    # up"): every given term must be explicitly "absent", matched or not; a
    # term the model calls present or unclear, or leaves out, blocks.
    for term in terms:
        if _key(term) not in model_absent:
            return WARNING_REASON.format(term=_canon(term))
    if any(_reports_medication_change(text, prescribed) for text in (original, reply)):
        return MEDICATION_REASON
    allowed = {name.lower() for name in prescribed}
    unprescribed = sorted((lexicon.find(original) | lexicon.find(reply)) - allowed)
    if unprescribed:
        return DRUG_REASON.format(name=unprescribed[0])
    if not terms:
        return NO_TERMS_REASON
    if other_concerns is not False:
        return OTHER_CONCERNS_REASON
    return None


def intake_rounds(q: CareTeamQuestion) -> list[dict]:
    """Every round of the clone's intake, in order: the completed rounds
    ([{asked, reply}], `intake`) and the round still waiting for a reply or
    just answered (`asked`, `patient_reply`) when `intake` does not end with it.
    A row from before rounds has no `intake`: its one round is asked/patient_reply.
    An interview's `intake` (a dict with items) keeps its rounds under "rounds"."""
    listed = q.intake.get("rounds") if isinstance(q.intake, dict) else q.intake
    rounds = [{"asked": r.get("asked"), "reply": r.get("reply")}
              for r in (listed if isinstance(listed, list) else []) if isinstance(r, dict)]
    if q.asked and not (rounds and rounds[-1]["asked"] == q.asked):
        rounds.append({"asked": q.asked, "reply": q.patient_reply})
    return rounds


def intake_summary(q: CareTeamQuestion) -> Optional[str]:
    """An interview's "Known / Not sure / Not answered" lines for the doctor,
    shown above the rounds; None for any other intake."""
    iv = load_interview(q.intake)
    return (summary_text(iv) or None) if iv is not None else None


async def pending_clarify(session: AsyncSession, document_id: int,
                          doctor_id: int) -> Optional[tuple[ChatMessage, CareTeamQuestion]]:
    """The clone's open clarify in this patient-doctor conversation: the latest
    assistant message, when it is a clarify still marked pending, with the
    care-team question it created. None otherwise (another doctor's
    conversation never counts). The question may have left waiting_on_patient
    (the doctor picked it up); the caller then leaves it alone."""
    message = await session.scalar(
        select(ChatMessage)
        .where(ChatMessage.document_id == document_id, ChatMessage.doctor_id == doctor_id,
               ChatMessage.role == "assistant")
        .order_by(ChatMessage.id.desc())
        .limit(1)
    )
    payload = message.payload if message is not None else None
    if not isinstance(payload, dict) or payload.get("kind") != "clarify" or not payload.get("pending"):
        return None
    question_id = payload.get("waiting_question_id")
    if not isinstance(question_id, int):
        return None
    question = await session.scalar(
        select(CareTeamQuestion).where(
            CareTeamQuestion.id == question_id,
            CareTeamQuestion.document_id == document_id,
            CareTeamQuestion.doctor_id == doctor_id,
        )
    )
    if question is None:
        return None
    return message, question


async def doctor_warning_terms(session: AsyncSession, doctor_id: int) -> list[str]:
    """The union of warning_terms over every live warning habit of this doctor
    (not suppressed, entry active), read directly: doctor_habits keeps one
    habit per statement, and a dropped duplicate's terms must still block a
    close. In order of first appearance."""
    rows = await session.scalars(
        select(GuidanceRule.warning_terms)
        .join(DoctorGuidance, DoctorGuidance.id == GuidanceRule.guidance_id)
        .where(
            DoctorGuidance.doctor_id == doctor_id,
            DoctorGuidance.active.is_(True),
            GuidanceRule.doctor_id == doctor_id,
            GuidanceRule.kind == "habit",
            GuidanceRule.habit_type == "warning",
            GuidanceRule.suppressed.is_(False),
        )
        .order_by(GuidanceRule.id)
    )
    terms: dict[str, None] = {}
    for found in rows:
        for term in found or []:
            if isinstance(term, str) and term.strip():
                terms.setdefault(term, None)
    return list(terms)
