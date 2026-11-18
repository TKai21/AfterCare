import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, StringConstraints, model_validator
from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agent import (
    HISTORY_LIMIT, AgentResult, GuidanceUnavailable, PendingClarify, clarify_questions,
    dedupe_terms, emergency_result, fallback_answer, judge_reply, rounds_text, run_agent,
    strip_markers,
)
from auth import current_patient
from closing import (
    EMERGENCY_REASON, WAITING, WARNING_REASON, close_blocker, doctor_warning_terms,
    pending_clarify,
)
from confidence import LOW_CONFIDENCE_REASON, score_answer
from db import get_session
from doctors import doctor_label, require_doctor
from druglabels import patient_rxcuis
from drugnames import allowed_drug_names, get_lexicon, prescribed_drug_names
from guidance import Unit, doctor_habits, retrieve
from interview import (
    ITEM_ATTEMPTS_MAX, NOT_SURE_TEXT, Interview, as_pending, ask_next, asked_item, build_items,
    done, dump_interview, extract, habit_ids, is_interview, live_habit_texts, load_interview,
    mark, open_safety_item, similar_case_habits, so_far,
)
from llm import LLMError
from models import CareTeamQuestion, ChatMessage, User
from routes_care_team import MAX_QUESTION_CHARS, add_question
from routes_documents import active_document, confirmed_document, load_items
from safety import _canon, detect_emergency, match_terms, warning_terms

router = APIRouter()
log = logging.getLogger(__name__)

# A care-team question with this reason is added at most once while open.
SOURCES_DISAGREE = "Sources disagree"

# The waiting question of a clarify (spec 4.3-4.4): its reason once it is the
# doctor's, and what stopped the clone from closing (CareTeamQuestion.stopped_by).
NEEDS_DOCTOR = "Needs doctor after questions"
NOT_ANSWERED = "Patient did not answer the clone's questions"
LOW_CONFIDENCE_STOP = "Low confidence"
COULD_NOT_ANSWER = "Clone could not answer"
EMERGENCY_ANSWER_STOP = "Emergency answer"
HABITS_UNAVAILABLE = "The doctor's warning signs could not be loaded"
STOPPED_BY_CHARS = 100


@dataclass
class Settle:
    """What a reply turn does to the waiting question."""
    status: str  # open | resolved | waiting_on_patient (the clone asks another round)
    reason: Optional[str] = None  # set when it becomes open
    stopped_by: Optional[str] = None
    answer: Optional[str] = None  # set when the clone resolved it
    asked: Optional[str] = None  # the new round's questions, when it keeps waiting


def handoff_result(doctor_name: str) -> AgentResult:
    """The patient's side of a handoff: the fixed sentence. The waiting question
    is the doctor's copy, so no second question is added."""
    return AgentResult(kind="not_covered", answer=fallback_answer(doctor_name), sources=[],
                       flagged=True, care_team_question=None)


def settle_reply(result: AgentResult, terms: list[str], blocker: Optional[str],
                 gate_passed: bool, doctor_name: str) -> tuple[AgentResult, Settle]:
    """The delivered result and the waiting question's fate on a reply turn.
    `result` is the agent's (before the gate's low-confidence swap); `blocker`
    is close_blocker's reason for a close or an answer; `gate_passed` is False
    only when the confidence gate ran and failed. A reported symptom always
    reaches the doctor unless an unblocked close or answer settled it: a
    blocked answer is still delivered, but the question goes to the doctor.

    Emergency terms in the reply: whatever kind the model gave (a close, an
    answer, anything but its own validated emergency text) the patient gets
    the fixed emergency answer, and the case goes to the doctor."""
    if terms:
        delivered = (replace(result, care_team_question=None, flagged=False)
                     if result.kind == "emergency" else emergency_result())
        return delivered, Settle("open", NEEDS_DOCTOR, EMERGENCY_REASON)
    if result.kind == "emergency":
        return (replace(result, care_team_question=None, flagged=False),
                Settle("open", NEEDS_DOCTOR, EMERGENCY_ANSWER_STOP))
    if result.replies_to_questions is False:
        # A blocker (say a warning sign in the new message) is the more useful
        # summary for the doctor; otherwise the patient simply moved on.
        settle = (Settle("open", NEEDS_DOCTOR, blocker) if blocker is not None
                  else Settle("open", NOT_ANSWERED, NOT_ANSWERED))
        if result.kind in ("close", "clarify"):  # neither is a reply to a new question
            return handoff_result(doctor_name), settle
        # The new question is answered as usual (its own gate and question).
        return result, settle
    if result.kind == "clarify":
        # Another round only while nothing in the intake so far could stop a
        # close: a warning sign, a medication change, another concern or an
        # unreadable judge hands the case to the doctor now.
        if blocker is not None or result.replies_to_questions is not True:
            return handoff_result(doctor_name), Settle(
                "open", NEEDS_DOCTOR, blocker or COULD_NOT_ANSWER)
        return (replace(result, care_team_question=None, flagged=False),
                Settle(WAITING, asked=strip_markers(result.answer)))
    if result.kind == "close":
        stop = blocker or (None if gate_passed else LOW_CONFIDENCE_STOP)
        if stop is not None:
            return handoff_result(doctor_name), Settle("open", NEEDS_DOCTOR, stop)
        return result, Settle("resolved", answer=strip_markers(result.answer))
    if result.kind == "answer":
        if not gate_passed:
            return handoff_result(doctor_name), Settle("open", NEEDS_DOCTOR, LOW_CONFIDENCE_STOP)
        if blocker is not None:
            return (replace(result, care_team_question=None),
                    Settle("open", NEEDS_DOCTOR, blocker))
        if result.care_team_question:  # the clone flagged it: the doctor sees the reply
            stop = (result.care_team_reason or COULD_NOT_ANSWER)[:STOPPED_BY_CHARS]
            return (replace(result, care_team_question=None),
                    Settle("open", NEEDS_DOCTOR, stop))
        return result, Settle("resolved", answer=strip_markers(result.answer))
    # not_covered (rule 6 or 7), out_of_scope or the validation fallback.
    return handoff_result(doctor_name), Settle("open", NEEDS_DOCTOR, COULD_NOT_ANSWER)


SETTLED_MEANWHILE = "Case was settled before this reply"


async def settle_waiting(session: AsyncSession, waiting_id: int, settle: Settle,
                         pending: PendingClarify, reply: str, user_id: int,
                         document_id: int, doctor_id: int,
                         intake: Optional[dict] = None) -> Optional[int]:
    """Settle the waiting question, only while it is still waiting: a
    conditional UPDATE, so of two concurrent replies only the first owns it.
    The reply is appended to the question's rounds (`intake`), whatever the
    outcome. A question the inbox timed out (open, unanswered, reason
    NOT_ANSWERED and no reply yet) still counts as waiting for this reply: the
    reply fills that same card, resolves it on a close, or reopens it as
    waiting when the clone asks another round, so the doctor never sees two
    cards. When it is no longer waiting (the doctor answered it, or another
    reply settled it, say a double-tapped send), a handoff or another round
    becomes a new open question for the doctor with every round, so a reply is
    never lost and a settled case is never interviewed again; a resolving
    outcome then only delivers its answer.

    An interview (item shape) passes its state as `intake`, stored in place of
    the rounds list; a new question for the doctor leaves out the round that
    was never delivered (its reply None).

    Returns the id of the row waiting for the patient's next reply (another
    round), else None: for another round, None means the case went to the
    doctor instead, and the caller must not deliver the questions."""
    rounds = [{"asked": strip_markers(r["asked"]), "reply": r["reply"]}
              for r in pending.all_rounds(reply)]
    values = {"status": settle.status, "patient_reply": reply, "stopped_by": settle.stopped_by,
              "intake": rounds if intake is None else intake}
    if settle.reason is not None:
        values["reason"] = settle.reason
    if settle.answer is not None:
        values["answer"] = settle.answer
    if settle.status == WAITING:
        # The next round: its questions wait for a reply, and the timeout restarts.
        values.update(asked=settle.asked, patient_reply=None, reason="",
                      last_asked_at=datetime.now(timezone.utc).replace(tzinfo=None))
    still_waiting = or_(
        CareTeamQuestion.status == WAITING,
        and_(CareTeamQuestion.status == "open", CareTeamQuestion.answer.is_(None),
             CareTeamQuestion.reason == NOT_ANSWERED, CareTeamQuestion.patient_reply.is_(None)),
    )
    done = await session.execute(
        update(CareTeamQuestion)
        .where(CareTeamQuestion.id == waiting_id, still_waiting)
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    if done.rowcount == 1:
        return waiting_id if settle.status == WAITING else None
    if settle.status not in ("open", WAITING):
        return None
    row = await add_question(
        session, user_id, document_id, question=pending.question,
        reason=(settle.reason if settle.status == "open" else None) or NEEDS_DOCTOR,
        source="agent", doctor_id=doctor_id, asked=strip_markers(pending.asked))
    row.patient_reply = reply
    row.stopped_by = settle.stopped_by if settle.status == "open" else SETTLED_MEANWHILE
    row.intake = rounds if intake is None else {
        **intake, "rounds": [r for r in intake.get("rounds") or [] if r.get("reply") is not None]}
    return None


class AskBody(BaseModel):
    # Optional only so a missing id reaches the emergency backstop; any other
    # message without a valid doctor is refused (422).
    doctor_id: Optional[int] = None
    # Empty only with not_sure: the server then stores NOT_SURE_TEXT.
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, max_length=MAX_QUESTION_CHARS)
    ]
    # The interview's "Not sure" button, for the question of round `round`.
    not_sure: bool = False
    round: Optional[int] = None

    @model_validator(mode="after")
    def _question_or_not_sure(self):
        if self.not_sure:
            self.question = NOT_SURE_TEXT  # the client's text is never used
        elif not self.question:
            raise ValueError("A question is required.")
        return self


def answer_payload(result, terms: list[str], doctor_id: Optional[int],
                   doctor_name: Optional[str], confidence: Optional[float] = None) -> dict:
    return {
        "kind": result.kind,
        "answer": result.answer,
        "sources": [
            {"number": e.number, "kind": e.kind, "title": e.title, "url": e.url,
             "content": e.content, "guidance_id": e.guidance_id}
            for e in result.sources
        ],
        "flagged": result.flagged,
        "emergency": {"terms": terms, "banner": bool(terms) or result.kind == "emergency"},
        # Read by the doctor's guidance list to count citations.
        "guidance_ids": result.guidance_ids,
        "habit_rule_ids": result.habit_rule_ids,
        "doctor_id": doctor_id,
        "doctor_name": doctor_name,
        # The confidence gate's score of the clone's answer (None: not scored).
        "confidence": confidence,
    }


def low_confidence_result(question: str, doctor_name: str) -> AgentResult:
    return AgentResult(
        kind="not_covered", answer=fallback_answer(doctor_name), sources=[], flagged=True,
        care_team_question=question, care_team_reason=LOW_CONFIDENCE_REASON,
    )


def message_view(m: ChatMessage) -> dict:
    return {
        "role": m.role,
        "content": m.content,
        "payload": m.payload,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


async def recent_history(
    session: AsyncSession, document_id: int, doctor_id: Optional[int]
) -> list[dict]:
    """The last messages of one clone's conversation about a document. With no
    doctor (a question nobody has picked up) there is no conversation to show."""
    if doctor_id is None:
        return []
    rows = await session.scalars(
        select(ChatMessage)
        .where(ChatMessage.document_id == document_id, ChatMessage.doctor_id == doctor_id)
        .order_by(ChatMessage.id.desc())
        .limit(HISTORY_LIMIT)
    )
    return [{"role": m.role, "content": m.content} for m in reversed(list(rows))]


# ---------- the clone interview (spec 2026-11-17, sections 4.1-4.3 and 4.7) ----------

def interview_backstop(terms: list[str], reply: str, iv: Interview) -> list[str]:
    """The emergency terms that still count on an interview reply turn. The one
    place the denied-danger-signs rule (spec 4.4) will relax the backstop; until
    then every match counts, so an interview reply gets today's emergency answer."""
    return terms


@dataclass
class InterviewStep:
    """What an interview reply turn does next: hand off now (`settle`), ask the
    next question (`asked`: item index, text, is the doctor's words), or
    neither, when the interview is over and the agent answers or closes."""
    settle: Optional[Settle] = None
    asked: Optional[tuple[int, str, bool]] = None
    habit_texts: dict[int, str] = field(default_factory=dict)


def interview_clarify(iv: Interview, asked: tuple[int, str, bool], habit_texts: dict[int, str],
                      max_rounds: int) -> tuple[AgentResult, dict]:
    """The clarify of one interview question and what its payload adds."""
    n, text, doctor = asked
    habit_id = next((h for h in iv.items[n].habits
                     if doctor and " ".join(habit_texts.get(h, "").split()) == text), None)
    result = AgentResult(kind="clarify", answer=text, sources=[], flagged=False,
                         care_team_question=None,
                         habit_rule_ids=[habit_id] if habit_id is not None else [])
    return result, {
        "questions": [{"text": text, "doctor": doctor}],
        "interview": {"so_far": so_far(iv), "round": len(iv.rounds), "max_rounds": max_rounds,
                      "item": iv.items[n].label},
    }


async def start_interview(session: AsyncSession, llm, question: str, doctor_id: int,
                          doctor_name: str, document_terms: list[str], allowed_drugs,
                          settings) -> Optional[tuple[Interview, AgentResult, dict]]:
    """A first turn the 4.7 intake would act on becomes the interview's first
    question: (state, clarify, payload additions). None keeps the agent's
    result (4.7): no similar case habits nor document warning terms, the
    patient reports no symptom, no item, every item answered already, or any
    failure (the embedding, the database, a model call)."""
    try:
        asks, warns = await similar_case_habits(session, llm, doctor_id, question,
                                                settings.interview_cases)
        for habit in (*asks, *warns):
            session.expunge(habit)  # stays loaded after the commit
        await session.commit()
    except Exception:
        log.exception("Reading the similar cases' habits failed")
        try:
            await session.rollback()
        except Exception:
            log.exception("Rollback after failing to read the similar cases failed")
        return None
    if not asks and not warns and not document_terms:
        return None
    built = await build_items(llm, question, asks, warns, document_terms, allowed_drugs,
                              max_items=settings.interview_max_items,
                              max_safety=settings.interview_max_safety)
    if built is None or not built[0] or not built[1]:
        return None
    iv = Interview(items=built[1])
    await extract(llm, iv, question, "")
    texts = {h.id: h.quote for h in asks if isinstance(h.quote, str)}
    asked = await ask_next(llm, iv, texts, allowed_drugs, doctor_name,
                           settings.interview_max_rounds)
    if asked is None:
        return None
    result, extra = interview_clarify(iv, asked, texts, settings.interview_max_rounds)
    return iv, result, extra


def _term_key(term: str) -> str:
    return _canon(term).casefold()


async def read_interview_reply(llm, iv: Interview, original: str, reply: str,
                               match_pool: list[str]) -> Optional[Settle]:
    """Apply a typed reply to the interview: the reply judge reads the open
    safety items and every warning term the reply names, then the extractor
    fills the items the patient's words answer. A Settle when the case goes to
    the doctor now (the judge failed, the patient asked something else, or a
    danger sign is present), else None."""
    n = asked_item(iv)
    open_safety = [i.label for i in iv.items if i.safety and i.status == "unknown"]
    matched = match_terms(reply, list(dict.fromkeys([*match_pool, *open_safety])))
    judge_terms = dedupe_terms([*open_safety, *matched])
    judged = await judge_reply(llm, as_pending(iv, original), reply, judge_terms)
    if judged is None:
        return Settle("open", NEEDS_DOCTOR, COULD_NOT_ANSWER)
    if not judged.replies:
        return Settle("open", NOT_ANSWERED, NOT_ANSWERED)
    statuses: dict[str, set] = {}
    for sign in judged.signs:
        statuses.setdefault(_term_key(sign["term"]), set()).add(sign["status"])
    for term in judge_terms:
        if "present" in statuses.get(_term_key(term), set()):
            return Settle("open", NEEDS_DOCTOR, WARNING_REASON.format(term=term))
    # Absent counts only for the safety item just asked or one the reply names:
    # the judge calls a term nobody mentioned "absent" too.
    named = {_term_key(t) for t in matched}
    for k, item in enumerate(iv.items):
        if (item.safety and item.status == "unknown"
                and (k == n or _term_key(item.label) in named)
                and statuses.get(_term_key(item.label)) == {"absent"}):
            mark(iv, k, "answered")
    await extract(llm, iv, original, reply)
    if n is not None and iv.items[n].status == "unknown" \
            and iv.items[n].attempts >= ITEM_ATTEMPTS_MAX:
        mark(iv, n, "unsure")  # asked twice, still unclear
    return None


async def interview_turn(session: AsyncSession, llm, iv: Interview, original: str, reply: str,
                         *, not_sure: bool, doctor_id: int, doctor_name: str, allowed_drugs,
                         settings, match_pool: list[str], terms_failed: bool) -> InterviewStep:
    """One reply turn of the interview, the agent not called: record the reply
    on the asked round, read it ("Not sure" marks the item unsure, no judge),
    then pick and ask the next item. Fails closed into a handoff."""
    if iv.rounds:
        iv.rounds[-1]["reply"] = reply
    if not iv.items or not iv.rounds:  # an unreadable intake
        return InterviewStep(settle=Settle("open", NEEDS_DOCTOR, COULD_NOT_ANSWER))
    if terms_failed:
        return InterviewStep(settle=Settle("open", NEEDS_DOCTOR, HABITS_UNAVAILABLE))
    if not_sure:
        n = asked_item(iv)
        if n is not None:
            mark(iv, n, "unsure")
    else:
        stop = await read_interview_reply(llm, iv, original, reply, match_pool)
        if stop is not None:
            return InterviewStep(settle=stop)
    max_rounds = settings.interview_max_rounds
    if done(iv, max_rounds):
        return InterviewStep()
    try:
        texts = await live_habit_texts(session, doctor_id, habit_ids(iv))
        await session.commit()
    except Exception:
        log.exception("Reading the interview's habits failed")
        try:
            await session.rollback()
        except Exception:
            log.exception("Rollback after failing to read the interview's habits failed")
        return InterviewStep(settle=Settle("open", NEEDS_DOCTOR, COULD_NOT_ANSWER))
    asked = await ask_next(llm, iv, texts, allowed_drugs, doctor_name, max_rounds)
    return InterviewStep(asked=asked, habit_texts=texts)


async def end_clarify_message(session: AsyncSession, clarify_id: Optional[int]) -> None:
    """Any reply turn ends the clarify; a new dict so the JSON change is saved.
    Every writer sets the same value, so a concurrent reply cannot undo it."""
    if clarify_id is None:
        return
    message = await session.get(ChatMessage, clarify_id)
    if message is not None:
        message.payload = {**(message.payload or {}), "pending": False}


async def deliver_interview_step(session: AsyncSession, step: InterviewStep, iv: Interview,
                                 pending: PendingClarify, reply: str, *, clarify_id: int,
                                 waiting_id: int, user_id: int, doc_id: int, doctor_id: int,
                                 doctor_name: str, max_rounds: int) -> dict:
    """Save and return an interview reply turn that asks again or hands off.
    The waiting row moves only while it is still waiting (settle_waiting); a
    case settled meanwhile is handed off instead of asked again."""
    extra = None
    if step.asked is not None:
        result, extra = interview_clarify(iv, step.asked, step.habit_texts, max_rounds)
        settle = Settle(WAITING, asked=result.answer)
    else:
        result, settle = handoff_result(doctor_name), step.settle
    await end_clarify_message(session, clarify_id)
    next_id = await settle_waiting(session, waiting_id, settle, pending, reply, user_id, doc_id,
                                   doctor_id, intake=dump_interview(iv))
    if settle.status == WAITING and next_id is None:
        result, extra = handoff_result(doctor_name), None
    payload = answer_payload(result, [], doctor_id, doctor_name)
    if extra is not None:
        payload.update(pending=True, waiting_question_id=next_id, **extra)
    session.add(ChatMessage(document_id=doc_id, doctor_id=doctor_id, role="user", content=reply))
    session.add(ChatMessage(document_id=doc_id, doctor_id=doctor_id, role="assistant",
                            content=result.answer, payload=payload))
    await session.commit()
    return payload


@router.post("/api/ask")
async def ask(
    body: AskBody,
    request: Request,
    user: User = Depends(current_patient),
    session: AsyncSession = Depends(get_session),
):
    doc = await confirmed_document(session, user)
    items = await load_items(session, doc)
    # Plain values: a failed guidance search rolls the session back, which
    # expires the ORM objects, and expired objects cannot be reloaded here.
    user_id, doc_id = user.id, doc.id
    # Read here, before the transaction ends: only these drugs' labels may be quoted.
    library_rxcuis = await patient_rxcuis(session, doc_id)
    # The backstop runs first on every message; the emergency path never depends
    # on the model (R19).
    terms = detect_emergency(body.question, warning_terms(items))
    try:
        doctor = await require_doctor(session, body.doctor_id)
    except HTTPException:
        if not terms:
            raise
        # No conversation to save it in, so the fixed answer is only returned.
        return answer_payload(emergency_result(), terms, None, None)
    doctor_id, name = doctor.id, doctor_label(doctor)
    llm = request.app.state.llm
    if llm is None and not terms:
        raise HTTPException(status_code=503, detail="The language model is not configured.")
    # A cheap read, also on an emergency turn: the waiting question must reach
    # the doctor. The pending clarify is this conversation's only (Focus 4).
    try:
        # A savepoint: a failed lookup rolls back only itself, so the user,
        # document and items read above stay loaded.
        async with session.begin_nested():
            found = await pending_clarify(session, doc_id, doctor_id)
    except Exception:
        if not terms:
            raise
        # The emergency answer never fails on the waiting question's lookup.
        log.exception("Looking up the pending clarify failed")
        found = None
    pending: Optional[PendingClarify] = None
    iv: Optional[Interview] = None  # an interview reply turn (item-shape intake)
    clarify_id = waiting_id = None
    if found is not None:
        clarify_message, waiting = found
        clarify_id, waiting_id = clarify_message.id, waiting.id
        if is_interview(waiting.intake):
            # An unreadable item-shape intake is kept as no items: the turn hands off.
            iv = load_interview(waiting.intake) or Interview(items=[], rounds=[
                {"asked": waiting.asked or clarify_message.content, "item": None, "reply": None}])
            pending = as_pending(iv, waiting.question)
        else:
            pending = PendingClarify(
                question=waiting.question, asked=waiting.asked or clarify_message.content,
                rounds=[r for r in (waiting.intake or []) if isinstance(r, dict)])
    if body.not_sure and (iv is None or len(iv.rounds) != body.round):
        # A double tap or a stale button: the current state, no new round.
        if found is None:
            raise HTTPException(status_code=409, detail="There is no question to answer.")
        return found[0].payload
    if iv is not None and terms:
        terms = interview_backstop(terms, body.question, iv)
    # On a reply turn the patient's original message also names drugs they
    # asked about (with the lexicon's equivalents).
    asked_text = body.question if pending is None else "\n".join(
        [body.question, pending.question, *(r.get("reply") or "" for r in pending.rounds)])
    allowed_drugs = frozenset(
        allowed_drug_names(items, library_rxcuis, asked_text, get_lexicon()))
    gate_score: Optional[float] = None  # the confidence gate's score, when it ran
    first_iv: Optional[Interview] = None  # a first turn that starts the interview
    interview_extra: dict = {}
    gate_passed = True
    blocker: Optional[str] = None  # why a close may not be delivered
    if llm is None:
        result = emergency_result()
    else:
        history = await recent_history(session, doc_id, doctor_id)
        await session.commit()  # end the read transaction before the slow model calls
        # A failed guidance search rolls the session back, which expires every
        # object in it; detached, the user, document and items the turn reads
        # stay loaded (the prefetch can fail before the agent has read them).
        for loaded in (user, doc, *items):
            session.expunge(loaded)
        habits = []
        doctor_terms: list[str] = []  # every live warning habit's terms (reply turns)
        terms_failed = False
        if not terms:  # an emergency answer does not wait on memory
            try:
                habits = await doctor_habits(session, doctor_id)
                for habit in habits:
                    session.expunge(habit)  # stays loaded after any later rollback
                await session.commit()
            except Exception:
                log.exception("Loading the doctor's habits failed")
                habits = []
                try:
                    await session.rollback()
                except Exception:
                    log.exception("Rollback after failing to load habits failed")
            try:
                doctor_terms = await doctor_warning_terms(session, doctor_id)
                await session.commit()
            except Exception:
                # Fail closed: without the doctor's warning signs no close and
                # no ask-first questions.
                log.exception("Loading the doctor's warning terms failed")
                doctor_terms, terms_failed = [], True
                try:
                    await session.rollback()
                except Exception:
                    log.exception("Rollback after failing to load warning terms failed")
        document_terms = warning_terms(items)
        settings = request.app.state.settings
        if iv is not None and not terms:
            step = await interview_turn(
                session, llm, iv, pending.question, body.question, not_sure=body.not_sure,
                doctor_id=doctor_id, doctor_name=name, allowed_drugs=allowed_drugs,
                settings=settings, match_pool=[*doctor_terms, *document_terms],
                terms_failed=terms_failed)
            if step.settle is not None or step.asked is not None:
                return await deliver_interview_step(
                    session, step, iv, pending, body.question, clarify_id=clarify_id,
                    waiting_id=waiting_id, user_id=user_id, doc_id=doc_id, doctor_id=doctor_id,
                    doctor_name=name, max_rounds=settings.interview_max_rounds)
            # The interview is over: the agent answers or closes, no more questions.
        memory_failed = False

        async def guidance_search(query: str) -> list[Unit]:
            # The guidance memory must never break a turn (an emergency answer
            # included): any failure other than the model's becomes a tool error.
            nonlocal memory_failed
            if memory_failed:
                raise GuidanceUnavailable("The guidance memory failed earlier in this turn.")
            try:
                hits = await retrieve(session, llm, user, doc, items, query, doctor_id=doctor_id,
                                      floor=settings.guidance_floor, margin=settings.guidance_margin,
                                      limit=settings.guidance_top_k)
                await session.commit()  # no read transaction held across model calls
            except LLMError:
                await session.commit()
                raise
            except Exception as exc:
                memory_failed = True
                log.exception("Doctor guidance search failed")
                try:
                    await session.rollback()
                except Exception:
                    log.exception("Rollback after a failed guidance search failed")
                raise GuidanceUnavailable(str(exc)) from exc
            return [unit for unit, _score in hits]

        search = guidance_search
        prefetch = None
        if not terms:  # an emergency answer does not wait on the guidance memory
            try:
                prefetch = await search(body.question)
            except (LLMError, GuidanceUnavailable):
                prefetch = None  # the turn goes on without it, as the tool does

        try:
            result = await run_agent(
                llm, request.app.state.store, request.app.state.foods,
                doc, items, history, body.question, terms,
                guidance_search=search, doctor_name=name, prefetch=prefetch,
                library_rxcuis=library_rxcuis, allowed_drugs=allowed_drugs,
                habits=habits, pending=pending,
                warning_terms=[*doctor_terms, *document_terms] if pending is not None else (),
                first_turn_terms=(None if terms_failed else [*doctor_terms, *document_terms]),
                max_clarify_rounds=(len(pending.rounds) + 1 if iv is not None
                                    else settings.clarify_max_rounds),
            )
        except LLMError as exc:
            if not terms:
                raise HTTPException(
                    status_code=502,
                    detail="The assistant could not answer right now. Please try again.",
                ) from exc
            result = emergency_result()

        if pending is None and not terms and result.intake_eligible:
            started = await start_interview(session, llm, body.question, doctor_id, name,
                                            document_terms, allowed_drugs, settings)
            if started is not None:
                first_iv, result, interview_extra = started

        if pending is not None and not terms:
            # The server, not the model, decides whether a close is delivered
            # (4.4); the same checks decide whether an answer settles the case,
            # and name what the doctor must see when the patient asked something else.
            lexicon = get_lexicon()
            # Every reply of the intake counts, not only this one.
            replies = "\n".join(r["reply"] for r in pending.all_rounds(body.question))
            blocker = HABITS_UNAVAILABLE if terms_failed else close_blocker(
                pending.question, replies, terms, doctor_terms, document_terms,
                result.warning_signs, prescribed_drug_names(items, library_rxcuis, lexicon),
                lexicon, other_concerns=result.other_concerns)
            if blocker is None and iv is not None:
                # A danger sign the patient was asked about and did not deny.
                label = open_safety_item(iv)
                if label is not None:
                    blocker = WARNING_REASON.format(term=label)

        threshold = request.app.state.clone_threshold
        # The gate never runs when the backstop saw an emergency term: that
        # answer must not wait on another model call. A clarify delivers no
        # medical fact and is not scored; a close is scored like an answer,
        # except the judge's close, which is the doctor's own words built by code.
        if (threshold is not None and result.kind in ("answer", "close") and not terms
                and not result.doctor_words_only
                and not (result.kind == "close" and blocker is not None)):
            scored = body.question
            if pending is not None:  # a reply is read with the question it answers
                scored = (f"Original question: {pending.question}\n"
                          f"The clone asked and the patient replied:\n"
                          f"{rounds_text(pending.all_rounds(body.question))}")
            try:
                confidence = await score_answer(llm, scored, result.answer, result.sources,
                                                result.seen)
            except (LLMError, ValueError) as exc:
                log.warning("Answer verifier failed, escalating: %s", exc)
                confidence = None  # fail closed
            gate_score = confidence
            gate_passed = confidence is not None and confidence >= threshold

    settle: Optional[Settle] = None
    if pending is not None:
        result, settle = settle_reply(result, terms, blocker, gate_passed, name)
        if result.replies_to_questions is False and result.kind == "answer" and not gate_passed:
            result = low_confidence_result(body.question, name)  # the new question's gate
    elif not gate_passed:
        result = low_confidence_result(body.question, name)
    if result.kind == "clarify":
        # The waiting question is the clone's only question this turn.
        result = replace(result, care_team_question=None, flagged=False)

    payload = answer_payload(result, terms, doctor_id, name, gate_score)
    next_waiting_id: Optional[int] = None
    await end_clarify_message(session, clarify_id)
    if waiting_id is not None and settle is not None and pending is not None:
        if iv is not None and iv.rounds:
            iv.rounds[-1]["reply"] = body.question  # also on an emergency turn
        next_waiting_id = await settle_waiting(
            session, waiting_id, settle, pending, body.question, user_id, doc_id, doctor_id,
            intake=dump_interview(iv) if iv is not None else None)
        if settle.status == WAITING and next_waiting_id is None:
            # The case was settled meanwhile (the doctor answered it, or a
            # concurrent reply won): it went to the doctor, so no more questions.
            result = handoff_result(name)
            payload = answer_payload(result, terms, doctor_id, name, gate_score)
    if result.kind == "clarify":
        if pending is None:  # the first round: the waiting question is the clone's own
            waiting_row = await add_question(
                session, user_id, doc_id, question=body.question, reason="", source="agent",
                doctor_id=doctor_id, status=WAITING, asked=strip_markers(result.answer))
            if first_iv is not None:
                waiting_row.intake = dump_interview(first_iv)
            next_waiting_id = waiting_row.id
        payload["pending"] = True
        payload["waiting_question_id"] = next_waiting_id
        # Each question marked as the doctor's (a habit) or the clone's own.
        payload.update(interview_extra or clarify_questions(result.answer, result.sources))
    session.add(ChatMessage(document_id=doc_id, doctor_id=doctor_id, role="user",
                            content=body.question))
    session.add(ChatMessage(document_id=doc_id, doctor_id=doctor_id, role="assistant",
                            content=result.answer, payload=payload))
    if result.care_team_question:
        await add_question(
            session, user_id, doc_id,
            question=result.care_team_question,
            reason=result.care_team_reason,
            source=result.care_team_source,
            doctor_id=doctor_id,
            dedupe=result.care_team_reason == SOURCES_DISAGREE,
        )
    await session.commit()
    return payload


@router.get("/api/conversation")
async def conversation(
    doctor_id: int,
    user: User = Depends(current_patient),
    session: AsyncSession = Depends(get_session),
):
    await require_doctor(session, doctor_id)
    try:
        doc = await active_document(session, user)
    except HTTPException:
        return {"messages": []}
    rows = await session.scalars(
        select(ChatMessage)
        .where(ChatMessage.document_id == doc.id, ChatMessage.doctor_id == doctor_id)
        .order_by(ChatMessage.id)
    )
    return {"messages": [message_view(m) for m in rows]}
