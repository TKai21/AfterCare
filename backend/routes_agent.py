import logging
from dataclasses import dataclass, replace
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, StringConstraints
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agent import (
    HISTORY_LIMIT, AgentResult, GuidanceUnavailable, PendingClarify, emergency_result,
    fallback_answer, run_agent, strip_markers,
)
from auth import current_patient
from closing import (
    EMERGENCY_REASON, WAITING, close_blocker, doctor_warning_terms, pending_clarify,
)
from confidence import LOW_CONFIDENCE_REASON, score_answer
from db import get_session
from doctors import doctor_label, require_doctor
from druglabels import patient_rxcuis
from drugnames import allowed_drug_names, get_lexicon, prescribed_drug_names
from guidance import Unit, doctor_habits, retrieve
from llm import LLMError
from models import CareTeamQuestion, ChatMessage, User
from routes_care_team import MAX_QUESTION_CHARS, add_question
from routes_documents import active_document, confirmed_document, load_items
from safety import detect_emergency, warning_terms

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
    status: str  # open | resolved
    reason: Optional[str] = None  # set when it becomes open
    stopped_by: Optional[str] = None
    answer: Optional[str] = None  # set when the clone resolved it


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
    blocked answer is still delivered, but the question goes to the doctor."""
    if terms or result.kind == "emergency":
        stop = EMERGENCY_REASON if terms else EMERGENCY_ANSWER_STOP
        return (replace(result, care_team_question=None, flagged=False),
                Settle("open", NEEDS_DOCTOR, stop))
    if result.replies_to_questions is False:
        # A blocker (say a warning sign in the new message) is the more useful
        # summary for the doctor; otherwise the patient simply moved on.
        settle = (Settle("open", NEEDS_DOCTOR, blocker) if blocker is not None
                  else Settle("open", NOT_ANSWERED, NOT_ANSWERED))
        if result.kind == "close":  # a close that is not a reply is never delivered
            return handoff_result(doctor_name), settle
        # The new question is answered as usual (its own gate and question).
        return result, settle
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


async def settle_waiting(session: AsyncSession, waiting_id: int, settle: Settle,
                         pending: PendingClarify, reply: str, user_id: int,
                         document_id: int, doctor_id: int) -> None:
    """Settle the waiting question, only while it is still waiting: a
    conditional UPDATE, so of two concurrent replies only the first owns it.
    When it is no longer waiting (the doctor answered it, or another reply
    settled it), a handoff becomes a new open question so a reply is never
    lost; a resolving outcome then only delivers its answer."""
    values = {"status": settle.status, "patient_reply": reply, "stopped_by": settle.stopped_by}
    if settle.reason is not None:
        values["reason"] = settle.reason
    if settle.answer is not None:
        values["answer"] = settle.answer
    done = await session.execute(
        update(CareTeamQuestion)
        .where(CareTeamQuestion.id == waiting_id, CareTeamQuestion.status == WAITING)
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    if done.rowcount == 1 or settle.status != "open":
        return
    row = await add_question(
        session, user_id, document_id, question=pending.question,
        reason=settle.reason or NEEDS_DOCTOR, source="agent", doctor_id=doctor_id,
        asked=strip_markers(pending.asked))
    row.patient_reply = reply
    row.stopped_by = settle.stopped_by


class AskBody(BaseModel):
    # Optional only so a missing id reaches the emergency backstop; any other
    # message without a valid doctor is refused (422).
    doctor_id: Optional[int] = None
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
    ]


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
    allowed_drugs = frozenset(
        allowed_drug_names(items, library_rxcuis, body.question, get_lexicon()))
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
    clarify_id = waiting_id = None
    if found is not None:
        clarify_message, waiting = found
        clarify_id, waiting_id = clarify_message.id, waiting.id
        pending = PendingClarify(question=waiting.question,
                                 asked=waiting.asked or clarify_message.content)
    gate_score: Optional[float] = None  # the confidence gate's score, when it ran
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
            if pending is not None:
                try:
                    doctor_terms = await doctor_warning_terms(session, doctor_id)
                    await session.commit()
                except Exception:
                    # Fail closed: without the doctor's warning signs no close.
                    log.exception("Loading the doctor's warning terms failed")
                    doctor_terms, terms_failed = [], True
                    try:
                        await session.rollback()
                    except Exception:
                        log.exception("Rollback after failing to load warning terms failed")
        document_terms = warning_terms(items)
        settings = request.app.state.settings
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
            )
        except LLMError as exc:
            if not terms:
                raise HTTPException(
                    status_code=502,
                    detail="The assistant could not answer right now. Please try again.",
                ) from exc
            result = emergency_result()

        if pending is not None and not terms:
            # The server, not the model, decides whether a close is delivered
            # (4.4); the same checks decide whether an answer settles the case,
            # and name what the doctor must see when the patient asked something else.
            lexicon = get_lexicon()
            blocker = HABITS_UNAVAILABLE if terms_failed else close_blocker(
                pending.question, body.question, terms, doctor_terms, document_terms,
                result.warning_signs, prescribed_drug_names(items, library_rxcuis, lexicon),
                lexicon)

        threshold = request.app.state.clone_threshold
        # The gate never runs when the backstop saw an emergency term: that
        # answer must not wait on another model call. A clarify delivers no
        # medical fact and is not scored; a close is scored like an answer.
        if (threshold is not None and result.kind in ("answer", "close") and not terms
                and not (result.kind == "close" and blocker is not None)):
            scored = body.question
            if pending is not None:  # a reply is read with the question it answers
                scored = (f"Original question: {pending.question}\n"
                          f"The clone asked: {strip_markers(pending.asked)}\n"
                          f"The patient replied: {body.question}")
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
    if result.kind == "clarify":
        waiting_row = await add_question(
            session, user_id, doc_id, question=body.question, reason="", source="agent",
            doctor_id=doctor_id, status=WAITING, asked=strip_markers(result.answer))
        payload["pending"] = True
        payload["waiting_question_id"] = waiting_row.id
    if clarify_id is not None:
        # Any reply turn ends the clarify; a new dict so the JSON change is saved.
        # Every writer sets the same value, so a concurrent reply cannot undo it.
        message = await session.get(ChatMessage, clarify_id)
        if message is not None:
            message.payload = {**(message.payload or {}), "pending": False}
    if waiting_id is not None and settle is not None and pending is not None:
        await settle_waiting(session, waiting_id, settle, pending, body.question,
                             user_id, doc_id, doctor_id)
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
