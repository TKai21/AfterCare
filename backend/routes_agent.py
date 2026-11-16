import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, StringConstraints
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent import (
    HISTORY_LIMIT, AgentResult, GuidanceUnavailable, emergency_result, fallback_answer, run_agent,
)
from auth import current_patient
from confidence import LOW_CONFIDENCE_REASON, score_answer
from db import get_session
from doctors import doctor_label, require_doctor
from druglabels import patient_rxcuis
from drugnames import allowed_drug_names, get_lexicon
from guidance import Unit, doctor_habits, retrieve
from llm import LLMError
from models import ChatMessage, User
from routes_care_team import MAX_QUESTION_CHARS, add_question
from routes_documents import active_document, confirmed_document, load_items
from safety import detect_emergency, warning_terms

router = APIRouter()
log = logging.getLogger(__name__)

# A care-team question with this reason is added at most once while open.
SOURCES_DISAGREE = "Sources disagree"


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
        "doctor_id": doctor_id,
        "doctor_name": doctor_name,
        # The confidence gate's score of the clone's answer (None: not scored).
        "confidence": confidence,
    }


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
    gate_score: Optional[float] = None  # the confidence gate's score, when it ran
    if llm is None:
        if not terms:
            raise HTTPException(status_code=503, detail="The language model is not configured.")
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
                habits=habits,
            )
        except LLMError as exc:
            if not terms:
                raise HTTPException(
                    status_code=502,
                    detail="The assistant could not answer right now. Please try again.",
                ) from exc
            result = emergency_result()

        threshold = request.app.state.clone_threshold
        # The gate never runs when the backstop saw an emergency term: that
        # answer must not wait on another model call.
        if threshold is not None and result.kind == "answer" and not terms:
            try:
                confidence = await score_answer(llm, body.question, result.answer, result.sources,
                                                result.seen)
            except (LLMError, ValueError) as exc:
                log.warning("Answer verifier failed, escalating: %s", exc)
                confidence = None  # fail closed
            gate_score = confidence
            if confidence is None or confidence < threshold:
                result = AgentResult(
                    kind="not_covered", answer=fallback_answer(name), sources=[], flagged=True,
                    care_team_question=body.question, care_team_reason=LOW_CONFIDENCE_REASON,
                )

    payload = answer_payload(result, terms, doctor_id, name, gate_score)
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
