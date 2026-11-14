from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, StringConstraints
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent import DEFAULT_DOCTOR_NAME, HISTORY_LIMIT, emergency_result, run_agent
from auth import current_patient
from db import get_session
from guidance import retrieve
from llm import LLMError
from models import ChatMessage, DoctorGuidance, User
from routes_care_team import add_question
from routes_documents import active_document, confirmed_document, load_items
from safety import detect_emergency, warning_terms

router = APIRouter()

MAX_QUESTION_CHARS = 2000
# A care-team question with this reason is added at most once while open.
SOURCES_DISAGREE = "Sources disagree"


class AskBody(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
    ]


def message_view(m: ChatMessage) -> dict:
    return {
        "role": m.role,
        "content": m.content,
        "payload": m.payload,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


async def recent_history(session: AsyncSession, document_id: int) -> list[dict]:
    rows = await session.scalars(
        select(ChatMessage)
        .where(ChatMessage.document_id == document_id)
        .order_by(ChatMessage.id.desc())
        .limit(HISTORY_LIMIT)
    )
    return [{"role": m.role, "content": m.content} for m in reversed(list(rows))]


async def doctor_name(session: AsyncSession) -> str:
    """The display name of the doctor account (the lowest id one, if several),
    its username if the display name is empty, or "your doctor"."""
    doctor = await session.scalar(
        select(User).where(User.role == "doctor").order_by(User.id).limit(1)
    )
    if doctor is None:
        return DEFAULT_DOCTOR_NAME
    return (doctor.display_name or "").strip() or doctor.username


@router.post("/api/ask")
async def ask(
    body: AskBody,
    request: Request,
    user: User = Depends(current_patient),
    session: AsyncSession = Depends(get_session),
):
    doc = await confirmed_document(session, user)
    items = await load_items(session, doc)
    # The backstop runs first on every message; the emergency path never depends
    # on the model (R19).
    terms = detect_emergency(body.question, warning_terms(items))
    name = await doctor_name(session)
    llm = request.app.state.llm
    if llm is None:
        if not terms:
            raise HTTPException(status_code=503, detail="The language model is not configured.")
        result = emergency_result()
    else:
        history = await recent_history(session, doc.id)
        await session.commit()  # end the read transaction before the slow model calls
        min_similarity = request.app.state.settings.guidance_min_similarity

        async def guidance_search(query: str) -> list[DoctorGuidance]:
            try:
                hits = await retrieve(session, llm, user, doc, items, query, min_similarity)
            finally:
                await session.commit()  # no read transaction held across model calls
            return [entry for entry, _score in hits]

        try:
            result = await run_agent(
                llm, request.app.state.store, request.app.state.foods,
                doc, items, history, body.question, terms,
                guidance_search=guidance_search, doctor_name=name,
            )
        except LLMError as exc:
            if not terms:
                raise HTTPException(
                    status_code=502,
                    detail="The assistant could not answer right now. Please try again.",
                ) from exc
            result = emergency_result()

    payload = {
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
        "doctor_name": name,
    }
    session.add(ChatMessage(document_id=doc.id, role="user", content=body.question))
    session.add(ChatMessage(document_id=doc.id, role="assistant", content=result.answer,
                            payload=payload))
    if result.care_team_question:
        await add_question(
            session, user.id, doc.id,
            question=result.care_team_question,
            reason=result.care_team_reason,
            source=result.care_team_source,
            dedupe=result.care_team_reason == SOURCES_DISAGREE,
        )
    await session.commit()
    return payload


@router.get("/api/conversation")
async def conversation(
    user: User = Depends(current_patient), session: AsyncSession = Depends(get_session)
):
    try:
        doc = await active_document(session, user)
    except HTTPException:
        return {"messages": []}
    rows = await session.scalars(
        select(ChatMessage).where(ChatMessage.document_id == doc.id).order_by(ChatMessage.id)
    )
    return {"messages": [message_view(m) for m in rows]}
