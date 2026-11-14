from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, StringConstraints
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import current_patient
from db import get_session
from doctors import doctor_label, doctor_labels, list_doctors, require_doctor
from models import CareTeamQuestion, User
from safety import detect_emergency, warning_terms
from voice import voice_views

MAX_QUESTION_CHARS = 2000

router = APIRouter()


def question_view(q: CareTeamQuestion, doctor_names: dict[int, str] | None = None,
                  voices: dict | None = None) -> dict:
    result = {
        "id": q.id,
        "question": q.question,
        "reason": q.reason,
        "source": q.source,
        "status": q.status,
        "answer": q.answer,
        "answered_by_name": (doctor_names or {}).get(q.answered_by) if q.answered_by else None,
        "doctor_id": q.doctor_id,
        "doctor_name": (doctor_names or {}).get(q.doctor_id) if q.doctor_id else None,
        "answered_at": q.answered_at.isoformat() if q.answered_at else None,
        "created_at": q.created_at.isoformat() if q.created_at else None,
        "document_id": q.document_id,
    }
    if voices and q.id in voices:
        result["voice"] = voices[q.id]
    return result


async def add_question(
    session: AsyncSession,
    user_id: int,
    document_id: int | None,
    question: str,
    reason: str,
    source: str,
    dedupe: bool = False,
    doctor_id: int | None = None,
) -> CareTeamQuestion | None:
    """Add a question to the user's care-team list. The caller commits.

    With dedupe=True, returns None (inserting nothing) when the user already has
    an open question with the same text and reason for the same doctor."""
    if dedupe:
        existing = await session.scalar(
            select(CareTeamQuestion.id).where(
                CareTeamQuestion.user_id == user_id,
                CareTeamQuestion.status == "open",
                CareTeamQuestion.question == question,
                CareTeamQuestion.reason == reason,
                CareTeamQuestion.doctor_id == doctor_id
                if doctor_id is not None else CareTeamQuestion.doctor_id.is_(None),
            ).limit(1)
        )
        if existing is not None:
            return None
    row = CareTeamQuestion(
        user_id=user_id,
        document_id=document_id,
        question=question,
        reason=reason,
        source=source,
        doctor_id=doctor_id,
    )
    session.add(row)
    await session.flush()
    return row


class DirectQuestion(BaseModel):
    doctor_id: int
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
    ]


@router.get("/api/doctors")
async def doctors(
    user: User = Depends(current_patient), session: AsyncSession = Depends(get_session)
):
    return {"doctors": [
        {"id": d.id, "name": doctor_label(d)} for d in await list_doctors(session)
    ]}


@router.post("/api/care-team")
async def ask_doctor(
    body: DirectQuestion,
    user: User = Depends(current_patient),
    session: AsyncSession = Depends(get_session),
):
    # routes_documents imports add_question from this module, so import it lazily.
    from routes_documents import active_document, load_items

    doctor = await require_doctor(session, body.doctor_id)
    try:
        doc = await active_document(session, user)
    except HTTPException:
        doc = None
    items = await load_items(session, doc) if doc is not None else []
    terms = detect_emergency(body.question, warning_terms(items))
    row = await add_question(
        session, user.id, doc.id if doc is not None else None,
        body.question, "Asked by the patient", "patient", doctor_id=doctor.id,
    )
    await session.commit()
    await session.refresh(row)
    return {
        "question": question_view(row, {doctor.id: doctor_label(doctor)}),
        "emergency": {"terms": terms, "banner": bool(terms)},
    }


@router.get("/api/care-team")
async def list_questions(
    user: User = Depends(current_patient), session: AsyncSession = Depends(get_session)
):
    rows = list(await session.scalars(
        select(CareTeamQuestion)
        .where(CareTeamQuestion.user_id == user.id)
        .order_by(CareTeamQuestion.id.desc())
    ))
    names = await doctor_labels(
        session, {i for q in rows for i in (q.answered_by, q.doctor_id)}
    )
    voices = await voice_views(session, [q.id for q in rows])
    return {"questions": [question_view(q, names, voices) for q in rows]}


@router.post("/api/care-team/{question_id}/resolve")
async def resolve_question(
    question_id: int,
    user: User = Depends(current_patient),
    session: AsyncSession = Depends(get_session),
):
    row = await session.scalar(
        select(CareTeamQuestion).where(
            CareTeamQuestion.id == question_id, CareTeamQuestion.user_id == user.id
        )
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Question not found")
    row.status = "resolved"
    await session.commit()
    names = await doctor_labels(session, {row.answered_by, row.doctor_id})
    return question_view(row, names, await voice_views(session, [row.id]))
