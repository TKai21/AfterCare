from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import current_patient
from db import get_session
from models import CareTeamQuestion, User

router = APIRouter()


def question_view(q: CareTeamQuestion, doctor_names: dict[int, str] | None = None) -> dict:
    return {
        "id": q.id,
        "question": q.question,
        "reason": q.reason,
        "source": q.source,
        "status": q.status,
        "answer": q.answer,
        "answered_by_name": (doctor_names or {}).get(q.answered_by) if q.answered_by else None,
        "answered_at": q.answered_at.isoformat() if q.answered_at else None,
        "created_at": q.created_at.isoformat() if q.created_at else None,
        "document_id": q.document_id,
    }


async def add_question(
    session: AsyncSession,
    user_id: int,
    document_id: int | None,
    question: str,
    reason: str,
    source: str,
    dedupe: bool = False,
) -> CareTeamQuestion | None:
    """Add a question to the user's care-team list. The caller commits.

    With dedupe=True, returns None (inserting nothing) when the user already has
    an open question with the same text and reason."""
    if dedupe:
        existing = await session.scalar(
            select(CareTeamQuestion.id).where(
                CareTeamQuestion.user_id == user_id,
                CareTeamQuestion.status == "open",
                CareTeamQuestion.question == question,
                CareTeamQuestion.reason == reason,
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
    )
    session.add(row)
    await session.flush()
    return row


@router.get("/api/care-team")
async def list_questions(
    user: User = Depends(current_patient), session: AsyncSession = Depends(get_session)
):
    rows = list(await session.scalars(
        select(CareTeamQuestion)
        .where(CareTeamQuestion.user_id == user.id)
        .order_by(CareTeamQuestion.id.desc())
    ))
    doctor_ids = {q.answered_by for q in rows if q.answered_by}
    names: dict[int, str] = {}
    if doctor_ids:
        doctors = await session.scalars(select(User).where(User.id.in_(doctor_ids)))
        names = {d.id: d.display_name or d.username for d in doctors}
    return {"questions": [question_view(q, names) for q in rows]}


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
    names: dict[int, str] = {}
    if row.answered_by:
        doctor = await session.get(User, row.answered_by)
        if doctor is not None:
            names = {doctor.id: doctor.display_name or doctor.username}
    return question_view(row, names)
