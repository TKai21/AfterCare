import re
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, StringConstraints
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import current_patient
from db import get_session
from doctors import doctor_label, doctor_labels, doctor_specialty, list_doctors, require_doctor
from models import CareTeamQuestion, User
from safety import detect_emergency, warning_terms
from voice import voice_views

MAX_QUESTION_CHARS = 2000

router = APIRouter()


def intake_rounds(q: CareTeamQuestion) -> list[dict]:
    """Every round of the clone's intake, in order: the completed rounds
    ([{asked, reply}], `intake`) and the round still waiting for a reply or
    just answered (`asked`, `patient_reply`) when `intake` does not end with it.
    A row from before rounds has no `intake`: its one round is asked/patient_reply."""
    rounds = [{"asked": r.get("asked"), "reply": r.get("reply")}
              for r in (q.intake or []) if isinstance(r, dict)]
    if q.asked and not (rounds and rounds[-1]["asked"] == q.asked):
        rounds.append({"asked": q.asked, "reply": q.patient_reply})
    return rounds


def question_view(q: CareTeamQuestion, doctor_names: dict[int, str] | None = None,
                  voices: dict | None = None) -> dict:
    result = {
        "id": q.id,
        "question": q.question,
        "reason": q.reason,
        "source": q.source,
        "status": q.status,
        "answer": q.answer,
        "asked": q.asked,
        "patient_reply": q.patient_reply,
        "intake": list(q.intake or []),
        "rounds": intake_rounds(q),
        "stopped_by": q.stopped_by,
        "answered_by_name": (doctor_names or {}).get(q.answered_by) if q.answered_by else None,
        "doctor_id": q.doctor_id,
        "doctor_name": (doctor_names or {}).get(q.doctor_id) if q.doctor_id else None,
        "answered_at": q.answered_at.isoformat() if q.answered_at else None,
        "created_at": q.created_at.isoformat() if q.created_at else None,
        "document_id": q.document_id,
        "asked_count": q.asked_count or 1,
    }
    if voices and q.id in voices:
        result["voice"] = voices[q.id]
    return result


_SPACES = re.compile(r"\s+")
_TRAILING = re.compile(r"[\s.,;:!?\u3002\uff01\uff1f]+$")


def normalize_question(text: str) -> str:
    """The text two asks of the same question share: case, spacing and trailing
    punctuation do not tell them apart."""
    return _TRAILING.sub("", _SPACES.sub(" ", text.casefold()).strip())


def same_open_question(q: CareTeamQuestion, other: CareTeamQuestion) -> bool:
    """Whether the doctor would see these two open rows as one question."""
    return (q.user_id == other.user_id and q.document_id == other.document_id
            and q.doctor_id == other.doctor_id
            and normalize_question(q.question) == normalize_question(other.question))


def mergeable(q: CareTeamQuestion) -> bool:
    """An open, unanswered question the doctor can answer in text. Waiting rows
    belong to the clone's clarify and voice questions carry a clip."""
    return q.status == "open" and q.answer is None and q.source != "patient_voice"


async def add_question(
    session: AsyncSession,
    user_id: int,
    document_id: int | None,
    question: str,
    reason: str,
    source: str,
    dedupe: bool = False,
    doctor_id: int | None = None,
    status: str = "open",
    asked: str | None = None,
) -> CareTeamQuestion | None:
    """Add a question to the user's care-team list. The caller commits.

    With dedupe=True, returns None (inserting nothing) when the user already has
    an open question with the same text and reason for the same doctor.

    A new open question the patient has already asked (same document, same
    doctor, same text once normalised) and the doctor has not answered is not
    added again: that row's asked_count goes up and it counts as asked now. The
    row is returned (None with dedupe=True, which asks only for a new row)."""
    if status == "open" and source != "patient_voice":
        candidate = CareTeamQuestion(user_id=user_id, document_id=document_id,
                                     question=question, doctor_id=doctor_id)
        rows = await session.scalars(
            select(CareTeamQuestion).where(
                CareTeamQuestion.user_id == user_id,
                CareTeamQuestion.status == "open",
                CareTeamQuestion.answer.is_(None),
                CareTeamQuestion.source != "patient_voice",
                CareTeamQuestion.document_id == document_id
                if document_id is not None else CareTeamQuestion.document_id.is_(None),
                CareTeamQuestion.doctor_id == doctor_id
                if doctor_id is not None else CareTeamQuestion.doctor_id.is_(None),
            ).order_by(CareTeamQuestion.id.desc())
        )
        for row in rows:
            if same_open_question(row, candidate):
                row.asked_count = (row.asked_count or 1) + 1
                row.last_asked_at = datetime.now(timezone.utc).replace(tzinfo=None)
                if asked is not None:
                    row.asked = asked
                await session.flush()
                return None if dedupe else row
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
        status=status,
        asked=asked,
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
        {"id": d.id, "name": doctor_label(d), "specialty": doctor_specialty(d)}
        for d in await list_doctors(session)
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
        # A question the clone is still asking the patient about is not yet
        # the care team's: the patient sees the clone's questions on Ask.
        .where(CareTeamQuestion.user_id == user.id,
               CareTeamQuestion.status != "waiting_on_patient")
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
    if row.status == "waiting_on_patient":
        # The clone is waiting for the patient's reply; resolving it here would
        # lose the reply on its way to the doctor.
        raise HTTPException(status_code=409,
                            detail="The clone is still waiting for your reply on Ask.")
    row.status = "resolved"
    await session.commit()
    names = await doctor_labels(session, {row.answered_by, row.doctor_id})
    return question_view(row, names, await voice_views(session, [row.id]))
