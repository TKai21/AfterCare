"""Doctors as a patient sees them: who can be asked, and how they are named."""
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import User

CHOOSE_DOCTOR_ERROR = "Choose one of the doctors."


async def list_doctors(session: AsyncSession) -> list[User]:
    return list(await session.scalars(
        select(User).where(User.role == "doctor").order_by(User.id)
    ))


def doctor_label(user: User) -> str:
    return (user.display_name or "").strip() or user.username


def doctor_specialty(user: User) -> str | None:
    """The specialty to show next to the doctor's name, or None when unset."""
    return (user.specialty or "").strip() or None


async def doctor_labels(session: AsyncSession, ids) -> dict[int, str]:
    """Labels for the given user ids, in one query."""
    ids = {i for i in ids if i}
    if not ids:
        return {}
    rows = await session.scalars(select(User).where(User.id.in_(ids)))
    return {u.id: doctor_label(u) for u in rows}


async def require_doctor(session: AsyncSession, doctor_id: int) -> User:
    user = await session.scalar(
        select(User).where(User.id == doctor_id, User.role == "doctor")
    )
    if user is None:
        raise HTTPException(status_code=422, detail=CHOOSE_DOCTOR_ERROR)
    return user
