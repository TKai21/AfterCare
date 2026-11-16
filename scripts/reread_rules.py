"""Re-read the rules and habits of every active guidance entry of one doctor.

    .venv/bin/python scripts/reread_rules.py USERNAME

Uses the database and model from the project .env. Each active entry is read
again with rules.index_entry (one model call per entry, plus a conflict check
when it has rules), so an answer saved before habits existed gets its habits.
Habits the doctor deleted stay deleted. Prints the rules and habits of each
entry. Inactive entries are left untouched.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from sqlalchemy import select  # noqa: E402

from config import get_settings  # noqa: E402
from db import init_db, make_engine, make_session_factory  # noqa: E402
from knowledge import open_store  # noqa: E402
from models import DoctorGuidance, GuidanceRule, User  # noqa: E402
from rules import index_entry  # noqa: E402


async def reread(session_factory, llm, store, username: str) -> list[str]:
    """Re-read every active entry of the doctor; returns the lines to print."""
    lines: list[str] = []
    async with session_factory() as session:
        doctor = await session.scalar(select(User).where(User.username == username))
        if doctor is None or doctor.role != "doctor":
            raise ValueError(f"No doctor named {username!r}.")
        entries = list(await session.scalars(
            select(DoctorGuidance)
            .where(DoctorGuidance.doctor_id == doctor.id, DoctorGuidance.active.is_(True))
            .order_by(DoctorGuidance.id)
        ))
        for entry in entries:
            await index_entry(session, llm, store, entry)
            await session.refresh(entry)
            rows = list(await session.scalars(
                select(GuidanceRule)
                .where(GuidanceRule.guidance_id == entry.id, GuidanceRule.suppressed.is_(False))
                .order_by(GuidanceRule.position, GuidanceRule.id)
            ))
            lines.append(f"Entry {entry.id} ({entry.rules_status or 'no status'}): {entry.question}")
            for rule in rows:
                tag = "habit " + (rule.habit_type or "") if rule.kind == "habit" else "rule"
                lines.append(f"  [{tag.strip()}] {rule.statement}")
    lines.append(f"Read {len(entries)} active entries.")
    return lines


async def main(username: str) -> None:
    from llm import LLMClient
    settings = get_settings()
    llm = LLMClient(settings)
    engine = make_engine(settings.database_url)
    await init_db(engine)
    try:
        for line in await reread(make_session_factory(engine), llm, open_store(settings, llm), username):
            print(line)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: reread_rules.py USERNAME")
    try:
        asyncio.run(main(sys.argv[1]))
    except ValueError as error:
        sys.exit(str(error))
