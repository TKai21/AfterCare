"""Seed a doctor cloned from the PortalPal physician into the database named by
DATABASE_URL, for trying the clone by hand.

    DATABASE_URL=sqlite+aiosqlite:////path/to/db.sqlite3 \\
        .venv/bin/python scripts/seed_portalpal_doctor.py

Creates the doctor `dr.portal` / `doctor-password-1` ("Dr. Helen Okafor",
Primary care). The 60 memory rows of eval/portalpal/split.json become that
doctor's answered questions, saved through the doctor answer route, so they are
embedded and read into rules and habits exactly as a real doctor's answers are
(this calls the configured language model and embedding model). The 40 tuning
and test rows become patients `eval-pp-NNN` / `eval-password-1`, each with the
row's EHR summary as a confirmed document, to ask the clone questions it has
not seen. Refuses to run when `dr.portal` already exists.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_portalpal  # noqa: E402  (also puts backend/ on sys.path)
from eval_clone import PATIENT_PASSWORD, open_app  # noqa: E402
from eval_portalpal import build_portal_world, load_split  # noqa: E402
from portalpal_data import load_rows  # noqa: E402

from sqlalchemy import select  # noqa: E402

import config  # noqa: E402
from auth import hash_password  # noqa: E402
from knowledge import open_store  # noqa: E402
from db import init_db, make_engine  # noqa: E402
from models import User  # noqa: E402

USERNAME = "dr.portal"
DOCTOR_PASSWORD = "doctor-password-1"
# The demo patient demo2's gout document (demo/cases/gout.json) names this
# attending, so the demo shows one doctor throughout.
DISPLAY_NAME = "Dr. Helen Okafor"


async def main() -> int:
    settings = config.get_settings()
    engine = make_engine(settings.database_url)
    await init_db(engine)
    await engine.dispose()

    from llm import LLMClient
    llm = LLMClient(settings)
    rows = load_rows()
    split = load_split()
    eval_portalpal.DOCTOR = (USERNAME, DISPLAY_NAME, "Primary care")
    async with open_app(settings, llm, open_store(settings, llm)) as (client, session):
        if await session.scalar(select(User).where(User.username == USERNAME)) is not None:
            print(f"{USERNAME} already exists in this database; nothing changed.")
            return 1
        world = await build_portal_world(client, session, rows, split["memory"],
                                         [*split["tune"], *split["test"]])
        doctor = await session.get(User, world.doctors[USERNAME])
        doctor.password_hash = hash_password(DOCTOR_PASSWORD)
        await session.commit()
    print(f"Doctor {USERNAME} / {DOCTOR_PASSWORD}: {len(split['memory'])} answers, {len(world.habits)} habits.")
    print(f"Patients eval-<row id> / {PATIENT_PASSWORD}, for example "
          f"eval-{split['test'][0]} (unseen rows: tuning and test).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
