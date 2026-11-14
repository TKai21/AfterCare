"""Create a doctor account in the database named by DATABASE_URL (project .env).

    .venv/bin/python scripts/create_doctor.py USERNAME "Display Name" ["Specialty"]

Prompts for the password twice (minimum 8 characters).
"""
import asyncio
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from auth import create_doctor  # noqa: E402
from config import get_settings  # noqa: E402
from db import init_db, make_engine, make_session_factory  # noqa: E402


async def main(username: str, display_name: str, password: str, specialty: str | None = None) -> None:
    engine = make_engine(get_settings().database_url)
    await init_db(engine)
    async with make_session_factory(engine)() as session:
        await create_doctor(session, username, password, display_name, specialty)
    await engine.dispose()
    label = f"{display_name}, {specialty}" if specialty else display_name
    print(f"Created doctor {username!r} ({label}).")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit('usage: create_doctor.py USERNAME "Display Name" ["Specialty"]')
    password = getpass.getpass("Password: ")
    if len(password) < 8:
        sys.exit("Password must be at least 8 characters.")
    if getpass.getpass("Repeat password: ") != password:
        sys.exit("Passwords do not match.")
    try:
        asyncio.run(main(sys.argv[1], sys.argv[2], password, sys.argv[3] if len(sys.argv) == 4 else None))
    except ValueError as error:
        sys.exit(str(error))
