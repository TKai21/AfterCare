"""init_db brings an existing database up to the current tables without data loss."""
import sqlite3

from sqlalchemy import select

from db import init_db, make_engine, make_session_factory
from models import GuidanceRule, User


async def test_init_db_adds_the_specialty_column_to_an_old_users_table(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:  # the users table as it shipped before specialty
        conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, username VARCHAR(50) NOT NULL UNIQUE,"
            " password_hash VARCHAR(255) NOT NULL, role VARCHAR(20) NOT NULL DEFAULT 'patient',"
            " display_name VARCHAR(100), created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO users (username, password_hash, role, display_name)"
            " VALUES ('dr.ting', 'x', 'doctor', 'Dr. Ting Kai')"
        )
    conn.close()
    engine = make_engine(f"sqlite+aiosqlite:///{path}")
    try:
        await init_db(engine)
        await init_db(engine)  # a second start finds the column and changes nothing
        async with make_session_factory(engine)() as session:
            (user,) = (await session.scalars(select(User))).all()
            assert (user.username, user.display_name, user.specialty) == ("dr.ting", "Dr. Ting Kai", None)
            user.specialty = "Rheumatology"
            await session.commit()
            assert (await session.get(User, user.id)).specialty == "Rheumatology"
    finally:
        await engine.dispose()


async def test_existing_guidance_rules_table_gains_habit_columns(tmp_path):
    path = tmp_path / "old_rules.db"
    with sqlite3.connect(path) as conn:  # guidance_rules as it shipped before habits
        conn.execute(
            "CREATE TABLE guidance_rules (id INTEGER PRIMARY KEY, guidance_id INTEGER NOT NULL,"
            " doctor_id INTEGER NOT NULL, position INTEGER NOT NULL, statement TEXT NOT NULL,"
            " quote TEXT NOT NULL, embedding JSON, embedding_model VARCHAR(100))"
        )
        conn.execute(
            "INSERT INTO guidance_rules (guidance_id, doctor_id, position, statement, quote)"
            " VALUES (1, 2, 0, 'Take it with food', 'with food')"
        )
    conn.close()
    engine = make_engine(f"sqlite+aiosqlite:///{path}")
    try:
        await init_db(engine)
        await init_db(engine)
        async with make_session_factory(engine)() as session:
            (rule,) = (await session.scalars(select(GuidanceRule))).all()
            assert (rule.kind, rule.habit_type, rule.warning_terms, rule.suppressed) == (
                "advice", None, [], False)
            rule.kind, rule.habit_type, rule.warning_terms = "habit", "warning", ["fever"]
            await session.commit()
            session.expunge_all()
            got = await session.get(GuidanceRule, rule.id)
            assert (got.kind, got.habit_type, got.warning_terms) == ("habit", "warning", ["fever"])
    finally:
        await engine.dispose()
