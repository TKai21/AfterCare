"""init_db brings an existing database up to the current tables without data loss."""
import sqlite3

from sqlalchemy import select

from db import init_db, make_engine, make_session_factory
from models import User


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
