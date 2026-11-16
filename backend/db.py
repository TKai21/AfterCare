from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool
from fastapi import Request


class Base(DeclarativeBase):
    pass


def make_engine(url: str) -> AsyncEngine:
    if url.startswith("sqlite"):
        kwargs = {}
        if ":memory:" in url or url.endswith("///"):
            # One shared connection so every session sees the same in-memory DB.
            kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
        return create_async_engine(url, **kwargs)
    # Force every connection to UTC so DATETIME columns are read/written as UTC.
    return create_async_engine(
        url,
        pool_pre_ping=True,
        connect_args={"init_command": "SET time_zone = '+00:00'"},
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session(request: Request):
    async with request.app.state.session_factory() as session:
        yield session


async def init_db(engine: AsyncEngine) -> None:
    import models  # noqa: F401  (register tables on Base.metadata)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


# Columns added after a table first shipped. create_all (and schema.sql's
# CREATE TABLE IF NOT EXISTS) never alters an existing table, so an existing
# database gets each one here, on SQLite and MySQL alike, with no data loss.
ADDED_COLUMNS = (
    ("users", "specialty", "VARCHAR(100) NULL"),
    ("doctor_guidance", "rules_status", "VARCHAR(20) NULL"),
    ("guidance_rules", "kind", "VARCHAR(20) NOT NULL DEFAULT 'advice'"),
    ("guidance_rules", "habit_type", "VARCHAR(20) NULL"),
    # Nullable: MySQL has no portable NOT NULL JSON default; the ORM reads NULL as [].
    ("guidance_rules", "warning_terms", "JSON NULL"),
    ("guidance_rules", "suppressed", "BOOLEAN NOT NULL DEFAULT 0"),
    ("care_team_questions", "asked", "TEXT NULL"),
    ("care_team_questions", "patient_reply", "TEXT NULL"),
    ("care_team_questions", "stopped_by", "VARCHAR(100) NULL"),
)


def _add_missing_columns(conn) -> None:
    inspector = inspect(conn)
    for table, column, ddl in ADDED_COLUMNS:
        if column not in {c["name"] for c in inspector.get_columns(table)}:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
