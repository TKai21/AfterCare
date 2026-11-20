"""Decision memory of one doctor: read the history, and measure what was learned.

    .venv/bin/python scripts/decision_memory.py USERNAME            # the replay only
    .venv/bin/python scripts/decision_memory.py USERNAME --backfill # read the history first

Uses the database and model from the project .env.

--backfill reads each piece of the doctor's earlier advice that has no case row
yet into one (one small model call and one embedding each; text sent to the
model is de-identified as in the app). It writes to the database.

The replay makes no model call. It hides each stored case in turn, names the
decision of its nearest cases, and prints how often that was the doctor's
decision, next to how often always naming her most common decision is right.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from sqlalchemy import select  # noqa: E402

from config import get_settings  # noqa: E402
from db import init_db, make_engine, make_session_factory  # noqa: E402
from decisions import (  # noqa: E402
    KIND_LABELS, KINDS, backfill_history, is_evidence, learned, replay,
)
from models import CaseDecision, User  # noqa: E402


def report(result: dict) -> list[str]:
    cases = result["cases"]
    if not cases:
        return ["No cases with a decision yet."]
    lines = [f"Cases with a decision: {cases}",
             f"Nearest cases named the doctor's decision: {result['right']} of {cases}"
             f" ({result['right'] / cases:.0%}); no similar case for {cases - result['proposed']}",
             f"Always naming the most common decision: {result['baseline']} of {cases}"
             f" ({result['baseline'] / cases:.0%})"]
    for kind in KINDS:
        if kind in result["per_kind"]:
            tally = result["per_kind"][kind]
            lines.append(f"  {KIND_LABELS[kind]}: {tally['right']} of {tally['cases']}")
    lines.append("Clearly above the baseline." if learned(result) else
                 "Not clearly above the baseline: the cases do not yet tell when the doctor "
                 "decides what.")
    return lines


async def main(username: str, backfill: bool) -> None:
    from drugnames import get_lexicon
    from llm import LLMClient
    from privacy import PrivateLLM, vault_loader
    settings = get_settings()
    engine = make_engine(settings.database_url)
    await init_db(engine)
    session_factory = make_session_factory(engine)
    try:
        async with session_factory() as session:
            doctor = await session.scalar(select(User).where(User.username == username))
            if doctor is None or doctor.role != "doctor":
                raise SystemExit(f"No doctor named {username!r}.")
            if backfill:
                llm = LLMClient(settings)
                if settings.privacy_masking:
                    lexicon = get_lexicon()
                    llm = PrivateLLM(llm, vault_loader(session_factory),
                                     is_medical=lambda word: bool(lexicon.find(word)))
                print(f"Read {await backfill_history(session, llm, doctor.id)} past replies.")
            rows = [r for r in await session.scalars(select(CaseDecision).where(
                CaseDecision.doctor_id == doctor.id, CaseDecision.embedding.is_not(None))
                .order_by(CaseDecision.id)) if is_evidence(r)]
            print("\n".join(report(replay([(r.id, r.kind, r.embedding) for r in rows]))))
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("username")
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.username, args.backfill))
