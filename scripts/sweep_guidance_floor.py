"""Choose the guidance retrieval floor and margin. Dev only, run on demand; never part of pytest.

Retrieval keeps the top K rules whose score is at least FLOOR and within MARGIN
of the best score (backend/guidance.py). This sweeps both on the calibration
split of eval/clone_questions.json, in the calibration world (the doctors'
history from eval/clone_guidance.json read into rules, the demo patients), and
chooses the pair; it never tunes on the test split. The world is built once
(the chat model reads the rules, so it varies run to run) and every
(floor, margin) pair is scored on it: embeddings are deterministic.

    .venv/bin/python scripts/sweep_guidance_floor.py
    .venv/bin/python scripts/sweep_guidance_floor.py --out eval/calibration/floor-sweep.json

Which guidance entry an answerable question needs is not recorded in the
questions file. It is derived: the entries of the question's doctor whose
answer text contains the question's key facts (every fact, at least 75% of its
content words), best match only. A question with no such entry is not about
the guidance (its facts come from the discharge document) and is left out.
A tie, such as the fact "call the clinic", accepts a rule of any tied entry.

For a question that should be escalated, the cost is the mean number of rules
retrieved: advice the clone would be tempted to act on. The chosen pair keeps
a rule of the needed entry for as many answerable questions as possible among the
pairs returning no more rules per escalate question than the old fixed 0.45
floor does on the same world (a recall-only rule let unrelated advice through);
ties go to the higher floor, then the smaller margin. The same numbers
are reported on the test split and on the preference cases (eval/clone/cases.json:
whether a rule of the case's own dimension is returned, per case and doctor),
next to the fixed 0.45 floor this replaces. The API key is never printed.
"""
import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

QUESTIONS = ROOT / "eval" / "clone_questions.json"
GUIDANCE = ROOT / "eval" / "clone_guidance.json"
OUT = ROOT / "eval" / "calibration" / "floor-sweep.json"

FLOORS = [round(0.10 + 0.05 * i, 2) for i in range(8)]    # 0.10 .. 0.45
MARGINS = [round(0.05 + 0.05 * i, 2) for i in range(6)]   # 0.05 .. 0.30
OLD_FLOOR = 0.45  # the fixed floor this replaces (no margin)
LINK_COVERAGE = 0.75
TOP_K = 6

_STOP = set("a an the of to in is are be it you your for and or on at with while can if as by that this do not no".split())


# ---------- linking questions to entries, scoring, choosing (pure; tested) ----------

def _tokens(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if w not in _STOP}


def _coverage(fact: str, answer: str) -> float:
    wanted = _tokens(fact)
    return len(wanted & _tokens(answer)) / len(wanted) if wanted else 0.0


def needed_entries(question: dict, entries: list[dict]) -> list[str]:
    """Questions (texts) of the question's doctor's entries that hold all its
    key facts: the best match at 75% or more of each fact's content words,
    every tied entry. Empty when the facts are not in the guidance."""
    scored = []
    for e in entries:
        if e["doctor"] != question["doctor"] or not question["facts"]:
            continue
        scored.append((min(_coverage(f, e["answer"]) for f in question["facts"]), e["question"]))
    best = max((c for c, _ in scored), default=0.0)
    if best < LINK_COVERAGE:
        return []
    return [name for c, name in scored if abs(c - best) < 1e-9]


def evaluate(rows: list[dict], floor: float, margin: float | None, limit: int = TOP_K) -> dict:
    """What a (floor, margin) pair does on `rows` (see score_questions). `kept`
    counts answerable questions with a rule of a needed entry returned;
    `escalate_rules` is the mean number of rules returned for the questions to
    escalate. A margin of None is no margin: the old fixed floor."""
    from guidance import select_hits
    kept = answerable = 0
    returned = []
    for row in rows:
        hits = select_hits([(key, score) for key, score in row["hits"]], floor,
                           float("inf") if margin is None else margin, limit)
        if row["expect"] == "escalate":
            returned.append(len(hits))
        elif row["needed"]:
            answerable += 1
            kept += any(key in row["needed"] for key, _ in hits)
    return {"floor": floor, "margin": margin, "kept": kept, "answerable": answerable,
            "escalate_rules": round(sum(returned) / len(returned), 3) if returned else 0.0,
            "escalate_questions": len(returned)}


def sweep(rows: list[dict]) -> list[dict]:
    return [evaluate(rows, floor, margin) for floor in FLOORS for margin in MARGINS]


def choose(table: list[dict], cap: float) -> dict:
    """Among the pairs returning no more than `cap` rules per escalate question
    (what the old fixed floor returns on the same world), the one keeping the
    most answerable questions' rules; ties go to the higher floor, then the
    smaller margin. Raises ValueError when no pair is within the cap."""
    allowed = [r for r in table if r["escalate_rules"] <= cap + 1e-9]
    if not allowed:
        raise ValueError(f"no (floor, margin) pair returns {cap} rules or fewer per escalate question")
    return min(allowed, key=lambda r: (-r["kept"], -r["floor"], r["margin"]))


async def score_questions(session, llm, questions: list[dict], entries: list[dict],
                          users: dict, docs: dict) -> list[dict]:
    """Per question: expect, needed entries, and every rule or whole entry the
    gate lets through with its score as (entry question, score), best first
    (no floor, no margin, no limit). `users` and `docs` map the patient's
    username to their user and document ids."""
    from sqlalchemy import select
    from guidance import retrieve, unit_entry
    from models import DischargeDocument, ExtractedItem, User
    rows = []
    for q in questions:
        user = await session.get(User, users[q["patient"]])
        doc = await session.get(DischargeDocument, docs[q["patient"]])
        items = list(await session.scalars(select(ExtractedItem).where(ExtractedItem.document_id == doc.id)))
        doctor = await session.scalar(select(User).where(User.username == q["doctor"]))
        found = await retrieve(session, llm, user, doc, items, q["question"], doctor_id=doctor.id,
                               floor=-1.0, margin=float("inf"), limit=10**6)
        rows.append({"id": q["id"], "expect": q["expect"],
                     "needed": needed_entries(q, entries) if q["expect"] == "answer" else [],
                     "hits": [(unit_entry(unit).question, round(score, 4)) for unit, score in found]})
    return rows


# ---------- running ----------

def split_summary(rows: list[dict], chosen: dict) -> dict:
    old = evaluate(rows, OLD_FLOOR, None)
    return {"questions": len(rows), "chosen": evaluate(rows, chosen["floor"], chosen["margin"]), "old": old}


def format_table(table: list[dict], chosen: dict) -> str:
    lines = ["floor margin  kept/answerable  escalate rules"]
    for r in table:
        mark = "  <- chosen" if (r["floor"], r["margin"]) == (chosen["floor"], chosen["margin"]) else ""
        lines.append(f"{r['floor']:.2f}  {r['margin']:.2f}   {r['kept']:>3}/{r['answerable']:<3}"
                     f"          {r['escalate_rules']:.3f}{mark}")
    return "\n".join(lines)


async def run(out: Path) -> int:
    from sqlalchemy import select
    from calibrate_clone import history_world
    from config import get_settings
    from eval_clone import retrieval_check
    from models import DischargeDocument, User

    questions = json.loads(QUESTIONS.read_text())["questions"]
    entries = json.loads(GUIDANCE.read_text())["entries"]
    async with history_world() as (app, llm, settings):
        async with app.state.session_factory() as session:
            users = {u.username: u.id for u in await session.scalars(select(User))}
            name_of = {i: n for n, i in users.items()}
            docs = {name_of[d.active_user_id]: d.id
                    for d in await session.scalars(select(DischargeDocument)) if d.active_user_id}
            by_split = {split: await score_questions(session, llm, [q for q in questions if q["split"] == split],
                                                     entries, users, docs)
                        for split in ("calibration", "test")}
    table = sweep(by_split["calibration"])
    cap = evaluate(by_split["calibration"], OLD_FLOOR, None)["escalate_rules"]
    chosen = choose(table, cap)
    print(format_table(table, chosen))
    print(f"\nChosen on the calibration split: floor {chosen['floor']}, margin {chosen['margin']} "
          f"({chosen['kept']}/{chosen['answerable']} answerable kept, "
          f"{chosen['escalate_rules']} escalate rules)")
    result = {"date": datetime.now(timezone.utc).date().isoformat(), "chat_model": settings.llm_model,
              "embedding_model": settings.embedding_model, "top_k": TOP_K,
              "escalate_cap": cap,
              "note": ("Single run: the rules are read by the chat model once per world build. Linking of "
                       "questions to entries: facts matched to entry answers (see the script docstring). "
                       "'old' is the fixed 0.45 floor, no margin."),
              "chosen": {"floor": chosen["floor"], "margin": chosen["margin"]},
              "calibration": {**split_summary(by_split["calibration"], chosen), "table": table},
              "test": {**split_summary(by_split["test"], chosen), "table": sweep(by_split["test"])}}
    for split in ("calibration", "test"):
        s = result[split]
        print(f"{split}: chosen {s['chosen']['kept']}/{s['chosen']['answerable']} kept, "
              f"{s['chosen']['escalate_rules']} escalate rules; old floor 0.45: "
              f"{s['old']['kept']}/{s['old']['answerable']}, {s['old']['escalate_rules']}")
    cases = {}
    for name, floor, margin in (("chosen", chosen["floor"], chosen["margin"]), ("old", OLD_FLOOR, None)):
        # The calibration world's temporary library index is gone by now: the check
        # builds its own world from the real settings.
        rows = await retrieval_check(get_settings(), llm, floor=floor,
                                     margin=float("inf") if margin is None else margin)
        cases[name] = {"floor": floor, "margin": margin, "pairs": len(rows),
                       "own": sum(r["own"] for r in rows),
                       "mean_hits": round(sum(len(r["hits"]) for r in rows) / len(rows), 3),
                       "rows": [{"case": r["case"], "dimension": r["dimension"], "doctor": r["doctor"],
                                 "own": r["own"], "hits": r["hits"]} for r in rows]}
        print(f"cases ({name}, floor {floor}, margin {margin}): own-dimension rule returned for "
              f"{cases[name]['own']} of {cases[name]['pairs']} case-doctor pairs, "
              f"{cases[name]['mean_hits']} rules per pair")
    result["cases"] = cases
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(OUT), help="where to write the sweep (default eval/calibration/floor-sweep.json)")
    args = parser.parse_args(argv)
    from calibrate_clone import library_index_problem
    problem = library_index_problem()
    if problem:
        print(f"Error: {problem}", file=sys.stderr)
        return 1
    return asyncio.run(run(Path(args.out)))


if __name__ == "__main__":
    sys.exit(main())
