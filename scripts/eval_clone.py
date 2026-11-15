"""Doctor clone preference evaluation. Dev only, run on demand; never part of pytest.

Asks every case in eval/clone/cases.json of every doctor's clone, with that
doctor's memory entries active and inactive, through the real /api/ask route
(in-process ASGI client, the model and embeddings from .env), scores each
answer with program checks and a judge model, and writes the report.

    .venv/bin/python scripts/eval_clone.py                    # run (resumes), then report
    .venv/bin/python scripts/eval_clone.py --repeats 1 --out-dir /tmp/quick
    .venv/bin/python scripts/eval_clone.py --human-sheet      # 20 judge checks for people to grade
    .venv/bin/python scripts/eval_clone.py --agreement        # human against judge, Cohen's kappa

Everything synthetic stays out of the configured databases: each run uses a
temporary SQLite database and a temporary copy of the library index (plus the
demo colchicine and naproxen labels, read offline from demo/labels), both
deleted at the end. The configured index is only read, through SQLite's backup
API; the configured DATABASE_URL is never opened.

Outputs (in --out-dir, default eval/clone/): results.jsonl (a header line with
the model names, then one record per answer, appended as it is made, so an
interrupted run resumes), results.md and results.json (the report),
human_review.csv and human_key.json (the human sheet and its hidden key), and
agreement.json. The API key is never printed.
"""
import argparse
import asyncio
import csv
import json
import random
import re
import secrets
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from clone_eval_data import DATA_DIR, DOCTORS, load, prescription_names  # also puts backend/ on sys.path

import httpx  # noqa: E402
from sqlalchemy import delete, select, update  # noqa: E402

import config  # noqa: E402
from agent import fallback_answer  # noqa: E402
from app import create_app  # noqa: E402
from auth import hash_password  # noqa: E402
from clone_eval_score import (  # noqa: E402
    JUDGE_ITEMS,
    _SENTENCE,
    _question,
    checks_for,
    cohen_kappa,
    distinctness,
    hit_rates,
    judge_messages,
    parse_judge,
    recommended_unprescribed,
    safety,
    score_answer,
    scrub_doctor_names,
)
from druglabels import DEMO_LABELS, LabelClient, fixture_transport  # noqa: E402
from eval_library import copy_index, demo_label_records, existing_index_path  # noqa: E402
from guidance import eligible, embed_entry, patient_text  # noqa: E402
from knowledge import open_store  # noqa: E402
from models import (  # noqa: E402
    ChatMessage, DischargeDocument, DoctorGuidance, DocumentDrug, DrugLabel, ExtractedItem, User,
)

CONDITIONS = ("with_memory", "without_memory")
DIMENSIONS = ("D1", "D2", "D3", "D4")
PATIENT_PASSWORD = "eval-password-1"
# The demo labels the synthetic patients' drugs get (prednisone has none in demo/labels).
LABEL_DRUGS = ("colchicine", "naproxen")
SHEET_COLUMNS = ("row", "question", "prescription", "answer", "item", "item_question", "human_met")
YES = {"yes", "y", "true", "1"}
NO = {"no", "n", "false", "0"}
SYNTHETIC_NOTE = ("All doctors, patients, memory entries and questions in this evaluation are synthetic "
                  "test settings, pending clinical review. Nothing here is medication advice.")


@dataclass
class World:
    """Ids of what build_world created in the temporary database."""
    doctors: dict[str, int]  # username -> user id
    names: dict[str, str]  # username -> display name
    patients: dict[str, int]  # patient id -> user id
    documents: dict[str, int]  # patient id -> document id
    guidance: dict[str, list[int]]  # username -> memory entry ids, ascending
    labels: list[dict]  # the approved label records, for the library copy
    tokens: dict[str, str] = field(default_factory=dict)  # patient id -> bearer token


def _label_row(record: dict) -> DrugLabel:
    match = re.fullmatch(r"DailyMed label: (\S+) \((.*)\)", record["title"], re.S)
    if match is None:
        raise ValueError(f"unexpected label title: {record['title']!r}")
    collected = record.get("collected_at")
    return DrugLabel(
        rxcui=record["rxcui"], ingredient=match.group(1), title=match.group(2), status="approved",
        setid=record["url"].rsplit("setid=", 1)[-1], url=record["url"], text=record["text"],
        sha256=record["sha256"], collected_at=datetime.fromisoformat(collected) if collected else None,
        reviewed_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )


async def build_world(session, llm, patients: list[dict], memory: dict,
                      labels: list[dict] | None = None) -> World:
    """The three doctors with specialties, one user per synthetic patient
    (`eval-<id>`) with a confirmed document, its items and resolved drugs, the
    approved colchicine and naproxen labels, and every doctor's memory entries,
    embedded and active. `labels` defaults to the demo label records."""
    if labels is None:
        labels = await demo_label_records()
    labels = [r for r in labels if _label_row(r).ingredient in LABEL_DRUGS]
    rxcuis = {_label_row(r).ingredient: r["rxcui"] for r in labels}
    session.add_all(_label_row(r) for r in labels)

    doctors = {}
    for username, name, specialty in DOCTORS:
        doctors[username] = User(username=username, password_hash=hash_password(secrets.token_urlsafe(16)),
                                 role="doctor", display_name=name, specialty=specialty)
    session.add_all(doctors.values())

    users, documents = {}, {}
    password = hash_password(PATIENT_PASSWORD)
    for patient in patients:
        user = User(username=f"eval-{patient['id']}", password_hash=password)
        session.add(user)
        await session.flush()
        doc = DischargeDocument(
            user_id=user.id, active_user_id=user.id, status="confirmed", filename=f"{patient['id']}.txt",
            text=patient["text"], unverified=[], confirmed_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(doc)
        await session.flush()
        for raw in patient["items"]:
            drug = (next(iter(prescription_names({"items": [raw]})), None)
                    if raw["category"] == "medication" else None)
            item = ExtractedItem(document_id=doc.id, category=raw["category"], text=raw["text"],
                                 source_quote=raw["source_quote"], status="active",
                                 details={"name": drug.capitalize()} if drug else {})
            session.add(item)
            await session.flush()
            if drug in rxcuis:
                session.add(DocumentDrug(document_id=doc.id, extracted_item_id=item.id, name=drug,
                                         rxcui=rxcuis[drug], status="resolved"))
        users[patient["id"]], documents[patient["id"]] = user.id, doc.id

    await session.flush()
    guidance: dict[str, list[int]] = {}
    for username, _, _ in DOCTORS:
        entries = []
        for raw in memory.get(username, []):
            if raw.get("scope", "general") != "general":
                raise ValueError(f"memory entry of {username} is not general: {raw['question']!r}")
            entries.append(DoctorGuidance(
                doctor_id=doctors[username].id, question=raw["question"], answer=raw["answer"],
                scope="general", drug_tags=list(raw.get("drug_tags", [])),
                condition_tags=list(raw.get("condition_tags", [])), active=True))
        session.add_all(entries)
        await session.flush()
        for entry in entries:
            if not await embed_entry(llm, entry):
                raise RuntimeError(f"Could not embed a memory entry of {username}: {entry.question!r}")
        guidance[username] = sorted(e.id for e in entries)
    await session.commit()
    return World(doctors={u: d.id for u, d in doctors.items()},
                 names={u: n for u, n, _ in DOCTORS}, patients=users, documents=documents,
                 guidance=guidance, labels=labels)


async def set_memory(session, world: World, username: str, active: bool) -> None:
    await session.execute(update(DoctorGuidance)
                          .where(DoctorGuidance.id.in_(world.guidance[username]))
                          .values(active=active))
    await session.commit()


async def _token(client, world: World, patient_id: str) -> str:
    if patient_id not in world.tokens:
        resp = await client.post("/api/login", json={"username": f"eval-{patient_id}",
                                                     "password": PATIENT_PASSWORD})
        resp.raise_for_status()
        world.tokens[patient_id] = resp.json()["token"]
    return world.tokens[patient_id]


async def _eligible_ids(session, world: World, patient_id: str, username: str) -> list[int]:
    """The doctor's active entries the guidance gate lets reach this patient."""
    doc_id = world.documents[patient_id]
    doc = await session.get(DischargeDocument, doc_id)
    items = list(await session.scalars(
        select(ExtractedItem).where(ExtractedItem.document_id == doc_id)
        .execution_options(populate_existing=True)))
    entries = await session.scalars(
        select(DoctorGuidance)
        .where(DoctorGuidance.doctor_id == world.doctors[username], DoctorGuidance.active.is_(True))
        .order_by(DoctorGuidance.id).execution_options(populate_existing=True))
    text = patient_text(doc, items)
    ids = [e.id for e in entries if eligible(e, text)]
    await session.commit()
    return ids


async def ask_turn(client, session, world: World, case: dict, username: str) -> dict:
    """One question to one clone, with no earlier conversation: the patient's
    chat history for the document is deleted first."""
    patient_id = case["patient"]
    await session.execute(delete(ChatMessage).where(ChatMessage.document_id == world.documents[patient_id]))
    await session.commit()
    resp = await client.post(
        "/api/ask", json={"question": case["question"], "doctor_id": world.doctors[username]},
        headers={"Authorization": f"Bearer {await _token(client, world, patient_id)}"})
    if resp.status_code != 200:
        raise RuntimeError(f"/api/ask answered {resp.status_code} for {case['id']} {username}: {resp.text}")
    body = resp.json()
    return {
        "case": case["id"], "doctor": username, "kind": body["kind"], "answer": body["answer"],
        "guidance_ids": body["guidance_ids"],
        "fallback": body["answer"] == fallback_answer(world.names[username]),
        "eligible_guidance_ids": await _eligible_ids(session, world, patient_id, username),
    }


def _reasons(raw: str) -> dict:
    """Reason strings from a judge reply that parse_judge already accepted."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    return {cid: str(v.get("reason", "")) for cid, v in json.loads(text)["checks"].items()
            if isinstance(v, dict)}


async def judge_turn(judge_llm, case: dict, patient: dict, turn: dict) -> dict:
    """The scored fields of a record: items, conflicts_met, blocked, judge.
    A blocked answer (fallback or not_covered) is never sent to the judge."""
    doctor = turn["doctor"]
    items = case["expected"][doctor]
    checks, owner = checks_for(case, doctor)
    blocked = turn["kind"] == "not_covered" or bool(turn["fallback"])
    judge: dict[str, dict] = {}
    if checks and not blocked:
        reply = await judge_llm.chat(judge_messages(case, patient, turn["answer"], checks), json_mode=True)
        raw = reply.get("content") or ""
        verdicts = parse_judge(raw, list(checks))
        reasons = _reasons(raw)
        judge = {cid: {"kind": kind, "expected": value, "owner": owner[cid], "met": verdicts[cid],
                       "reason": reasons.get(cid, "")} for cid, (kind, value) in checks.items()}
    judged_own = {v["kind"]: v["met"] for v in judge.values() if v["owner"] == "own"}
    scored = score_answer(turn, items, judged_own, set(patient["prescribed"]))
    conflicts = [v["kind"] for v in judge.values() if v["owner"] == "other" and v["met"]]
    return {"items": scored["items"], "conflicts_met": conflicts, "blocked": scored["blocked"],
            "judge": judge}


def _key(record: dict) -> tuple:
    return (record["case"], record["doctor"], record["condition"], record["repeat"])


def read_results(path: Path) -> tuple[dict | None, list[dict]]:
    """(header, records) of a results file; (None, []) when it does not exist.
    A last line cut off mid-write (no newline, not JSON) is removed from the file."""
    if not path.is_file():
        return None, []
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    parsed = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except ValueError:
            if number == len(lines) and not content.endswith("\n"):
                path.write_text(content[:content.rfind("\n") + 1], encoding="utf-8")
                break
            raise SystemExit(f"{path} line {number} is not valid JSON; fix or delete the file.")
    else:
        if content and not content.endswith("\n"):
            path.write_text(content + "\n", encoding="utf-8")
    if not parsed:
        return None, []
    header, records = parsed[0], parsed[1:]
    if "clone_model" not in header or "judge_model" not in header:
        raise SystemExit(f"{path} line 1 is not a results header; fix or delete the file.")
    return header, records


def _append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value) + "\n")


@asynccontextmanager
async def open_world(settings, llm, store, patients: list[dict], memory: dict, labels: list[dict] | None = None):
    """The app on `settings` (its own database) with the world built and the
    approved labels in `store`; yields (client, session, world)."""
    label_client = LabelClient(fixture_transport(DEMO_LABELS))
    app = create_app(settings, llm=llm, store=store, label_client=label_client)
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://eval") as client:
                async with app.state.session_factory() as session:
                    world = await build_world(session, llm, patients, memory, labels)
                    # After the lifespan pruned labels that are not approved (none were yet).
                    for record in world.labels:
                        await store.add(record)
                    yield client, session, world
    finally:
        # The app's lifespan closes these only when its body ends without an
        # error; both calls are safe to repeat.
        await label_client.aclose()
        await app.state.engine.dispose()


async def run_eval(settings, llm, judge_llm, *, repeats: int, results_path: Path,
                   cases: list[dict] | None = None, patients: list[dict] | None = None,
                   memory: dict | None = None, judge_model: str | None = None) -> list[dict]:
    """Every case x doctor x condition x repeat not already in `results_path`,
    each appended there as soon as it is judged. Returns every planned record
    (resumed and new) in file order. `judge_model` defaults to the judge
    client's configured model."""
    cases = load("cases") if cases is None else cases
    patients = load("patients") if patients is None else patients
    memory = load("memory") if memory is None else memory
    judge_model = judge_model or judge_llm.settings.llm_model
    by_patient = {p["id"]: p for p in patients}

    header, existing = read_results(results_path)
    if header is None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text("", encoding="utf-8")
        _append(results_path, {"clone_model": settings.llm_model, "judge_model": judge_model,
                               "started": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        existing = []
    elif (header["clone_model"], header["judge_model"]) != (settings.llm_model, judge_model):
        raise SystemExit(
            f"{results_path} was started with clone model {header['clone_model']!r} and judge model "
            f"{header['judge_model']!r}; this run uses {settings.llm_model!r} and {judge_model!r}. "
            "Use another --out-dir, or delete the file to start over.")

    plan = [(case, username, condition, repeat)
            for repeat in range(repeats) for condition in CONDITIONS
            for case in cases for username, _, _ in DOCTORS]
    planned = {(c["id"], u, cond, r) for c, u, cond, r in plan}
    records = [r for r in existing if _key(r) in planned]
    done = {_key(r) for r in records}
    if len(done) == len(plan):
        return records

    with tempfile.TemporaryDirectory(prefix="aftercare-clone-eval-") as scratch:
        # The configured index is only read; the run works on a copy that is deleted afterwards.
        copy = Path(scratch) / "index.sqlite3"
        copy_index(existing_index_path(settings.rag_db_path), copy)
        run_settings = replace(settings, database_url=f"sqlite+aiosqlite:///{Path(scratch) / 'eval.sqlite3'}",
                               rag_db_path=str(copy), jwt_secret=secrets.token_urlsafe(32))
        store = open_store(run_settings, llm)
        async with open_world(run_settings, llm, store, patients, memory) as (client, session, world):
            memory_on: dict[str, bool] = {}
            for number, (case, username, condition, repeat) in enumerate(plan, 1):
                if (case["id"], username, condition, repeat) in done:
                    continue
                active = condition == "with_memory"
                if memory_on.get(username) is not active:
                    await set_memory(session, world, username, active)
                    memory_on[username] = active
                patient = by_patient[case["patient"]]
                turn = await ask_turn(client, session, world, case, username)
                record = {"case": case["id"], "patient": patient["id"], "question": case["question"],
                          "prescribed": list(patient["prescribed"]), "dimension": case["dimension"],
                          "doctor": username, "condition": condition, "repeat": repeat}
                record.update({k: v for k, v in turn.items() if k not in ("case", "doctor")})
                record.update(await judge_turn(judge_llm, case, patient, turn))
                _append(results_path, record)
                records.append(record)
                print(f"[{number}/{len(plan)}] {case['id']} {username} {condition} {repeat}: {record['kind']}",
                      flush=True)
    return records


# ---------------------------------------------------------------- report


def _flagged_sentences(record: dict) -> list[str]:
    prescribed = set(record.get("prescribed", []))
    return [s.strip() for s in _SENTENCE.split(record["answer"])
            if s.strip() and recommended_unprescribed(s, prescribed)]


def safety_flags(records: list[dict]) -> list[dict]:
    """Every answer the program check flagged as recommending a drug that is
    not prescribed, with the sentences, for a person to confirm (the check
    errs toward flagging)."""
    flags = []
    for r in records:
        drugs = recommended_unprescribed(r["answer"], set(r.get("prescribed", [])))
        if drugs:
            flags.append({"case": r["case"], "doctor": r["doctor"], "condition": r["condition"],
                          "repeat": r["repeat"], "drugs": drugs, "sentences": _flagged_sentences(r)})
    return flags


def blocked_counts(records: list[dict]) -> dict:
    """Per "doctor|condition": answers, blocked answers (fallback, not_covered),
    judge items not met because the answer was blocked, judge items not met in
    an answer that was not blocked, and with-memory answers whose gate let no
    entry through."""
    out: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        c = out[f"{r['doctor']}|{r['condition']}"]
        c["answers"] += 1
        c["blocked"] += bool(r["blocked"])
        c["fallback"] += bool(r["fallback"])
        c["not_covered"] += r["kind"] == "not_covered"
        missed = sum(1 for k, v in r["items"].items() if k in JUDGE_ITEMS and not v)
        c["items_blocked" if r["blocked"] else "items_not_followed"] += missed
        c["no_eligible_entry"] += r["condition"] == "with_memory" and not r["eligible_guidance_ids"]
    return {k: dict(v) for k, v in out.items()}


def _rate(entry: dict | None) -> str:
    if entry is None:
        return "-"
    return f"{entry['mean']:.2f} ({entry['min']:.2f}-{entry['max']:.2f})"


def _overall(rates: dict, condition: str) -> float | None:
    values = [v["mean"] for k, v in rates.items() if k.endswith("|" + condition)]
    return mean(values) if values else None


def write_report(records: list[dict], out_md: Path, out_json: Path, *, meta: dict | None = None,
                 agreement: dict | None = None) -> None:
    meta = meta or {}
    rates = hit_rates(records)
    distinct = distinctness(records)
    safe = safety(records)
    flags = safety_flags(records)
    blocked = blocked_counts(records)
    written = datetime.now(timezone.utc).isoformat(timespec="seconds")
    repeats = sorted({r["repeat"] for r in records})
    cited: dict[tuple, list[bool]] = defaultdict(list)
    for r in records:
        if r["condition"] == "with_memory":
            cited[(r["doctor"], r["dimension"])].append(bool(r["guidance_ids"]))

    md = ["# Doctor clone preference evaluation", "", f"> {SYNTHETIC_NOTE}", "",
          f"- Clone model: {meta.get('clone_model', '-')}",
          f"- Judge model: {meta.get('judge_model', '-')}",
          f"- Run started: {meta.get('started', '-')}; report written: {written}",
          f"- Answers: {len(records)}; repeats: {len(repeats)}", ""]
    overall = [(c, _overall(rates, c)) for c in CONDITIONS]
    md += ["## Habit hit rate", "",
           "Share of the doctor's judged items met (" + ", ".join(JUDGE_ITEMS) + "), per repeat; "
           "mean (min-max) over repeats. Descriptive only, no significance test. A blocked answer "
           "counts its items as not met.", "",
           "Overall mean: " + "; ".join(f"{c} {v:.2f}" if v is not None else f"{c} -" for c, v in overall), "",
           "| Doctor | Dimension | With memory | Without memory | Answers (with/without) | Cited guidance (with) |",
           "|---|---|---|---|---|---|"]
    for username, _, _ in DOCTORS:
        for dim in DIMENSIONS:
            with_, without = (rates.get(f"{username}|{dim}|{c}") for c in CONDITIONS)
            if with_ is None and without is None:
                continue
            share = cited.get((username, dim))
            md.append(f"| {username} | {dim} | {_rate(with_)} | {_rate(without)} | "
                      f"{with_['n'] if with_ else 0}/{without['n'] if without else 0} | "
                      f"{f'{sum(share) / len(share):.2f}' if share else '-'} |")

    md += ["", "## Cross-doctor distinctness (with memory)", "",
           "Share of repeats where the clone met all its own items and none of another doctor's "
           "conflicting items.", "", "| Case | Dimension | " + " | ".join(u for u, _, _ in DOCTORS) + " |",
           "|---|---|" + "---|" * len(DOCTORS)]
    dims = {r["case"]: r["dimension"] for r in records}
    for case in sorted({k.split("|")[0] for k in distinct}):
        cells = [distinct.get(f"{case}|{u}") for u, _, _ in DOCTORS]
        md.append(f"| {case} | {dims[case]} | "
                  + " | ".join(f"{c['distinct']:.2f}" if c else "-" for c in cells) + " |")

    md += ["", "## Safety", "",
           f"- Answers recommending a drug that is not prescribed: {safe['unprescribed_recommendations']} (target 0)",
           f"- No-memory cases that cited a guidance entry: {safe['no_memory_cases_citing_guidance']} (target 0)",
           f"- Blocked answers (fallback or not_covered): {safe['blocked']}", "",
           "### Flagged sentences", "",
           "The check errs toward flagging; confirm each one by reading it.", ""]
    md += [f"- {f['case']} {f['doctor']} {f['condition']} repeat {f['repeat']} ({', '.join(f['drugs'])}): "
           + " / ".join(f'"{s}"' for s in f["sentences"]) for f in flags] or ["None."]

    md += ["", "## Blocked by the clone's rules against memory not followed", "",
           "| Doctor | Condition | Answers | Blocked | Fallback | not_covered | Items not met, blocked "
           "| Items not met, not blocked | With memory, no eligible entry |",
           "|---|---|---|---|---|---|---|---|---|"]
    for username, _, _ in DOCTORS:
        for condition in CONDITIONS:
            c = blocked.get(f"{username}|{condition}")
            if c:
                md.append(f"| {username} | {condition} | {c['answers']} | {c.get('blocked', 0)} | "
                          f"{c.get('fallback', 0)} | {c.get('not_covered', 0)} | {c.get('items_blocked', 0)} | "
                          f"{c.get('items_not_followed', 0)} | {c.get('no_eligible_entry', 0)} |")

    if agreement is not None:
        md += ["", "## Judge-human agreement", "",
               f"- Rows graded: {agreement['rows']}",
               f"- Raw agreement: {agreement['agreement']:.2f}",
               f"- Cohen's kappa: {agreement['kappa']:.2f}"]
    md.append("")
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json.write_text(json.dumps({
        "meta": meta, "written": written, "note": SYNTHETIC_NOTE, "hit_rates": rates,
        "distinctness": distinct, "safety": safe, "safety_flags": flags, "blocked": blocked,
        "agreement": agreement,
    }, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- human sheet


def _bare_names() -> re.Pattern:
    names = {u for u, _, _ in DOCTORS}
    for _, display, _ in DOCTORS:
        words = display.removeprefix("Dr. ").split()
        names.add(" ".join(words))
        names.update(words)
    ordered = sorted(names, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(n).replace(r"\ ", r"\s+") for n in ordered) + r")\b")


_BARE_NAMES = _bare_names()


def scrub_for_review(text: str) -> str:
    """The judge's scrub, then any doctor name or username left without a title."""
    return _BARE_NAMES.sub("the doctor", scrub_doctor_names(text))


def write_human_sheet(records: list[dict], path: Path, n: int = 20, seed: int = 542,
                      key_path: Path | None = None) -> None:
    """`n` of the doctors' own judge checks, sampled with `seed`, for people to
    grade blind: the CSV has no doctor, condition or verdict. The key (row ->
    record and judge verdict) goes to `key_path`, default human_key.json beside it."""
    key_path = key_path or path.with_name("human_key.json")
    pool = [(index, cid) for index, r in enumerate(records)
            for cid, v in (r.get("judge") or {}).items() if v["owner"] == "own"]
    picks = random.Random(seed).sample(pool, min(n, len(pool)))
    key = {}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SHEET_COLUMNS)
        for row, (index, cid) in enumerate(picks, 1):
            r, check = records[index], records[index]["judge"][cid]
            writer.writerow([row, r["question"], ", ".join(r["prescribed"]) or "none",
                             scrub_for_review(r["answer"]), check["kind"],
                             _question(check["kind"], check["expected"]), ""])
            key[str(row)] = {"case": r["case"], "doctor": r["doctor"], "condition": r["condition"],
                             "repeat": r["repeat"], "check": cid, "kind": check["kind"],
                             "expected": check["expected"], "judge_met": check["met"],
                             "judge_reason": check["reason"]}
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")


def agreement(sheet: Path, key: Path) -> dict:
    """Raw agreement and Cohen's kappa of the filled sheet against the judge.
    Every row must say yes or no."""
    verdicts = json.loads(key.read_text(encoding="utf-8"))
    with sheet.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{sheet} has no rows.")
    blank = [r["row"] for r in rows if not (r.get("human_met") or "").strip()]
    if blank:
        raise ValueError(f"human_met is blank in row {', row '.join(blank)} of {sheet}; fill in yes or no.")
    human, judge = [], []
    for r in rows:
        value = r["human_met"].strip().lower()
        if value not in YES | NO:
            raise ValueError(f"row {r['row']}: {r['human_met'].strip()!r} is not yes or no.")
        if r["row"] not in verdicts:
            raise ValueError(f"row {r['row']} is not in {key}.")
        human.append(value in YES)
        judge.append(bool(verdicts[r["row"]]["judge_met"]))
    return {"rows": len(rows), "agreement": sum(h == j for h, j in zip(human, judge)) / len(rows),
            "kappa": cohen_kappa(human, judge), "human_yes": sum(human), "judge_yes": sum(judge)}


# ---------------------------------------------------------------- CLI


def _fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def _summary(records: list[dict]) -> str:
    rates = hit_rates(records)
    safe = safety(records)
    parts = [f"{c}: {v:.2f}" for c in CONDITIONS if (v := _overall(rates, c)) is not None]
    return (f"{len(records)} answers. Habit hit rate " + (", ".join(parts) or "-")
            + f". Unprescribed recommendations: {safe['unprescribed_recommendations']}; "
            f"no-memory cases citing guidance: {safe['no_memory_cases_citing_guidance']}; "
            f"blocked: {safe['blocked']}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=3, help="runs per case, doctor and condition (default 3)")
    parser.add_argument("--strong-model", default="gpt-4o", help="the judge model (default gpt-4o)")
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--human-sheet", action="store_true",
                      help="write human_review.csv and human_key.json from results.jsonl")
    step.add_argument("--agreement", action="store_true",
                      help="human against judge on the filled human_review.csv")
    parser.add_argument("--out-dir", default=str(DATA_DIR), help="where outputs go (default eval/clone)")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        return _fail("--repeats must be at least 1.")

    out = Path(args.out_dir)
    results, md, js = out / "results.jsonl", out / "results.md", out / "results.json"
    sheet, key, agreed = out / "human_review.csv", out / "human_key.json", out / "agreement.json"

    if args.human_sheet:
        if sheet.exists():
            return _fail(f"{sheet} already exists (it may hold grades); delete it to sample a new sheet.")
        _, records = read_results(results)
        if not records:
            return _fail(f"{results} has no answers yet; run the evaluation first.")
        write_human_sheet(records, sheet, key_path=key)
        print(f"Wrote {sheet} and {key}. Fill human_met with yes or no, then run --agreement.")
        return 0

    if args.agreement:
        try:
            result = agreement(sheet, key)
        except (ValueError, OSError) as exc:
            return _fail(str(exc))
        agreed.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        header, records = read_results(results)
        if records:
            write_report(records, md, js, meta=header, agreement=result)
        print(f"{result['rows']} rows: raw agreement {result['agreement']:.2f}, "
              f"Cohen's kappa {result['kappa']:.2f}")
        return 0

    settings = config.get_settings()
    if not settings.llm_model:
        return _fail("LLM_MODEL is not set. Configure the clone's model in the project .env.")
    if not settings.llm_api_key:
        return _fail("LLM_API_KEY is not set. Configure the API key in the project .env.")
    if not (DATA_DIR / "memory.json").is_file():
        return _fail(f"{DATA_DIR / 'memory.json'} is missing; generate the memory entries first.")

    from llm import LLMClient

    judge = LLMClient(replace(settings, llm_model=args.strong_model))
    records = asyncio.run(run_eval(settings, LLMClient(settings), judge, repeats=args.repeats,
                                   results_path=results, judge_model=args.strong_model))
    header, _ = read_results(results)
    write_report(records, md, js, meta=header)
    print(_summary(records))
    print(f"Report: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
