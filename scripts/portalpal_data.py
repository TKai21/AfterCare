"""Data for the PortalPal evaluation of the doctor clone. Pure: no model, no
network, no database.

eval/portalpal/syppm.csv is PortalPal-AI/Patient-Message-Response-Drafting
(CC BY-NC 4.0; see eval/portalpal/README.md): 100 rows of a semi-synthetic
patient message, a de-identified EHR summary and one primary care physician's
reply. The file is untrusted data, read with the csv module only.

- `load_rows`: the rows with ids "pp-000".."pp-099" (the row index).
- `split`: the fixed memory / tune / test split (60 / 15 / 25), committed as
  eval/portalpal/split.json.
- `ehr_document`: an EHR summary as the patient's confirmed document: its
  text, one medication item per active medication and one item per problem.
- `is_emergency_response`: the program rule for an emergency referral.
- `check_reply_set`: the problems of one normal row's scripted turn-2 replies
  (eval/portalpal/replies.json), empty when the set is usable.
- `check_entry`: the same for any row type (normal, no_sign, emergency).
"""
import csv
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "eval" / "portalpal"
CSV_PATH = DATA_DIR / "syppm.csv"

COLUMNS = {"message": "Patient Message", "ehr": "Patient EHR Summary", "response": "Clinician Response"}
SPLIT_SIZES = (("memory", 60), ("tune", 15), ("test", 25))
SEED = 542
REPLY_KINDS = ("benign", "warning", "warning_paraphrase")
REPLY_FIELDS = ("benign", "warning", "warning_sign", "warning_paraphrase", "warning_paraphrase_of")
ROW_TYPES = ("normal", "no_sign", "emergency")
MAX_REPLY_CHARS = 1000
EM_DASH = chr(0x2014)

_PROBLEMS = "###Full Active Problem List###:"
_ACTIVE = "Active (Start Date Before Message, Not Yet Ended):"
_NO_MEDICATIONS = "*No Active Medications*"


def row_id(index: int) -> str:
    return f"pp-{index:03d}"


def load_rows(path: Path = CSV_PATH) -> list[dict]:
    """Every row as {id, message, ehr, response}, in file order."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS.values() if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} has no column {', '.join(repr(c) for c in missing)}.")
        return [{"id": row_id(i), **{key: (raw[col] or "").strip() for key, col in COLUMNS.items()}}
                for i, raw in enumerate(reader)]


def split(rows: list[dict], seed: int = SEED) -> dict[str, list[str]]:
    """The row ids shuffled with `seed`, then cut into memory, tune and test;
    each list sorted. Needs exactly as many rows as the three parts hold."""
    total = sum(n for _, n in SPLIT_SIZES)
    if len(rows) != total:
        raise ValueError(f"the split needs {total} rows, got {len(rows)}.")
    ids = [r["id"] for r in rows]
    random.Random(seed).shuffle(ids)
    out, start = {}, 0
    for name, n in SPLIT_SIZES:
        out[name] = sorted(ids[start:start + n])
        start += n
    return out


def _section(lines: list[str], heading: str) -> list[str]:
    """The lines after `heading` up to the next heading or sub-heading."""
    try:
        start = [line.strip() for line in lines].index(heading)
    except ValueError:
        return []
    out = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if stripped.startswith("###") or stripped.startswith("Recently Ended"):
            break
        out.append(stripped)
    return out


def ehr_document(ehr: str) -> tuple[str, list[dict]]:
    """The document text (the summary itself) and its items: one "other" item
    per active problem ("Active problem: ...") and one "medication" item per
    active medication line; "*No Active Medications*" gives none. Each item's
    source_quote is copied from the text."""
    text = ehr.strip()
    lines = text.splitlines()
    items = []
    for line in _section(lines, _PROBLEMS):
        for problem in line.split(" - "):
            problem = problem.strip(" -")
            if problem:
                items.append({"category": "other", "text": f"Active problem: {problem}",
                              "source_quote": problem})
    for line in _section(lines, _ACTIVE):
        if line == _NO_MEDICATIONS or not line.startswith("-"):
            continue
        medication = line.lstrip("-").strip()
        if medication:
            items.append({"category": "medication", "text": f"Active medication: {medication}",
                          "source_quote": medication})
    return text, items


# ---------------------------------------------------------------- scripted replies

_FUNCTION_WORDS = frozenset("""
a an and any are as at be been by for from has have having i if in is it its me more my new no not of on or
our so some than that the their them then there these they this those to up was were with you your
""".split())


def key_words(phrase: str) -> set[str]:
    """The words of a warning sign that a paraphrase must not reuse."""
    return {w for w in re.findall(r"[a-z]+", phrase.lower()) if w not in _FUNCTION_WORDS}


def _stem(word: str) -> str:
    return word[:5] if len(word) > 5 else word


def reuses_key_words(text: str, phrase: str) -> list[str]:
    """The key words of `phrase` that `text` uses, by a shared stem
    ("feverish" reuses "fever", "painful" reuses "pain")."""
    words = re.findall(r"[a-z]+", text.lower())
    return sorted(k for k in key_words(phrase) if any(w.startswith(_stem(k)) for w in words))


def check_reply_set(entry: dict, response: str) -> list[str]:
    """The problems of one row's replies, given the physician's response; []
    when usable. The warning reply contains `warning_sign`, a sign the
    physician's response names; the paraphrase describes the same sign
    (`warning_paraphrase_of`) without its key words; the benign reply does
    not contain the sign."""
    missing = [f for f in REPLY_FIELDS if not isinstance(entry.get(f), str) or not entry[f].strip()]
    if missing:
        return [f"missing or empty: {', '.join(missing)}"]
    problems = []
    sign = entry["warning_sign"].strip()
    if sign.lower() not in response.lower():
        problems.append(f"warning_sign {sign!r} is not a sign the physician's response names")
    if sign.lower() not in entry["warning"].lower():
        problems.append(f"the warning reply does not contain its warning_sign {sign!r}")
    if entry["warning_paraphrase_of"].strip().lower() != sign.lower():
        problems.append(f"warning_paraphrase_of {entry['warning_paraphrase_of']!r} is not the warning_sign "
                        f"{sign!r}")
    reused = reuses_key_words(entry["warning_paraphrase"], sign)
    if reused:
        problems.append(f"the warning paraphrase reuses the key word(s) {', '.join(reused)} of {sign!r}")
    # The phrase, not its words: a benign reply about stomach pain may say
    # "pain" when the sign is "increased abdominal pain". A sign slipped in
    # other words is for the hand review to catch.
    if sign.lower() in entry["benign"].lower():
        problems.append(f"the benign reply mentions the warning sign {sign!r}")
    for kind in REPLY_KINDS:
        if len(entry[kind]) > MAX_REPLY_CHARS:
            problems.append(f"the {kind} reply is longer than {MAX_REPLY_CHARS} characters")
    if any(EM_DASH in entry[f] for f in REPLY_FIELDS):
        problems.append("use plain dashes, never long dashes")
    return problems


# ---------------------------------------------------------------- row types

_REFERRAL_PLACE = re.compile(r"\b911\b|\bemergency department\b", re.I)
_REFERRAL_URGENCY = re.compile(r"\b(?:immediately|right away|now)\b", re.I)


def is_emergency_response(response: str) -> bool:
    """True when the physician's response is an emergency referral: it names
    "911" or "emergency department" and says "immediately", "right away" or
    "now". A warning sign for the emergency department ("if you have
    vomiting, go to the emergency department") is advice, not a referral. The
    program decides this, never the model."""
    return bool(_REFERRAL_PLACE.search(response) and _REFERRAL_URGENCY.search(response))


def check_entry(entry: dict, response: str) -> list[str]:
    """The problems of one row's entry in replies.json, given the physician's
    response; [] when usable. `row_type` is "emergency" (no replies; the run
    sends turn 1 only), "no_sign" (a benign reply only; the response names no
    warning sign) or "normal" (all three replies, `check_reply_set`). The row
    type must agree with `is_emergency_response`."""
    row_type = entry.get("row_type")
    if row_type not in ROW_TYPES:
        return [f"row_type {row_type!r} is not one of {', '.join(ROW_TYPES)}"]
    if is_emergency_response(response) != (row_type == "emergency"):
        return [f"row_type {row_type!r} disagrees with the emergency rule for the physician's response"]
    body = {k: v for k, v in entry.items() if k != "row_type"}
    if row_type == "emergency":
        return ["an emergency row has no replies"] if body else []
    if row_type == "normal":
        return check_reply_set(body, response)
    if not isinstance(body.get("benign"), str) or not body["benign"].strip():
        return ["missing or empty: benign"]
    problems = []
    if set(body) != {"benign"}:
        problems.append("a no_sign row has only a benign reply")
    if len(body["benign"]) > MAX_REPLY_CHARS:
        problems.append(f"the benign reply is longer than {MAX_REPLY_CHARS} characters")
    if EM_DASH in body["benign"]:
        problems.append("use plain dashes, never long dashes")
    return problems
