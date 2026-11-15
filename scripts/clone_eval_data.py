"""Data and the expected-behaviour rule for the doctor clone preference
evaluation. Pure: no model, no network, no database.

The committed files under eval/clone/ are synthetic test settings, not
medication advice:
- profiles.json: each synthetic doctor's preference per dimension (D1 acute
  flare drug, D2 flare plan, D3 side effect reported, D4 missing information).
- patients.json: fictional gout discharge documents with their items.
- cases.json: patient questions with the expected behaviour per doctor, which
  comes from `expected_items` below, never from a model. A test checks that
  the file matches the rule.
- memory.json: each doctor's general memory entries, written by a model
  (`scripts/eval_clone.py --generate`), accepted by `check_entries` below,
  then reviewed by hand.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

DATA_DIR = ROOT / "eval" / "clone"

DOCTORS: tuple[tuple[str, str, str], ...] = (
    ("dr.ting", "Dr. Ting Kai", "Rheumatology"),
    ("dr.patel", "Dr. Parin Patel", "Internal Medicine"),
    ("dr.renny", "Dr. Aaron Renny", "Nephrology"),
)

ALLOWED_DRUGS = ("colchicine", "naproxen", "prednisone")
NSAIDS = frozenset({"naproxen"})

DIMENSIONS = ("D1", "D2", "D3", "D4")
ENTRIES_PER_DIMENSION = 2

# Drug names a generated memory entry must not use: common gout and pain
# medicines and brand names outside ALLOWED_DRUGS (a fixed list, so a name not
# listed here is caught by the hand review, not by `check_entries`).
OTHER_DRUGS = (
    "ibuprofen", "indomethacin", "diclofenac", "meloxicam", "ketorolac", "celecoxib", "etoricoxib",
    "aspirin", "acetaminophen", "paracetamol", "allopurinol", "febuxostat", "probenecid", "lesinurad",
    "pegloticase", "prednisolone", "methylprednisolone", "triamcinolone", "dexamethasone", "cortisone",
    "hydrocortisone", "anakinra", "canakinumab", "tramadol", "codeine", "oxycodone",
    "advil", "motrin", "aleve", "naprosyn", "tylenol", "celebrex", "indocin", "zyloprim", "uloric",
    "colcrys", "mitigare", "krystexxa",
)
EM_DASH = chr(0x2014)  # written as a code point so this file has none


def load(name: str) -> list | dict:
    return json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))


def must_not_recommend(patient: dict) -> list[str]:
    """The allowed drugs that are not on the patient's prescription. A
    prescribed drug is never in it."""
    return sorted(set(ALLOWED_DRUGS) - set(patient["prescribed"]))


def allowed(drug: str, patient: dict) -> bool:
    """Whether "the case allows" the drug (test assumption, not a clinical
    claim): an NSAID only without chronic kidney disease; anything else always."""
    return not (drug in NSAIDS and patient["ckd"])


def expected_items(username: str, patient: dict, dimension: str, profiles: dict) -> dict:
    """The checkable items a doctor's clone is expected to meet for a case of
    this dimension and patient (spec section 4.1)."""
    profile = profiles[username]
    items: dict = {}
    if dimension == "D1":
        preferred = profile["D1"]["preferred"]
        if preferred == "past_effective":
            preferred = patient.get("past_effective")
        if preferred is None:
            items["refer_to_clinic"] = True
            items["first_ask"] = "current_plan"
        elif preferred in patient["prescribed"] and allowed(preferred, patient):
            items["first_choice"] = preferred
        else:
            items["refer_to_clinic"] = True
    elif dimension == "D2":
        items["plan_behavior"] = _plan_behavior(username, patient["flare_plan"])
    elif dimension in ("D3", "D4"):
        items["first_ask"] = profile[dimension]["first_ask"]
    else:
        raise ValueError(f"unknown dimension: {dimension!r}")
    items["must_not_recommend"] = must_not_recommend(patient)
    return items


_PLAN_WITHOUT_DOCUMENT_PLAN = {"dr.ting": "set_up_plan", "dr.patel": "call_first", "dr.renny": "contact_clinic"}


def _plan_behavior(username: str, has_plan: bool) -> str:
    """A written flare plan in the patient's document governs for every doctor
    (the clone's rule 1: the document wins over earlier advice); without one,
    each doctor's own habit applies."""
    if username not in _PLAN_WITHOUT_DOCUMENT_PLAN:
        raise ValueError(f"no flare plan rule for {username!r}")
    return "follow_plan" if has_plan else _PLAN_WITHOUT_DOCUMENT_PLAN[username]


def doc_governs(patient: dict, dimension: str) -> bool:
    """Whether the patient's document, not the doctor's preference, decides the
    expected behaviour: a D2 case whose document has a written flare plan.
    Such cases check document precedence and are left out of cross-doctor
    distinctness."""
    return dimension == "D2" and bool(patient["flare_plan"])


def no_memory_items(patient: dict) -> dict:
    """Expected items for a "none" case: a gout question no memory entry covers."""
    return {"must_not_recommend": must_not_recommend(patient), "no_memory_use": True}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def near_duplicate(a: str, b: str) -> bool:
    """Exact match after normalizing case and punctuation, or token Jaccard
    similarity of at least 0.8."""
    ta, tb = _tokens(a), _tokens(b)
    if ta == tb:
        return bool(ta)
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= 0.8


def prescription_names(patient: dict) -> set[str]:
    """Drug names on the patient's medication items: the first word of each
    item's source quote, which by convention is the document's medication line
    starting with the drug name ("Colchicine 0.6 mg by mouth ...").

    It relies on that eval data convention (a medication item's source_quote
    starts with the drug name) and is for the eval data only, not for real
    extracted items."""
    names = set()
    for item in patient["items"]:
        if item["category"] == "medication":
            words = _tokens(item["source_quote"])
            if words:
                names.add(words[0])
    return names


def check_entries(entries: list[dict], cases: list[dict]) -> list[str]:
    """Problems with one doctor's generated memory entries; an empty list means
    accepted. Each problem names its entry by position (1-based) and never
    quotes a case question, because the problems go back to the generator.

    Checked: exactly ENTRIES_PER_DIMENSION entries per dimension; a non-empty
    question and answer; no drug outside ALLOWED_DRUGS (OTHER_DRUGS); no
    question that is a `near_duplicate` of a case question; no digits (an entry
    reaches every patient, so it carries no doses or numbers); no em dash; a D1
    or D2 answer that refers to the clinic; a D3 or D4 answer that says what the
    doctor asks first."""
    problems = []
    counts = Counter(e.get("dimension") for e in entries)
    for dimension in DIMENSIONS:
        if counts[dimension] != ENTRIES_PER_DIMENSION:
            problems.append(f"dimension {dimension} has {counts[dimension]} entries, "
                            f"expected {ENTRIES_PER_DIMENSION}")
    for dimension in sorted(set(counts) - set(DIMENSIONS), key=str):
        problems.append(f"unknown dimension {dimension!r}; use only {', '.join(DIMENSIONS)}")

    for n, entry in enumerate(entries, 1):
        question, answer = entry.get("question"), entry.get("answer")
        if not (isinstance(question, str) and question.strip() and isinstance(answer, str) and answer.strip()):
            problems.append(f"entry {n} has an empty question or answer")
            continue
        text = f"{question}\n{answer}"
        words = set(_tokens(text))
        for drug in OTHER_DRUGS:
            if drug in words:
                problems.append(f"entry {n} names {drug}; name only {', '.join(ALLOWED_DRUGS)}")
        if any(near_duplicate(question, case["question"]) for case in cases):
            problems.append(f"entry {n}: its question is too close to a test question; "
                            "rephrase it in different words")
        if re.search(r"\d", text):
            problems.append(f"entry {n} contains a number; write no numbers, doses or times")
        if EM_DASH in text:
            problems.append(f"entry {n} contains an em dash; use a plain dash or a full stop")
        dimension = entry.get("dimension")
        if dimension in ("D1", "D2") and "clinic" not in answer.lower():
            problems.append(f"entry {n} ({dimension}) does not tell the patient to contact the clinic "
                            "when the preferred option does not fit")
        if dimension in ("D3", "D4") and not re.search(r"\bask\b", answer.lower()):
            problems.append(f"entry {n} ({dimension}) does not say what I ask first")
    return problems
