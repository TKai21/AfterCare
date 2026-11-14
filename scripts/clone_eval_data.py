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
"""
import json
import re
import sys
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


def _plan_behavior(username: str, has_plan: bool) -> str:
    if username == "dr.ting":
        return "point_to_plan" if has_plan else "set_up_plan"
    if username == "dr.patel":
        return "call_first"
    if username == "dr.renny":
        return "follow_plan" if has_plan else "contact_clinic"
    raise ValueError(f"no flare plan rule for {username!r}")


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
    starting with the drug name ("Colchicine 0.6 mg by mouth ...")."""
    names = set()
    for item in patient["items"]:
        if item["category"] == "medication":
            words = _tokens(item["source_quote"])
            if words:
                names.add(words[0])
    return names
