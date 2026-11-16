"""Build data/drug_names.json from RxNorm via the public RxNav API (run once).

Fetches every ingredient (IN) and brand name (BN) concept, then each brand's
ingredients. Raw responses are cached on disk so a rerun resumes. Sends no
personal data. Exclusions (everyday words that are also RxNorm names) live in
data/drug_name_exclusions.json, which is hand-maintained, never written by this
script, and applied at load time.

Usage: python scripts/build_drug_names.py [cache_dir]  (default .cache/rxnav, git-ignored)
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import httpx

BASE = "https://rxnav.nlm.nih.gov/REST"
HEADERS = {"User-Agent": "AfterCare-ClassProject/1.0"}
MIN_INTERVAL = 0.08  # about 12 requests per second, under RxNav's 20
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "rxnav"
DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "drug_names.json"


_last = 0.0


def fetch(client: httpx.Client, path: str, cache: Path, key: str) -> dict:
    global _last
    f = cache / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    for attempt in range(5):
        wait = MIN_INTERVAL - (time.monotonic() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()
        try:
            r = client.get(BASE + path, timeout=30)
            r.raise_for_status()
            data = r.json()
            f.write_text(json.dumps(data))
            return data
        except (httpx.HTTPError, ValueError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"RxNav request failed: {path}")


def main(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS) as client:
        allc = fetch(client, "/Prescribe/allconcepts.json?tty=IN+BN", cache, "allconcepts")
        concepts = allc["minConceptGroup"]["minConcept"]
        names: dict[str, dict] = {}
        brands = []
        for c in concepts:
            name = c["name"].strip().lower()
            if c["tty"] == "IN":
                names[name] = {"rxcui": c["rxcui"], "tty": "IN", "ingredients": [c["rxcui"]]}
            else:
                brands.append((name, c["rxcui"]))
        for i, (name, rxcui) in enumerate(brands):
            rel = fetch(client, f"/rxcui/{rxcui}/related.json?tty=IN", cache, f"bn-{rxcui}")
            groups = rel.get("relatedGroup", {}).get("conceptGroup", []) or []
            ings = sorted({p["rxcui"] for g in groups if g.get("tty") == "IN"
                           for p in g.get("conceptProperties", [])})
            if ings and name not in names:
                names[name] = {"rxcui": rxcui, "tty": "BN", "ingredients": ings}
            if i % 500 == 0:
                print(f"{i}/{len(brands)} brands", file=sys.stderr)
    out = {"source": "RxNorm via RxNav (rxnav.nlm.nih.gov), tty IN and BN",
           "date": date.today().isoformat(), "names": dict(sorted(names.items()))}
    OUT.write_text(json.dumps(out, indent=0, ensure_ascii=False) + "\n")
    n_in = sum(1 for v in names.values() if v["tty"] == "IN")
    print(f"wrote {OUT}: {n_in} ingredients, {len(names) - n_in} brands")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CACHE)
