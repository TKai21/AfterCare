"""Build data/drug_names.json from RxNorm via the public RxNav API (run once).

Fetches every ingredient (IN) and brand name (BN) concept, then each brand's
ingredients. Raw responses are cached on disk so a rerun resumes. Sends no
personal data. Exclusions (everyday words that are also RxNorm names) live in
data/drug_name_exclusions.json and are applied at load time, not here.

Usage: python scripts/build_drug_names.py [cache_dir]
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
DEFAULT_CACHE = Path("/private/tmp/claude-501/-Users-wutingkai-Desktop-USC-EE542-Final/"
                     "8e6fb667-f613-4b9d-8d7f-417eab02d382/scratchpad/rxnav-cache")
DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "drug_names.json"
EXCLUSIONS_OUT = DATA / "drug_name_exclusions.json"

# RxNorm names that are also everyday diet, English or label-filler words. Each was
# found by intersecting the lexicon with (a) backend/safety.py STOP_WORDS (no hits),
# (b) every food name, alias and word in data/foods.json, (c) the nutrient keys of
# foods.json, (d) /usr/share/dict/words (321 hits, reviewed by hand), (e) every
# word and bigram in this repo's knowledge, demo, eval, docs and backend text.
EXCLUDED_WORDS = """
water, air, oxygen, nitrogen, helium, caffeine, cholesterol
calcium, potassium, sodium, magnesium, iron, zinc, copper, selenium, manganese, molybdenum
iodine, boron, silicon, silver, aluminum, titanium
glucose, fructose, sucrose, lactose, maltose, galactose, mannose, trehalose, corn syrup
high fructose corn syrup, maltose syrup, invert sugar, raw sugar, sea salt, sodium chloride
potassium chloride, sodium phosphate, sorbitol, xylitol, erythritol, aspartame, sucralose
starch, corn starch, potato starch, cellulose, microcrystalline cellulose, pectin, gelatin
collagen, inulin, lecithin, bran, oat bran, rice bran, rice bran oil, hemp, licorice, molasses
niacin, riboflavin, thiamine, biotin, choline, pyridoxine, tocopherol, lutein, lycopene
curcumin, quercetin, taurine, glycine, glutamate
vitamin a, vitamin b complex, vitamin b6, vitamin b12, vitamin d, vitamin e, vitamin k
vitamin k1, vitamin k2, vitamin mk6, vitamin mk7, folic acid, ascorbic acid, citric acid
malic acid, tartaric acid
apple juice, apricot juice, coconut juice, cranberry juice, cucumber juice, kiwi juice
lemon juice, mango juice, orange juice, pineapple juice, pomegranate juice, raspberry juice
strawberry juice, skim milk, apple cider vinegar, cider vinegar, cocoa butter
olive oil, coconut oil, canola oil, corn oil, sesame oil, sunflower oil, palm oil, soybean oil
cod liver oil, avocado oil, almond oil, cottonseed oil, argan oil
green tea leaf extract, white tea extract
succinate, glycolate, propylene, hydroxyproline, silicon dioxide, titanium dioxide
today, alli, align, amber, dragon, muse, perform, performs, react, rid, sojourn, tandem, thrive
vanquish, pronto, prolate, ella, indigo, agar, mica, talc, rosin, shellac, beeswax
morning after, new day
"""

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


def write_exclusions() -> None:
    foods = json.loads((DATA / "foods.json").read_text())["foods"]
    keys = {k for f in foods for k in f["nutrients"]}
    nutrients = sorted({k.rsplit("_", 1)[0].replace("_", " ") for k in keys})
    words = sorted({w.strip() for w in EXCLUDED_WORDS.replace("\n", ",").split(",") if w.strip()})
    EXCLUSIONS_OUT.write_text(json.dumps({"nutrients": nutrients, "words": words}, indent=1) + "\n")


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
    write_exclusions()
    out = {"source": "RxNorm via RxNav (rxnav.nlm.nih.gov), tty IN and BN",
           "date": date.today().isoformat(), "names": dict(sorted(names.items()))}
    OUT.write_text(json.dumps(out, indent=0, ensure_ascii=False) + "\n")
    n_in = sum(1 for v in names.values() if v["tty"] == "IN")
    print(f"wrote {OUT}: {n_in} ingredients, {len(names) - n_in} brands")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CACHE)
