"""Drug-name lexicon built from RxNorm ingredients (IN) and brand names (BN).

`find` returns the canonical lower-case names a text mentions, as whole words,
case-insensitively, forgiving a trailing "'s" and a plural "s". Names that are
also everyday words (water, sodium, vitamin d, today) are dropped at load time
via data/drug_name_exclusions.json, so a food answer never trips on them.
"""
import json
import re
from collections.abc import Collection
from functools import lru_cache
from pathlib import Path

from config import PROJECT_DIR

DATA = PROJECT_DIR / "data"


class DrugLexicon:
    def __init__(self, names: dict[str, dict]):
        self.names = names
        self._by_rxcui: dict[str, set[str]] = {}
        self._brands_of: dict[str, set[str]] = {}
        for name, info in names.items():
            self._by_rxcui.setdefault(info["rxcui"], set()).add(name)
            if info["tty"] == "BN":
                for ing in info["ingredients"]:
                    self._brands_of.setdefault(ing, set()).add(name)
        pattern = "|".join(re.escape(n) for n in sorted(names, key=lambda n: (-len(n), n)))
        self._regex = (re.compile(rf"(?<![a-z0-9])(?:{pattern})(?:['’]s|s)?(?![a-z0-9])",
                                  re.IGNORECASE) if names else None)

    @classmethod
    def load(cls, path: Path = DATA / "drug_names.json",
             exclusions: Path = DATA / "drug_name_exclusions.json") -> "DrugLexicon":
        names = json.loads(Path(path).read_text())["names"]
        ex = json.loads(Path(exclusions).read_text())
        excluded = {w.lower() for w in ex.get("words", [])} | {n.lower() for n in ex.get("nutrients", [])}
        return cls({n: v for n, v in names.items() if n not in excluded})

    def find(self, text: str) -> set[str]:
        if self._regex is None:
            return set()
        found = set()
        for m in self._regex.finditer(text):
            word = m.group(0).lower()
            for base in (word, re.sub(r"['’]s$", "", word), word[:-1]):
                if base in self.names:
                    found.add(base)
                    break
        return found

    def equivalents(self, name: str) -> set[str]:
        name = name.lower()
        info = self.names.get(name)
        if info is None:
            return {name}
        out = {name}
        for ing in info["ingredients"]:
            out |= {n for n in self._by_rxcui.get(ing, ()) if self.names[n]["tty"] == "IN"}
        if info["tty"] == "IN":
            out |= self._brands_of.get(info["rxcui"], set())
        return out

    def names_for_rxcuis(self, rxcuis: Collection[str]) -> set[str]:
        out: set[str] = set()
        for r in rxcuis:
            out |= self._by_rxcui.get(str(r), set())
        return out


@lru_cache(maxsize=1)
def get_lexicon() -> DrugLexicon:
    return DrugLexicon.load()
