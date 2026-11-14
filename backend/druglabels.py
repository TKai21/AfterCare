"""FDA drug labels: RxNorm lookup and DailyMed label client.

Only the medication name, the dose text, RxCUIs and setids ever leave the server.
"""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from knowledge import DAILYMED_SPL_XML, USER_AGENT, dailymed_sections

RXNAV_APPROXIMATE = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
RXNAV_RELATED = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json"
DAILYMED_SPLS = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
MAX_CANDIDATES = 5
REQUEST_TIMEOUT = 15.0

# LOINC codes: indications, dosage and administration, contraindications,
# warnings, warnings and precautions, adverse reactions.
LABEL_SECTIONS = ["34067-9", "34068-7", "34070-3", "34071-1", "43685-7", "34084-4"]

_LABELER = re.compile(r"\s*\[([^\]]*)\]\s*$")
_DOSAGE_FORM = re.compile(
    r"\b(TABLET|CAPSULE|SUSPENSION|SOLUTION|CREAM|GEL|PATCH|INJECTION|POWDER|SYRUP|LIQUID|"
    r"SPRAY|OINTMENT|LOTION|KIT|ELIXIR|EMULSION|GRANULE|LOZENGE|SUPPOSITORY|DROPS)\b.*$"
)
_COMBINATION = re.compile(r" AND |/|,")


class LabelServiceError(Exception):
    """RxNav or DailyMed could not be reached or answered unusably."""


@dataclass
class Resolution:
    ingredients: list[tuple[str, str]]  # (rxcui, lower-case ingredient name)
    strength_rxcui: str | None = None


def _is_combination(product: str) -> bool:
    """True when a DailyMed title (labeler removed) names more than one active ingredient.

    Titles look like "BRAND (INGREDIENT, INGREDIENT) FORM" or "INGREDIENT AND INGREDIENT FORM".
    When the title has a parenthesized ingredient list, only that list is judged, so brand
    words such as "COLD AND SINUS" do not count. Otherwise the product name is judged after
    cutting off the dosage form, whose own comma ("TABLET, FILM COATED") is not a list.
    """
    inner = re.search(r"\((.*)\)", product)
    names = inner.group(1) if inner else _DOSAGE_FORM.sub("", product)
    return bool(_COMBINATION.search(names.upper()))


def _iso_date(text: str) -> str:
    return datetime.strptime(text, "%b %d, %Y").date().isoformat()


class LabelClient:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            raise LabelServiceError(f"Label service request failed: {type(error).__name__}") from error

    async def _json(self, url: str, params: dict | None = None) -> dict:
        response = await self._get(url, params)
        try:
            data = response.json()
        except ValueError as error:
            raise LabelServiceError("Label service sent unreadable data.") from error
        if not isinstance(data, dict):
            raise LabelServiceError("Label service sent unexpected data.")
        return data

    async def _approximate(self, term: str) -> str | None:
        data = await self._json(RXNAV_APPROXIMATE, {"term": term, "maxEntries": "1"})
        group = data.get("approximateGroup")
        found = group.get("candidate") if isinstance(group, dict) else None
        if not found:
            return None
        rxcui = found[0].get("rxcui") if isinstance(found[0], dict) else None
        return str(rxcui) if rxcui else None

    async def resolve(self, name: str, dose: str | None) -> Resolution:
        name, dose = name.strip(), (dose or "").strip()
        rxcui = await self._approximate(f"{name} {dose}") if dose else None
        if rxcui is None:
            rxcui = await self._approximate(name)
        if rxcui is None:
            return Resolution([])
        data = await self._json(RXNAV_RELATED.format(rxcui=rxcui), {"tty": "IN"})
        ingredients: list[tuple[str, str]] = []
        try:
            for group in data.get("relatedGroup", {}).get("conceptGroup") or []:
                for item in group.get("conceptProperties") or []:
                    pair = (str(item["rxcui"]), str(item["name"]).lower())
                    if pair not in ingredients:
                        ingredients.append(pair)
        except (AttributeError, KeyError, TypeError) as error:
            raise LabelServiceError("Label service sent unexpected data.") from error
        if not ingredients:
            return Resolution([])
        strength = rxcui if rxcui not in {i for i, _ in ingredients} else None
        return Resolution(ingredients, strength)

    async def candidates(self, rxcui: str, ingredient: str) -> list[dict]:
        """Up to 5 newest single-ingredient labels for this RxCUI.

        `ingredient` is only the name of the drug being looked up; the title rule does not need it.
        """
        data = await self._json(DAILYMED_SPLS, {"rxcui": rxcui})
        found = []
        try:
            for item in data.get("data") or []:
                title = str(item["title"])
                match = _LABELER.search(title)
                product = title[: match.start()] if match else title
                if _is_combination(product):
                    continue
                found.append({
                    "setid": str(item["setid"]),
                    "title": title,
                    "published": _iso_date(str(item["published_date"])),
                    "labeler": match.group(1) if match else "",
                })
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise LabelServiceError("Label service sent unexpected data.") from error
        found.sort(key=lambda label: label["published"], reverse=True)
        return found[:MAX_CANDIDATES]

    async def label_text(self, setid: str) -> str:
        response = await self._get(DAILYMED_SPL_XML.format(setid=setid))
        try:
            return dailymed_sections(response.text, LABEL_SECTIONS)
        except (ValueError, ET.ParseError) as error:  # no matching sections or invalid XML
            raise LabelServiceError("Label text could not be read.") from error


def _request_key(request: httpx.Request) -> str:
    query = "&".join(f"{key}={value}" for key, value in sorted(request.url.params.multi_items()))
    return request.url.path + (f"?{query}" if query else "")


def fixture_transport(directory: Path) -> httpx.MockTransport:
    """Serve recorded RxNav and DailyMed responses from `directory`/index.json; 404 otherwise."""
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())

    def handler(request: httpx.Request) -> httpx.Response:
        name = index.get(_request_key(request))
        if name is None:
            return httpx.Response(404, text="not in fixtures")
        return httpx.Response(200, text=(directory / name).read_text())

    return httpx.MockTransport(handler)


def failing_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("label services are down", request=request)

    return httpx.MockTransport(handler)
