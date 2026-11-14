"""FDA drug labels: RxNorm lookup and DailyMed label client.

Only the medication name, the dose text, RxCUIs and setids ever leave the server.
"""

import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from db import make_engine, make_session_factory
from knowledge import DAILYMED_SPL_XML, USER_AGENT, dailymed_sections
from models import DischargeDocument, DocumentDrug, DrugLabel, ExtractedItem

logger = logging.getLogger(__name__)

DEMO_LABELS = Path(__file__).resolve().parent.parent / "demo" / "labels"
DAILYMED_PAGE = "https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={setid}"
INTERRUPTED = "Interrupted, try again"
UNEXPECTED = "Unexpected error, try again"

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


# ---------- background jobs (own sessions, never raise) ----------

async def _set_status(session_factory, model, row_id: int, status: str, error: str | None = None) -> None:
    async with session_factory() as session:
        await session.execute(update(model).where(model.id == row_id).values(status=status, error=error))
        await session.commit()


async def _find_candidates(client: LabelClient, resolution: Resolution, rxcui: str, ingredient: str) -> list[dict]:
    """Strength-level candidates for a single-ingredient product, else (or when
    that finds nothing) the ingredient's own."""
    if resolution.strength_rxcui and len(resolution.ingredients) == 1:
        try:
            found = await client.candidates(resolution.strength_rxcui, ingredient)
        except LabelServiceError:
            found = []  # the ingredient query below raises if the service is really down
        if found:
            return found
    return await client.candidates(rxcui, ingredient)


async def fetch_label(session_factory, client: LabelClient, label_id: int, setid: str | None = None) -> None:
    """Fetch the text of one label (the newest candidate unless `setid` is given)
    and leave it pending, no_label or failed. Never raises."""
    try:
        async with session_factory() as session:
            label = await session.get(DrugLabel, label_id)
            if label is None:
                return
            label.status, label.error = "fetching", None
            rxcui, ingredient, candidates = label.rxcui, label.ingredient, list(label.candidates or [])
            await session.commit()
        try:
            if not candidates:
                candidates = await client.candidates(rxcui, ingredient)
                async with session_factory() as session:
                    await session.execute(
                        update(DrugLabel).where(DrugLabel.id == label_id).values(candidates=candidates)
                    )
                    await session.commit()
            if setid is None and not candidates:
                await _set_status(session_factory, DrugLabel, label_id, "no_label")
                return
            chosen = setid or candidates[0]["setid"]
            title = next((c["title"] for c in candidates if c["setid"] == chosen), "")
            text = await client.label_text(chosen)
        except LabelServiceError as error:
            await _set_status(session_factory, DrugLabel, label_id, "failed", str(error))
            return
        async with session_factory() as session:
            await session.execute(
                update(DrugLabel).where(DrugLabel.id == label_id).values(
                    status="pending", error=None, setid=chosen, title=title,
                    url=DAILYMED_PAGE.format(setid=chosen), text=text,
                    sha256=hashlib.sha256(text.encode()).hexdigest(),
                    collected_at=datetime.now(timezone.utc).replace(tzinfo=None),
                    reviewed_by=None, reviewed_at=None,
                )
            )
            await session.commit()
    except Exception:
        logger.exception("Fetching drug label %s failed", label_id)
        await _record_failure(session_factory, DrugLabel, label_id)


async def _record_failure(session_factory, model, row_id: int) -> None:
    try:
        await _set_status(session_factory, model, row_id, "failed", UNEXPECTED)
    except Exception:
        logger.exception("Could not record the failure of %s %s", model.__tablename__, row_id)


async def _ensure_label(session_factory, client: LabelClient, resolution: Resolution, rxcui: str, ingredient: str) -> None:
    """Create the label for a new ingredient and fetch it. An ingredient with a
    row (any status) is left alone, so a rejected drug is not fetched again."""
    async with session_factory() as session:
        if await session.scalar(select(DrugLabel.id).where(DrugLabel.rxcui == rxcui)) is not None:
            return
        label = DrugLabel(rxcui=rxcui, ingredient=ingredient, status="fetching", candidates=[])
        session.add(label)
        try:
            await session.commit()
        except IntegrityError:  # another job claimed the ingredient first
            return
        label_id = label.id
    try:
        candidates = await _find_candidates(client, resolution, rxcui, ingredient)
    except LabelServiceError as error:
        await _set_status(session_factory, DrugLabel, label_id, "failed", str(error))
        return
    if not candidates:
        await _set_status(session_factory, DrugLabel, label_id, "no_label")
        return
    async with session_factory() as session:
        await session.execute(update(DrugLabel).where(DrugLabel.id == label_id).values(candidates=candidates))
        await session.commit()
    await fetch_label(session_factory, client, label_id)


async def _resolve_row(session_factory, client: LabelClient, row_id: int) -> None:
    async with session_factory() as session:
        row = await session.get(DocumentDrug, row_id)
        if row is None:
            return
        item = await session.get(ExtractedItem, row.extracted_item_id)
        details = (item.details if item else None) or {}
        name = str(details.get("name") or "").strip()
        dose = str(details.get("dose") or "").strip() or None
        row.status, row.error = "fetching", None
        await session.commit()
    if not name:
        await _set_status(session_factory, DocumentDrug, row_id, "unresolved")
        return
    try:
        resolution = await client.resolve(name, dose)
    except LabelServiceError as error:
        await _set_status(session_factory, DocumentDrug, row_id, "failed", str(error))
        return
    if not resolution.ingredients:
        await _set_status(session_factory, DocumentDrug, row_id, "unresolved")
        return
    async with session_factory() as session:
        row = await session.get(DocumentDrug, row_id)
        taken = set(await session.scalars(
            select(DocumentDrug.rxcui).where(DocumentDrug.extracted_item_id == row.extracted_item_id)
        ))
        row_free = row.rxcui is None
        for rxcui, ingredient in resolution.ingredients:
            if rxcui in taken:
                continue
            if row_free:
                row.rxcui, row.status, row_free = rxcui, "resolved", False
            else:
                session.add(DocumentDrug(
                    document_id=row.document_id, extracted_item_id=row.extracted_item_id,
                    name=row.name, rxcui=rxcui, status="resolved",
                ))
        if row_free:  # every ingredient already has its own row
            await session.delete(row)
        await session.commit()
    for rxcui, ingredient in resolution.ingredients:
        await _ensure_label(session_factory, client, resolution, rxcui, ingredient)


async def resolve_document(session_factory, client: LabelClient, document_id: int) -> None:
    """Resolve every medication of a confirmed document to ingredients and labels.
    Never raises."""
    try:
        async with session_factory() as session:
            items = list(await session.scalars(
                select(ExtractedItem)
                .where(ExtractedItem.document_id == document_id,
                       ExtractedItem.category == "medication",
                       ExtractedItem.status == "active")
                .order_by(ExtractedItem.id)
            ))
            rows = [
                DocumentDrug(
                    document_id=document_id, extracted_item_id=item.id,
                    name=str((item.details or {}).get("name") or "").strip()[:200], status="fetching",
                )
                for item in items
            ]
            session.add_all(rows)
            await session.commit()
            row_ids = [row.id for row in rows]
    except Exception:
        logger.exception("Resolving drugs of document %s failed", document_id)
        return
    for row_id in row_ids:
        await _resolve_row_safely(session_factory, client, row_id)


async def _resolve_row_safely(session_factory, client: LabelClient, row_id: int) -> None:
    try:
        await _resolve_row(session_factory, client, row_id)
    except Exception:
        logger.exception("Resolving document drug %s failed", row_id)
        await _record_failure(session_factory, DocumentDrug, row_id)


async def resolve_document_drug(session_factory, client: LabelClient, document_drug_id: int) -> None:
    """Retry one document_drugs row. Never raises."""
    await _resolve_row_safely(session_factory, client, document_drug_id)


async def mark_interrupted(session_factory) -> int:
    """Rows left `fetching` by a restart become `failed`. Returns how many."""
    count = 0
    async with session_factory() as session:
        for model in (DrugLabel, DocumentDrug):
            result = await session.execute(
                update(model).where(model.status == "fetching").values(status="failed", error=INTERRUPTED)
            )
            count += result.rowcount
        await session.commit()
    return count


async def patient_rxcuis(session, document_id: int | None) -> frozenset[str]:
    """The RxCUIs whose labels this patient may read: resolved drugs of a confirmed
    document whose label a doctor has approved. Empty for None or any other status.

    Requiring the approval here, too, means a label document left in the library
    store (say a restored index) is still never shown unless its row is approved."""
    if document_id is None:
        return frozenset()
    rows = await session.scalars(
        select(DocumentDrug.rxcui)
        .join(DischargeDocument, DischargeDocument.id == DocumentDrug.document_id)
        .join(DrugLabel, DrugLabel.rxcui == DocumentDrug.rxcui)
        .where(DocumentDrug.document_id == document_id,
               DischargeDocument.status == "confirmed",
               DocumentDrug.status == "resolved",
               DrugLabel.status == "approved")
    )
    return frozenset(rows)


async def approved_rxcuis(session_factory) -> frozenset[str]:
    """RxCUIs of every approved label."""
    async with session_factory() as session:
        return frozenset(await session.scalars(select(DrugLabel.rxcui).where(DrugLabel.status == "approved")))


LABEL_TOPIC = "Medications"
LABEL_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"


def label_record(label: DrugLabel) -> dict:
    """The education library record of an approved label."""
    return {
        "id": f"dailymed-rxcui-{label.rxcui}",
        "title": f"DailyMed label: {label.ingredient} ({label.title})",
        "url": LABEL_URL.format(setid=label.setid),
        "topic": LABEL_TOPIC,
        "text": label.text,
        "sha256": label.sha256,
        "collected_at": label.collected_at.isoformat() if label.collected_at else None,
        "rxcui": label.rxcui,
    }


async def approved_label_records(database_url: str) -> list[dict]:
    """Library records of every approved label, for a full index rebuild."""
    engine = make_engine(database_url)
    try:
        async with make_session_factory(engine)() as session:
            rows = await session.scalars(
                select(DrugLabel).where(DrugLabel.status == "approved").order_by(DrugLabel.id))
            return [label_record(row) for row in rows]
    finally:
        await engine.dispose()
