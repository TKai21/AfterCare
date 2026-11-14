from pathlib import Path

import httpx
import pytest

from druglabels import LabelClient, LabelServiceError, failing_transport, fixture_transport

FIXTURES = Path(__file__).resolve().parent.parent / "demo" / "labels"


@pytest.fixture
async def labels():
    label_client = LabelClient(fixture_transport(FIXTURES))
    yield label_client
    await label_client.aclose()


def json_transport(routes):
    """MockTransport answering by path with the given JSON bodies; records requests."""
    seen = []

    def handler(request):
        seen.append(request)
        for path, body in routes.items():
            if request.url.path.endswith(path):
                return httpx.Response(200, json=body)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    transport.seen = seen
    return transport


async def test_resolve_colchicine_with_dose(labels):
    result = await labels.resolve("Colchicine", "0.6 mg")
    assert result.ingredients == [("2683", "colchicine")]
    assert result.strength_rxcui is not None


async def test_resolve_falls_back_to_name_only():
    terms = []

    def handler(request):
        if request.url.path.endswith("approximateTerm.json"):
            term = request.url.params["term"]
            terms.append(term)
            if term == "Colchicine 0.6 mg":
                return httpx.Response(200, json={"approximateGroup": {"inputTerm": None}})
            return httpx.Response(200, json={"approximateGroup": {"candidate": [{"rxcui": "2683"}]}})
        return httpx.Response(200, json={"relatedGroup": {"conceptGroup": [
            {"tty": "IN", "conceptProperties": [{"rxcui": "2683", "name": "colchicine"}]}]}})

    label_client = LabelClient(httpx.MockTransport(handler))
    result = await label_client.resolve("Colchicine", "0.6 mg")
    await label_client.aclose()
    assert terms == ["Colchicine 0.6 mg", "Colchicine"]
    assert result.ingredients == [("2683", "colchicine")]
    assert result.strength_rxcui is None


async def test_resolve_unknown_name_is_empty():
    transport = json_transport({"approximateTerm.json": {"approximateGroup": {"inputTerm": None}}})
    label_client = LabelClient(transport)
    result = await label_client.resolve("Zzqxv", None)
    await label_client.aclose()
    assert result.ingredients == []


async def test_resolve_service_down_raises():
    label_client = LabelClient(failing_transport())
    with pytest.raises(LabelServiceError):
        await label_client.resolve("Colchicine", "0.6 mg")
    await label_client.aclose()


async def test_candidates_newest_first_max_five_no_combinations():
    titles = [
        ("IBUPROFEN TABLET, FILM COATED [A]", "Jan 05, 2026"),
        ("IBUPROFEN AND FAMOTIDINE TABLET [B]", "Dec 30, 2026"),
        ("ADVIL (IBUPROFEN) TABLET, COATED [C]", "Mar 10, 2026"),
        ("IBUPROFEN CAPSULE, LIQUID FILLED [D]", "Nov 09, 2026"),
        ("DUAL ACTION (ACETAMINOPHEN, IBUPROFEN) TABLET [E]", "Dec 01, 2026"),
        ("IBUPROFEN TABLET [F]", "Feb 02, 2026"),
        ("IBUPROFEN TABLET, FILM COATED [G]", "Oct 20, 2026"),
        ("IBUPROFEN TABLET, FILM COATED [H]", "Jun 15, 2026"),
    ]
    data = [{"setid": f"s{i}", "title": t, "published_date": d} for i, (t, d) in enumerate(titles)]
    label_client = LabelClient(json_transport({"spls.json": {"data": data}}))
    result = await label_client.candidates("5640", "ibuprofen")
    await label_client.aclose()
    assert [c["labeler"] for c in result] == ["D", "G", "H", "C", "F"]
    assert [c["published"] for c in result] == [
        "2026-11-09", "2026-10-20", "2026-06-15", "2026-03-10", "2026-02-02"]
    assert all("FAMOTIDINE" not in c["title"] and "ACETAMINOPHEN" not in c["title"] for c in result)
    assert set(result[0]) == {"setid", "title", "published", "labeler"}


async def test_candidates_from_fixture_skip_combinations(labels):
    result = await labels.candidates("2683", "colchicine")
    assert 0 < len(result) <= 5
    assert all("PROBENECID" not in c["title"] for c in result)
    assert result[0]["published"] == "2026-11-09"


async def test_label_text_has_dosage_and_warnings(labels):
    text = await labels.label_text("5c1ceff7-8f00-48cc-8ea4-678106455ff1")
    assert text.strip()
    assert "dosage" in text.lower() and "warnings" in text.lower()


async def test_label_text_unknown_setid_raises(labels):
    with pytest.raises(LabelServiceError):
        await labels.label_text("00000000-0000-0000-0000-000000000000")


async def test_only_names_leave_the_server():
    requests = []
    inner = fixture_transport(FIXTURES)

    def handler(request):
        requests.append(request)
        return inner.handle_request(request)

    label_client = LabelClient(httpx.MockTransport(handler))
    resolution = await label_client.resolve("Colchicine", "0.6 mg")
    await label_client.candidates("2683", "colchicine")
    await label_client.label_text("5c1ceff7-8f00-48cc-8ea4-678106455ff1")
    await label_client.aclose()
    allowed = {"Colchicine", "Colchicine 0.6 mg", "1", "tty=IN", "IN", "2683", resolution.strength_rxcui,
               "5c1ceff7-8f00-48cc-8ea4-678106455ff1"}
    assert requests
    for request in requests:
        for key, value in request.url.params.items():
            assert key in {"term", "maxEntries", "tty", "rxcui"}
            assert value in allowed
        assert request.headers["user-agent"].startswith("AfterCare")


# ---------- confirm hook and background jobs ----------

import json as _json

from sqlalchemy import select

from conftest import auth_headers
from druglabels import (
    DEMO_LABELS, INTERRUPTED, mark_interrupted, patient_rxcuis, resolve_document_drug,
)
from models import DischargeDocument, DocumentDrug, DrugLabel


def med(name, dose=None):
    details = {"name": name}
    if dose:
        details["dose"] = dose
    text = f"Take {name} {dose or ''}".strip()
    return {"category": "medication", "text": text, "source_quote": text, "details": details}


def doc_text(*meds):
    return "Discharge date: 2026-03-02\n\n" + "\n\n".join(m["source_quote"] for m in meds)


async def confirm_meds(http, fake_llm, username, *meds):
    h = await auth_headers(http, username)
    fake_llm.script({"content": _json.dumps({"discharge_date": "2026-03-02", "items": list(meds)})})
    up = await http.post("/api/documents", data={"text": doc_text(*meds)}, headers=h)
    assert up.status_code == 200, up.text
    resp = await http.post("/api/documents/active/confirm", json={}, headers=h)
    assert resp.status_code == 200, resp.text
    return resp.json()["id"], h


async def rows(http, model):
    async with http.app.state.session_factory() as session:
        return list(await session.scalars(select(model).order_by(model.id)))


GOUT = (med("Colchicine", "0.6 mg"), med("Naproxen", "500 mg"))


async def test_confirm_resolves_medications(client, fake_llm):
    await confirm_meds(client, fake_llm, "alice", *GOUT)
    drugs = await rows(client, DocumentDrug)
    assert [(d.name, d.rxcui, d.status) for d in drugs] == [
        ("Colchicine", "2683", "resolved"), ("Naproxen", "7258", "resolved"),
    ]
    labels = await rows(client, DrugLabel)
    assert [(l.ingredient, l.status) for l in labels] == [("colchicine", "pending"), ("naproxen", "pending")]
    assert all(l.title and l.text and l.setid and l.sha256 and l.url for l in labels)


async def test_confirm_succeeds_when_services_down(client, fake_llm):
    client.app.state.label_client = LabelClient(failing_transport())
    doc_id, _ = await confirm_meds(client, fake_llm, "alice", *GOUT)
    drugs = await rows(client, DocumentDrug)
    assert [d.status for d in drugs] == ["failed", "failed"]
    assert all(d.error for d in drugs)
    assert await rows(client, DrugLabel) == []


async def test_same_ingredient_two_patients_one_label(client, fake_llm):
    first, _ = await confirm_meds(client, fake_llm, "alice", med("Colchicine", "0.6 mg"))
    second, _ = await confirm_meds(client, fake_llm, "bob", med("Colchicine", "0.6 mg"))
    assert [l.rxcui for l in await rows(client, DrugLabel)] == ["2683"]
    async with client.app.state.session_factory() as session:
        assert await patient_rxcuis(session, first) == frozenset({"2683"})
        assert await patient_rxcuis(session, second) == frozenset({"2683"})


async def test_rejected_ingredient_not_requeued(client, fake_llm):
    await confirm_meds(client, fake_llm, "alice", med("Colchicine", "0.6 mg"))
    async with client.app.state.session_factory() as session:
        label = (await session.scalars(select(DrugLabel))).one()
        label.status = "rejected"
        await session.commit()
    await confirm_meds(client, fake_llm, "bob", med("Colchicine", "0.6 mg"))
    assert [l.status for l in await rows(client, DrugLabel)] == ["rejected"]
    assert [d.status for d in await rows(client, DocumentDrug)] == ["resolved", "resolved"]


async def test_unresolvable_name_is_unresolved(client, fake_llm):
    nothing = {"approximateTerm.json": {"approximateGroup": {"inputTerm": None}}}
    client.app.state.label_client = LabelClient(json_transport(nothing))
    await confirm_meds(client, fake_llm, "alice", med("Multivitamin"))
    (drug,) = await rows(client, DocumentDrug)
    assert (drug.status, drug.rxcui) == ("unresolved", None)
    assert await rows(client, DrugLabel) == []


async def test_retry_resolves_a_failed_row(client, fake_llm):
    real = client.app.state.label_client
    client.app.state.label_client = LabelClient(failing_transport())
    await confirm_meds(client, fake_llm, "alice", med("Colchicine", "0.6 mg"))
    (drug,) = await rows(client, DocumentDrug)
    assert drug.status == "failed"
    await resolve_document_drug(client.app.state.session_factory, real, drug.id)
    (drug,) = await rows(client, DocumentDrug)
    assert (drug.status, drug.rxcui, drug.error) == ("resolved", "2683", None)
    assert [l.status for l in await rows(client, DrugLabel)] == ["pending"]


async def test_patient_rxcuis_empty_without_confirmed_document(client, fake_llm):
    async with client.app.state.session_factory() as session:
        assert await patient_rxcuis(session, None) == frozenset()
    h = await auth_headers(client, "alice")
    fake_llm.script({"content": _json.dumps({"discharge_date": "2026-03-02", "items": [GOUT[0]]})})
    up = await client.post("/api/documents", data={"text": doc_text(GOUT[0])}, headers=h)
    async with client.app.state.session_factory() as session:
        assert await patient_rxcuis(session, up.json()["id"]) == frozenset()


async def test_mark_interrupted(client, fake_llm):
    doc_id, _ = await confirm_meds(client, fake_llm, "alice", med("Colchicine", "0.6 mg"))
    async with client.app.state.session_factory() as session:
        for model in (DrugLabel, DocumentDrug):
            for row in await session.scalars(select(model)):
                row.status = "fetching"
        await session.commit()
    assert await mark_interrupted(client.app.state.session_factory) == 2
    for model in (DrugLabel, DocumentDrug):
        (row,) = await rows(client, model)
        assert (row.status, row.error) == ("failed", INTERRUPTED)


async def test_no_real_network_in_default_app_client(client):
    transport = client.app.state.label_client._client._transport
    assert isinstance(transport, httpx.MockTransport)


async def test_missing_name_is_unresolved_without_lookup(client, fake_llm):
    client.app.state.label_client = LabelClient(failing_transport())
    unnamed = {"category": "medication", "text": "Take your pills.", "source_quote": "Take your pills.", "details": {}}
    await confirm_meds(client, fake_llm, "alice", unnamed)
    (drug,) = await rows(client, DocumentDrug)
    assert (drug.status, drug.rxcui, drug.error) == ("unresolved", None, None)


async def test_approved_label_records_build_library_records(tmp_path):
    from datetime import datetime
    from db import init_db, make_engine, make_session_factory
    from druglabels import approved_label_records

    url = f"sqlite+aiosqlite:///{tmp_path / 'labels.db'}"
    engine = make_engine(url)
    await init_db(engine)
    async with make_session_factory(engine)() as session:
        session.add_all([
            DrugLabel(rxcui="2683", ingredient="colchicine", status="approved", setid="abc-1",
                      title="Colcrys tablet", text="Label text.", sha256="1" * 64,
                      collected_at=datetime(2026, 11, 14, 1, 2, 3)),
            DrugLabel(rxcui="7258", ingredient="naproxen", status="pending", setid="def-2",
                      title="Naprosyn", text="Other.", sha256="2" * 64),
        ])
        await session.commit()
    await engine.dispose()
    records = await approved_label_records(url)
    assert records == [{
        "id": "dailymed-rxcui-2683",
        "title": "DailyMed label: colchicine (Colcrys tablet)",
        "url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc-1",
        "topic": "Medications", "text": "Label text.", "sha256": "1" * 64,
        "collected_at": "2026-11-14T01:02:03", "rxcui": "2683",
    }]
