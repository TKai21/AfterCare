import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, update

from conftest import auth_headers, doctor_headers
from druglabels import DEMO_LABELS, LabelClient, failing_transport, fixture_transport, patient_rxcuis
from llm import LLMError
from models import DischargeDocument, DocumentDrug, DrugLabel
from test_druglabels import GOUT, confirm_meds, med, rows

COLCHICINE_SETID = "5c1ceff7-8f00-48cc-8ea4-678106455ff1"


async def setup(client, fake_llm, *meds):
    doc_id, patient = await confirm_meds(client, fake_llm, "gout-patient", *(meds or GOUT))
    return doc_id, patient, await doctor_headers(client)


async def label_by_ingredient(client, name):
    return next(l for l in await rows(client, DrugLabel) if l.ingredient == name)


async def test_patient_gets_403(client, fake_llm):
    _, patient, _ = await setup(client, fake_llm)
    for method, path, body in [
        ("get", "/api/doctor/labels", None),
        ("post", "/api/doctor/labels/1/approve", None),
        ("post", "/api/doctor/labels/1/reject", None),
        ("post", "/api/doctor/labels/1/candidate", {"setid": COLCHICINE_SETID}),
        ("post", "/api/doctor/labels/1/retry", None),
        ("post", "/api/doctor/document-drugs/1/retry", None),
    ]:
        resp = await getattr(client, method)(path, headers=patient, **({"json": body} if body else {}))
        assert resp.status_code == 403, path


async def test_unknown_id_404(client, fake_llm):
    doctor = await doctor_headers(client)
    for path, body in [
        ("/api/doctor/labels/999/approve", None),
        ("/api/doctor/labels/999/reject", None),
        ("/api/doctor/labels/999/candidate", {"setid": "x"}),
        ("/api/doctor/labels/999/retry", None),
        ("/api/doctor/document-drugs/999/retry", None),
    ]:
        resp = await client.post(path, headers=doctor, **({"json": body} if body else {}))
        assert resp.status_code == 404, path


async def test_list_shows_pending_with_sources(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    data = (await client.get("/api/doctor/labels", headers=doctor)).json()
    assert data["unresolved"] == []
    colchicine = next(l for l in data["labels"] if l["ingredient"] == "colchicine")
    assert set(colchicine) == {
        "id", "rxcui", "ingredient", "status", "title", "url", "labeler", "published", "text", "error",
        "candidates", "sources", "reviewed_by_name", "reviewed_at"}
    assert colchicine["status"] == "pending" and colchicine["rxcui"] == "2683"
    assert colchicine["sources"] == [{"medication": "Take Colchicine 0.6 mg", "username": "gout-patient"}]
    assert colchicine["text"] and colchicine["candidates"] and colchicine["reviewed_by_name"] is None


async def test_list_orders_pending_first_and_hides_old_closed(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    colchicine = await label_by_ingredient(client, "colchicine")
    naproxen = await label_by_ingredient(client, "naproxen")
    async with client.app.state.session_factory() as session:
        await session.execute(update(DrugLabel).where(DrugLabel.id == colchicine.id)
                              .values(status="failed", error="x"))
        await session.commit()
    ids = [l["id"] for l in (await client.get("/api/doctor/labels", headers=doctor)).json()["labels"]]
    assert ids == [naproxen.id, colchicine.id]
    async with client.app.state.session_factory() as session:
        await session.execute(update(DrugLabel).where(DrugLabel.id == naproxen.id).values(
            status="rejected", reviewed_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=31)))
        await session.commit()
    ids = [l["id"] for l in (await client.get("/api/doctor/labels", headers=doctor)).json()["labels"]]
    assert ids == [colchicine.id]


async def test_list_shows_unresolved_drugs(client, fake_llm):
    client.app.state.label_client = LabelClient(failing_transport())
    _, _, doctor = await setup(client, fake_llm)
    data = (await client.get("/api/doctor/labels", headers=doctor)).json()
    assert data["labels"] == []
    assert [(u["medication"], u["username"], u["status"]) for u in data["unresolved"]] == [
        ("Take Colchicine 0.6 mg", "gout-patient", "failed"), ("Take Naproxen 500 mg", "gout-patient", "failed")]
    assert all(u["error"] for u in data["unresolved"])


async def test_approve_adds_to_store_and_is_retrievable_for_patient_only(client, fake_llm):
    doc_id, _, doctor = await setup(client, fake_llm)
    stone_doc, _ = await confirm_meds(client, fake_llm, "stone-patient", med("Tamsulosin", "0.4 mg"))
    colchicine = await label_by_ingredient(client, "colchicine")
    resp = await client.post(f"/api/doctor/labels/{colchicine.id}/approve", headers=doctor)
    assert resp.status_code == 200, resp.text
    view = resp.json()
    assert view["status"] == "approved" and view["reviewed_by_name"] == "Dr. Ting Kai" and view["reviewed_at"]
    store = client.app.state.store
    async with client.app.state.session_factory() as session:
        gout, stone = await patient_rxcuis(session, doc_id), await patient_rxcuis(session, stone_doc)
    hits = await store.retrieve("colchicine dosage gout flares", rxcuis=gout)
    assert any(h["source_id"] == "dailymed-rxcui-2683" for h in hits)
    assert not any(h["source_id"] == "dailymed-rxcui-2683"
                   for h in await store.retrieve("colchicine dosage gout flares", rxcuis=stone))


async def test_approve_embedding_failure_stays_pending(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    colchicine = await label_by_ingredient(client, "colchicine")

    async def broken(texts):
        raise LLMError("embedding down")
    fake_llm.embed = broken
    resp = await client.post(f"/api/doctor/labels/{colchicine.id}/approve", headers=doctor)
    assert resp.status_code == 502
    assert resp.json()["detail"] == "Could not add the label to the library. Try again."
    label = await label_by_ingredient(client, "colchicine")
    assert (label.status, label.reviewed_by, label.reviewed_at) == ("pending", None, None)
    assert client.app.state.store.status()["documents"] == []


async def test_approve_without_store_503(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    client.app.state.store = None
    colchicine = await label_by_ingredient(client, "colchicine")
    resp = await client.post(f"/api/doctor/labels/{colchicine.id}/approve", headers=doctor)
    assert (resp.status_code, resp.json()["detail"]) == (503, "The education library is not available right now.")
    assert (await label_by_ingredient(client, "colchicine")).status == "pending"


async def test_approve_and_reject_need_pending(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    colchicine = await label_by_ingredient(client, "colchicine")
    path = f"/api/doctor/labels/{colchicine.id}"
    assert (await client.post(f"{path}/reject", headers=doctor)).json()["status"] == "rejected"
    assert (await client.post(f"{path}/reject", headers=doctor)).status_code == 409
    assert (await client.post(f"{path}/approve", headers=doctor)).status_code == 409
    assert client.app.state.store.status()["documents"] == []


async def test_concurrent_approve_and_reject_one_wins_409(file_client, fake_llm):
    _, _, doctor = await setup(file_client, fake_llm)
    colchicine = await label_by_ingredient(file_client, "colchicine")
    path = f"/api/doctor/labels/{colchicine.id}"
    approve, reject = await asyncio.gather(
        file_client.post(f"{path}/approve", headers=doctor), file_client.post(f"{path}/reject", headers=doctor))
    assert sorted([approve.status_code, reject.status_code]) == [200, 409]
    final = (await label_by_ingredient(file_client, "colchicine")).status
    in_store = [d["id"] for d in file_client.app.state.store.status()["documents"]]
    assert (final, in_store) == (("approved", ["dailymed-rxcui-2683"]) if approve.status_code == 200
                                 else ("rejected", []))


async def test_reject_during_slow_add_cannot_win(file_client, fake_llm):
    _, _, doctor = await setup(file_client, fake_llm)
    colchicine = await label_by_ingredient(file_client, "colchicine")
    path = f"/api/doctor/labels/{colchicine.id}"
    entered, release = asyncio.Event(), asyncio.Event()
    real_embed = fake_llm.embed

    async def slow(texts):
        entered.set()
        await release.wait()
        return await real_embed(texts)
    fake_llm.embed = slow
    approving = asyncio.create_task(file_client.post(f"{path}/approve", headers=doctor))
    await entered.wait()
    assert (await file_client.post(f"{path}/reject", headers=doctor)).status_code == 409
    release.set()
    assert (await approving).status_code == 200
    assert (await label_by_ingredient(file_client, "colchicine")).status == "approved"
    assert [d["id"] for d in file_client.app.state.store.status()["documents"]] == ["dailymed-rxcui-2683"]


async def test_pick_candidate_on_rejected_refetches(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    colchicine = await label_by_ingredient(client, "colchicine")
    path = f"/api/doctor/labels/{colchicine.id}"
    await client.post(f"{path}/reject", headers=doctor)
    resp = await client.post(f"{path}/candidate", json={"setid": colchicine.candidates[0]["setid"]}, headers=doctor)
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "pending" and resp.json()["reviewed_at"] is None


async def test_pick_candidate_validates_setid_before_any_outside_call(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    colchicine = await label_by_ingredient(client, "colchicine")
    client.app.state.label_client = LabelClient(failing_transport())
    resp = await client.post(f"/api/doctor/labels/{colchicine.id}/candidate",
                             json={"setid": "../../etc/passwd"}, headers=doctor)
    assert resp.status_code == 422
    assert (await label_by_ingredient(client, "colchicine")).status == "pending"


async def test_pick_candidate_on_approved_409(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    colchicine = await label_by_ingredient(client, "colchicine")
    path = f"/api/doctor/labels/{colchicine.id}"
    assert (await client.post(f"{path}/approve", headers=doctor)).status_code == 200
    resp = await client.post(f"{path}/candidate", json={"setid": colchicine.candidates[0]["setid"]}, headers=doctor)
    assert resp.status_code == 409
    assert (await label_by_ingredient(client, "colchicine")).status == "approved"


async def test_retry_label_only_failed_or_no_label(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    colchicine = await label_by_ingredient(client, "colchicine")
    path = f"/api/doctor/labels/{colchicine.id}/retry"
    assert (await client.post(path, headers=doctor)).status_code == 409
    async with client.app.state.session_factory() as session:
        await session.execute(update(DrugLabel).where(DrugLabel.id == colchicine.id)
                              .values(status="failed", error="down"))
        await session.commit()
    resp = await client.post(path, headers=doctor)
    assert resp.status_code == 200 and resp.json()["status"] == "pending" and resp.json()["error"] is None


async def test_retry_failed_document_drug_resolves(client, fake_llm):
    real = client.app.state.label_client
    client.app.state.label_client = LabelClient(failing_transport())
    _, _, doctor = await setup(client, fake_llm, med("Colchicine", "0.6 mg"))
    (drug,) = await rows(client, DocumentDrug)
    assert drug.status == "failed"
    client.app.state.label_client = real
    path = f"/api/doctor/document-drugs/{drug.id}/retry"
    resp = await client.post(path, headers=doctor)
    assert resp.status_code == 200, resp.text
    assert resp.json()["unresolved"] == []
    assert [l["ingredient"] for l in resp.json()["labels"]] == ["colchicine"]
    (drug,) = await rows(client, DocumentDrug)
    assert (drug.status, drug.rxcui) == ("resolved", "2683")
    assert (await client.post(path, headers=doctor)).status_code == 409
