import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import select, update

from conftest import auth_headers, doctor_headers
from druglabels import DEMO_LABELS, LabelClient, failing_transport, fixture_transport, patient_rxcuis
from llm import LLMError
from models import DischargeDocument, DocumentDrug, DrugLabel
from test_routes_agent import answer, post_ask, tool, tool_results
from test_druglabels import GOUT, confirm_meds, json_transport, med, rows

COLCHICINE_SETID = "5c1ceff7-8f00-48cc-8ea4-678106455ff1"
HIKMA_SETID = "ee1583ac-c308-4beb-b602-9ecac4977026"


async def setup(client, fake_llm, *meds):
    doc_id, patient = await confirm_meds(client, fake_llm, "gout-patient", *(meds or GOUT))
    return doc_id, patient, await doctor_headers(client)


async def label_by_ingredient(client, name):
    return next(l for l in await rows(client, DrugLabel) if l.ingredient == name)


async def review(client, label, action, headers, setid=None):
    """Approve or reject `label`, naming the label text the doctor saw (its setid)."""
    return await client.post(f"/api/doctor/labels/{label.id}/{action}",
                             json={"setid": setid or label.setid}, headers=headers)


async def test_patient_gets_403(client, fake_llm):
    _, patient, _ = await setup(client, fake_llm)
    for method, path, body in [
        ("get", "/api/doctor/labels", None),
        ("post", "/api/doctor/labels/1/approve", {"setid": COLCHICINE_SETID}),
        ("post", "/api/doctor/labels/1/reject", {"setid": COLCHICINE_SETID}),
        ("post", "/api/doctor/labels/1/candidate", {"setid": COLCHICINE_SETID}),
        ("post", "/api/doctor/labels/1/retry", None),
        ("post", "/api/doctor/document-drugs/1/retry", None),
    ]:
        resp = await getattr(client, method)(path, headers=patient, **({"json": body} if body else {}))
        assert resp.status_code == 403, path


async def test_unknown_id_404(client, fake_llm):
    doctor = await doctor_headers(client)
    for path, body in [
        ("/api/doctor/labels/999/approve", {"setid": "x"}),
        ("/api/doctor/labels/999/reject", {"setid": "x"}),
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
        "id", "rxcui", "ingredient", "status", "setid", "title", "url", "labeler", "published", "text", "error",
        "candidates", "sources", "reviewed_by_name", "reviewed_at"}
    assert colchicine["status"] == "pending" and colchicine["rxcui"] == "2683"
    assert colchicine["setid"] == COLCHICINE_SETID
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


async def library_passages(client, fake_llm, headers):
    """Passages the library tool returns when this patient asks about colchicine."""
    fake_llm.script(
        tool("search_education_library", {"query": "colchicine dosage gout flares"}),
        answer("The approved sources do not say.", [], kind="not_covered"),
    )
    before = len(fake_llm.calls)
    resp = await post_ask(client, headers, "How much colchicine should I take?")
    assert resp.status_code == 200, resp.text
    (result,) = tool_results(fake_llm, before + 1)
    return result["passages"]


async def test_approve_adds_to_store_and_is_retrievable_for_patient_only(client, fake_llm):
    doc_id, gout_patient, doctor = await setup(client, fake_llm)
    stone_doc, stone_patient = await confirm_meds(client, fake_llm, "stone-patient", med("Tamsulosin", "0.4 mg"))
    colchicine = await label_by_ingredient(client, "colchicine")
    resp = await review(client, colchicine, "approve", doctor)
    assert resp.status_code == 200, resp.text
    view = resp.json()
    assert view["status"] == "approved" and view["reviewed_by_name"] == "Dr. Ting Kai" and view["reviewed_at"]
    async with client.app.state.session_factory() as session:
        assert await patient_rxcuis(session, doc_id) == {"2683", "7258"}
        assert await patient_rxcuis(session, stone_doc) == {"77492"}
    gout = await library_passages(client, fake_llm, gout_patient)
    assert gout and all("colchicine" in p["title"].lower() and "setid=" in p["url"] for p in gout)
    assert await library_passages(client, fake_llm, stone_patient) == []


async def test_approve_embedding_failure_stays_pending(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    colchicine = await label_by_ingredient(client, "colchicine")

    async def broken(texts):
        raise LLMError("embedding down")
    fake_llm.embed = broken
    resp = await review(client, colchicine, "approve", doctor)
    assert resp.status_code == 502
    assert resp.json()["detail"] == "Could not add the label to the library. Try again."
    label = await label_by_ingredient(client, "colchicine")
    assert (label.status, label.reviewed_by, label.reviewed_at) == ("pending", None, None)
    assert client.app.state.store.status()["documents"] == []


async def test_approve_without_store_503(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    client.app.state.store = None
    colchicine = await label_by_ingredient(client, "colchicine")
    resp = await review(client, colchicine, "approve", doctor)
    assert (resp.status_code, resp.json()["detail"]) == (503, "The education library is not available right now.")
    assert (await label_by_ingredient(client, "colchicine")).status == "pending"


async def test_approve_and_reject_need_pending(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    colchicine = await label_by_ingredient(client, "colchicine")
    assert (await review(client, colchicine, "reject", doctor)).json()["status"] == "rejected"
    assert (await review(client, colchicine, "reject", doctor)).status_code == 409
    assert (await review(client, colchicine, "approve", doctor)).status_code == 409
    assert client.app.state.store.status()["documents"] == []


async def test_concurrent_approve_and_reject_one_wins_409(file_client, fake_llm):
    _, _, doctor = await setup(file_client, fake_llm)
    colchicine = await label_by_ingredient(file_client, "colchicine")
    approve, reject = await asyncio.gather(
        review(file_client, colchicine, "approve", doctor), review(file_client, colchicine, "reject", doctor))
    assert sorted([approve.status_code, reject.status_code]) == [200, 409]
    final = (await label_by_ingredient(file_client, "colchicine")).status
    in_store = [d["id"] for d in file_client.app.state.store.status()["documents"]]
    assert (final, in_store) == (("approved", ["dailymed-rxcui-2683"]) if approve.status_code == 200
                                 else ("rejected", []))


async def test_reject_during_slow_add_cannot_win(file_client, fake_llm):
    _, _, doctor = await setup(file_client, fake_llm)
    colchicine = await label_by_ingredient(file_client, "colchicine")
    entered, release = asyncio.Event(), asyncio.Event()
    real_embed = fake_llm.embed

    async def slow(texts):
        entered.set()
        await release.wait()
        return await real_embed(texts)
    fake_llm.embed = slow
    approving = asyncio.create_task(review(file_client, colchicine, "approve", doctor))
    await entered.wait()
    assert (await review(file_client, colchicine, "reject", doctor)).status_code == 409
    release.set()
    assert (await approving).status_code == 200
    assert (await label_by_ingredient(file_client, "colchicine")).status == "approved"
    assert [d["id"] for d in file_client.app.state.store.status()["documents"]] == ["dailymed-rxcui-2683"]


async def test_pick_candidate_on_rejected_refetches(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    colchicine = await label_by_ingredient(client, "colchicine")
    path = f"/api/doctor/labels/{colchicine.id}"
    other = next(c for c in colchicine.candidates if c["setid"] == HIKMA_SETID)
    assert other["setid"] != colchicine.setid
    await review(client, colchicine, "reject", doctor)
    resp = await client.post(f"{path}/candidate", json={"setid": HIKMA_SETID}, headers=doctor)
    assert resp.status_code == 200, resp.text
    view = resp.json()
    assert view["status"] == "pending"
    assert view["title"] == other["title"] != colchicine.title
    assert view["url"].endswith(f"setid={HIKMA_SETID}") and view["url"] != colchicine.url
    assert view["labeler"] == other["labeler"] and view["text"] != colchicine.text
    assert view["reviewed_by_name"] is None and view["reviewed_at"] is None
    assert (await label_by_ingredient(client, "colchicine")).setid == HIKMA_SETID


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
    assert (await review(client, colchicine, "approve", doctor)).status_code == 200
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


async def test_retry_document_drug_still_failing_shows_the_new_error(client, fake_llm):
    client.app.state.label_client = LabelClient(failing_transport())
    _, _, doctor = await setup(client, fake_llm, med("Colchicine", "0.6 mg"))
    (drug,) = await rows(client, DocumentDrug)
    first_error = drug.error
    client.app.state.label_client = LabelClient(httpx.MockTransport(lambda request: httpx.Response(500)))
    resp = await client.post(f"/api/doctor/document-drugs/{drug.id}/retry", headers=doctor)
    assert resp.status_code == 200, resp.text
    (entry,) = resp.json()["unresolved"]
    assert entry["status"] == "failed" and entry["error"] and entry["error"] != first_error
    (drug,) = await rows(client, DocumentDrug)
    assert entry["error"] == drug.error


async def test_retry_document_drug_that_turns_unresolved_shows_it(client, fake_llm):
    client.app.state.label_client = LabelClient(failing_transport())
    _, _, doctor = await setup(client, fake_llm, med("Multivitamin"))
    (drug,) = await rows(client, DocumentDrug)
    assert drug.status == "failed"
    nothing = {"approximateTerm.json": {"approximateGroup": {"inputTerm": None}}}
    client.app.state.label_client = LabelClient(json_transport(nothing))
    resp = await client.post(f"/api/doctor/document-drugs/{drug.id}/retry", headers=doctor)
    assert resp.status_code == 200, resp.text
    (entry,) = resp.json()["unresolved"]
    assert (entry["status"], entry["error"]) == ("unresolved", None)


@pytest.mark.parametrize("action", ["approve", "reject"])
async def test_review_of_label_text_since_replaced_409(client, fake_llm, action):
    """A doctor reviews the text they saw: after another doctor picks another
    candidate, a stale approve or reject (old setid) changes nothing."""
    _, _, doctor = await setup(client, fake_llm)
    seen = await label_by_ingredient(client, "colchicine")
    assert seen.setid == COLCHICINE_SETID
    other_doctor = await doctor_headers(client, "dr.renny", "Dr. Aaron Renny")
    picked = await client.post(f"/api/doctor/labels/{seen.id}/candidate",
                               json={"setid": HIKMA_SETID}, headers=other_doctor)
    assert picked.status_code == 200 and picked.json()["status"] == "pending"
    resp = await review(client, seen, action, doctor, setid=COLCHICINE_SETID)
    assert (resp.status_code, resp.json()["detail"]) == (409, "This label changed. Review it again.")
    now = await label_by_ingredient(client, "colchicine")
    assert (now.status, now.setid, now.reviewed_by) == ("pending", HIKMA_SETID, None)
    assert client.app.state.store.status()["documents"] == []
    assert (await review(client, now, action, doctor)).status_code == 200


async def test_review_needs_the_setid(client, fake_llm):
    _, _, doctor = await setup(client, fake_llm)
    colchicine = await label_by_ingredient(client, "colchicine")
    for action in ("approve", "reject"):
        resp = await client.post(f"/api/doctor/labels/{colchicine.id}/{action}", headers=doctor)
        assert resp.status_code == 422
    assert (await label_by_ingredient(client, "colchicine")).status == "pending"
