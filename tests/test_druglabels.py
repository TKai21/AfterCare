from pathlib import Path

import httpx
import pytest

from druglabels import LabelClient, LabelServiceError, failing_transport, fixture_transport

FIXTURES = Path(__file__).resolve().parent.parent / "demo" / "labels"


@pytest.fixture
async def client():
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


async def test_resolve_colchicine_with_dose(client):
    result = await client.resolve("Colchicine", "0.6 mg")
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


async def test_candidates_from_fixture_skip_combinations(client):
    result = await client.candidates("2683", "colchicine")
    assert 0 < len(result) <= 5
    assert all("PROBENECID" not in c["title"] for c in result)
    assert result[0]["published"] == "2026-11-09"


async def test_label_text_has_dosage_and_warnings(client):
    text = await client.label_text("5c1ceff7-8f00-48cc-8ea4-678106455ff1")
    assert text.strip()
    assert "dosage" in text.lower() and "warnings" in text.lower()


async def test_label_text_unknown_setid_raises(client):
    with pytest.raises(LabelServiceError):
        await client.label_text("00000000-0000-0000-0000-000000000000")


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
