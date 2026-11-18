import json

import httpx
import pytest
import pytest_asyncio
from config import Settings
from app import create_app
from fake_llm import FakeLLM
from auth import create_doctor
from druglabels import DEMO_LABELS, LabelClient, fixture_transport
from knowledge import KnowledgeStore
import interview
import reranker


@pytest.fixture(autouse=True)
def no_real_reranker(tmp_path_factory, monkeypatch):
    """open_store would load models/ when present; tests never load a real model.
    The skip-if-missing real-model tests in test_reranker.py use the path they
    captured at import."""
    monkeypatch.setattr(reranker, "MODELS_DIR", tmp_path_factory.mktemp("no-models"))


def no_interview_items(messages):
    return {"content": json.dumps({"reports_symptom": False, "items": []})}


@pytest.fixture
def fake_llm():
    """By default the clone interview finds no items, so a first turn keeps the
    4.7 intake and never consumes the scripted queue; interview tests route
    interview.BUILD_PROMPT themselves."""
    llm = FakeLLM()
    llm.route(interview.BUILD_PROMPT, no_interview_items)
    return llm


@pytest_asyncio.fixture
async def client(fake_llm, tmp_path):
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        jwt_secret="test-secret-not-for-production",
    )
    # An empty store under tmp_path, so tests never create the repo's index file.
    app = create_app(settings, llm=fake_llm, store=KnowledgeStore(tmp_path / "index.sqlite3", fake_llm),
                     label_client=LabelClient(fixture_transport(DEMO_LABELS)))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.app = app
            yield c


async def auth_headers(client, username="alice"):
    resp = await client.post(
        "/api/signup", json={"username": username, "password": "password123"}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


async def doctor_headers(client, username="dr.ting", display_name="Dr. Ting Kai"):
    async with client.app.state.session_factory() as session:
        await create_doctor(session, username, "doctor-password-1", display_name)
    resp = await client.post(
        "/api/login", json={"username": username, "password": "doctor-password-1"}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


@pytest_asyncio.fixture
async def file_client(fake_llm, tmp_path):
    """Like `client`, but on a file-backed SQLite DB so sessions get separate
    connections (needed for real concurrency tests)."""
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        jwt_secret="test-secret-not-for-production",
    )
    # An empty store under tmp_path, so tests never create the repo's index file.
    app = create_app(settings, llm=fake_llm, store=KnowledgeStore(tmp_path / "index.sqlite3", fake_llm),
                     label_client=LabelClient(fixture_transport(DEMO_LABELS)))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.app = app
            yield c


@pytest_asyncio.fixture
async def swap_labels(client):
    """swap_labels(transport) gives the app a new label client on `transport`.
    Every client made this way is closed at teardown and the original comes back."""
    original = client.app.state.label_client
    made = []

    def swap(transport):
        made.append(LabelClient(transport))
        client.app.state.label_client = made[-1]
        return made[-1]

    yield swap
    client.app.state.label_client = original
    for label_client in made:
        await label_client.aclose()
