import json
import logging
import os

import pytest

import manage_knowledge
import reranker
from app import create_app
from config import Settings, get_settings
from fake_llm import FakeLLM


@pytest.fixture
def clean_settings_cache(monkeypatch):
    """get_settings is cached and loads the real .env: start and end empty, and
    undo what load_dotenv writes into os.environ."""
    saved = dict(os.environ)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    os.environ.clear()
    os.environ.update(saved)


def test_settings_parse_rerank_env(monkeypatch, clean_settings_cache):
    monkeypatch.setenv("RERANK_ENABLED", "0")
    monkeypatch.setenv("RAG_MIN_RERANK", "-2.5")
    monkeypatch.setenv("RERANK_MODEL", "x")
    settings = get_settings()
    assert settings.rerank_enabled is False
    assert settings.rag_min_rerank == -2.5
    assert settings.rerank_model == "x"


def missing_model_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(reranker, "MODELS_DIR", tmp_path / "models")
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        jwt_secret="test-secret-not-for-production",
        rag_db_path=str(tmp_path / "index.sqlite3"),
        rerank_model="ms-marco-MiniLM-L-6-v2",
    )


async def test_app_starts_without_model_files_and_reports_off(tmp_path, monkeypatch, caplog):
    settings = missing_model_settings(tmp_path, monkeypatch)
    app = create_app(settings, llm=FakeLLM())
    with caplog.at_level(logging.INFO, logger="app"):
        async with app.router.lifespan_context(app):
            assert app.state.reranker is None
            status = app.state.store.status()
    assert status["reranker"] == "off"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "Library reranker" in r.getMessage()]
    assert warnings and "fetch-reranker" in warnings[0].getMessage()


async def test_passed_store_keeps_its_own_reranker(tmp_path):
    from knowledge import KnowledgeStore

    class Fixed:
        def score(self, query, passages):
            return [0.0] * len(passages)

    fixed = Fixed()
    store = KnowledgeStore(tmp_path / "i.sqlite3", FakeLLM(), reranker=fixed)
    settings = Settings(database_url="sqlite+aiosqlite:///:memory:", jwt_secret="test-secret-not-for-production")
    app = create_app(settings, llm=FakeLLM(), store=store)
    assert app.state.reranker is fixed


async def test_app_without_store_has_no_reranker(tmp_path):
    settings = Settings(database_url="sqlite+aiosqlite:///:memory:", jwt_secret="test-secret-not-for-production")
    app = create_app(settings, llm=None, store=None)
    assert app.state.store is None and app.state.reranker is None


def test_cli_status_includes_reranker_field(tmp_path, monkeypatch, capsys):
    settings = missing_model_settings(tmp_path, monkeypatch)
    assert manage_knowledge.run(["status"], settings) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["reranker"] == "off"
    assert "fetch-reranker" in status["reranker_note"]
