import json
import logging
import os
from dataclasses import replace

import pytest

import knowledge
import manage_knowledge
import reranker
from app import create_app
from config import Settings, get_settings
from fake_llm import FakeLLM


@pytest.fixture
def clean_settings_cache():
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


def test_rerank_is_on_by_default_and_off_with_rerank_enabled_0(monkeypatch, clean_settings_cache):
    # On by default: the reranker beat embeddings on test-split hit@1, MRR and
    # false hits (README, Library search).
    assert Settings().rerank_enabled is True
    for value, expected in ((None, True), ("", True), ("1", True), (" 1 ", True), ("0", False)):
        get_settings.cache_clear()  # get_settings is cached; each environment needs a fresh read
        if value is None:
            monkeypatch.delenv("RERANK_ENABLED", raising=False)
        else:
            monkeypatch.setenv("RERANK_ENABLED", value)
        assert get_settings().rerank_enabled is expected, value


@pytest.mark.parametrize("value", ["true", "false", "off", "yes", "2"])
def test_rerank_enabled_refuses_anything_but_0_or_1(monkeypatch, clean_settings_cache, value):
    monkeypatch.setenv("RERANK_ENABLED", value)
    with pytest.raises(RuntimeError, match="RERANK_ENABLED must be 0 or 1"):
        get_settings()


def test_rag_min_rerank_default_is_the_calibrated_cut():
    assert Settings().rag_min_rerank == -6.5


def missing_model_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(reranker, "MODELS_DIR", tmp_path / "models")
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        jwt_secret="test-secret-not-for-production",
        rag_db_path=str(tmp_path / "index.sqlite3"),
        rerank_model="ms-marco-MiniLM-L-6-v2",
        rerank_enabled=True,
    )


async def test_app_starts_without_model_files_and_reports_off(tmp_path, monkeypatch, caplog):
    settings = missing_model_settings(tmp_path, monkeypatch)
    app = create_app(settings, llm=FakeLLM())
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        async with app.router.lifespan_context(app):
            assert app.state.reranker is None
            status = app.state.store.status()
    assert status["reranker"] == "off"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "Library reranker" in r.getMessage()]
    assert warnings and "fetch-reranker" in warnings[0].getMessage()


async def test_app_with_rerank_disabled_logs_info_not_warning(tmp_path, monkeypatch, caplog):
    settings = replace(missing_model_settings(tmp_path, monkeypatch), rerank_enabled=False)
    app = create_app(settings, llm=FakeLLM())
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        async with app.router.lifespan_context(app):
            assert app.state.store.reranker_note == "off: disabled by RERANK_ENABLED=0"
    lines = [r for r in caplog.records if "Library reranker" in r.getMessage()]
    assert [r.levelno for r in lines] == [logging.INFO]


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


async def test_open_store_passes_reranker_and_logs_on(tmp_path, monkeypatch, caplog):
    class Fake:
        def score(self, query, passages):
            return [0.0] * len(passages)

    fake = Fake()
    monkeypatch.setattr(knowledge, "load_reranker", lambda settings: (fake, "on: fake"))
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        jwt_secret="test-secret-not-for-production",
        rag_db_path=str(tmp_path / "index.sqlite3"),
        rag_min_rerank=-1.5,
    )
    store = knowledge.open_store(settings, FakeLLM())
    assert store.reranker is fake and store.min_rerank == -1.5 and store.reranker_note == "on: fake"
    app = create_app(settings, llm=FakeLLM())
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        async with app.router.lifespan_context(app):
            assert app.state.reranker is fake
    infos = [r for r in caplog.records if r.levelno == logging.INFO and r.getMessage() == "Library reranker: on: fake"]
    assert len(infos) == 1
