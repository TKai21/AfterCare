import hashlib
import math

import httpx
import pytest

import manage_knowledge
import reranker
from config import Settings
from reranker import MODELS_DIR, RerankerError, fetch_reranker, load_reranker

REAL_NAME = "ms-marco-MiniLM-L-6-v2"
HAVE_MODEL = (MODELS_DIR / REAL_NAME / "model.onnx").exists()
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="run fetch-reranker to download the model")

MODEL_BYTES = b"fake-onnx-bytes"
TOKENIZER_BYTES = b'{"fake": "tokenizer"}'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fake_entry(model_bytes=MODEL_BYTES):
    return {
        "repo": "acme/tiny",
        "revision": "abc123",
        "files": {
            "onnx/model.onnx": ("model.onnx", sha(model_bytes)),
            "tokenizer.json": ("tokenizer.json", sha(TOKENIZER_BYTES)),
        },
    }


def serve(model_bytes, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path.endswith("model.onnx"):
            return httpx.Response(200, content=model_bytes)
        return httpx.Response(200, content=TOKENIZER_BYTES)

    return httpx.MockTransport(handler)


async def test_fetch_writes_verified_files(tmp_path, monkeypatch):
    monkeypatch.setitem(reranker.RERANKERS, "fake", fake_entry())
    seen = []
    folder = await fetch_reranker("fake", dest=tmp_path, transport=serve(MODEL_BYTES, seen))
    assert folder == tmp_path / "fake"
    assert (folder / "model.onnx").read_bytes() == MODEL_BYTES
    assert (folder / "tokenizer.json").read_bytes() == TOKENIZER_BYTES
    assert not list(folder.glob("*.part"))
    assert sorted(seen) == [
        "https://huggingface.co/acme/tiny/resolve/abc123/onnx/model.onnx",
        "https://huggingface.co/acme/tiny/resolve/abc123/tokenizer.json",
    ]
    # A second run finds both files verified and downloads nothing.
    seen.clear()
    await fetch_reranker("fake", dest=tmp_path, transport=serve(MODEL_BYTES, seen))
    assert seen == []


async def test_fetch_sha_mismatch_raises_and_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setitem(reranker.RERANKERS, "fake", fake_entry())
    with pytest.raises(RerankerError, match="model.onnx"):
        await fetch_reranker("fake", dest=tmp_path, transport=serve(b"wrong", []))
    folder = tmp_path / "fake"
    assert not (folder / "model.onnx").exists()
    assert not list(folder.glob("*.part"))


async def test_fetch_later_mismatch_keeps_verified_earlier_file(tmp_path, monkeypatch):
    monkeypatch.setitem(reranker.RERANKERS, "fake", fake_entry())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("model.onnx"):
            return httpx.Response(200, content=MODEL_BYTES)
        return httpx.Response(200, content=b"bad tokenizer")

    with pytest.raises(RerankerError, match="tokenizer.json"):
        await fetch_reranker("fake", dest=tmp_path, transport=httpx.MockTransport(handler))
    folder = tmp_path / "fake"
    assert (folder / "model.onnx").read_bytes() == MODEL_BYTES
    assert not (folder / "tokenizer.json").exists()
    assert not list(folder.glob("*.part"))


async def test_fetch_unknown_model(tmp_path):
    with pytest.raises(RerankerError, match="nope"):
        await fetch_reranker("nope", dest=tmp_path)


def test_load_reranker_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(reranker, "MODELS_DIR", tmp_path)
    assert load_reranker(Settings(rerank_enabled=False)) == (None, "off: RERANK_ENABLED=0")
    got, note = load_reranker(Settings(rerank_model="nope"))
    assert got is None and note == "off: unknown RERANK_MODEL nope"
    got, note = load_reranker(Settings())
    assert got is None
    assert note == "off: model files missing, run python backend/manage_knowledge.py fetch-reranker"
    folder = tmp_path / REAL_NAME
    folder.mkdir()
    (folder / "model.onnx").write_bytes(b"garbage")
    (folder / "tokenizer.json").write_bytes(b"garbage")
    got, note = load_reranker(Settings())
    assert got is None
    assert note.startswith("off: failed to load (") and note.endswith(")")


def test_cli_fetch_reranker_unknown_model_exits_2():
    assert manage_knowledge.run(["fetch-reranker"], settings=Settings(rerank_model="nope")) == 2


def test_cli_fetch_reranker_prints_folder(tmp_path, monkeypatch, capsys):
    async def fake_fetch(name, dest=None, transport=None):
        return tmp_path / name

    monkeypatch.setattr(reranker, "fetch_reranker", fake_fetch)
    assert manage_knowledge.run(["fetch-reranker"], settings=Settings()) == 0
    assert str(tmp_path / REAL_NAME) in capsys.readouterr().out


def test_cli_fetch_reranker_error_exits_1(monkeypatch, capsys):
    async def boom(name, dest=None, transport=None):
        raise RerankerError("bad hash")

    monkeypatch.setattr(reranker, "fetch_reranker", boom)
    assert manage_knowledge.run(["fetch-reranker"], settings=Settings()) == 1
    assert "Error: bad hash" in capsys.readouterr().err


def test_cli_fetch_reranker_os_and_stream_errors_exit_1(monkeypatch, capsys):
    for exc in (OSError("disk full"), httpx.StreamClosed()):
        async def boom(name, dest=None, transport=None, exc=exc):
            raise exc

        monkeypatch.setattr(reranker, "fetch_reranker", boom)
        assert manage_knowledge.run(["fetch-reranker"], settings=Settings()) == 1
        assert "Error:" in capsys.readouterr().err


def test_load_reranker_logs_failure(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(reranker, "MODELS_DIR", tmp_path)
    folder = tmp_path / REAL_NAME
    folder.mkdir()
    (folder / "model.onnx").write_bytes(b"garbage")
    (folder / "tokenizer.json").write_bytes(b"garbage")
    with caplog.at_level("WARNING", logger="reranker"):
        load_reranker(Settings())
    assert any(r.levelname == "WARNING" and r.exc_info for r in caplog.records)


@pytest.fixture(scope="module")
def real():
    return reranker.OnnxReranker(MODELS_DIR / REAL_NAME)


@needs_model
def test_real_reranker_orders_relevant_first(real):
    query = "What should I do if I miss a dose of colchicine?"
    good, bad = real.score(query, [
        "If you miss a dose of colchicine, take it as soon as you remember.",
        "Drink plenty of water to prevent kidney stones.",
    ])
    assert good > bad


@needs_model
def test_real_reranker_truncates_long_query(real):
    scores = real.score("colchicine dose " * 400, ["a", "b b b", "colchicine " * 600])
    assert len(scores) == 3 and all(math.isfinite(s) for s in scores)


@needs_model
def test_real_reranker_empty_list(real):
    assert real.score("anything", []) == []
