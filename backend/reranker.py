"""Local cross-encoder reranker run through ONNX Runtime (no PyTorch).

The server never downloads a model: `fetch_reranker` is only called by
`manage_knowledge.py fetch-reranker`, and every file is pinned to a revision and
checked against a sha256.
"""
import hashlib
import logging
from pathlib import Path

import httpx

from config import PROJECT_DIR

logger = logging.getLogger(__name__)
MODELS_DIR = PROJECT_DIR / "models"
MAX_TOKENS = 512
HF_URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
FETCH_HINT = "run python backend/manage_knowledge.py fetch-reranker"

RERANKERS = {
    "ms-marco-MiniLM-L-6-v2": {
        "repo": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        "files": {  # repo path -> (local name, sha256)
            "onnx/model.onnx": ("model.onnx", "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a"),
            "tokenizer.json": ("tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
        },
    },
}


class RerankerError(Exception):
    pass


class OnnxReranker:
    """Scores (query, passage) pairs; higher means more relevant. Synchronous."""

    def __init__(self, model_dir: Path):
        # Imported here so that the app starts, with the reranker off, without them.
        import onnxruntime
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.enable_padding()
        self._session = onnxruntime.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        import numpy as np

        encodings = self._tokenizer.encode_batch([(query, p) for p in passages])
        features = {
            "input_ids": [e.ids for e in encodings],
            "attention_mask": [e.attention_mask for e in encodings],
            "token_type_ids": [e.type_ids for e in encodings],
        }
        feeds = {
            name: np.array(rows, dtype=np.int64)
            for name, rows in features.items()
            if name in self._inputs
        }
        logits = self._session.run(None, feeds)[0]
        return [float(x) for x in logits[:, 0]]


def load_reranker(settings) -> tuple[OnnxReranker | None, str]:
    """The reranker and a note saying why it is on or off. Never raises."""
    if not settings.rerank_enabled:
        return None, "off: disabled, set RERANK_ENABLED=1 to turn it on"
    name = settings.rerank_model
    spec = RERANKERS.get(name)
    if spec is None:
        return None, f"off: unknown RERANK_MODEL {name}"
    folder = MODELS_DIR / name
    if not all((folder / local).is_file() for local, _ in spec["files"].values()):
        return None, f"off: model files missing, {FETCH_HINT}"
    try:
        return OnnxReranker(folder), f"on: {name}"
    except Exception as exc:
        logger.warning("Reranker %s failed to load", name, exc_info=True)
        return None, f"off: failed to load ({type(exc).__name__})"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


async def fetch_reranker(name: str, dest: Path | None = None, transport=None) -> Path:
    """Download the pinned files into dest/<name> (default: the current MODELS_DIR),
    each verified before it is kept."""
    dest = MODELS_DIR if dest is None else dest
    spec = RERANKERS.get(name)
    if spec is None:
        raise RerankerError(f"unknown RERANK_MODEL {name}")
    folder = dest / name
    folder.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(
        transport=transport,
        follow_redirects=True,
        timeout=60,
        headers={"User-Agent": "AfterCare-ClassProject/1.0"},
    ) as client:
        for repo_path, (local, expected) in spec["files"].items():
            target = folder / local
            if target.is_file() and _sha256(target) == expected:
                continue
            part = folder / f"{local}.part"
            url = HF_URL.format(repo=spec["repo"], revision=spec["revision"], path=repo_path)
            digest = hashlib.sha256()
            try:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with part.open("wb") as handle:
                        async for block in response.aiter_bytes():
                            digest.update(block)
                            handle.write(block)
                if digest.hexdigest() != expected:
                    raise RerankerError(f"{local}: sha256 does not match the pinned value")
                part.replace(target)
            finally:
                part.unlink(missing_ok=True)
    return folder
