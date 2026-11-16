import math

import httpx

from config import Settings

EMBED_BATCH_SIZE = 64

# Token usage per model since the process started (or reset_usage()), read by the
# evaluation scripts to report spend. Recording changes no behaviour.
_USAGE: dict[str, dict[str, int]] = {}


def _record_usage(model: str, data: dict) -> None:
    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    entry = _USAGE.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    entry["calls"] += 1
    for key in ("prompt_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            entry[key] += value


def usage_snapshot() -> dict[str, dict[str, int]]:
    return {model: dict(entry) for model, entry in _USAGE.items()}


def reset_usage() -> None:
    _USAGE.clear()


class LLMError(Exception):
    """Raised when the model service fails or returns an unexpected shape."""


class LLMClient:
    """OpenAI-compatible /chat/completions and /embeddings client."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    @property
    def embedding_model(self) -> str:
        """Name recorded with every stored vector so a model change forces a re-index."""
        return self.settings.embedding_model

    async def _post(self, url: str, api_key: str, payload: dict) -> dict:
        # A keyless self-hosted server (vLLM) gets no Authorization header at all.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            async with httpx.AsyncClient(timeout=90, transport=self.transport) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"Model service request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMError("Model service returned an unexpected response.")
        return data

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False) -> dict:
        payload: dict = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            payload["tools"] = tools
        # With tools, JSON output only where the server supports both together.
        if json_mode and (not tools or self.settings.llm_json_with_tools):
            payload["response_format"] = {"type": "json_object"}
        data = await self._post(
            self.settings.llm_api_base.rstrip("/") + "/chat/completions",
            self.settings.llm_api_key,
            payload,
        )
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Model service returned an unexpected chat response.") from exc
        if not isinstance(message, dict):
            raise LLMError("Model service returned an unexpected chat response.")
        _record_usage(self.settings.llm_model, data)
        return message

    async def embed(self, texts: list[str]) -> list[list[float]]:
        url = self.settings.embedding_api_base.rstrip("/") + "/embeddings"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            data = await self._post(
                url,
                self.settings.embedding_api_key,
                {"model": self.settings.embedding_model, "input": batch, "encoding_format": "float"},
            )
            try:
                items = sorted(data["data"], key=lambda item: item["index"])
                embeddings = [item["embedding"] for item in items]
            except (KeyError, TypeError) as exc:
                raise LLMError("Embedding service returned an unexpected response.") from exc
            if len(embeddings) != len(batch):
                raise LLMError("Embedding service returned an incomplete batch.")
            _record_usage(self.settings.embedding_model, data)
            vectors.extend(embeddings)
        self._validate(vectors)
        return vectors

    @staticmethod
    def _validate(vectors: list) -> None:
        width = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise LLMError("Embedding service returned invalid vectors.")
            if width is None:
                width = len(vector)
            if len(vector) != width:
                raise LLMError("Embedding service returned invalid vectors.")
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                raise LLMError("Embedding service returned invalid vectors.")
            if math.sqrt(sum(x * x for x in vector)) == 0:
                raise LLMError("Embedding service returned invalid vectors.")
