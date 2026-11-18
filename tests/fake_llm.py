import hashlib
import json
import math
import re

import agent
import drafts
import rules


def no_paragraphs(messages):
    return {"role": "assistant", "content": json.dumps({"paragraphs": []})}


class FakeLLM:
    """Scripted stand-in for LLMClient. Same chat/embed interface."""

    DIM = 64

    def __init__(self, embedding_model: str = "fake-bag-of-words-64"):
        self.queue: list[dict] = []
        self.calls: list[tuple] = []
        # System prompt -> responder(messages) -> message. Routed calls skip the
        # scripted queue and are recorded in routed_calls, not calls.
        # By default the rule reader gets one rule per sentence and the conflict
        # check finds none, so indexing never consumes the scripted queue. The
        # second look at the document finds no paragraph.
        self.routes: dict = {agent.SECOND_LOOK_PROMPT: no_paragraphs,
                             rules.RULES_PROMPT: sentence_rules,
                             rules.CONFLICT_PROMPT: no_conflicts,
                             drafts.SUPPORT_PROMPT: all_supported}
        self.routed_calls: list[tuple] = []
        self._embedding_model = embedding_model
        # Scripted transcripts for transcribe(): a string, or an exception to raise.
        self.transcripts: list = []
        self.transcribe_calls: list[tuple] = []

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    def script(self, *messages: dict) -> None:
        self.queue.extend(messages)

    def route(self, system_prompt: str, responder) -> None:
        self.routes[system_prompt] = responder

    async def chat(self, messages, tools=None, json_mode=False):
        system = messages[0].get("content") if messages else None
        if system in self.routes:
            self.routed_calls.append((messages, tools, json_mode))
            return self.routes[system](messages)
        self.calls.append((messages, tools, json_mode))
        if not self.queue:
            raise AssertionError(
                f"FakeLLM.chat called with no scripted response left (call #{len(self.calls)})"
            )
        return self.queue.pop(0)

    async def transcribe(self, data: bytes, filename: str, content_type: str) -> str:
        self.transcribe_calls.append((data, filename, content_type))
        result = self.transcripts.pop(0) if self.transcripts else ""
        if isinstance(result, Exception):
            raise result
        return result

    async def embed(self, texts):
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
            values[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:
            values[0], norm = 1.0, 1.0
        return [v / norm for v in values]


def sentence_rules(messages):
    """Default rule reader: one rule per sentence of the answer, the sentence
    as both statement and quote."""
    answer = messages[1]["content"].split("Doctor's answer:\n", 1)[1]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    return {"content": json.dumps({"rules": [{"kind": "advice", "statement": s, "quote": s} for s in sentences]})}


def no_conflicts(messages):
    return {"content": json.dumps({"conflicts": []})}


def all_supported(messages):
    return {"content": json.dumps({"unsupported": []})}
