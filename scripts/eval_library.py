"""Library search evaluation. Dev only, run on demand; never part of pytest.

Runs every question in eval/library_questions.json through the real index
(knowledge/index.sqlite3) and the real embedding API (from .env) in the three
search modes, and prints hit@1, hit@5, mean reciprocal rank and false hits on
the questions the library cannot answer, plus the reranker time.

    .venv/bin/python scripts/eval_library.py                 # all questions
    .venv/bin/python scripts/eval_library.py --split test
    .venv/bin/python scripts/eval_library.py --sweep         # calibrate RAG_MIN_RERANK
    .venv/bin/python scripts/eval_library.py --model NAME    # another reranker

Every RxCUI present in the index is passed to the search, so approved drug
labels are eligible. The reranker time is measured around `reranker.score`
only. The API key is never printed.
"""
import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

QUESTIONS = ROOT / "eval" / "library_questions.json"
MODES = ("embedding", "hybrid", "rerank")
SWEEP = [step / 2 for step in range(-20, 21)]  # -10.0 to 10.0 in steps of 0.5


def metrics(results: list[list[str]], expected: list[list[str]]) -> dict:
    """Ranking quality of `results` (source ids, best first) against `expected`.

    hit@k and mrr are over the answerable questions (non-empty expected); a hit
    means any expected id is in the top k, and the reciprocal rank uses the
    first expected id returned (0 when none is). A no-answer question (empty
    expected) that returns anything is a false hit.
    """
    if len(results) != len(expected):
        raise ValueError(f"{len(results)} result lists for {len(expected)} questions.")
    answerable = no_answer = false_hits = 0
    hit1 = hit5 = reciprocal = 0.0
    for returned, wanted in zip(results, expected):
        if not wanted:
            no_answer += 1
            false_hits += bool(returned)
            continue
        answerable += 1
        rank = next((i for i, source in enumerate(returned, 1) if source in wanted), None)
        if rank is not None:
            hit1 += rank == 1
            hit5 += rank <= 5
            reciprocal += 1 / rank
    share = (lambda value: value / answerable) if answerable else (lambda value: 0.0)
    return {"hit@1": share(hit1), "hit@5": share(hit5), "mrr": share(reciprocal),
            "no_answer_false_hits": false_hits, "answerable": answerable, "no_answer": no_answer}


def load_questions(split: str) -> list[dict]:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    return [q for q in questions if split == "all" or q["split"] == split]


class CachedEmbeddings:
    """The real client, asked once per distinct text: the three modes and the
    sweep embed the same questions."""

    def __init__(self, llm):
        self._llm = llm
        self._cache: dict[str, list[float]] = {}
        self.embedding_model = llm.embedding_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self._cache.update(zip(missing, await self._llm.embed(missing)))
        return [self._cache[t] for t in texts]


class TimedReranker:
    """Wraps the real reranker and records milliseconds spent in `score`."""

    def __init__(self, inner):
        self._inner = inner
        self.times_ms: list[float] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        start = time.perf_counter()
        try:
            return self._inner.score(query, passages)
        finally:
            self.times_ms.append((time.perf_counter() - start) * 1000)


def index_rxcuis(store) -> list[str]:
    with store.connect() as db:
        return [row[0] for row in db.execute(
            "SELECT DISTINCT rxcui FROM documents WHERE rxcui IS NOT NULL ORDER BY rxcui")]


async def search(store, questions: list[dict], mode: str, rxcuis: list[str]) -> list[list[dict]]:
    return [await store.retrieve(q["question"], rxcuis=rxcuis, mode=mode) for q in questions]


def _ids(hits: list[dict]) -> list[str]:
    return [hit["source_id"] for hit in hits]


def _row(label: str, result: dict, latency: str) -> str:
    return (f"{label:<10} {result['hit@1']:>6.3f} {result['hit@5']:>6.3f} {result['mrr']:>6.3f} "
            f"{result['no_answer_false_hits']:>4}/{result['no_answer']:<4} {result['answerable']:>5}  {latency}")


async def sweep(store, rxcuis: list[str]) -> None:
    questions = load_questions("tune")
    expected = [q["expected"] for q in questions]
    # One rerank pass with no cut. Results are sorted by score, so applying a
    # cut to the top 5 equals running the search with that cut.
    store.min_rerank = -math.inf
    hits = await search(store, questions, "rerank", rxcuis)
    uncut = metrics([_ids(h) for h in hits], expected)
    print(f"Tune split: {len(questions)} questions, {uncut['answerable']} answerable, "
          f"{uncut['no_answer']} no-answer")
    print(f"{'cut':>6} {'hit@5':>6} {'mrr':>6} false_hits")
    print(f"{'none':>6} {uncut['hit@5']:>6.3f} {uncut['mrr']:>6.3f} "
          f"{uncut['no_answer_false_hits']:>4}/{uncut['no_answer']}")
    chosen = None
    for cut in SWEEP:
        result = metrics([_ids([x for x in h if x["score"] >= cut]) for h in hits], expected)
        print(f"{cut:>6.1f} {result['hit@5']:>6.3f} {result['mrr']:>6.3f} "
              f"{result['no_answer_false_hits']:>4}/{result['no_answer']}")
        if result["hit@5"] == uncut["hit@5"]:
            chosen = cut
    print(f"Highest cut that keeps the uncut hit@5: {chosen}")


async def measure(store, timed: TimedReranker, rxcuis: list[str], split: str) -> None:
    questions = load_questions(split)
    expected = [q["expected"] for q in questions]
    print(f"Split {split}: {len(questions)} questions, RAG_MIN_RERANK={store.min_rerank}, "
          f"RxCUIs passed: {len(rxcuis)}")
    print(f"{'mode':<10} {'hit@1':>6} {'hit@5':>6} {'mrr':>6} {'false':>9} {'n_ans':>5}  rerank ms (median / worst)")
    for mode in MODES:
        timed.times_ms.clear()
        hits = await search(store, questions, mode, rxcuis)
        latency = "-"
        if mode == "rerank" and timed.times_ms:
            latency = f"{statistics.median(timed.times_ms):.0f} / {max(timed.times_ms):.0f}"
        print(_row(mode, metrics([_ids(h) for h in hits], expected), latency))


async def run(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=("tune", "test", "all"), default="all")
    parser.add_argument("--sweep", action="store_true",
                        help="print tune-split hit@5 and false hits for RAG_MIN_RERANK -10 to 10")
    parser.add_argument("--model", default=None, help="reranker name from reranker.RERANKERS")
    args = parser.parse_args(argv)

    from config import get_settings
    from knowledge import open_store
    from llm import LLMClient

    settings = get_settings()
    settings = replace(settings, rerank_enabled=True, rerank_model=args.model or settings.rerank_model)
    store = open_store(settings, CachedEmbeddings(LLMClient(settings)))
    if store.reranker is None:
        print(f"Reranker {settings.rerank_model} {store.reranker_note}", file=sys.stderr)
        return 1
    print(f"Reranker: {settings.rerank_model}")
    timed = TimedReranker(store.reranker)
    store.reranker = timed
    rxcuis = index_rxcuis(store)
    if args.sweep:
        await sweep(store, rxcuis)
    else:
        await measure(store, timed, rxcuis, args.split)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(argv))


if __name__ == "__main__":
    sys.exit(main())
