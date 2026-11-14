"""Education library curator commands.

  status   show what is indexed and whether it matches EMBEDDING_MODEL
  collect  fetch every manifest URL into knowledge/sources/ as an unreviewed snapshot
  index    embed every reviewed snapshot and curated summary into the index
"""
import argparse
import asyncio
import json
import sys
from urllib.parse import urlparse

from sqlalchemy import make_url
from sqlalchemy.exc import SQLAlchemyError

from config import Settings, get_settings
from druglabels import approved_label_records
from knowledge import DATA, KnowledgeError, KnowledgeStore, collect_sources, index_sources
from llm import LLMClient, LLMError


KEYED_HOSTS = frozenset({"api.openai.com"})


def needs_key(settings: Settings) -> bool:
    """Hosted APIs need a key; a self-hosted embeddings server (vLLM) may not."""
    return (urlparse(settings.embedding_api_base).hostname or "") in KEYED_HOSTS


def run(argv: list[str], settings: Settings | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "collect", "index"])
    args = parser.parse_args(argv)
    settings = settings or get_settings()
    llm = LLMClient(settings)
    try:
        if args.command == "collect":
            report = asyncio.run(collect_sources(DATA))
            print(json.dumps(report, indent=2))
            failed = [r["id"] for r in report if not r["ok"]]
            if failed:
                print("Collection failed for: " + ", ".join(failed), file=sys.stderr)
                return 1
            print("Review each snapshot in knowledge/sources/, then set \"reviewed\": true.", file=sys.stderr)
            return 0
        if args.command == "index" and needs_key(settings) and not settings.embedding_api_key:
            print("Set EMBEDDING_API_KEY (or LLM_API_KEY) in the project .env before indexing.", file=sys.stderr)
            return 2
        store = KnowledgeStore(settings.rag_db_path, llm, settings.rag_min_similarity)
        if args.command == "index":
            try:
                labels = asyncio.run(approved_label_records(settings.database_url))
            except (SQLAlchemyError, OSError) as exc:
                # No fallback to an empty list: the index is a full rebuild, so
                # that would silently drop the labels already indexed.
                shown = make_url(settings.database_url).render_as_string(hide_password=True)
                print(f"Cannot read approved drug labels from {shown}: {type(exc).__name__}. "
                      "Check DATABASE_URL and that the database is running and initialised.",
                      file=sys.stderr)
                return 1
            result = asyncio.run(index_sources(store, DATA, extra_records=labels))
            for source_id, chunks in result["indexed"].items():
                print(f"{source_id}: {chunks} chunks")
            if result["skipped"]:
                print("Skipped (not reviewed): " + ", ".join(result["skipped"]), file=sys.stderr)
        print(json.dumps(store.status(), indent=2))
        return 0
    except (KnowledgeError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
