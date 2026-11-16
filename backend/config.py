import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
PLACEHOLDER_SECRET = "please_change_me_to_a_long_random_value"


@dataclass(frozen=True)
class Settings:
    database_url: str = "mysql+asyncmy://aftercare:aftercare@localhost:3306/aftercare"
    jwt_secret: str = ""
    jwt_expire_minutes: int = 43200
    llm_api_base: str = "https://api.openai.com/v1"
    llm_model: str = ""
    llm_api_key: str = ""
    embedding_api_base: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    rag_db_path: str = "knowledge/index.sqlite3"
    rag_min_similarity: float = 0.25
    guidance_min_similarity: float = 0.45
    foods_path: str = "data/foods.json"
    # Dev only: answer RxNav and DailyMed from the recorded demo/labels fixtures.
    label_fixtures: bool = False
    rerank_model: str = "ms-marco-MiniLM-L-6-v2"
    # On by default: on the evaluation set's test split the reranker beat
    # embedding-only search on hit@1, MRR and no-answer false hits, at the cost
    # of one hit@5 question (README, "Library search"). RERANK_ENABLED=0 turns
    # it off; without the model files search is embedding-only either way.
    rerank_enabled: bool = True
    # Calibrated by scripts/eval_library.py --sweep on the tune split.
    rag_min_rerank: float = -6.5

    def __post_init__(self):
        # Embedding settings fall back to the chat settings when unset.
        if not self.embedding_api_base:
            object.__setattr__(self, "embedding_api_base", self.llm_api_base)
        if not self.embedding_api_key:
            object.__setattr__(self, "embedding_api_key", self.llm_api_key)

    def validate(self) -> None:
        if not self.jwt_secret or self.jwt_secret == PLACEHOLDER_SECRET:
            raise RuntimeError("Set a unique JWT_SECRET in the project .env file.")


def _flag(env, name: str, default: bool) -> bool:
    """"1" is on, "0" is off, unset or empty is the default; anything else is
    refused, so a typo such as "false" cannot silently mean the opposite."""
    value = env.get(name, "").strip()
    if not value:
        return default
    if value in ("0", "1"):
        return value == "1"
    raise RuntimeError(f"{name} must be 0 or 1, not {value!r}.")


@lru_cache
def get_settings() -> Settings:
    load_dotenv(PROJECT_DIR / ".env")
    env = os.environ
    defaults = Settings()
    llm_base = env.get("LLM_API_BASE", defaults.llm_api_base).rstrip("/")
    return Settings(
        database_url=env.get("DATABASE_URL", defaults.database_url),
        jwt_secret=env.get("JWT_SECRET", ""),
        jwt_expire_minutes=int(env.get("JWT_EXPIRE_MINUTES", defaults.jwt_expire_minutes)),
        llm_api_base=llm_base,
        llm_model=env.get("LLM_MODEL", ""),
        llm_api_key=env.get("LLM_API_KEY", ""),
        embedding_api_base=env.get("EMBEDDING_API_BASE", llm_base).rstrip("/"),
        embedding_model=env.get("EMBEDDING_MODEL", defaults.embedding_model),
        embedding_api_key=env.get("EMBEDDING_API_KEY", env.get("LLM_API_KEY", "")),
        rag_db_path=env.get("RAG_DB_PATH", defaults.rag_db_path),
        rag_min_similarity=float(env.get("RAG_MIN_SIMILARITY", defaults.rag_min_similarity)),
        guidance_min_similarity=float(
            env.get("GUIDANCE_MIN_SIMILARITY", defaults.guidance_min_similarity)
        ),
        foods_path=env.get("FOODS_PATH", defaults.foods_path),
        label_fixtures=env.get("LABEL_FIXTURES", "") == "1",
        rerank_model=env.get("RERANK_MODEL", defaults.rerank_model),
        rerank_enabled=_flag(env, "RERANK_ENABLED", defaults.rerank_enabled),
        rag_min_rerank=float(env.get("RAG_MIN_RERANK", defaults.rag_min_rerank)),
    )
