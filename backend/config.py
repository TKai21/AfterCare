import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

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
    # Off by default: on the evaluation set's test split the reranker lost one
    # hit@5 to embedding-only search (README, "Library search"). RERANK_ENABLED=1
    # turns it on. Without it, library search is embedding-only.
    rerank_enabled: bool = False
    # Calibrated by scripts/eval_library.py --sweep on the tune split.
    rag_min_rerank: float = -6.5
    # The clone's confidence gate. CLONE_MIN_CONFIDENCE wins; else the threshold
    # in the calibration file, when it was calibrated for LLM_MODEL; else no gate.
    clone_min_confidence: Optional[float] = None
    clone_calibration_path: str = "eval/clone_calibration.json"

    def __post_init__(self):
        # Embedding settings fall back to the chat settings when unset.
        if not self.embedding_api_base:
            object.__setattr__(self, "embedding_api_base", self.llm_api_base)
        if not self.embedding_api_key:
            object.__setattr__(self, "embedding_api_key", self.llm_api_key)

    def validate(self) -> None:
        if not self.jwt_secret or self.jwt_secret == PLACEHOLDER_SECRET:
            raise RuntimeError("Set a unique JWT_SECRET in the project .env file.")


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
        rerank_enabled=env.get("RERANK_ENABLED", "") == "1",
        rag_min_rerank=float(env.get("RAG_MIN_RERANK", defaults.rag_min_rerank)),
        clone_min_confidence=(
            float(env["CLONE_MIN_CONFIDENCE"]) if env.get("CLONE_MIN_CONFIDENCE") else None
        ),
        clone_calibration_path=env.get("CLONE_CALIBRATION_PATH", defaults.clone_calibration_path),
    )
