import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import auth
import routes_agent
import routes_care_team
import routes_checklist
import routes_doctor
import routes_labels
import routes_documents
from config import PROJECT_DIR, Settings, get_settings
from druglabels import DEMO_LABELS, LabelClient, approved_rxcuis, fixture_transport, mark_interrupted
from db import init_db, make_engine, make_session_factory
from foods import FoodTable
from knowledge import open_store
from llm import LLMClient

FRONTEND_DIR = PROJECT_DIR / "frontend"

logger = logging.getLogger(__name__)
# uvicorn configures only its own loggers (uvicorn.error is at INFO), so the startup
# status line goes there to be shown with uvicorn's normal format.
startup_logger = logging.getLogger("uvicorn.error")

# The frontend is unbundled files with stable names, so a browser must check
# back (a cheap 304 via ETag) every time, or it keeps running an old version.
REVALIDATE = {"Cache-Control": "no-cache"}


class FrontendFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.update(REVALIDATE)
        return response


def create_app(
    settings: Optional[Settings] = None, llm=None, store=None, foods=None, label_client=None
) -> FastAPI:
    """`llm`, `store` and `foods` default to what `settings` configures. Without
    LLM_MODEL there is no model (model routes answer 503) and no library store."""
    settings = settings or get_settings()
    settings.validate()
    if llm is None and settings.llm_model:
        llm = LLMClient(settings)  # no API key required, so a self-hosted server works
    if store is None and llm is not None:
        store = open_store(settings, llm)
    if foods is None:
        foods_path = Path(settings.foods_path)
        foods = FoodTable.load(foods_path if foods_path.is_absolute() else PROJECT_DIR / foods_path)
    if label_client is None:
        # LABEL_FIXTURES=1 (dev only) answers from the recorded demo labels, offline.
        label_client = LabelClient(fixture_transport(DEMO_LABELS) if settings.label_fixtures else None)
    engine = make_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Tables are created automatically; sql/schema.sql is the manual alternative.
        await init_db(engine)
        # Assumes one server process: a row another process is still fetching
        # would be marked failed here.
        await mark_interrupted(app.state.session_factory)
        if app.state.store is not None:
            # Only approved labels may be in the library; anything else (a restored
            # index, a label the demo seed reset to pending) leaves before serving.
            removed = app.state.store.prune_labels(await approved_rxcuis(app.state.session_factory))
            if removed:
                logger.warning("Removed drug labels that are not approved from the library: %s",
                               ", ".join(removed))
            # A warning only when the reranker was asked for and is not available.
            unavailable = settings.rerank_enabled and app.state.store.reranker is None
            (startup_logger.warning if unavailable else startup_logger.info)(
                "Library reranker: %s", app.state.store.reranker_note)
        yield
        await label_client.aclose()
        await engine.dispose()

    app = FastAPI(title="AfterCare", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    app.state.llm = llm
    app.state.store = store
    # store.reranker is the source of truth; this is a convenience alias for the spec.
    app.state.reranker = store.reranker if store is not None else None
    app.state.foods = foods
    app.state.label_client = label_client

    app.include_router(auth.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_care_team.router)
    app.include_router(routes_doctor.router)
    app.include_router(routes_labels.router)
    app.include_router(routes_checklist.router)
    app.include_router(routes_agent.router)

    def page(name: str):
        async def handler():
            path = FRONTEND_DIR / name
            if not path.is_file():
                raise HTTPException(status_code=404, detail="Page not found")
            return FileResponse(path, headers=REVALIDATE)
        return handler

    app.add_api_route("/", page("index.html"), include_in_schema=False)
    app.add_api_route("/app", page("app.html"), include_in_schema=False)
    app.add_api_route("/doctor", page("doctor.html"), include_in_schema=False)
    # Mounted last so it never shadows /api routes.
    app.mount("/static", FrontendFiles(directory=FRONTEND_DIR), name="static")
    return app


def __getattr__(name: str):
    # `uvicorn app:app` resolves this lazily so importing the module (tests)
    # does not require a configured JWT_SECRET.
    if name == "app":
        instance = create_app()
        globals()["app"] = instance
        return instance
    raise AttributeError(name)
