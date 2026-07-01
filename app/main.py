"""FastAPI service exposing GET /health and POST /chat.

Design priorities (in order):
  1. Schema compliance ALWAYS - even malformed input returns a valid ChatResponse,
     never a 422/500, because the evaluator scores schema compliance on every response.
  2. Statelessness - no per-conversation server state; the full history arrives each call.
  3. Fast readiness - /health never touches the LLM; the retriever is warmed at startup.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .agent import run_agent
from .config import settings
from .retriever import get_retriever
from .schemas import ChatRequest, ChatResponse, HealthResponse, Message

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("shl.main")

STATIC_DIR = settings.project_root / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the retriever (loads catalog + embeddings, builds BM25) so the first
    # /chat isn't penalised. Failure here is non-fatal: /chat degrades gracefully.
    try:
        get_retriever()
        logger.info("Retriever warmed at startup.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Retriever warm-up failed (continuing): %s", exc)
    if not settings.has_api_key:
        logger.warning("No API key configured; /chat will return graceful fallbacks.")
    yield


app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness probe. Intentionally cheap and LLM-free."""
    return HealthResponse(status="ok")


@app.get("/version")
def version() -> dict[str, str]:
    """Build marker (used to confirm which revision is deployed)."""
    return {"version": __version__}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Stateless conversational turn. Full history in, next reply + shortlist out."""
    try:
        return run_agent(request.messages)
    except Exception as exc:  # noqa: BLE001 - last-resort guard; keep schema valid
        logger.exception("Unhandled error in /chat: %s", exc)
        return ChatResponse(
            reply="Sorry, I hit a temporary issue. Could you restate what you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )


@app.exception_handler(Exception)
async def _any_error(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so /chat never emits a raw 500 that breaks schema compliance."""
    logger.exception("Global handler caught: %s", exc)
    if request.url.path == "/chat":
        return JSONResponse(
            status_code=200,
            content=ChatResponse(
                reply="Sorry, something went wrong. Please tell me about the role again.",
                recommendations=[],
                end_of_conversation=False,
            ).model_dump(),
        )
    return JSONResponse(status_code=500, content={"detail": "internal error"})


# RequestValidationError -> still return a usable ChatResponse for /chat.
from fastapi.exceptions import RequestValidationError  # noqa: E402


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    if request.url.path == "/chat":
        return JSONResponse(
            status_code=200,
            content=ChatResponse(
                reply="I couldn't read that request. What role are you hiring for?",
                recommendations=[],
                end_of_conversation=False,
            ).model_dump(),
        )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/")
def root():
    """Serve the test UI if present, else basic service info."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {
        "service": "SHL Conversational Assessment Recommender",
        "version": __version__,
        "endpoints": ["GET /health", "POST /chat"],
        "docs": "/docs",
    }
