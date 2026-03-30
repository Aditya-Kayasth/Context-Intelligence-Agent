"""Context Intelligence Agent — FastAPI application entry point."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routers import context, profile


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Graceful startup and shutdown for the Context Agent."""
    logger.info("[START] Context Intelligence Agent starting")
    yield
    from app.cache.redis_cache import close_redis
    logger.info("[STOP] Context Agent shutting down -- closing Redis")
    await close_redis()


# ── App factory ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Context Intelligence Agent",
    description="Data ingestion gateway for the multi-agent architecture.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(profile.router)
app.include_router(context.router)


@app.get("/health", tags=["ops"], summary="Health check")
async def health() -> dict:
    """Return service liveness status."""
    from app.config import settings
    from app.cache.redis_cache import ContextCache
    
    provider = str(getattr(settings, "llm_provider", "unknown"))
    model = ""
    if provider == "azure_openai":
        model = getattr(settings, "azure_openai_deployment_name", "")
    elif provider == "groq":
        model = getattr(settings, "groq_model", "")
    elif provider == "openai":
        model = "gpt-4o"
    elif provider == "ollama":
        model = getattr(settings, "ollama_model", "")

    cache = ContextCache()
    redis_alive = await cache.ping()
    await cache.close()

    return {
        "status": "healthy",
        "llm_provider": provider,
        "llm_model": model,
        "redis_status": "online" if redis_alive else "offline (caching disabled)",
    }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status code, and elapsed time for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "%s %s  %s  %sms",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unhandled errors and return a structured 500 JSON response."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "type": type(exc).__name__,
            "message": str(exc),
        },
    )
