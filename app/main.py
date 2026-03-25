"""
Context Intelligence Agent — FastAPI application entry point.
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routers import context, profile
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Context Intelligence Agent",
    description="Data ingestion gateway for the multi-agent architecture.",
    version="1.0.0",
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(profile.router)
app.include_router(context.router)

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"], summary="Health check")
async def health() -> dict:
    return {"status": "healthy", "llm_provider": "grok" if getattr(settings, "xai_api_key", "") else settings.llm_provider}

# ── Request timing middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    logger.info("%s %s  %s  %sms", request.method, request.url.path, response.status_code, elapsed)
    return response

# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "type": type(exc).__name__,
            "message": str(exc),
        },
    )
