"""Context Intelligence Agent — FastAPI application entry point."""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routers import context, profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Context Intelligence Agent",
    description="Data ingestion gateway for the multi-agent architecture.",
    version="1.0.0",
)

app.include_router(profile.router)
app.include_router(context.router)


@app.get("/health", tags=["ops"], summary="Health check")
async def health() -> dict:
    """Return service liveness status."""
    return {"status": "healthy"}


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
