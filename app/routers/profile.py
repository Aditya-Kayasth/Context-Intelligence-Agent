"""
Profile router — two endpoints:

  POST /profile         Sync: run full pipeline, return ContextObject.
  POST /profile/stream  SSE:  stream pipeline progress events, final context_id.

SSE format
----------
Each event is a UTF-8 string:

  data: {"stage": "<name>", "pct": <0-100>}\n\n          (progress)
  data: {"stage": "complete", "context_id": "<key>"}\n\n  (final)
  data: {"stage": "error", "message": "<text>"}\n\n        (on failure)

The generator is an async def that yields strings.  FastAPI's StreamingResponse
wraps it in a chunked HTTP/1.1 response with Content-Type: text/event-stream.
The client can consume it with EventSource or any SSE-capable HTTP client.
"""
from __future__ import annotations

import json
import logging
from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.cache.redis_cache import ContextCache
from app.models.context import ContextObject
from app.models.sources import DataSource
from app.utils.pipeline import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/profile", tags=["profile"])


from collections.abc import AsyncGenerator

async def get_cache() -> AsyncGenerator[ContextCache, None]:
    cache = ContextCache()
    try:
        yield cache
    finally:
        await cache.close()


# ── Request body ──────────────────────────────────────────────────────────────

class ProfileRequest(BaseModel):
    source: DataSource = Field(..., discriminator="type")


# ── POST /profile  (sync) ─────────────────────────────────────────────────────

@router.post("", response_model=ContextObject, summary="Profile a data source")
async def profile_source(
    body: ProfileRequest,
    cache: Annotated[ContextCache, Depends(get_cache)],
) -> ContextObject:
    """Run the full pipeline synchronously and return the completed ContextObject."""
    context: ContextObject | None = None
    async for _stage, _pct, payload in run_pipeline(body.source, cache):
        if payload is not None:
            context = payload  # type: ignore[assignment]
    if context is None:
        raise RuntimeError("Pipeline completed without producing a ContextObject.")
    return context


# ── POST /profile/stream  (SSE) ───────────────────────────────────────────────

@router.post("/stream", summary="Profile a data source with SSE progress stream")
async def profile_source_stream(body: ProfileRequest) -> StreamingResponse:
    """
    Stream pipeline progress as Server-Sent Events.
    Connect with EventSource or: curl -N -X POST .../profile/stream -d '...'
    """
    return StreamingResponse(
        _sse_generator(body.source),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


async def _sse_generator(source: DataSource) -> AsyncGenerator[str, None]:
    """
    Async generator that drives run_pipeline and formats each step as an SSE frame.

    SSE wire format requires every message to end with two newlines (\n\n).
    We use the `data:` field only (no `event:` or `id:` fields needed here).
    """
    cache = ContextCache()
    try:
        async for stage, pct, payload in run_pipeline(source, cache):
            if stage == "complete" and payload is not None:
                context: ContextObject = payload  # type: ignore[assignment]
                event = json.dumps({"stage": "complete", "context_id": context.source_id})
            else:
                event = json.dumps({"stage": stage, "pct": pct})
            yield f"data: {event}\n\n"
    except Exception as exc:
        logger.exception("SSE pipeline error")
        error_event = json.dumps({"stage": "error", "message": str(exc)})
        yield f"data: {error_event}\n\n"
    finally:
        await cache.close()
