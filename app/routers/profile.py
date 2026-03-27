"""Profile router — sync and SSE-streaming pipeline endpoints."""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.cache.redis_cache import ContextCache
from app.models.context import ContextObject
from app.models.sources import DataSource
from app.utils.pipeline import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/profile", tags=["profile"])


def get_cache() -> ContextCache:
    """Dependency that provides a ContextCache instance."""
    return ContextCache()


class ProfileRequest(BaseModel):
    """Request body for profile endpoints."""

    source: DataSource = Field(..., discriminator="type")


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


@router.post("/stream", summary="Profile a data source with SSE progress stream")
async def profile_source_stream(body: ProfileRequest) -> StreamingResponse:
    """Stream pipeline progress as Server-Sent Events."""
    return StreamingResponse(
        _sse_generator(body.source),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _sse_generator(source: DataSource) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted strings as each pipeline stage completes."""
    cache = ContextCache()
    try:
        async for stage, pct, payload in run_pipeline(source, cache):
            if stage == "complete" and payload is not None:
                context: ContextObject = payload  # type: ignore[assignment]
                event = json.dumps({"stage": "complete", "context_id": context.source_id})
            else:
                event = json.dumps({"stage": stage, "pct": pct})
            yield f"data: {event}\n\n"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("SSE pipeline error")
        yield f"data: {json.dumps({'stage': 'error', 'message': str(exc)})}\n\n"
    finally:
        await cache.close()
