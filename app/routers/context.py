"""
Context router — two endpoints:

  GET  /context/{context_id}    Fetch a cached ContextObject.
  POST /refresh/{context_id}    Re-run the pipeline and overwrite the cache entry.

The refresh endpoint requires the original DataSource config to be re-submitted
in the request body because the cache stores only the result, not the source
descriptor.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.cache.redis_cache import ContextCache
from app.models.context import ContextObject
from app.models.sources import DataSource
from app.utils.pipeline import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["context"])


from collections.abc import AsyncGenerator

async def get_cache() -> AsyncGenerator[ContextCache, None]:
    cache = ContextCache()
    try:
        yield cache
    finally:
        await cache.close()


# ── GET /context/{context_id} ─────────────────────────────────────────────────

@router.get(
    "/context/{context_id}",
    response_model=ContextObject,
    summary="Retrieve a cached ContextObject",
)
async def get_context(
    context_id: str,
    cache: Annotated[ContextCache, Depends(get_cache)],
) -> ContextObject:
    context = await cache.get_context(context_id)
    if context is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached context found for id: {context_id!r}",
        )
    return context


# ── POST /refresh/{context_id} ────────────────────────────────────────────────

class RefreshRequest(BaseModel):
    source: DataSource = Field(..., discriminator="type")


@router.post(
    "/refresh/{context_id}",
    response_model=ContextObject,
    summary="Re-profile a data source and refresh the cache",
)
async def refresh_context(
    context_id: str,
    body: RefreshRequest,
    cache: Annotated[ContextCache, Depends(get_cache)],
) -> ContextObject:
    """
    Re-runs the full pipeline for the given source, overwrites the cached entry,
    and returns the updated ContextObject.

    The context_id in the path is informational; the canonical key is always
    re-derived from the source descriptor so it stays consistent.
    """
    context: ContextObject | None = None
    async for _stage, _pct, payload in run_pipeline(body.source, cache):
        if payload is not None:
            context = payload  # type: ignore[assignment]

    if context is None:
        raise HTTPException(status_code=500, detail="Pipeline completed without a result.")

    return context
