"""Context router — cache retrieval and pipeline refresh endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.cache.redis_cache import ContextCache
from app.models.context import ContextObject
from app.models.sources import DataSource
from app.utils.pipeline import run_pipeline

router = APIRouter(tags=["context"])


def get_cache() -> ContextCache:
    """Dependency that provides a ContextCache instance."""
    return ContextCache()


@router.get(
    "/context/{context_id}",
    response_model=ContextObject,
    summary="Retrieve a cached ContextObject",
)
async def get_context(
    context_id: str,
    cache: Annotated[ContextCache, Depends(get_cache)],
) -> ContextObject:
    """Return the cached ContextObject for context_id, or 404 if not found."""
    context = await cache.get_context(context_id)
    if context is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached context found for id: {context_id!r}",
        )
    return context


class RefreshRequest(BaseModel):
    """Request body for the refresh endpoint."""

    source: DataSource = Field(..., discriminator="type")


@router.post(
    "/refresh/{context_id}",
    response_model=ContextObject,
    summary="Re-profile a data source and refresh the cache",
)
async def refresh_context(
    context_id: str,  # pylint: disable=unused-argument  # kept for URL symmetry
    body: RefreshRequest,
    cache: Annotated[ContextCache, Depends(get_cache)],
) -> ContextObject:
    """Re-run the full pipeline for the given source and overwrite the cache entry."""
    context: ContextObject | None = None
    async for _stage, _pct, payload in run_pipeline(body.source, cache):
        if payload is not None:
            context = payload  # type: ignore[assignment]
    if context is None:
        raise HTTPException(status_code=500, detail="Pipeline completed without a result.")
    return context
