"""Async Redis cache for ContextObject results."""
from __future__ import annotations

import hashlib
import json
from typing import Optional

from redis.asyncio import Redis, from_url

from app.config import settings
from app.models.context import ContextObject
from app.models.sources import DataSource

_KEY_PREFIX = "cia"


class ContextCache:
    """Async Redis-backed cache keyed by a SHA-256 hash of the source descriptor."""

    def __init__(self) -> None:
        self._redis: Optional[Redis] = None

    async def _get_client(self) -> Redis:
        """Return (or lazily create) the Redis async client."""
        if self._redis is None:
            self._redis = from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    @staticmethod
    def generate_key(source: DataSource) -> str:
        """Return a deterministic SHA-256 cache key for the given source descriptor.

        Credential fields are excluded so the same data with different creds
        maps to the same cache entry.
        """
        source_dict = source.model_dump()
        source_type = source_dict.get("type", "unknown")
        for sensitive in (
            "aws_access_key_id", "aws_secret_access_key",
            "password", "private_key_path",
            "credentials_path", "google_application_credentials",
        ):
            source_dict.pop(sensitive, None)
        canonical = json.dumps(source_dict, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return f"{_KEY_PREFIX}:{source_type}:{digest}"

    async def get_context(self, key: str) -> Optional[ContextObject]:
        """Return the cached ContextObject for key, or None on a miss."""
        client = await self._get_client()
        raw = await client.get(key)
        if raw is None:
            return None
        return ContextObject.model_validate_json(raw)

    async def set_context(self, key: str, context: ContextObject) -> None:
        """Persist a ContextObject under key with the configured TTL."""
        client = await self._get_client()
        serialised = context.model_dump_json()
        await client.set(key, serialised, ex=settings.context_ttl_seconds)
