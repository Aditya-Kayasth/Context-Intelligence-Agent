"""Async Redis cache for ContextObject results."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from redis.asyncio import Redis, from_url

from app.config import settings
from app.models.context import ContextObject
from app.models.sources import DataSource

logger = logging.getLogger(__name__)
_KEY_PREFIX = "cia"

# Global Redis client singleton
_redis_client: Optional[Redis] = None
_redis_disabled: bool = False

async def get_redis_client() -> Optional[Redis]:
    """Return the global Redis client, or None if disabled."""
    global _redis_client, _redis_disabled
    
    if _redis_disabled:
        return None
        
    if _redis_client is None:
        try:
            _redis_client = from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )
            await _redis_client.ping()
            logger.info("Connected to Redis at %s", settings.redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable at %s: %s -- caching disabled", settings.redis_url, exc)
            _redis_disabled = True
            if _redis_client:
                await _redis_client.aclose()
                _redis_client = None
            return None
    return _redis_client

async def close_redis() -> None:
    """Close the global Redis client."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis connection closed")

class ContextCache:
    """Async Redis-backed cache keyed by a SHA-256 hash of the source descriptor."""

    async def ping(self) -> bool:
        """Check if Redis is alive. Returns False if offline or ping fails."""
        client = await get_redis_client()
        if client is None:
            return False
        try:
            await client.ping()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """No-op for compatibility (use close_redis for the global client)."""
        pass

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
        """Return the cached ContextObject for key, or None on a miss or if Redis is offline."""
        client = await get_redis_client()
        if client is None:
            return None
        try:
            raw = await client.get(key)
            if raw is None:
                return None
            return ContextObject.model_validate_json(raw)
        except Exception as exc:
            logger.warning("Error reading from Redis: %s -- bypassing cache", exc)
            return None

    async def set_context(self, key: str, context: ContextObject) -> None:
        """Persist a ContextObject under key with the configured TTL. Bypasses if Redis is offline."""
        client = await get_redis_client()
        if client is None:
            return
        try:
            serialised = context.model_dump_json()
            await client.set(key, serialised, ex=settings.context_ttl_seconds)
        except Exception as exc:
            logger.warning("Error writing to Redis: %s", exc)
