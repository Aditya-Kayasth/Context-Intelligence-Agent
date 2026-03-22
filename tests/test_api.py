"""
API integration tests using httpx.AsyncClient against the FastAPI app.
The LLM call is mocked so no real Groq key is needed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_body(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.json() == {"status": "healthy"}


# ── POST /profile ─────────────────────────────────────────────────────────────

_MOCK_LLM_RESPONSE = {
    "semantic_types": {"name": "category", "age": "quantity", "email": "email"},
    "suggested_analyses": [
        "Analyse age distribution across categories",
        "Identify duplicate email addresses",
    ],
}

_MOCK_DF = pd.DataFrame({
    "name":  ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age":   [25, 30, 35, 40, 45],
    "email": [f"user{i}@example.com" for i in range(5)],
})


class TestProfileEndpoint:
    @pytest.mark.asyncio
    async def test_profile_local_csv(self, client: AsyncClient, tmp_path):
        """POST /profile with a local CSV source returns a valid ContextObject."""
        csv_file = tmp_path / "sample.csv"
        _MOCK_DF.to_csv(csv_file, index=False)

        with (
            patch(
                "app.llm.summarizer.SemanticEnricher.enrich_profile",
                new_callable=AsyncMock,
                return_value=_MOCK_LLM_RESPONSE,
            ),
            patch(
                "app.cache.redis_cache.ContextCache.get_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.cache.redis_cache.ContextCache.set_context",
                new_callable=AsyncMock,
            ),
        ):
            resp = await client.post(
                "/profile",
                json={"source": {"type": "local_file", "path": str(csv_file), "format": "csv"}},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["source_type"] == "local_file"
        assert body["row_count"] == 5
        assert body["column_count"] == 3
        assert isinstance(body["columns"], list)
        assert len(body["columns"]) == 3

    @pytest.mark.asyncio
    async def test_profile_column_names(self, client: AsyncClient, tmp_path):
        """Returned column profiles must include all DataFrame columns."""
        csv_file = tmp_path / "cols.csv"
        _MOCK_DF.to_csv(csv_file, index=False)

        with (
            patch("app.llm.summarizer.SemanticEnricher.enrich_profile", new_callable=AsyncMock, return_value=_MOCK_LLM_RESPONSE),
            patch("app.cache.redis_cache.ContextCache.get_context", new_callable=AsyncMock, return_value=None),
            patch("app.cache.redis_cache.ContextCache.set_context", new_callable=AsyncMock),
        ):
            resp = await client.post(
                "/profile",
                json={"source": {"type": "local_file", "path": str(csv_file), "format": "csv"}},
            )

        col_names = [c["name"] for c in resp.json()["columns"]]
        assert set(col_names) == {"name", "age", "email"}

    @pytest.mark.asyncio
    async def test_profile_semantic_types_applied(self, client: AsyncClient, tmp_path):
        """LLM semantic_types must be applied to the returned ColumnProfile objects."""
        csv_file = tmp_path / "sem.csv"
        _MOCK_DF.to_csv(csv_file, index=False)

        with (
            patch("app.llm.summarizer.SemanticEnricher.enrich_profile", new_callable=AsyncMock, return_value=_MOCK_LLM_RESPONSE),
            patch("app.cache.redis_cache.ContextCache.get_context", new_callable=AsyncMock, return_value=None),
            patch("app.cache.redis_cache.ContextCache.set_context", new_callable=AsyncMock),
        ):
            resp = await client.post(
                "/profile",
                json={"source": {"type": "local_file", "path": str(csv_file), "format": "csv"}},
            )

        cols = {c["name"]: c for c in resp.json()["columns"]}
        assert cols["name"]["semantic_type"] == "category"
        assert cols["age"]["semantic_type"] == "quantity"

    @pytest.mark.asyncio
    async def test_profile_cache_hit_skips_pipeline(self, client: AsyncClient):
        """When the cache returns a hit, the pipeline must not re-run."""
        from app.models.context import ContextObject

        cached = ContextObject(
            source_id="cia:local_file:abc123",
            source_type="local_file",
            row_count=100,
            column_count=2,
        )

        with patch(
            "app.cache.redis_cache.ContextCache.get_context",
            new_callable=AsyncMock,
            return_value=cached,
        ):
            resp = await client.post(
                "/profile",
                json={"source": {"type": "local_file", "path": "/any/path.csv", "format": "csv"}},
            )

        assert resp.status_code == 200
        assert resp.json()["row_count"] == 100


# ── GET /context/{context_id} ─────────────────────────────────────────────────

class TestContextEndpoint:
    @pytest.mark.asyncio
    async def test_get_context_404_on_miss(self, client: AsyncClient):
        with patch(
            "app.cache.redis_cache.ContextCache.get_context",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = await client.get("/context/nonexistent-key")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_context_returns_cached_object(self, client: AsyncClient):
        from app.models.context import ContextObject

        cached = ContextObject(
            source_id="cia:s3:deadbeef",
            source_type="s3",
            row_count=500,
            column_count=4,
        )
        with patch(
            "app.cache.redis_cache.ContextCache.get_context",
            new_callable=AsyncMock,
            return_value=cached,
        ):
            resp = await client.get("/context/cia:s3:deadbeef")

        assert resp.status_code == 200
        assert resp.json()["source_id"] == "cia:s3:deadbeef"
        assert resp.json()["row_count"] == 500


# ── Global exception handler ──────────────────────────────────────────────────

class TestExceptionHandler:
    @pytest.mark.asyncio
    async def test_unhandled_exception_returns_500(self, client: AsyncClient):
        with patch(
            "app.cache.redis_cache.ContextCache.get_context",
            side_effect=RuntimeError("boom"),
        ):
            resp = await client.get("/context/any-key")

        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] is True
        assert body["type"] == "RuntimeError"
        assert "boom" in body["message"]
