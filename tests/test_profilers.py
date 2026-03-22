"""
Unit tests for DataProfiler.
All tests operate on small in-memory DataFrames — no I/O required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.context import ColumnProfile
from app.profilers.schema_profiler import DataProfiler


def _profile(df: pd.DataFrame) -> list[ColumnProfile]:
    """Synchronous helper: run the profiler and return column profiles."""
    import asyncio
    return asyncio.get_event_loop().run_until_complete(DataProfiler(df).profile())


# ── Null statistics ───────────────────────────────────────────────────────────

class TestNullStats:
    def test_null_count_and_pct(self):
        df = pd.DataFrame({"col": [1.0, None, 3.0, None, 5.0]})
        profiles = _profile(df)
        p = profiles[0]
        assert p.null_count == 2
        assert pytest.approx(p.null_pct, abs=1e-4) == 0.4

    def test_zero_nulls(self):
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        p = _profile(df)[0]
        assert p.null_count == 0
        assert p.null_pct == 0.0

    def test_all_nulls(self):
        df = pd.DataFrame({"col": [None, None, None]})
        p = _profile(df)[0]
        assert p.null_count == 3
        assert pytest.approx(p.null_pct) == 1.0


# ── Unique counts ─────────────────────────────────────────────────────────────

class TestUniqueCounts:
    def test_unique_count_numeric(self):
        df = pd.DataFrame({"col": [1, 2, 2, 3, 3, 3]})
        p = _profile(df)[0]
        assert p.unique_count == 3

    def test_unique_count_string(self):
        df = pd.DataFrame({"col": ["a", "b", "a", "c"]})
        p = _profile(df)[0]
        assert p.unique_count == 3

    def test_unique_count_excludes_nulls(self):
        df = pd.DataFrame({"col": [1, 2, None, 2, None]})
        p = _profile(df)[0]
        assert p.unique_count == 2


# ── Cardinality classification ────────────────────────────────────────────────

class TestCardinality:
    def test_low_cardinality(self):
        df = pd.DataFrame({"col": ["a", "b", "c"] * 10})
        p = _profile(df)[0]
        assert p.cardinality == "low"

    def test_high_cardinality(self):
        df = pd.DataFrame({"col": list(range(200))})
        p = _profile(df)[0]
        assert p.cardinality == "high"

    def test_unique_cardinality(self):
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        p = _profile(df)[0]
        assert p.cardinality == "unique"

    def test_medium_cardinality(self):
        df = pd.DataFrame({"col": list(range(50))})
        p = _profile(df)[0]
        assert p.cardinality == "medium"


# ── Numeric stats ─────────────────────────────────────────────────────────────

class TestNumericStats:
    def test_min_max_mean_std(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = pd.DataFrame({"val": data})
        p = _profile(df)[0]
        assert pytest.approx(p.min) == 1.0
        assert pytest.approx(p.max) == 5.0
        assert pytest.approx(p.mean, abs=1e-4) == 3.0
        assert p.std is not None

    def test_quartiles_present(self):
        df = pd.DataFrame({"val": list(range(100))})
        p = _profile(df)[0]
        assert p.quartiles is not None
        assert "q25" in p.quartiles
        assert "q50" in p.quartiles
        assert "q75" in p.quartiles
        assert p.quartiles["q25"] < p.quartiles["q50"] < p.quartiles["q75"]


# ── Categorical / text stats ──────────────────────────────────────────────────

class TestCategoricalStats:
    def test_top_values_order(self):
        df = pd.DataFrame({"cat": ["a"] * 5 + ["b"] * 3 + ["c"] * 1})
        p = _profile(df)[0]
        assert len(p.top_values) > 0
        assert p.top_values[0]["value"] == "a"
        assert p.top_values[0]["count"] == 5

    def test_avg_length_computed(self):
        df = pd.DataFrame({"text": ["hello", "world", "hi"]})
        p = _profile(df)[0]
        assert p.avg_length is not None
        assert p.avg_length > 0


# ── Pattern detection ─────────────────────────────────────────────────────────

class TestPatternDetection:
    def test_email_pattern_detected(self):
        emails = [f"user{i}@example.com" for i in range(20)]
        df = pd.DataFrame({"email": emails})
        p = _profile(df)[0]
        assert p.has_pattern is True
        assert p.detected_pattern == "email"

    def test_uuid_pattern_detected(self):
        import uuid
        uuids = [str(uuid.uuid4()) for _ in range(20)]
        df = pd.DataFrame({"id": uuids})
        p = _profile(df)[0]
        assert p.has_pattern is True
        assert p.detected_pattern == "uuid"

    def test_no_pattern_on_mixed_text(self):
        df = pd.DataFrame({"notes": ["hello", "world", "foo@bar.com", "random text"] * 5})
        p = _profile(df)[0]
        # Mixed content should not trigger a pattern (< 80 % match)
        assert p.has_pattern is False

    def test_zip_pattern_detected(self):
        zips = ["90210", "10001", "30301", "60601", "77001"] * 4
        df = pd.DataFrame({"zip": zips})
        p = _profile(df)[0]
        assert p.has_pattern is True
        assert p.detected_pattern == "zip"


# ── Sample values ─────────────────────────────────────────────────────────────

class TestSampleValues:
    def test_sample_values_non_null(self):
        df = pd.DataFrame({"col": [1, 2, None, 4, 5]})
        p = _profile(df)[0]
        assert None not in p.sample_values
        assert len(p.sample_values) <= 5

    def test_sample_values_empty_on_all_null(self):
        df = pd.DataFrame({"col": pd.Series([None, None], dtype=object)})
        p = _profile(df)[0]
        assert p.sample_values == []


# ── Datetime columns ──────────────────────────────────────────────────────────

class TestDatetimeColumns:
    def test_date_range_computed(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2023-01-01", "2023-06-15", "2023-12-31"])})
        p = _profile(df)[0]
        assert p.date_range is not None
        assert "min" in p.date_range
        assert "max" in p.date_range
