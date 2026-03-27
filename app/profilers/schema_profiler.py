"""
DataProfiler — pure Pandas/NumPy, no LLM calls.

All heavy computation runs inside asyncio.to_thread so the FastAPI event loop
is never blocked.  With the sampler capping input at 50 000 rows this is fast
(typically < 1 s), but the thread offload keeps latency predictable under load.

Pattern detection
-----------------
For each text/object column we compile five anchored regexes (email, url, uuid,
phone, zip) and test them against every non-null value in the sample.  A pattern
is reported only when ≥ 80 % of non-null values match — this avoids false
positives on mixed-content columns.  The regexes are module-level compiled
constants so they are built once at import time, not per-column.
"""
from __future__ import annotations

import asyncio
import re
from typing import Optional

import numpy as np
import pandas as pd

from app.models.context import ColumnProfile

# ── Pattern regexes (compiled once) ──────────────────────────────────────────

_PATTERNS: dict[str, re.Pattern] = {
    "email":   re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"),
    "url":     re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE),
    "uuid":    re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    ),
    "phone":   re.compile(
        r"^\+?1?\s*[\-.]?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}$"
    ),
    "zip":     re.compile(r"^\d{5}(?:-\d{4})?$"),
}

_PATTERN_THRESHOLD = 0.80   # 80 % of non-null values must match
_TOP_N = 5                  # top-N frequent values for categorical columns
_SAMPLE_N = 5               # random sample values per column

# ── PII / sensitive-column detection ─────────────────────────────────────────

_REDACTED = ["[REDACTED_FOR_SECURITY]"]

# Substrings that flag a column name as PII-sensitive (case-insensitive)
_PII_NAME_KEYWORDS: frozenset[str] = frozenset({
    "email", "mail",
    "phone", "mobile", "cell",
    "ssn", "social",
    "password", "passwd", "secret", "token", "hash",
    "credit", "card", "cvv", "iban", "account",
    "name", "firstname", "lastname", "surname",
    "address", "street", "postcode", "zipcode",
    "dob", "birthdate", "birthday",
    "ip", "ipaddr",
    "passport", "license", "licence",
    "salary", "income",
    "gender", "race", "ethnicity", "religion",
})

# Regex-detected patterns that are inherently PII
_PII_PATTERNS: frozenset[str] = frozenset({"email", "phone"})

# LLM-assigned semantic types that are inherently PII
_PII_SEMANTIC_TYPES: frozenset[str] = frozenset({
    "email", "phone", "identifier", "url",
})


# ── Public class ──────────────────────────────────────────────────────────────

class DataProfiler:  # pylint: disable=too-few-public-methods
    """Profile a sampled DataFrame and return a list of ColumnProfile objects."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    async def profile(self) -> list[ColumnProfile]:
        """Async entry-point — offloads blocking work to a thread."""
        return await asyncio.to_thread(self._profile_sync)

    # ── Synchronous core ──────────────────────────────────────────────────────

    def _profile_sync(self) -> list[ColumnProfile]:
        df = self._df
        total_rows = len(df)
        profiles: list[ColumnProfile] = []

        for col_name in df.columns:
            series = df[col_name]
            col = self._profile_column(series, total_rows)
            _redact_if_pii(col)
            profiles.append(col)

        return profiles

    def _profile_column(self, series: pd.Series, total_rows: int) -> ColumnProfile:
        name = series.name
        null_count = int(series.isna().sum())
        null_pct = round(null_count / total_rows, 4) if total_rows else 0.0
        non_null = series.dropna()
        unique_count = int(series.nunique(dropna=True))
        dtype_str = str(series.dtype)

        # ── Cardinality ───────────────────────────────────────────────────────
        cardinality = _classify_cardinality(unique_count, len(non_null))

        # ── Sample values (3-5 random non-null) ───────────────────────────────
        sample_values = _draw_samples(non_null, _SAMPLE_N)

        # ── Branch by dtype ───────────────────────────────────────────────────
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnProfile(
                name=name,
                dtype=dtype_str,
                null_count=null_count,
                null_pct=null_pct,
                unique_count=unique_count,
                cardinality=cardinality,
                date_range=_date_range(non_null),
                sample_values=sample_values,
                semantic_type=None,
            )

        if pd.api.types.is_numeric_dtype(series):
            return ColumnProfile(
                name=name,
                dtype=dtype_str,
                null_count=null_count,
                null_pct=null_pct,
                unique_count=unique_count,
                cardinality=cardinality,
                min=_safe_scalar(non_null.min()),
                max=_safe_scalar(non_null.max()),
                mean=_safe_float(non_null.mean()),
                std=_safe_float(non_null.std()),
                quartiles=_quartiles(non_null),
                sample_values=sample_values,
                semantic_type=None,
            )

        # Categorical / text (object, string, bool, category …)
        top_values = _top_values(non_null, _TOP_N)
        avg_length = _avg_length(non_null)
        has_pattern, detected_pattern = _detect_pattern(non_null)

        return ColumnProfile(
            name=name,
            dtype=dtype_str,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
            cardinality=cardinality,
            top_values=top_values,
            avg_length=avg_length,
            has_pattern=has_pattern,
            detected_pattern=detected_pattern,
            sample_values=sample_values,
            semantic_type=None,
        )


# ── Helper functions ──────────────────────────────────────────────────────────

def _redact_if_pii(col: ColumnProfile) -> None:
    """Overwrite sample_values with a redaction sentinel if the column is PII-sensitive.

    Detection order (any match triggers redaction):
      1. Column name contains a known PII keyword (case-insensitive substring match).
      2. Regex pattern detector flagged the column as email or phone.
      3. LLM-assigned semantic_type is in the PII semantic-type set.
    """
    col_lower = col.name.lower().replace("_", "").replace("-", "").replace(" ", "")
    name_is_pii = any(kw in col_lower for kw in _PII_NAME_KEYWORDS)
    pattern_is_pii = col.detected_pattern in _PII_PATTERNS
    semantic_is_pii = col.semantic_type in _PII_SEMANTIC_TYPES

    if name_is_pii or pattern_is_pii or semantic_is_pii:
        col.sample_values = _REDACTED

def _classify_cardinality(
    unique_count: int, non_null_count: int
) -> str:
    if non_null_count > 0 and unique_count == non_null_count:
        return "unique"
    if unique_count < 20:
        return "low"
    if unique_count <= 100:
        return "medium"
    return "high"


def _draw_samples(non_null: pd.Series, n: int) -> list:
    if len(non_null) == 0:
        return []
    sample = non_null.sample(min(n, len(non_null)), random_state=42)
    return [_safe_scalar(v) for v in sample.tolist()]


def _quartiles(non_null: pd.Series) -> Optional[dict[str, float]]:
    if len(non_null) == 0:
        return None
    q = non_null.quantile([0.25, 0.50, 0.75])
    return {
        "q25": _safe_float(q[0.25]),
        "q50": _safe_float(q[0.50]),
        "q75": _safe_float(q[0.75]),
    }


def _top_values(non_null: pd.Series, n: int) -> list[dict]:
    if len(non_null) == 0:
        return []
    counts = non_null.value_counts().head(n)
    return [{"value": _safe_scalar(v), "count": int(c)} for v, c in counts.items()]


def _avg_length(non_null: pd.Series) -> Optional[float]:
    """Return the mean string length of non-null values, or None on failure."""
    try:
        lengths = non_null.astype(str).str.len()
        return round(float(lengths.mean()), 2)
    except (TypeError, ValueError):
        return None


def _date_range(non_null: pd.Series) -> Optional[dict[str, str]]:
    """Return {"min": ..., "max": ...} for a datetime series, or None on failure."""
    try:
        return {
            "min": str(non_null.min()),
            "max": str(non_null.max()),
        }
    except (TypeError, ValueError):
        return None


def _detect_pattern(non_null: pd.Series) -> tuple[bool, Optional[str]]:
    """
    Test each compiled regex against the string representation of non-null values.
    Returns (True, pattern_name) for the first pattern where ≥ 80 % of values match.
    Returns (False, None) if no pattern clears the threshold.
    """
    if len(non_null) == 0:
        return False, None

    str_series = non_null.astype(str)
    n = len(str_series)

    for pattern_name, regex in _PATTERNS.items():
        # Capture regex in default arg to avoid cell-var-from-loop
        match_count = str_series.apply(lambda v, r=regex: bool(r.match(v))).sum()
        if match_count / n >= _PATTERN_THRESHOLD:
            return True, pattern_name

    return False, None


def _safe_scalar(value):
    """Convert numpy scalars to native Python types for JSON serialisation."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _safe_float(value) -> Optional[float]:
    try:
        v = float(value)
        return round(v, 6) if not np.isnan(v) else None
    except (TypeError, ValueError):
        return None
