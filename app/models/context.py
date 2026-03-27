"""
ContextObject and ColumnProfile models.

Token-budget enforcement:
  - Heuristic: 1 token ≈ 4 characters  →  4096 tokens ≈ 16 384 chars
  - A @model_validator(mode='after') serialises the object to JSON and, if the
    payload exceeds the budget, progressively trims `sample_values` and
    `top_values` on every ColumnProfile until it fits.
  - A public `truncate_for_llm(max_tokens)` method exposes the same logic for
    callers that want an explicit, on-demand trim.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator

# 1 token ≈ 4 chars (conservative GPT-style heuristic)
_CHARS_PER_TOKEN: int = 4
_DEFAULT_TOKEN_BUDGET: int = 4096


def _char_budget(max_tokens: int = _DEFAULT_TOKEN_BUDGET) -> int:
    return max_tokens * _CHARS_PER_TOKEN


class ColumnProfile(BaseModel):
    """Statistical and semantic profile for a single DataFrame column."""
    name: str
    dtype: str
    null_count: int = 0
    null_pct: float = 0.0
    unique_count: Optional[int] = None
    # Cardinality bucket: low < 20, medium 20-100, high > 100, unique = all distinct
    cardinality: Optional[Literal["low", "medium", "high", "unique"]] = None
    # Numeric stats
    min: Optional[Any] = None
    max: Optional[Any] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    quartiles: Optional[dict[str, float]] = None   # {"q25": ..., "q50": ..., "q75": ...}
    # Categorical / text stats
    top_values: list[Any] = Field(default_factory=list)   # [{"value": x, "count": n}, ...]
    avg_length: Optional[float] = None
    # Pattern detection
    has_pattern: bool = False
    detected_pattern: Optional[str] = None   # "email" | "url" | "uuid" | "phone" | "zip" | None
    # Datetime stats
    date_range: Optional[dict[str, str]] = None   # {"min": ..., "max": ...}
    # LLM-facing
    sample_values: list[Any] = Field(default_factory=list)
    semantic_type: Optional[str] = None   # filled in by LLM in Phase 5


class ContextObject(BaseModel):
    """Enriched context payload returned to downstream agents."""
    source_id: str
    source_type: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile] = Field(default_factory=list)
    quality_score: Optional[float] = None   # 0–1 from great-expectations
    llm_summary: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ── Auto-trim on construction ─────────────────────────────────────────────

    @model_validator(mode="after")
    def _enforce_token_budget(self) -> "ContextObject":
        """Trim sample/top values if the serialised payload exceeds 4 096 tokens."""
        self._trim_to_budget(_DEFAULT_TOKEN_BUDGET)
        return self

    # ── Public helper ─────────────────────────────────────────────────────────

    def truncate_for_llm(self, max_tokens: int = _DEFAULT_TOKEN_BUDGET) -> "ContextObject":
        """Return *self* after trimming in-place to fit within `max_tokens`."""
        self._trim_to_budget(max_tokens)
        return self

    # ── Internal trim logic ───────────────────────────────────────────────────

    def _trim_to_budget(self, max_tokens: int) -> None:
        budget = _char_budget(max_tokens)

        if len(self._json_size()) <= budget:
            return  # already fits — nothing to do

        # Progressive trim: halve list lengths per column until payload fits
        # or every list is empty.
        while len(self._json_size()) > budget:
            trimmed_any = False
            for col in self.columns:
                if col.sample_values:
                    col.sample_values = col.sample_values[: max(0, len(col.sample_values) // 2)]
                    trimmed_any = True
                if col.top_values:
                    col.top_values = col.top_values[: max(0, len(col.top_values) // 2)]
                    trimmed_any = True
            if not trimmed_any:
                break  # nothing left to trim

    def _json_size(self) -> str:
        """Serialise to JSON string (used only for length measurement)."""
        return self.model_dump_json()
