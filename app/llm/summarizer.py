"""
SemanticEnricher — uses the configured LLM to:
  1. Guess a semantic_type for each column.
  2. Suggest 3-5 analyses the dataset is suited for.

JSON-only enforcement strategy
-------------------------------
Three layers work together to guarantee we always get parseable JSON back:

  Layer 1 — response_format: {"type": "json_object"}
    Passed to Groq (and OpenAI-compatible endpoints) at the API level.
    The model is contractually required to emit valid JSON; it will never
    wrap the output in markdown fences or add prose.

  Layer 2 — System prompt instruction
    The system prompt explicitly says "Return ONLY valid JSON. No markdown,
    no explanation, no code fences." Belt-and-suspenders for providers that
    don't support response_format.

  Layer 3 — Fence stripper + json.loads with fallback
    Before parsing, we strip any accidental ```json … ``` wrappers.
    If json.loads still fails we raise a structured LLMParseError so the
    caller can decide whether to retry or surface the error.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.llm.provider import get_llm_provider
from app.models.context import ColumnProfile

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_SYSTEM_PROMPT = """\
You are a data intelligence assistant. You will receive a JSON array describing \
columns from a dataset. For each column analyse the name, dtype, sample values, \
and statistics, then infer its real-world meaning.

Return ONLY valid JSON — no markdown, no explanation, no code fences.

The JSON object MUST have exactly two keys:
  "semantic_types": an object mapping each column name to one of:
    "identifier", "category", "currency", "quantity", "percentage",
    "datetime", "location", "text", "boolean", "url", "email", "phone",
    "uuid", "zip", "unknown"
  "suggested_analyses": an array of 3 to 5 short, actionable analysis ideas
    tailored to the dataset as a whole (e.g. "Analyse revenue trends over time").
"""


class LLMParseError(Exception):
    """Raised when the LLM response cannot be parsed as JSON."""


class SemanticEnricher:  # pylint: disable=too-few-public-methods
    """Calls the configured LLM to assign semantic types and suggest analyses."""

    def __init__(self) -> None:
        self._provider = get_llm_provider()

    async def enrich_profile(
        self, columns: list[ColumnProfile]
    ) -> dict[str, Any]:
        """
        Call the LLM with a stripped-down column summary.
        Mutates each ColumnProfile.semantic_type in-place.
        Returns the full parsed LLM response dict.
        """
        user_prompt = self._build_user_prompt(columns)
        raw = await self._provider.complete(_SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_response(raw)

        # Apply semantic types back onto the column objects
        semantic_map: dict[str, str] = parsed.get("semantic_types", {})
        for col in columns:
            if col.name in semantic_map:
                col.semantic_type = semantic_map[col.name]

        return parsed

    # ── Prompt construction ───────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(columns: list[ColumnProfile]) -> str:
        """
        Build a compact column summary — only the fields the LLM needs.
        Keeps the payload small to stay well within the token budget.
        """
        summary = []
        for col in columns:
            entry: dict[str, Any] = {
                "name": col.name,
                "dtype": col.dtype,
                "null_pct": col.null_pct,
                "cardinality": col.cardinality,
            }
            if col.sample_values:
                entry["sample_values"] = col.sample_values[:5]
            if col.top_values:
                entry["top_values"] = col.top_values[:5]
            if col.min is not None:
                entry["min"] = col.min
            if col.max is not None:
                entry["max"] = col.max
            if col.detected_pattern:
                entry["detected_pattern"] = col.detected_pattern
            if col.date_range:
                entry["date_range"] = col.date_range
            summary.append(entry)

        return json.dumps(summary, default=str)

    # ── Response parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        # Layer 3: strip accidental markdown fences
        fence_match = _FENCE_RE.search(raw)
        cleaned = fence_match.group(1) if fence_match else raw.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMParseError(
                f"LLM returned non-JSON output. Raw response: {raw[:300]!r}"
            ) from exc

        if not isinstance(parsed, dict):
            raise LLMParseError(f"Expected a JSON object, got {type(parsed).__name__}.")

        return parsed
