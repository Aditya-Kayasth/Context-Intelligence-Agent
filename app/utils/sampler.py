"""
Adaptive smart sampler.

Thresholds
----------
< 10 000 rows          → load everything
10 000 – 1 000 000     → 5 % random sample (or stratified if target_col given)
> 1 000 000            → reservoir sample capped at MAX_SAMPLE_ROWS

All public entry-points are async; blocking pandas/numpy work is offloaded to a
thread pool via asyncio.to_thread so the FastAPI event loop stays free.
"""
from __future__ import annotations

import asyncio
import math
import random
from typing import Optional

import pandas as pd

MAX_SAMPLE_ROWS: int = 50_000

_SMALL_THRESHOLD: int = 10_000
_MEDIUM_THRESHOLD: int = 1_000_000
_MEDIUM_FRAC: float = 0.05


# ── Public async API ──────────────────────────────────────────────────────────

async def smart_sample(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    max_sample_rows: int = MAX_SAMPLE_ROWS,
) -> pd.DataFrame:
    """Return a sampled DataFrame according to the adaptive thresholds."""
    return await asyncio.to_thread(_sample_sync, df, target_col, max_sample_rows)


async def reservoir_sample_iter(
    iterator,  # iterable of pd.DataFrame chunks
    max_rows: int = MAX_SAMPLE_ROWS,
) -> pd.DataFrame:
    """
    Reservoir-sample from a chunk iterator without loading everything into RAM.
    Useful for very large files read in chunks.
    """
    return await asyncio.to_thread(_reservoir_from_iter, iterator, max_rows)


# ── Synchronous implementations (run inside to_thread) ───────────────────────

def _sample_sync(
    df: pd.DataFrame,
    target_col: Optional[str],
    max_sample_rows: int,
) -> pd.DataFrame:
    n = len(df)

    if n < _SMALL_THRESHOLD:
        return df

    if n <= _MEDIUM_THRESHOLD:
        n_sample = max(1, math.ceil(n * _MEDIUM_FRAC))
        n_sample = min(n_sample, max_sample_rows)
        if target_col and target_col in df.columns:
            return _stratified_sample(df, target_col, n_sample)
        return df.sample(n=n_sample, random_state=42)

    # > 1 000 000 rows — reservoir sample
    return _reservoir_from_df(df, max_sample_rows)


def _stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    n_sample: int,
) -> pd.DataFrame:
    """Proportional stratified sample by target_col."""
    groups = df.groupby(target_col, group_keys=False)
    frac = n_sample / len(df)
    sampled = groups.apply(
        lambda g: g.sample(n=max(1, math.ceil(len(g) * frac)), random_state=42)
    )
    return sampled.reset_index(drop=True).head(n_sample)


def _reservoir_from_df(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Vitter's Algorithm R over a DataFrame."""
    n = len(df)
    if n <= k:
        return df
    indices = list(range(k))
    for i in range(k, n):
        j = random.randint(0, i)
        if j < k:
            indices[j] = i
    return df.iloc[sorted(indices)].reset_index(drop=True)


def _reservoir_from_iter(iterator, k: int) -> pd.DataFrame:
    """Reservoir sample from an iterable of DataFrame chunks."""
    reservoir: list[pd.Series] = []
    count = 0
    for chunk in iterator:
        for _, row in chunk.iterrows():
            count += 1
            if len(reservoir) < k:
                reservoir.append(row)
            else:
                j = random.randint(0, count - 1)
                if j < k:
                    reservoir[j] = row
    if not reservoir:
        return pd.DataFrame()
    return pd.DataFrame(reservoir).reset_index(drop=True)
