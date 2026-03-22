"""
PostgreSQL / generic SQL connector.

Sampling strategy (push-down first, Python fallback):
  1. If the query looks like a plain table name or simple SELECT, rewrite it to
     use TABLESAMPLE BERNOULLI(pct) so the DB does the heavy lifting.
  2. Otherwise wrap the caller's query in a sub-select with ORDER BY RANDOM()
     LIMIT <n> — still server-side, avoids pulling millions of rows.
  3. If neither rewrite is safe (e.g. the query already has LIMIT/TABLESAMPLE),
     execute as-is and apply Python-side smart_sample.

All SQLAlchemy calls run inside asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
import re
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from app.connectors.base import BaseConnector, ConnectorError
from app.config import settings
from app.models.sources import DatabaseSource
from app.utils.sampler import MAX_SAMPLE_ROWS, smart_sample

# Fraction used for TABLESAMPLE when total rows are unknown
_TABLESAMPLE_PCT = 5.0
_SIMPLE_TABLE_RE = re.compile(r"^\s*[\w\"\.\`]+\s*$")  # bare table name
_HAS_LIMIT_RE = re.compile(r"\bLIMIT\b", re.IGNORECASE)
_HAS_TABLESAMPLE_RE = re.compile(r"\bTABLESAMPLE\b", re.IGNORECASE)


class PostgresConnector(BaseConnector):
    def __init__(self, source: DatabaseSource) -> None:
        super().__init__(source)
        self._source: DatabaseSource = source
        self._engine = None

    async def connect(self) -> None:
        self._engine = await asyncio.to_thread(self._make_engine)

    def _make_engine(self):
        url = self._source.database_url or settings.database_url
        return create_engine(url, pool_pre_ping=True)

    async def sample(self, target_col: Optional[str] = None) -> pd.DataFrame:
        if self._engine is None:
            await self.connect()
        try:
            df = await asyncio.to_thread(self._read_with_pushdown)
            # Apply Python-side sampler only if push-down wasn't used
            return await smart_sample(df, target_col)
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError("database", str(exc)) from exc

    def _read_with_pushdown(self) -> pd.DataFrame:
        query = self._source.query.strip().rstrip(";")

        # Case 1: bare table name → TABLESAMPLE
        if _SIMPLE_TABLE_RE.match(query):
            sql = (
                f"SELECT * FROM {query} "
                f"TABLESAMPLE BERNOULLI({_TABLESAMPLE_PCT}) "
                f"LIMIT {MAX_SAMPLE_ROWS}"
            )
            return self._execute(sql)

        # Case 2: query already has LIMIT or TABLESAMPLE → run as-is
        if _HAS_LIMIT_RE.search(query) or _HAS_TABLESAMPLE_RE.search(query):
            return self._execute(query)

        # Case 3: arbitrary SELECT → wrap with ORDER BY RANDOM() LIMIT
        sql = (
            f"SELECT * FROM ({query}) AS _cia_sub "
            f"ORDER BY RANDOM() "
            f"LIMIT {MAX_SAMPLE_ROWS}"
        )
        return self._execute(sql)

    def _execute(self, sql: str) -> pd.DataFrame:
        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
