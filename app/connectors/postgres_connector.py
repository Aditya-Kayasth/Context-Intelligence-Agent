"""PostgreSQL / generic SQL connector with server-side sampling push-down."""
from __future__ import annotations

import asyncio
import re
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from app.config import settings
from app.connectors.base import BaseConnector, ConnectorError
from app.models.sources import DatabaseSource
from app.utils.sampler import MAX_SAMPLE_ROWS, smart_sample

_TABLESAMPLE_PCT = 5.0
_SIMPLE_TABLE_RE = re.compile(r"^\s*[\w\".\`]+\s*$")
_HAS_LIMIT_RE = re.compile(r"\bLIMIT\b", re.IGNORECASE)
_HAS_TABLESAMPLE_RE = re.compile(r"\bTABLESAMPLE\b", re.IGNORECASE)


class PostgresConnector(BaseConnector):
    """SQL connector that pushes sampling down to the database where possible."""

    def __init__(self, source: DatabaseSource) -> None:
        super().__init__(source)
        self._source: DatabaseSource = source
        self._engine = None

    async def connect(self) -> None:
        """Create the SQLAlchemy engine."""
        self._engine = await asyncio.to_thread(self._make_engine)

    def _make_engine(self):
        """Build a SQLAlchemy engine from the configured database URL."""
        url = self._source.database_url or settings.database_url
        return create_engine(url, pool_pre_ping=True)

    async def sample(self, target_col: Optional[str] = None) -> pd.DataFrame:
        """Return a sampled DataFrame, preferring server-side sampling."""
        if self._engine is None:
            await self.connect()
        try:
            df = await asyncio.to_thread(self._read_with_pushdown)
            return await smart_sample(df, target_col)
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError("database", str(exc)) from exc

    def _read_with_pushdown(self) -> pd.DataFrame:
        """Rewrite the query to use TABLESAMPLE or ORDER BY RANDOM() LIMIT."""
        query = self._source.query.strip().rstrip(";")
        if _SIMPLE_TABLE_RE.match(query):
            sql = (
                f"SELECT * FROM {query} "
                f"TABLESAMPLE BERNOULLI({_TABLESAMPLE_PCT}) "
                f"LIMIT {MAX_SAMPLE_ROWS}"
            )
            return self._execute(sql)
        if _HAS_LIMIT_RE.search(query) or _HAS_TABLESAMPLE_RE.search(query):
            return self._execute(query)
        sql = (
            f"SELECT * FROM ({query}) AS _cia_sub "
            f"ORDER BY RANDOM() "
            f"LIMIT {MAX_SAMPLE_ROWS}"
        )
        return self._execute(sql)

    def _execute(self, sql: str) -> pd.DataFrame:
        """Execute a SQL string and return the result as a DataFrame."""
        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
