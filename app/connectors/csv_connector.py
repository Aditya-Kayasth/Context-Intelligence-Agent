"""
Local CSV connector.

Blocking pandas I/O is wrapped in asyncio.to_thread so the FastAPI event loop
is never stalled during large file reads.

For files that may exceed 1 M rows the connector reads in chunks and feeds them
through the reservoir sampler so memory stays bounded.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import pandas as pd

from app.connectors.base import BaseConnector, ConnectorError
from app.models.sources import LocalFileSource
from app.utils.sampler import MAX_SAMPLE_ROWS, reservoir_sample_iter, smart_sample

_CHUNK_SIZE = 100_000  # rows per chunk when streaming large files


class CSVConnector(BaseConnector):
    def __init__(self, source: LocalFileSource) -> None:
        super().__init__(source)
        self._source: LocalFileSource = source

    async def connect(self) -> None:
        # No persistent connection needed for local files; validate path exists.
        import os
        if not os.path.exists(self._source.path):
            raise ConnectorError("local_file", f"File not found: {self._source.path}")

    async def sample(self, target_col: Optional[str] = None) -> pd.DataFrame:
        await self.connect()
        try:
            # Peek at row count without loading the whole file
            row_count = await asyncio.to_thread(self._count_rows)

            if row_count > 1_000_000:
                # Stream in chunks → reservoir sample (stays off the event loop)
                iterator = await asyncio.to_thread(
                    lambda: pd.read_csv(self._source.path, chunksize=_CHUNK_SIZE)
                )
                return await reservoir_sample_iter(iterator, MAX_SAMPLE_ROWS)

            df = await asyncio.to_thread(pd.read_csv, self._source.path)
            return await smart_sample(df, target_col)

        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError("local_file", str(exc)) from exc

    def _count_rows(self) -> int:
        """Fast row count via line iteration (excludes header)."""
        with open(self._source.path, "rb") as f:
            return sum(1 for _ in f) - 1
