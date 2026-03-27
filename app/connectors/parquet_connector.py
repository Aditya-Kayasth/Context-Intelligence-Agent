"""Local Parquet connector — uses PyArrow row-group sampling for large files."""
from __future__ import annotations

import asyncio
import math
import random
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.connectors.base import BaseConnector, ConnectorError
from app.models.sources import LocalFileSource
from app.utils.sampler import MAX_SAMPLE_ROWS, smart_sample

_MEDIUM_THRESHOLD = 1_000_000


class ParquetConnector(BaseConnector):
    """Read and adaptively sample a local Parquet file without blocking the event loop."""

    def __init__(self, source: LocalFileSource) -> None:
        super().__init__(source)
        self._source: LocalFileSource = source

    async def connect(self) -> None:
        """Verify the file exists; raise ConnectorError if not."""
        self._assert_file_exists(self._source.path)

    async def sample(self, target_col: Optional[str] = None) -> pd.DataFrame:
        """Return a sampled DataFrame from the Parquet file."""
        await self.connect()
        try:
            df = await asyncio.to_thread(self._read_with_sampling)
            return await smart_sample(df, target_col)
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError("local_file", str(exc)) from exc

    def _read_with_sampling(self) -> pd.DataFrame:
        """Read the file, sampling row-groups proportionally for large files."""
        pf = pq.ParquetFile(self._source.path)
        total_rows = pf.metadata.num_rows

        if total_rows <= _MEDIUM_THRESHOLD:
            return pf.read().to_pandas()

        num_groups = pf.metadata.num_row_groups
        target_groups = max(1, math.ceil(num_groups * (MAX_SAMPLE_ROWS / total_rows)))
        selected = random.sample(range(num_groups), min(target_groups, num_groups))
        tables = [pf.read_row_group(i) for i in sorted(selected)]
        return pa.concat_tables(tables).to_pandas()
