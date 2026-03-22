"""
Unit tests for data connectors.
All filesystem and network I/O is mocked — no real files or services needed.
"""
from __future__ import annotations

import pytest

from app.connectors.base import ConnectorError
from app.connectors.csv_connector import CSVConnector
from app.models.sources import LocalFileSource


# ── CSVConnector ──────────────────────────────────────────────────────────────

class TestCSVConnector:
    def _make_source(self, path: str) -> LocalFileSource:
        return LocalFileSource(type="local_file", path=path, format="csv")

    @pytest.mark.asyncio
    async def test_connect_raises_on_missing_file(self):
        """connect() must raise ConnectorError when the file does not exist."""
        source = self._make_source("/nonexistent/path/data.csv")
        connector = CSVConnector(source)
        with pytest.raises(ConnectorError) as exc_info:
            await connector.connect()
        assert "local_file" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_sample_raises_on_missing_file(self):
        """sample() must propagate ConnectorError for a missing file."""
        source = self._make_source("/nonexistent/path/data.csv")
        connector = CSVConnector(source)
        with pytest.raises(ConnectorError):
            await connector.sample()

    @pytest.mark.asyncio
    async def test_sample_returns_dataframe(self, tmp_path):
        """sample() returns a DataFrame when the file exists and is valid CSV."""
        import pandas as pd

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")

        source = self._make_source(str(csv_file))
        connector = CSVConnector(source)
        df = await connector.sample()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b", "c"]
        assert len(df) == 3

    @pytest.mark.asyncio
    async def test_sample_all_rows_below_threshold(self, tmp_path):
        """Files with < 10 000 rows must be returned in full (no sampling)."""
        import pandas as pd

        n_rows = 500
        df_orig = pd.DataFrame({"x": range(n_rows), "y": range(n_rows)})
        csv_file = tmp_path / "small.csv"
        df_orig.to_csv(csv_file, index=False)

        source = self._make_source(str(csv_file))
        connector = CSVConnector(source)
        df = await connector.sample()

        assert len(df) == n_rows

    @pytest.mark.asyncio
    async def test_connector_error_contains_source_type(self):
        """ConnectorError must expose the source_type attribute."""
        source = self._make_source("/bad/path.csv")
        connector = CSVConnector(source)
        with pytest.raises(ConnectorError) as exc_info:
            await connector.connect()
        assert exc_info.value.source_type == "local_file"
