"""Azure Blob Storage connector — downloads blob via URL."""
from __future__ import annotations

import asyncio
import io
from typing import Optional

import httpx
import pandas as pd

from app.connectors.base import BaseConnector, ConnectorError
from app.models.sources import AzureBlobSource
from app.utils.sampler import smart_sample


class BlobConnector(BaseConnector):
    """Read and adaptively sample an Azure Blob Storage object via URL."""

    def __init__(self, source: AzureBlobSource) -> None:
        super().__init__(source)
        self._source: AzureBlobSource = source

    async def connect(self) -> None:
        """Verify the URL is accessible."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.head(self._source.path)
                if resp.status_code >= 400:
                    raise ConnectorError("azure_blob", f"Blob URL not accessible: {resp.status_code}")
        except Exception as exc:
            raise ConnectorError("azure_blob", str(exc)) from exc

    async def sample(self, target_col: Optional[str] = None) -> pd.DataFrame:
        """Download and return a sampled DataFrame from the blob."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(self._source.path)
                if resp.status_code != 200:
                    raise ConnectorError("azure_blob", f"Failed to download blob: {resp.status_code}")
                
                content = resp.content
                buf = io.BytesIO(content)
                
                def _read():
                    if self._source.format == "parquet":
                        return pd.read_parquet(buf)
                    if self._source.format == "json":
                        return pd.read_json(buf)
                    return pd.read_csv(buf)

                df = await asyncio.to_thread(_read)
                return await smart_sample(df, target_col)
        except Exception as exc:
            raise ConnectorError("azure_blob", str(exc)) from exc
