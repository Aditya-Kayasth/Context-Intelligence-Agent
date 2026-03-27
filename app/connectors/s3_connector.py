"""AWS S3 connector — supports single keys and glob patterns."""
from __future__ import annotations

import asyncio
import fnmatch
import io
from typing import Optional

import boto3
import pandas as pd

from app.connectors.base import BaseConnector, ConnectorError
from app.models.sources import S3Source
from app.utils.sampler import smart_sample


class S3Connector(BaseConnector):
    """Read and adaptively sample one or more S3 objects (glob patterns supported)."""

    def __init__(self, source: S3Source) -> None:
        super().__init__(source)
        self._source: S3Source = source
        self._client = None

    async def connect(self) -> None:
        """Create the boto3 S3 client."""
        self._client = await asyncio.to_thread(self._make_client)

    def _make_client(self):
        """Build a boto3 S3 client, optionally using explicit credentials."""
        kwargs: dict = {"region_name": self._source.region}
        if self._source.aws_access_key_id:
            kwargs["aws_access_key_id"] = self._source.aws_access_key_id
            kwargs["aws_secret_access_key"] = self._source.aws_secret_access_key
        return boto3.client("s3", **kwargs)

    async def sample(self, target_col: Optional[str] = None) -> pd.DataFrame:
        """Return a sampled DataFrame from the resolved S3 object(s)."""
        if self._client is None:
            await self.connect()
        try:
            keys = await asyncio.to_thread(self._resolve_keys)
            if not keys:
                raise ConnectorError(
                    "s3",
                    f"No objects matched: s3://{self._source.bucket}/{self._source.key}",
                )
            frames = [await asyncio.to_thread(self._read_key, k) for k in keys]
            combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            return await smart_sample(combined, target_col)
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError("s3", str(exc)) from exc

    def _resolve_keys(self) -> list[str]:
        """Expand glob patterns via list_objects_v2; return exact key otherwise."""
        pattern = self._source.key
        if not any(c in pattern for c in ("*", "?", "[")):
            return [pattern]
        prefix = pattern.split("*")[0].split("?")[0].split("[")[0]
        paginator = self._client.get_paginator("list_objects_v2")
        matched = []
        for page in paginator.paginate(Bucket=self._source.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if fnmatch.fnmatch(obj["Key"], pattern):
                    matched.append(obj["Key"])
        return matched

    def _read_key(self, key: str) -> pd.DataFrame:
        """Download and parse a single S3 object into a DataFrame."""
        resp = self._client.get_object(Bucket=self._source.bucket, Key=key)
        buf = io.BytesIO(resp["Body"].read())
        if key.endswith(".parquet"):
            return pd.read_parquet(buf)
        if key.endswith(".json"):
            return pd.read_json(buf)
        return pd.read_csv(buf)
