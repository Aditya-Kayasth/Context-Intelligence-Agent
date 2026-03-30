"""Connector factory — maps a DataSource type to the correct connector class."""
from __future__ import annotations

from app.connectors.base import BaseConnector
from app.connectors.csv_connector import CSVConnector
from app.connectors.parquet_connector import ParquetConnector
from app.connectors.postgres_connector import PostgresConnector
from app.connectors.s3_connector import S3Connector
from app.connectors.blob_connector import BlobConnector
from app.models.sources import (
    AzureBlobSource,
    DataSource,
    DatabaseSource,
    LocalFileSource,
    S3Source,
)


def get_connector(source: DataSource) -> BaseConnector:
    """Return the appropriate connector instance for the given DataSource."""
    if isinstance(source, LocalFileSource):
        if source.format == "parquet":
            return ParquetConnector(source)
        return CSVConnector(source)
    if isinstance(source, S3Source):
        return S3Connector(source)
    if isinstance(source, DatabaseSource):
        return PostgresConnector(source)
    if isinstance(source, AzureBlobSource):
        return BlobConnector(source)
    raise NotImplementedError(
        f"No connector implemented for source type: {source.type!r}"  # type: ignore[union-attr]
    )
