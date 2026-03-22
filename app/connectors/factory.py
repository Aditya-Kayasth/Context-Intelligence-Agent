"""
Connector factory — maps a DataSource discriminator to the right connector class.
"""
from __future__ import annotations

from app.connectors.base import BaseConnector
from app.models.sources import DataSource


def get_connector(source: DataSource) -> BaseConnector:
    source_type = source.type  # type: ignore[union-attr]

    if source_type == "local_file":
        from app.models.sources import LocalFileSource
        assert isinstance(source, LocalFileSource)
        if source.format == "parquet":
            from app.connectors.parquet_connector import ParquetConnector
            return ParquetConnector(source)
        from app.connectors.csv_connector import CSVConnector
        return CSVConnector(source)

    if source_type == "s3":
        from app.connectors.s3_connector import S3Connector
        return S3Connector(source)  # type: ignore[arg-type]

    if source_type == "database":
        from app.connectors.postgres_connector import PostgresConnector
        return PostgresConnector(source)  # type: ignore[arg-type]

    raise NotImplementedError(f"No connector implemented for source type: {source_type!r}")
