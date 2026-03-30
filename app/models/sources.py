"""Discriminated-union DataSource model for all supported connector types."""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class LocalFileSource(BaseModel):
    """Source descriptor for a local CSV, Parquet, or JSON file."""

    type: Literal["local_file"] = "local_file"
    path: str
    format: Literal["csv", "parquet", "json"] = "csv"


class S3Source(BaseModel):
    """Source descriptor for an AWS S3 object or glob pattern."""

    type: Literal["s3"] = "s3"
    bucket: str
    key: str
    region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


class GCSSource(BaseModel):
    """Source descriptor for a Google Cloud Storage blob."""

    type: Literal["gcs"] = "gcs"
    bucket: str
    blob: str
    credentials_path: Optional[str] = None


class SFTPSource(BaseModel):
    """Source descriptor for a remote file accessible via SFTP."""

    type: Literal["sftp"] = "sftp"
    host: str
    port: int = 22
    username: str
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    remote_path: str


class DatabaseSource(BaseModel):
    """Source descriptor for a SQL database query."""

    type: Literal["database"] = "database"
    database_url: Optional[str] = None
    query: str


class KafkaSource(BaseModel):
    """Source descriptor for a Kafka topic."""

    type: Literal["kafka"] = "kafka"
    bootstrap_servers: str
    topic: str
    group_id: str = "cia-consumer"
    max_messages: int = 1000


class APISource(BaseModel):
    """Source descriptor for an HTTP API endpoint."""

    type: Literal["api"] = "api"
    url: str
    method: Literal["GET", "POST"] = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    body: Optional[dict] = None


class AzureBlobSource(BaseModel):
    """Source descriptor for an Azure Blob Storage blob."""

    type: Literal["azure_blob"] = "azure_blob"
    path: str  # URL to the blob
    format: Literal["csv", "parquet", "json"] = "csv"
    connection_string: Optional[str] = None


DataSource = Annotated[
    Union[
        LocalFileSource,
        S3Source,
        GCSSource,
        SFTPSource,
        DatabaseSource,
        KafkaSource,
        APISource,
        AzureBlobSource,
    ],
    Field(discriminator="type"),
]
