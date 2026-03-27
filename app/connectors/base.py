"""Abstract base connector — all concrete connectors inherit from BaseConnector."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from app.models.sources import DataSource


class ConnectorError(Exception):
    """Raised when a connector fails to read data."""

    def __init__(self, source_type: str, message: str) -> None:
        self.source_type = source_type
        super().__init__(f"[{source_type}] {message}")


class BaseConnector(ABC):
    """Contract that every data-source connector must satisfy."""

    def __init__(self, source: DataSource) -> None:
        self.source = source

    @abstractmethod
    async def connect(self) -> None:
        """Establish any necessary client connections / auth."""

    @abstractmethod
    async def sample(self, target_col: Optional[str] = None) -> pd.DataFrame:
        """Return a sampled DataFrame using the smart sampler."""

    def _assert_file_exists(self, path: str, source_type: str = "local_file") -> None:
        """Raise ConnectorError if path does not exist on disk."""
        if not os.path.exists(path):
            raise ConnectorError(source_type, f"File not found: {path}")
