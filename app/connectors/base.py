"""
Abstract base connector.

Every concrete connector must:
  1. Accept the matching DataSource model in __init__.
  2. Implement connect() to set up any client/session.
  3. Implement sample() to return a sampled pandas DataFrame via the smart sampler.
"""
from __future__ import annotations

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
    def __init__(self, source: DataSource) -> None:
        self.source = source

    @abstractmethod
    async def connect(self) -> None:
        """Establish any necessary client connections / auth."""

    @abstractmethod
    async def sample(self, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Return a sampled DataFrame using the smart sampler.
        Implementations must call connect() before reading data.
        """
