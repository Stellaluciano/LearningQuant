from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["bid_volume", "ask_volume", "symbol"]


@dataclass(frozen=True)
class MarketSchema:
    required: Iterable[str] = tuple(REQUIRED_COLUMNS)
    optional: Iterable[str] = tuple(OPTIONAL_COLUMNS)

    def missing_columns(self, columns: Iterable[str]) -> list[str]:
        existing = set(columns)
        return [col for col in self.required if col not in existing]


MARKET_SCHEMA = MarketSchema()
