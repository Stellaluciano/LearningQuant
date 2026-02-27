from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from learnquant.data.schema import MARKET_SCHEMA


class DataLoader:
    def __init__(self, base_dir: str | Path = "learnquant/data") -> None:
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def validate_schema(self, df: pd.DataFrame) -> None:
        missing = MARKET_SCHEMA.missing_columns(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def csv_to_parquet(self, csv_path: str | Path, dataset_name: str) -> Path:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        self.validate_schema(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        parquet_path = self.processed_dir / f"{dataset_name}.parquet"
        pl.from_pandas(df).write_parquet(parquet_path)
        return parquet_path

    def load_processed(self, dataset_name: str) -> pd.DataFrame:
        parquet_path = self.processed_dir / f"{dataset_name}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {parquet_path}")
        return pl.read_parquet(parquet_path).to_pandas()
