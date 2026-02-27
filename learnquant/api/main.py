from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile

from learnquant.backtest.engine import BacktestEngine
from learnquant.data.loader import DataLoader
from learnquant.experiments.run_manager import RunManager
from learnquant.features.feature import generate_features
from learnquant.ml.model_registry import ModelRegistry
from learnquant.ml.train import train
from learnquant.reports.report_generator import generate_report

ROOT = Path("learnquant")
CONFIG_PATH = ROOT / "config" / "default.yaml"
FEATURE_STORE = ROOT / "data" / "feature_store"
FEATURE_STORE.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LearnQuant API")
loader = DataLoader(ROOT / "data")
run_manager = RunManager(ROOT / "runs")
backtest_engine = BacktestEngine()
registry = ModelRegistry(ROOT / "runs")


def load_default_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _feature_path(dataset_name: str) -> Path:
    return FEATURE_STORE / f"{dataset_name}_features.parquet"


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), dataset_name: str = "market"):
    raw_path = loader.raw_dir / file.filename
    raw_path.write_bytes(await file.read())
    parquet_path = loader.csv_to_parquet(raw_path, dataset_name)
    return {"dataset": dataset_name, "raw_path": str(raw_path), "parquet_path": str(parquet_path)}


@app.post("/generate_features")
def run_generate_features(dataset_name: str = "market"):
    df = loader.load_processed(dataset_name)
    feat_df = generate_features(df)
    out_path = _feature_path(dataset_name)
    pl.from_pandas(feat_df).write_parquet(out_path)
    return {"dataset": dataset_name, "feature_path": str(out_path), "rows": len(feat_df)}


@app.post("/run_backtest")
def run_backtest(dataset_name: str = "market"):
    feature_path = _feature_path(dataset_name)
    if not feature_path.exists():
        raise HTTPException(404, f"Feature dataset not found. Run /generate_features first for {dataset_name}")

    features = pl.read_parquet(feature_path).to_pandas()
    config = load_default_config()
    run_id, run_dir = run_manager.create_run({**config, "task": "backtest", "dataset": dataset_name})

    bt = backtest_engine.run(features)
    bt_metrics = bt["metrics"]
    bt["trades"].to_csv(run_dir / "trade_logs.csv", index=False)
    bt["equity_curve"].to_csv(run_dir / "equity_curve.csv", index=False)
    run_manager.log_metrics(run_dir, {"backtest": bt_metrics})

    report = generate_report(run_dir, bt["equity_curve"], bt_metrics)
    return {"run_id": run_id, "metrics": bt_metrics, "report": str(report)}


@app.post("/run_train")
def run_train(dataset_name: str = "market", model_type: str = "sklearn"):
    feature_path = _feature_path(dataset_name)
    if not feature_path.exists():
        raise HTTPException(404, f"Feature dataset not found. Run /generate_features first for {dataset_name}")

    features = pl.read_parquet(feature_path).to_pandas()
    config = load_default_config()
    run_id, run_dir = run_manager.create_run(
        {**config, "task": "train", "dataset": dataset_name, "model_type": model_type}
    )

    result = train(features, model_type=model_type)
    registry.register(run_dir, result.model, {"ml": result.metrics})

    bt = backtest_engine.run(features)
    report = generate_report(
        run_dir,
        bt["equity_curve"],
        bt["metrics"],
        ml_metrics=result.metrics,
        feature_importance=result.feature_importance,
    )

    return {
        "run_id": run_id,
        "metrics": result.metrics,
        "feature_importance": dict(sorted(result.feature_importance.items())[:10]),
        "report": str(report),
    }


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    try:
        run = run_manager.get_run(run_id)
        metrics_path = Path(run["artifacts"].get("metrics.json", ""))
        if metrics_path.exists():
            run["metrics"] = json.loads(metrics_path.read_text())
        return run
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("learnquant.api.main:app", host="0.0.0.0", port=8000, reload=False)
