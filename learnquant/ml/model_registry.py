from __future__ import annotations

import json
from pathlib import Path

import joblib


class ModelRegistry:
    def __init__(self, base_dir: str | Path = "learnquant/runs") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def register(self, run_dir: Path, model, metrics: dict) -> tuple[Path, Path]:
        model_path = run_dir / "model.pkl"
        metrics_path = run_dir / "metrics.json"
        joblib.dump(model, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        return model_path, metrics_path
