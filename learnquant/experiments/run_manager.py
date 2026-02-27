from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class RunManager:
    def __init__(self, runs_root: str | Path = "learnquant/runs") -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def create_run(self, config: dict[str, Any]) -> tuple[str, Path]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        return run_id, run_dir

    def log_metrics(self, run_dir: Path, metrics: dict[str, Any]) -> Path:
        path = run_dir / "metrics.json"
        path.write_text(json.dumps(metrics, indent=2))
        return path

    def get_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.runs_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        payload = {"run_id": run_id, "artifacts": {}}
        for item in run_dir.iterdir():
            payload["artifacts"][item.name] = str(item)
        return payload
