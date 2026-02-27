# LearnQuant

A lightweight, modular research platform for quant analysts to ingest CSV market data, build features, run backtests, and train ML models with reproducible run artifacts.

## Why this design
- **Simplicity**: clear modules and small interfaces.
- **Modularity**: data, feature, backtest, ML, runs, and reporting are separated.
- **Reproducibility**: every run writes `config.yaml`, `metrics.json`, model artifact, and report under `learnquant/runs/<run_id>/`.
- **Analyst-first customization**: analysts only need to modify:
  - `learnquant/features/feature.py`
  - `learnquant/ml/train.py`

## Project layout

```text
learnquant/
  api/main.py
  data/loader.py
  data/schema.py
  features/feature.py
  backtest/engine.py
  backtest/portfolio.py
  backtest/execution.py
  ml/train.py
  ml/model_registry.py
  experiments/run_manager.py
  reports/report_generator.py
  storage/artifact_store.py
  config/default.yaml
```

## Quickstart

```bash
pip install -r requirements.txt
python api/main.py
```

Open API docs at: http://localhost:8000/docs

## Example flow

1. Upload CSV:
```bash
curl -X POST "http://localhost:8000/upload_csv?dataset_name=example_market" \
  -F "file=@learnquant/data/raw/example_market.csv"
```

2. Generate features:
```bash
curl -X POST "http://localhost:8000/generate_features?dataset_name=example_market"
```

3. Run backtest:
```bash
curl -X POST "http://localhost:8000/run_backtest?dataset_name=example_market"
```

4. Train model:
```bash
curl -X POST "http://localhost:8000/run_train?dataset_name=example_market&model_type=sklearn"
```

5. Inspect run:
```bash
curl "http://localhost:8000/runs/<run_id>"
```

## Example assets
- Example CSV: `learnquant/data/raw/example_market.csv`
- Example strategy: `learnquant/examples/strategy.py`
- Example training script: `learnquant/examples/example_train.py`

## Notes
- Input CSV is validated and converted to parquet in `learnquant/data/processed/`.
- Feature parquet files are stored in `learnquant/data/feature_store/`.
- Backtest and training both consume the same feature parquet dataset.
