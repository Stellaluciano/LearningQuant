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


## Backtest configuration

`learnquant/config/default.yaml` drives backtest behavior end-to-end via `/run_backtest` and `/run_train`:

- `backtest.initial_capital`: starting equity.
- `backtest.frequency`: annualization basis (`minute`, `hour`, `daily`, `weekly`, `monthly`). If omitted, metrics infer frequency from timestamps.
- `backtest.fees`: transaction fee per unit turnover (`abs(position_t-position_{t-1}) * fee`).
- `backtest.slippage.fixed_bps`: fixed slippage in basis points applied on turnover.
- `backtest.slippage.vol_multiplier`: dynamic slippage coefficient multiplied by rolling volatility.
- `backtest.risk.max_position`: hard cap for long/short position magnitude.
- `backtest.risk.stop_loss`, `take_profit`: optional single-trade exit thresholds.
- `backtest.risk.max_drawdown`: optional strategy-level kill switch.

Backtest artifacts are always written to `learnquant/runs/<run_id>/` as `metrics.json`, `trade_logs.csv`, `equity_curve.csv`, and `report.html`.

## Example assets
- Example CSV: `learnquant/data/raw/example_market.csv`
- Example strategy: `learnquant/examples/strategy.py`
- Example training script: `learnquant/examples/example_train.py`

## Notes
- Input CSV is validated and converted to parquet in `learnquant/data/processed/`.
- Feature parquet files are stored in `learnquant/data/feature_store/`.
- Backtest and training both consume the same feature parquet dataset.
