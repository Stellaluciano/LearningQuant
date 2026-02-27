from __future__ import annotations

import pandas as pd

from learnquant.backtest.execution import simulate_trades
from learnquant.backtest.portfolio import equity_curve, performance_metrics


def default_strategy(features: pd.DataFrame) -> pd.Series:
    return (features["momentum_10"] > 0).astype(float) - (features["momentum_10"] < 0).astype(float)


class BacktestEngine:
    def run(self, features: pd.DataFrame, strategy_fn=default_strategy) -> dict:
        data = features.copy()
        data["signal"] = strategy_fn(data)
        trades = simulate_trades(data)
        curve = equity_curve(trades)
        metrics = performance_metrics(trades, curve)
        return {"trades": trades, "equity_curve": curve, "metrics": metrics}
