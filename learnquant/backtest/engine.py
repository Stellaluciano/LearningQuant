from __future__ import annotations

from typing import Any

import pandas as pd

from learnquant.backtest.execution import simulate_trades
from learnquant.backtest.portfolio import equity_curve, performance_metrics


def default_strategy(features: pd.DataFrame) -> pd.Series:
    return (features["momentum_10"] > 0).astype(float) - (features["momentum_10"] < 0).astype(float)


class BacktestEngine:
    def run(self, features: pd.DataFrame, strategy_fn=default_strategy, config: dict[str, Any] | None = None) -> dict:
        cfg = config or {}
        initial_capital = float(cfg.get("initial_capital", 100_000.0))
        frequency = cfg.get("frequency")
        fee = float(cfg.get("fees", 0.0))
        slippage = cfg.get("slippage", {})
        risk = cfg.get("risk", {})

        data = features.copy()
        data["signal"] = strategy_fn(data)
        trades = simulate_trades(
            data,
            fee=fee,
            slippage_bps=float(slippage.get("fixed_bps", 0.0)),
            slippage_vol_multiplier=float(slippage.get("vol_multiplier", 0.0)),
            max_position=float(risk.get("max_position", 1.0)),
            stop_loss=risk.get("stop_loss"),
            take_profit=risk.get("take_profit"),
            max_drawdown=risk.get("max_drawdown"),
        )
        curve = equity_curve(trades, initial_capital=initial_capital)
        metrics = performance_metrics(trades, curve, frequency=frequency)
        return {"trades": trades, "equity_curve": curve, "metrics": metrics}
