from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(trades: pd.DataFrame, initial_capital: float = 100_000.0) -> pd.DataFrame:
    curve = trades[["timestamp", "strategy_ret"]].copy()
    curve["equity"] = initial_capital * (1.0 + curve["strategy_ret"]).cumprod()
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = (curve["equity"] - curve["peak"]) / curve["peak"]
    return curve


def performance_metrics(trades: pd.DataFrame, curve: pd.DataFrame) -> dict:
    rets = trades["strategy_ret"]
    sharpe = 0.0 if rets.std() == 0 else np.sqrt(252) * rets.mean() / rets.std()
    wins = (rets > 0).sum()
    losses = (rets < 0).sum()
    win_rate = 0.0 if (wins + losses) == 0 else wins / (wins + losses)
    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(curve["drawdown"].min()),
        "win_rate": float(win_rate),
        "turnover": float(trades["turnover"].mean()),
        "total_return": float(curve["equity"].iloc[-1] / curve["equity"].iloc[0] - 1.0),
    }
