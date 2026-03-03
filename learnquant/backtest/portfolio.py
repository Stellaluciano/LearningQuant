from __future__ import annotations

import numpy as np
import pandas as pd


def _periods_per_year(frequency: str | None, timestamps: pd.Series | None = None) -> int | None:
    mapping = {
        "minutely": 252 * 390,
        "minute": 252 * 390,
        "hourly": 252 * 6.5,
        "hour": 252 * 6.5,
        "daily": 252,
        "day": 252,
        "weekly": 52,
        "month": 12,
        "monthly": 12,
    }
    if frequency:
        value = mapping.get(str(frequency).lower())
        if value:
            return int(value)
    if timestamps is None or len(timestamps) < 3:
        return None

    ts = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    if len(ts) < 3:
        return None

    median_delta_seconds = ts.diff().dropna().dt.total_seconds().median()
    if median_delta_seconds <= 0:
        return None
    seconds_per_year = 365.25 * 24 * 3600
    return max(int(seconds_per_year / median_delta_seconds), 1)


def equity_curve(
    trades: pd.DataFrame,
    initial_capital: float = 100_000.0,
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    if symbol_col in trades.columns:
        rets = (
            trades.groupby("timestamp", as_index=False)
            .agg(strategy_ret=("strategy_ret", "mean"), turnover=("turnover", "sum"))
            .sort_values("timestamp")
        )
    else:
        rets = trades[["timestamp", "strategy_ret", "turnover"]].copy().sort_values("timestamp")

    rets["equity"] = initial_capital * (1.0 + rets["strategy_ret"]).cumprod()
    rets["peak"] = rets["equity"].cummax()
    rets["drawdown"] = (rets["equity"] - rets["peak"]) / rets["peak"]
    return rets


def performance_metrics(trades: pd.DataFrame, curve: pd.DataFrame, frequency: str | None = None) -> dict:
    rets = curve["strategy_ret"]
    periods = _periods_per_year(frequency, curve.get("timestamp"))

    mean_ret = rets.mean()
    vol = rets.std()
    downside = rets[rets < 0].std()
    sharpe = 0.0
    sortino = 0.0
    annual_return = 0.0

    if periods and periods > 0:
        sharpe = 0.0 if vol == 0 else np.sqrt(periods) * mean_ret / vol
        sortino = 0.0 if downside == 0 or np.isnan(downside) else np.sqrt(periods) * mean_ret / downside
        annual_return = (1.0 + mean_ret) ** periods - 1.0
    elif vol != 0:
        sharpe = mean_ret / vol
        sortino = 0.0 if downside == 0 or np.isnan(downside) else mean_ret / downside

    pnl = rets
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    if gross_loss == 0:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    wins = (rets > 0).sum()
    losses = (rets < 0).sum()
    win_rate = 0.0 if (wins + losses) == 0 else wins / (wins + losses)
    starting_equity = curve["equity"].iloc[0] if not curve.empty else 1.0
    ending_equity = curve["equity"].iloc[-1] if not curve.empty else starting_equity

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "profit_factor": float(profit_factor),
        "annualized_return": float(annual_return),
        "max_drawdown": float(curve["drawdown"].min()) if not curve.empty else 0.0,
        "win_rate": float(win_rate),
        "turnover": float(trades["turnover"].mean()) if not trades.empty else 0.0,
        "total_return": float(ending_equity / starting_equity - 1.0) if starting_equity != 0 else 0.0,
    }
