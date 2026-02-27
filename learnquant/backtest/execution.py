from __future__ import annotations

import pandas as pd


def simulate_trades(features: pd.DataFrame, signal_col: str = "signal") -> pd.DataFrame:
    trades = features[["timestamp", "close", signal_col]].copy()
    trades["position"] = trades[signal_col].shift(1).fillna(0.0)
    trades["asset_ret"] = features["close"].pct_change().fillna(0.0)
    trades["strategy_ret"] = trades["position"] * trades["asset_ret"]
    trades["turnover"] = trades["position"].diff().abs().fillna(0.0)
    return trades
