from __future__ import annotations

import pandas as pd


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Analysts can edit this function to define custom features."""
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    out["return_1"] = out["close"].pct_change()
    out["volatility_20"] = out["return_1"].rolling(20).std()
    out["momentum_10"] = out["close"] / out["close"].shift(10) - 1.0

    if {"bid_volume", "ask_volume"}.issubset(set(out.columns)):
        denom = (out["bid_volume"] + out["ask_volume"]).replace(0, pd.NA)
        out["order_book_imbalance"] = (out["bid_volume"] - out["ask_volume"]) / denom
    else:
        out["order_book_imbalance"] = 0.0

    out["target"] = out["return_1"].shift(-1)
    return out.dropna().reset_index(drop=True)
