from __future__ import annotations

import pandas as pd


def _generate_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    out["return_1"] = out["close"].pct_change()
    out["volatility_20"] = out["return_1"].rolling(20, min_periods=20).std().shift(1)
    out["momentum_10"] = (out["close"] / out["close"].shift(10) - 1.0).shift(1)

    if {"bid_volume", "ask_volume"}.issubset(set(out.columns)):
        denom = (out["bid_volume"] + out["ask_volume"]).replace(0, pd.NA)
        out["order_book_imbalance"] = ((out["bid_volume"] - out["ask_volume"]) / denom).shift(1)
    else:
        out["order_book_imbalance"] = 0.0

    out["target"] = out["close"].pct_change().shift(-1)
    return out


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Analysts can edit this function to define custom features."""
    if "symbol" in df.columns:
        out = (
            df.groupby("symbol", group_keys=False, sort=False)
            .apply(_generate_symbol_features)
            .reset_index(drop=True)
        )
    else:
        out = _generate_symbol_features(df)

    feature_cols = ["return_1", "volatility_20", "momentum_10", "order_book_imbalance", "target"]
    out[feature_cols] = out[feature_cols].replace([float("inf"), float("-inf")], pd.NA)
    return out.dropna(subset=feature_cols).reset_index(drop=True)
