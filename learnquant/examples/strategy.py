from __future__ import annotations

import pandas as pd


def mean_reversion_strategy(features: pd.DataFrame) -> pd.Series:
    zscore = (features["return_1"] - features["return_1"].rolling(20).mean()) / (
        features["return_1"].rolling(20).std().replace(0, 1)
    )
    return (zscore < -1).astype(float) - (zscore > 1).astype(float)
