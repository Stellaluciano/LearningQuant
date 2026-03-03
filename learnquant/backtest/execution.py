from __future__ import annotations

import pandas as pd


def _simulate_single_symbol(
    df: pd.DataFrame,
    signal_col: str,
    fee: float,
    slippage_bps: float,
    slippage_vol_multiplier: float,
    max_position: float,
    stop_loss: float | None,
    take_profit: float | None,
    max_drawdown: float | None,
) -> pd.DataFrame:
    trades = df[["timestamp", "close", signal_col]].copy()
    trades["asset_ret"] = trades["close"].pct_change().fillna(0.0)
    trades["volatility"] = trades["asset_ret"].rolling(20).std().shift(1).fillna(0.0)
    raw_position = trades[signal_col].shift(1).fillna(0.0).clip(-max_position, max_position)

    positions: list[float] = []
    turnovers: list[float] = []
    fee_costs: list[float] = []
    slippage_costs: list[float] = []
    strategy_rets: list[float] = []

    prev_position = 0.0
    equity = 1.0
    peak = 1.0
    entry_price: float | None = None
    halted = False

    for idx, row in trades.iterrows():
        position = 0.0 if halted else float(raw_position.loc[idx])
        if prev_position == 0.0 and position != 0.0:
            entry_price = float(row["close"])
        elif position == 0.0:
            entry_price = None

        turnover = abs(position - prev_position)
        fee_cost = turnover * fee
        slippage_cost = turnover * (slippage_bps / 10_000.0 + slippage_vol_multiplier * float(row["volatility"]))
        strategy_ret = position * float(row["asset_ret"]) - fee_cost - slippage_cost

        positions.append(position)
        turnovers.append(turnover)
        fee_costs.append(fee_cost)
        slippage_costs.append(slippage_cost)
        strategy_rets.append(strategy_ret)

        equity *= 1.0 + strategy_ret
        peak = max(peak, equity)
        drawdown = 0.0 if peak == 0 else (equity - peak) / peak

        if position != 0.0 and entry_price is not None:
            trade_pnl = position * (float(row["close"]) / entry_price - 1.0)
            if stop_loss is not None and trade_pnl <= -abs(stop_loss):
                prev_position = 0.0
                continue
            if take_profit is not None and trade_pnl >= abs(take_profit):
                prev_position = 0.0
                continue

        if max_drawdown is not None and drawdown <= -abs(max_drawdown):
            halted = True
            prev_position = 0.0
            continue

        prev_position = position

    trades["position"] = positions
    trades["turnover"] = turnovers
    trades["fee_cost"] = fee_costs
    trades["slippage_cost"] = slippage_costs
    trades["strategy_ret"] = strategy_rets
    return trades


def simulate_trades(
    features: pd.DataFrame,
    signal_col: str = "signal",
    symbol_col: str = "symbol",
    fee: float = 0.0,
    slippage_bps: float = 0.0,
    slippage_vol_multiplier: float = 0.0,
    max_position: float = 1.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    max_drawdown: float | None = None,
) -> pd.DataFrame:
    if symbol_col in features.columns:
        outputs = []
        for symbol, group in features.sort_values([symbol_col, "timestamp"]).groupby(symbol_col, sort=False):
            simulated = _simulate_single_symbol(
                group,
                signal_col=signal_col,
                fee=fee,
                slippage_bps=slippage_bps,
                slippage_vol_multiplier=slippage_vol_multiplier,
                max_position=max_position,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_drawdown=max_drawdown,
            )
            simulated[symbol_col] = symbol
            outputs.append(simulated)
        return pd.concat(outputs, ignore_index=True)

    return _simulate_single_symbol(
        features.sort_values("timestamp"),
        signal_col=signal_col,
        fee=fee,
        slippage_bps=slippage_bps,
        slippage_vol_multiplier=slippage_vol_multiplier,
        max_position=max_position,
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_drawdown=max_drawdown,
    )
