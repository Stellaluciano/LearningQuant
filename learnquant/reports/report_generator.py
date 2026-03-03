from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


METRIC_LABELS = {
    "total_return": "Total Return",
    "annualized_return": "Annualized Return",
    "sharpe": "Sharpe Ratio",
    "sortino": "Sortino Ratio",
    "profit_factor": "Profit Factor",
    "max_drawdown": "Max Drawdown",
    "win_rate": "Win Rate",
    "turnover": "Average Turnover",
}


def _format_metric(name: str, value: float) -> str:
    if name in {"total_return", "annualized_return", "max_drawdown", "win_rate", "turnover"}:
        return f"{value:.2%}"
    if value == float("inf"):
        return "inf"
    return f"{value:.6f}"


def generate_report(
    run_dir: Path,
    equity_df: pd.DataFrame,
    backtest_metrics: dict,
    ml_metrics: dict | None = None,
    feature_importance: dict | None = None,
) -> Path:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_df["timestamp"], y=equity_df["equity"], mode="lines", name="Equity"))
    equity_div = fig.to_html(full_html=False, include_plotlyjs="cdn")

    rows = "".join(
        f"<tr><td>{METRIC_LABELS.get(k, k)}</td><td>{_format_metric(k, v)}</td></tr>" for k, v in backtest_metrics.items()
    )
    ml_rows = ""
    if ml_metrics:
        ml_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in ml_metrics.items())

    fi_rows = ""
    if feature_importance:
        top = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        fi_rows = "".join(f"<tr><td>{k}</td><td>{v:.6f}</td></tr>" for k, v in top)

    html = f"""
    <html><body>
    <h1>LearnQuant Experiment Report</h1>
    <h2>Equity Curve</h2>
    {equity_div}
    <h2>Backtest Metrics</h2><table border='1'>{rows}</table>
    <h2>ML Metrics</h2><table border='1'>{ml_rows}</table>
    <h2>Feature Importance</h2><table border='1'>{fi_rows}</table>
    </body></html>
    """
    report_path = run_dir / "report.html"
    report_path.write_text(html)
    return report_path
