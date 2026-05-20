"""Plotly figure builders. All titles, labels, and legends are in Spanish."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

import strings as S


# Consistent palette across the app.
COLOR_PURO = "#1f77b4"      # blue
COLOR_TACTICO = "#d62728"   # red
COLOR_INVESTED = "#7f7f7f"  # grey
COLOR_PRICE = "#2ca02c"     # green
COLOR_MEAN = "#ff7f0e"      # orange — mean marker
COLOR_MEDIAN = "#9467bd"    # purple — median marker
COLOR_TARGET = "#17becf"    # cyan — target X marker
COLOR_RUIN = "#000000"      # black — 1€ ruin line


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def portfolio_value_chart(
    dates: pd.DatetimeIndex,
    values_puro: np.ndarray,
    values_tactico: np.ndarray,
    invested_reference: np.ndarray,
    investment_name: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values_puro, mode="lines",
        name=S.LABEL_PURO, line=dict(color=COLOR_PURO, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=values_tactico, mode="lines",
        name=S.LABEL_TACTICO, line=dict(color=COLOR_TACTICO, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=invested_reference, mode="lines",
        name=S.HIST_LEGEND_INVESTED,
        line=dict(color=COLOR_INVESTED, width=1, dash="dash"),
    ))
    fig.update_layout(
        title=f"{S.HIST_CHART_PORTFOLIO} — {investment_name}",
        xaxis_title=S.AXIS_DATE,
        yaxis_title=S.AXIS_VALUE_EUR,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def price_chart(
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    investment_name: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode="lines",
        name=investment_name, line=dict(color=COLOR_PRICE, width=2),
    ))
    fig.update_layout(
        title=f"{S.HIST_CHART_PRICE} — {investment_name}",
        xaxis_title=S.AXIS_DATE,
        yaxis_title=S.AXIS_PRICE_EUR,
        hovermode="x unified",
        showlegend=False,
    )
    return fig


# ---------------- Monte Carlo plots ----------------

def fan_chart(
    percentile_bands: dict,
    strategy_label: str,
    color: str,
    investment_name: str,
) -> go.Figure:
    """Shaded percentile bands. Expects keys 10, 30, 50, 70, 90 mapping to arrays."""
    n_steps = len(percentile_bands[50])
    months = np.arange(n_steps)
    fig = go.Figure()

    # Outer band: 10 – 90
    fig.add_trace(go.Scatter(
        x=months, y=percentile_bands[90], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=percentile_bands[10], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor=_hex_to_rgba(color, 0.15),
        name="P10–P90", hoverinfo="skip",
    ))
    # Inner band: 30 – 70
    fig.add_trace(go.Scatter(
        x=months, y=percentile_bands[70], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=percentile_bands[30], mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor=_hex_to_rgba(color, 0.30),
        name="P30–P70", hoverinfo="skip",
    ))
    # Median line
    fig.add_trace(go.Scatter(
        x=months, y=percentile_bands[50], mode="lines",
        line=dict(color=color, width=2),
        name="Mediana (P50)",
    ))

    fig.update_layout(
        title=f"{S.MC_FAN_CHART} — {strategy_label} — {investment_name}",
        xaxis_title=S.AXIS_MONTH,
        yaxis_title=S.AXIS_VALUE_EUR,
        hovermode="x unified",
    )
    return fig


def _vline(fig: go.Figure, x: float, color: str, label: str, dash: str = "dash") -> None:
    """Vertical reference line. Annotations are rotated 90° so neighbouring
    labels don't overlap horizontally even when the lines are close in x."""
    fig.add_vline(
        x=x,
        line=dict(color=color, width=2, dash=dash),
        annotation_text=label,
        annotation_position="top",
        annotation_textangle=-90,
        annotation_yanchor="bottom",
        annotation_font=dict(size=11, color=color),
        annotation_xshift=10,
    )


def histogram(
    final_values: np.ndarray,
    total_invested_mean: float,
    strategy_label: str,
    color: str,
    investment_name: str,
) -> go.Figure:
    mean_v = float(np.mean(final_values))
    median_v = float(np.median(final_values))
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_values, xbins=dict(size=1000),
        marker=dict(color=_hex_to_rgba(color, 0.6)),
        name=strategy_label,
    ))
    _vline(fig, mean_v, COLOR_MEAN, f"{S.MC_STAT_MEAN}: {mean_v:,.0f} €")
    _vline(fig, median_v, COLOR_MEDIAN, f"{S.MC_STAT_MEDIAN}: {median_v:,.0f} €")
    _vline(fig, total_invested_mean, COLOR_INVESTED,
           f"{S.HIST_LEGEND_INVESTED}: {total_invested_mean:,.0f} €", dash="dot")
    fig.update_layout(
        title=f"{S.MC_HIST_CHART} — {strategy_label} — {investment_name}",
        xaxis_title=S.AXIS_VALUE_EUR,
        yaxis_title=S.AXIS_FREQ,
        showlegend=False,
        margin=dict(t=110),  # room for rotated vertical-line labels
    )
    return fig


def pdf_chart(
    final_values: np.ndarray,
    strategy_label: str,
    color: str,
    investment_name: str,
) -> go.Figure:
    arr = np.asarray(final_values, dtype=float)
    # Clip the KDE evaluation range to avoid huge tails.
    lo = float(np.percentile(arr, 0.5))
    hi = float(np.percentile(arr, 99.5))
    xs = np.linspace(lo, hi, 500)
    kde = gaussian_kde(arr)
    ys = kde(xs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=color, width=2), fill="tozeroy",
        fillcolor=_hex_to_rgba(color, 0.25),
        name=strategy_label,
    ))
    fig.update_layout(
        title=f"{S.MC_PDF_CHART} — {strategy_label} — {investment_name}",
        xaxis_title=S.AXIS_VALUE_EUR,
        yaxis_title=S.AXIS_DENSITY,
        showlegend=False,
    )
    return fig


def cdf_chart(
    final_values: np.ndarray,
    total_invested_mean: float,
    target_x: float,
    strategy_label: str,
    color: str,
    investment_name: str,
) -> go.Figure:
    arr = np.sort(np.asarray(final_values, dtype=float))
    cdf = np.arange(1, arr.size + 1) / arr.size
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=arr, y=cdf, mode="lines",
        line=dict(color=color, width=2), name=strategy_label,
    ))
    _vline(fig, 1.0, COLOR_RUIN, "1 €")
    _vline(fig, total_invested_mean, COLOR_INVESTED,
           f"{S.HIST_LEGEND_INVESTED}: {total_invested_mean:,.0f} €", dash="dot")
    _vline(fig, total_invested_mean + target_x, COLOR_TARGET,
           f"+ X: {total_invested_mean + target_x:,.0f} €")
    fig.update_layout(
        title=f"{S.MC_CDF_CHART} — {strategy_label} — {investment_name}",
        xaxis_title=S.AXIS_VALUE_EUR,
        yaxis_title=S.AXIS_PROB,
        showlegend=False,
        margin=dict(t=110),
    )
    return fig


def compare_histograms(
    final_puro: np.ndarray,
    final_tactico: np.ndarray,
    investment_name: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_puro, xbins=dict(size=1000), name=S.LABEL_PURO,
        marker=dict(color=_hex_to_rgba(COLOR_PURO, 0.5)),
    ))
    fig.add_trace(go.Histogram(
        x=final_tactico, xbins=dict(size=1000), name=S.LABEL_TACTICO,
        marker=dict(color=_hex_to_rgba(COLOR_TACTICO, 0.5)),
    ))
    fig.update_layout(
        title=f"{S.MC_COMPARE_HIST} — {investment_name}",
        barmode="overlay",
        xaxis_title=S.AXIS_VALUE_EUR,
        yaxis_title=S.AXIS_FREQ,
    )
    return fig


def compare_cdfs(
    final_puro: np.ndarray,
    final_tactico: np.ndarray,
    investment_name: str,
) -> go.Figure:
    fig = go.Figure()
    for arr, label, color in [
        (final_puro, S.LABEL_PURO, COLOR_PURO),
        (final_tactico, S.LABEL_TACTICO, COLOR_TACTICO),
    ]:
        sorted_arr = np.sort(np.asarray(arr, dtype=float))
        cdf = np.arange(1, sorted_arr.size + 1) / sorted_arr.size
        fig.add_trace(go.Scatter(
            x=sorted_arr, y=cdf, mode="lines",
            line=dict(color=color, width=2), name=label,
        ))
    fig.update_layout(
        title=f"{S.MC_COMPARE_CDF} — {investment_name}",
        xaxis_title=S.AXIS_VALUE_EUR,
        yaxis_title=S.AXIS_PROB,
    )
    return fig


def compare_histograms_returns(
    returns_puro: np.ndarray,
    returns_tactico: np.ndarray,
    investment_name: str,
) -> go.Figure:
    """Overlaid histograms of total return in % across both strategies."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns_puro * 100.0, nbinsx=80, name=S.LABEL_PURO,
        marker=dict(color=_hex_to_rgba(COLOR_PURO, 0.5)),
    ))
    fig.add_trace(go.Histogram(
        x=returns_tactico * 100.0, nbinsx=80, name=S.LABEL_TACTICO,
        marker=dict(color=_hex_to_rgba(COLOR_TACTICO, 0.5)),
    ))
    fig.update_layout(
        title=f"{S.MC_COMPARE_HIST_RETURN} — {investment_name}",
        barmode="overlay",
        xaxis_title=S.AXIS_RETURN_PCT,
        yaxis_title=S.AXIS_FREQ,
    )
    return fig


def compare_cdfs_returns(
    returns_puro: np.ndarray,
    returns_tactico: np.ndarray,
    investment_name: str,
) -> go.Figure:
    fig = go.Figure()
    for arr, label, color in [
        (returns_puro, S.LABEL_PURO, COLOR_PURO),
        (returns_tactico, S.LABEL_TACTICO, COLOR_TACTICO),
    ]:
        sorted_arr = np.sort(np.asarray(arr, dtype=float)) * 100.0
        cdf = np.arange(1, sorted_arr.size + 1) / sorted_arr.size
        fig.add_trace(go.Scatter(
            x=sorted_arr, y=cdf, mode="lines",
            line=dict(color=color, width=2), name=label,
        ))
    fig.update_layout(
        title=f"{S.MC_COMPARE_CDF_RETURN} — {investment_name}",
        xaxis_title=S.AXIS_RETURN_PCT,
        yaxis_title=S.AXIS_PROB,
    )
    return fig


def spaghetti_chart(
    trajectories: np.ndarray,
    strategy_label: str,
    color: str,
    investment_name: str,
) -> go.Figure:
    """Plot all simulated trajectories with very low opacity (density visualization).

    Uses one WebGL trace with NaN separators between paths — fast enough for
    10 000 × ~240-step paths in the browser. Median overlaid on top in solid.
    """
    arr = np.asarray(trajectories, dtype=float)
    n_sims, n_steps = arr.shape
    months = np.arange(n_steps, dtype=float)

    # Build a single (x, y) sequence: [path_0, NaN, path_1, NaN, ..., path_{n-1}]
    sep = np.full(n_sims, np.nan)
    xs = np.concatenate([
        np.tile(months, n_sims).reshape(n_sims, n_steps),
        sep[:, None],
    ], axis=1).ravel()
    ys = np.concatenate([arr, sep[:, None]], axis=1).ravel()

    median = np.percentile(arr, 50, axis=0)
    final_vals = arr[:, -1]
    i_best = int(np.argmax(final_vals))
    i_worst = int(np.argmin(final_vals))

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="lines",
        line=dict(color=color, width=1),
        opacity=0.02,
        name="Trayectorias",
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=months, y=median, mode="lines",
        line=dict(color=color, width=2.5),
        name="Mediana",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=arr[i_best], mode="lines",
        line=dict(color=COLOR_PRICE, width=3),
        name=f"Mejor final ({final_vals[i_best]:,.0f} €)",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=arr[i_worst], mode="lines",
        line=dict(color="#d62728", width=3),
        name=f"Peor final ({final_vals[i_worst]:,.0f} €)",
    ))
    fig.update_layout(
        title=f"{S.MC_SPAGHETTI} — {strategy_label} — {investment_name}",
        xaxis_title=S.AXIS_MONTH,
        yaxis_title=S.AXIS_VALUE_EUR,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig
