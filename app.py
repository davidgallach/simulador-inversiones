"""Streamlit entry point for the DCA simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import strings as S
from data import DataError, fetch_prices
from metrics import (
    cagr,
    max_drawdown,
    percentiles,
    prob_loss_given_ruin,
    prob_profit_above,
    prob_ruin_path,
    sharpe_ratio,
    total_return,
    twr_annualized,
    volatility_annualized,
)
from montecarlo import (
    N_SIMS,
    SimResult,
    monthly_returns_from_prices,
    simulate,
)
from plots import (
    cdf_chart,
    compare_cdfs,
    compare_cdfs_returns,
    compare_histograms,
    compare_histograms_returns,
    fan_chart,
    histogram,
    pdf_chart,
    portfolio_value_chart,
    price_chart,
    spaghetti_chart,
    COLOR_PURO,
    COLOR_TACTICO,
)
from strategies import StrategyResult, dca_puro, dca_tactico


# ---------------- Auth ----------------

def _check_password() -> bool:
    if st.session_state.get("auth_ok"):
        return True

    try:
        expected = st.secrets["app_password"]
    except Exception:
        st.error(S.AUTH_MISSING_SECRET)
        return False

    st.title(S.AUTH_TITLE)
    with st.form("auth_form", clear_on_submit=False):
        pwd = st.text_input(S.AUTH_PROMPT, type="password")
        submitted = st.form_submit_button(S.AUTH_BUTTON)
    if submitted:
        if pwd == expected:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error(S.AUTH_ERROR)
    return False


# ---------------- Sidebar ----------------

def _render_sidebar() -> dict | None:
    st.sidebar.header(S.SIDEBAR_HEADER)
    name = st.sidebar.text_input(S.INPUT_NAME, value="Mi inversión")
    ticker = st.sidebar.text_input(S.INPUT_TICKER, value="SPY")
    initial = st.sidebar.number_input(
        S.INPUT_INITIAL, min_value=0.0, value=1000.0, step=100.0
    )
    monthly = st.sidebar.number_input(
        S.INPUT_MONTHLY, min_value=0.0, value=200.0, step=50.0
    )
    years_hist = st.sidebar.number_input(
        S.INPUT_YEARS_HIST, min_value=1, max_value=50, value=10, step=1
    )
    years_mc = st.sidebar.number_input(
        S.INPUT_YEARS_MC, min_value=1, max_value=50, value=20, step=1
    )
    run = st.sidebar.button(S.RUN_BUTTON, type="primary")
    st.sidebar.markdown("---")
    st.sidebar.caption(S.DISCLAIMER)

    if run:
        return {
            "name": name.strip() or "Mi inversión",
            "ticker": ticker.strip().upper(),
            "initial": float(initial),
            "monthly": float(monthly),
            "years_hist": int(years_hist),
            "years_mc": int(years_mc),
        }
    return None


# ---------------- Pipeline ----------------

def _run_pipeline(params: dict) -> bool:
    """Download data, run both strategies, compute metrics, cache in session_state.
    Returns True on success, False (and shows a Spanish error) on failure."""
    try:
        with st.spinner(f"Descargando histórico de {params['ticker']}..."):
            price_data = fetch_prices(params["ticker"], params["years_hist"])
    except DataError as exc:
        code = str(exc)
        if code.startswith("INSUFFICIENT_HISTORY"):
            st.error(S.ERR_INSUFFICIENT_HISTORY)
        elif code.startswith("ISIN_UNRESOLVED"):
            st.error(S.ERR_ISIN_UNRESOLVED)
        else:
            st.error(S.ERR_NO_DATA)
        return False

    prices = price_data.monthly.values.astype(float)
    dates = pd.DatetimeIndex(price_data.monthly.index)

    res_puro = dca_puro(prices, params["initial"], params["monthly"])
    res_tactico = dca_tactico(prices, params["initial"], params["monthly"])

    st.session_state["sim"] = {
        "params": params,
        "ticker_resolved": price_data.ticker,
        "dates": dates,
        "prices": prices,
        "puro": res_puro,
        "tactico": res_tactico,
    }

    # Run Monte Carlo right after the historical backtest.
    historical_returns = monthly_returns_from_prices(prices)
    n_months_mc = params["years_mc"] * 12
    progress = st.progress(0.0, text=S.PROGRESS_MC)

    def _cb(frac: float) -> None:
        progress.progress(min(max(frac, 0.0), 1.0), text=S.PROGRESS_MC)

    mc_result = simulate(
        historical_monthly_returns=historical_returns,
        initial=params["initial"],
        monthly=params["monthly"],
        n_months=n_months_mc,
        n_sims=N_SIMS,
        seed=None,
        progress_cb=_cb,
    )
    progress.empty()
    st.session_state["mc"] = mc_result
    return True


# ---------------- Histórico tab ----------------

def _format_eur(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def _format_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x * 100:.2f} %".replace(".", ",")


def _format_ratio(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.2f}".replace(".", ",")


def _summary_row(label: str, res: StrategyResult) -> dict:
    return {
        S.HIST_COL_STRATEGY: label,
        S.HIST_COL_INVESTED: _format_eur(res.total_invested),
        S.HIST_COL_COMMITTED: _format_eur(res.committed),
        S.HIST_COL_FINAL: _format_eur(res.final_value),
        S.HIST_COL_TOTAL_RETURN: _format_pct(total_return(res)),
        S.HIST_COL_CAGR: _format_pct(cagr(res)),
        S.HIST_COL_TWR: _format_pct(twr_annualized(res)),
        S.HIST_COL_VOL: _format_pct(volatility_annualized(res)),
        S.HIST_COL_SHARPE: _format_ratio(sharpe_ratio(res)),
        S.HIST_COL_MDD: _format_pct(max_drawdown(res.values)),
    }


def _render_historical_tab() -> None:
    sim = st.session_state.get("sim")
    if sim is None:
        st.info("Ejecuta una simulación desde la barra lateral para ver el histórico.")
        return

    name = sim["params"]["name"]
    res_puro: StrategyResult = sim["puro"]
    res_tactico: StrategyResult = sim["tactico"]

    st.subheader(S.HIST_SUMMARY_TITLE)
    st.caption(S.HIST_SHARPE_NOTE)
    table = pd.DataFrame([
        _summary_row(S.LABEL_PURO, res_puro),
        _summary_row(S.LABEL_TACTICO, res_tactico),
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.plotly_chart(
        portfolio_value_chart(
            dates=sim["dates"],
            values_puro=res_puro.values,
            values_tactico=res_tactico.values,
            invested_reference=res_puro.invested,  # both share initial + monthly schedule
            investment_name=name,
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        price_chart(
            dates=sim["dates"],
            prices=sim["prices"],
            investment_name=name,
        ),
        use_container_width=True,
    )


# ---------------- Montecarlo tab ----------------

def _format_eur_with_return(value: float, base_invested: float) -> str:
    """'12,345 € (+23.4 %)' — return % computed relative to base_invested."""
    if base_invested is None or base_invested <= 0:
        return _format_eur(value)
    pct = (value / base_invested - 1.0) * 100.0
    sign = "+" if pct >= 0 else ""
    pct_str = f"{sign}{pct:.1f} %".replace(".", ",")
    return f"{_format_eur(value)} ({pct_str})"


def _strategy_stats_table(
    label: str, final_values: np.ndarray, total_invested_mean: float
) -> pd.DataFrame:
    pct = percentiles(final_values)
    rows = [
        (S.MC_STAT_MEAN, float(np.mean(final_values))),
        (S.MC_STAT_MEDIAN, float(np.median(final_values))),
    ]
    for p in (10, 20, 30, 40, 60, 70, 80, 90):
        rows.append((f"P{p}", pct[p]))
    df = pd.DataFrame(rows, columns=["Estadístico", label])
    df[label] = df[label].map(lambda v: _format_eur_with_return(v, total_invested_mean))
    return df


def _render_montecarlo_tab() -> None:
    sim = st.session_state.get("sim")
    mc: SimResult | None = st.session_state.get("mc")
    if sim is None or mc is None:
        st.info("Ejecuta una simulación desde la barra lateral para ver el Montecarlo.")
        return

    name = sim["params"]["name"]

    # --- Live X filter (no re-simulation) ---
    target_x = st.number_input(
        S.MC_TARGET_INPUT, min_value=0.0, value=50000.0, step=1000.0,
        key="mc_target_x",
    )
    st.caption(S.MC_TARGET_CAPTION)
    st.caption(
        f"Simulaciones: {mc.n_sims:,} · Horizonte: {mc.n_months} meses "
        f"({mc.n_months / 12:.0f} años)"
    )

    # --- Combined stats table (Puro + Táctico side by side) ---
    st.subheader(S.MC_STATS_TITLE)
    mean_inv_puro = float(np.mean(mc.puro.total_invested))
    mean_inv_tactico = float(np.mean(mc.tactico.total_invested))
    stats_puro = _strategy_stats_table(S.LABEL_PURO, mc.puro.final_values, mean_inv_puro)
    stats_tactico = _strategy_stats_table(
        S.LABEL_TACTICO, mc.tactico.final_values, mean_inv_tactico
    )
    combined = stats_puro.merge(stats_tactico, on="Estadístico")
    st.dataframe(combined, use_container_width=True, hide_index=True)
    st.caption(
        "Entre paréntesis: rendimiento total (%) = valor / total invertido medio − 1."
    )

    # --- Probability table (recomputes live from cached arrays when X changes) ---
    st.subheader(S.MC_PROBS_TITLE)
    prob_rows = []
    for label, strat in [(S.LABEL_PURO, mc.puro), (S.LABEL_TACTICO, mc.tactico)]:
        prob_rows.append({
            S.HIST_COL_STRATEGY: label,
            S.MC_PROB_RUIN: _format_pct(prob_ruin_path(strat.trajectories)),
            S.MC_PROB_LOSS_GIVEN_RUIN: _format_pct(
                prob_loss_given_ruin(
                    strat.trajectories, strat.final_values, strat.total_invested
                )
            ),
            S.MC_PROB_NEG: _format_pct(
                float(np.mean(strat.final_values < strat.total_invested))
            ),
            S.MC_PROB_TARGET: _format_pct(
                prob_profit_above(strat.final_values, strat.total_invested, target_x)
            ),
        })
    st.dataframe(pd.DataFrame(prob_rows), use_container_width=True, hide_index=True)

    # --- Per-strategy charts ---
    for label, strat, color in [
        (S.LABEL_PURO, mc.puro, COLOR_PURO),
        (S.LABEL_TACTICO, mc.tactico, COLOR_TACTICO),
    ]:
        st.markdown(f"### {label}")
        mean_invested = float(np.mean(strat.total_invested))

        st.plotly_chart(
            spaghetti_chart(strat.trajectories, label, color, name),
            use_container_width=True,
        )
        st.plotly_chart(
            fan_chart(strat.percentile_bands, label, color, name),
            use_container_width=True,
        )
        col_h, col_p = st.columns(2)
        with col_h:
            st.plotly_chart(
                histogram(strat.final_values, mean_invested, label, color, name),
                use_container_width=True,
            )
        with col_p:
            st.plotly_chart(
                pdf_chart(strat.final_values, label, color, name),
                use_container_width=True,
            )
        st.plotly_chart(
            cdf_chart(strat.final_values, mean_invested, target_x, label, color, name),
            use_container_width=True,
        )

    # --- Side-by-side comparison ---
    st.markdown("### Comparativa")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            compare_histograms(mc.puro.final_values, mc.tactico.final_values, name),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            compare_cdfs(mc.puro.final_values, mc.tactico.final_values, name),
            use_container_width=True,
        )

    # Same comparison but on % total return (per-sim value / per-sim invested - 1).
    returns_puro = mc.puro.final_values / np.where(
        mc.puro.total_invested > 0, mc.puro.total_invested, np.nan
    ) - 1.0
    returns_tactico = mc.tactico.final_values / np.where(
        mc.tactico.total_invested > 0, mc.tactico.total_invested, np.nan
    ) - 1.0
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(
            compare_histograms_returns(returns_puro, returns_tactico, name),
            use_container_width=True,
        )
    with col_d:
        st.plotly_chart(
            compare_cdfs_returns(returns_puro, returns_tactico, name),
            use_container_width=True,
        )


# ---------------- Main ----------------

def main() -> None:
    st.set_page_config(page_title=S.APP_TITLE, layout="wide")

    if not _check_password():
        st.stop()

    st.title(S.APP_TITLE)
    st.write(S.APP_SUBTITLE)

    params = _render_sidebar()
    if params is not None:
        _run_pipeline(params)

    tab_hist, tab_mc = st.tabs([S.TAB_HISTORICAL, S.TAB_MONTECARLO])
    with tab_hist:
        _render_historical_tab()
    with tab_mc:
        _render_montecarlo_tab()


if __name__ == "__main__":
    main()
