# simulador-inversiones

Streamlit app that simulates and compares two DCA (Dollar Cost Averaging) strategies — **DCA Puro** and **DCA Táctico** — on a single stock, ETF, or fund.

The simulator pulls historical prices from Yahoo Finance, runs both strategies on the real series (a "Histórico" tab with summary metrics and charts), and then runs a 10 000-path bootstrap Monte Carlo (a "Montecarlo" tab with percentile bands, histograms, PDF, and CDF). All user-facing text is in Spanish (Castilian); code, comments, and this README are in English.

## What each strategy does

- **DCA Puro** — Month 0 invests the initial deposit. Every subsequent month invests the monthly contribution at that month's price.
- **DCA Táctico** — Month 0 invests only the initial deposit. From month 1 onward, the decision depends on the asset's return over the immediately preceding month:
  - If the prior monthly return was negative, deploy the entire pool plus this month's contribution.
  - Otherwise, add this month's contribution to the pool and don't invest.
  - Any cash left in the pool at the end is ignored (does not count toward the final value).

Both strategies report `total invertido`, `capital comprometido`, `valor final`, money-weighted CAGR (IRR annualized), annualized volatility, Sharpe ratio (risk-free = 0), and max drawdown.

The Monte Carlo bootstrap samples monthly returns with replacement from the historical window and builds 10 000 synthetic price paths to estimate the distribution of final portfolio values.

## Running locally

Requires Python 3.11+ (developed and tested on 3.12).

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

Then create `.streamlit/secrets.toml` (this file is gitignored) by copying the example and setting your own password:

```toml
# .streamlit/secrets.toml
app_password = "your-password-here"
```

Launch the app:

```bash
streamlit run app.py
```

## Running the tests

```bash
pytest tests/ -v
```

## Deploying to Streamlit Community Cloud

1. Push the repo to GitHub. Make sure `.streamlit/secrets.toml` is **not** committed (the `.gitignore` already excludes it).
2. Sign in at <https://share.streamlit.io> with GitHub.
3. Click **New app**, point it at this repository and `app.py`.
4. Open the app's **Settings → Secrets** in the Streamlit Cloud UI and paste:

   ```toml
   app_password = "your-password-here"
   ```

   Save. Streamlit Cloud reads this exactly like a local `secrets.toml`.
5. Deploy. The app's public URL (e.g. `https://<your-app>.streamlit.app`) appears once the build finishes.

If you ever need to pin a specific Python version on Streamlit Cloud, set it under **Settings → Advanced**; the default works for this project.

## Project layout

```
app.py            # Streamlit entry point: auth, sidebar, tabs
data.py           # yfinance download + first-business-day monthly resample
strategies.py     # dca_puro, dca_tactico (pure functions on a price array)
metrics.py        # CAGR/IRR, volatility, Sharpe, drawdown, probability helpers
montecarlo.py     # vectorized 10 000-path bootstrap
plots.py          # plotly figure builders (Spanish labels)
strings.py        # centralized Spanish UI strings
tests/            # pytest suite
.streamlit/       # config.toml, secrets.toml.example
```

## Disclaimer

Esta herramienta tiene fines educativos. No constituye asesoramiento financiero. Los rendimientos pasados no garantizan rendimientos futuros.
