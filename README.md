# Macro News → Market Returns (USA)

End-to-end research pipeline that links **US macro news features** (GDELT-style daily aggregates) to **next-day returns** on ETFs and BTC, with a **PyTorch MLP** as the main neural model alongside classical baselines, walk-forward evaluation, threshold backtests, and automated model selection.

---

## Project Overview

This repository implements a reproducible machine learning workflow for a common quant research question: **do daily news signals contain information about short-horizon asset returns?**

The pipeline:

1. Merges filtered news exports with Yahoo Finance market data.
2. Builds a supervised dataset with **15 return targets** (sector ETFs, rates, dollar, gold, crypto).
3. Trains and compares **six model families** (zero/last baselines, kNN, Ridge, Random Forest, **MLP**).
4. Evaluates with **time-ordered walk-forward folds** (no random shuffle).
5. Runs **long-flat / long-short backtests** with transaction costs and **confidence thresholds**.
6. Selects a final setup globally or **per target** using Sharpe on held-out predictions.

This is a **research and portfolio project**, not a production trading system.

---

## Motivation

Macro news moves markets, but the signal is noisy and non-stationary. Simple “predict tomorrow’s return from today’s news” setups are easy to get wrong through leakage, weak baselines, or overfitting.

This project focuses on **rigorous comparison**: strong baselines, multi-target evaluation, and economic sanity checks (directional accuracy, threshold sweeps, cost-aware backtests) rather than claiming alpha from a single metric.

---

## Key Features

- **News feature engineering** — daily GDELT aggregates: tone, themes, organisations (hashed), article counts.
- **Multi-target regression** — 15 `y_*` next-day return columns in one framework.
- **PyTorch MLP** — configurable feed-forward network with preprocessing fit **inside each fold only**.
- **Baselines** — `baseline_zero`, `baseline_last`, kNN “similar days”, Ridge, Random Forest.
- **Walk-forward evaluation** — expanding train window; configurable step and test window.
- **Rich metrics** — MAE, RMSE, R², directional accuracy, Pearson, Spearman (per target and averaged).
- **Backtesting** — long-flat and long-short rules, 1 bps default cost, threshold grid search.
- **Model selection** — `select_setup` picks global or per-target model + threshold by Sharpe.
- **One-command runner** — `python -m news_impact.run_all` for train → evaluate → leaderboard → backtest.

---

## Technical Stack

| Area | Tools |
|------|--------|
| Language | Python 3.10+ |
| Deep learning | PyTorch (MLP) |
| ML / stats | scikit-learn, pandas, NumPy |
| Data | Parquet dataset, CSV inputs |
| News | GDELT export (user-provided) |
| Markets | yfinance daily prices |
| Packaging | `pyproject.toml`, editable install |

---

## End-to-End Workflow

```
data/raw/dataset.csv          ──┐
data/raw/market_data.csv      ──┼──► prepare_dataset ──► ml_dataset.parquet
                                │
                                ├──► train (6 models) ──► models/<name>/
                                │
                                ├──► evaluate (walk-forward) ──► reports/metrics/
                                │
                                ├──► backtest + threshold sweep ──► reports/metrics/backtest_*.csv
                                │
                                └──► select_setup ──► reports/selection/final_selection_*.csv
```

**Typical commands**

```bash
# 1) Build dataset (after placing raw CSVs)
python -m news_impact.prepare_dataset \
  --news data/raw/dataset.csv \
  --market data/raw/market_data.csv \
  --out data/processed/ml_dataset.parquet

# 2) Full pipeline (quick mode by default)
python -m news_impact.run_all --data data/processed/ml_dataset.parquet

# 3) Full mode (more thorough walk-forward)
python -m news_impact.run_all --data data/processed/ml_dataset.parquet --mode full
```

See `reports/summary.md` for the latest run diagnostics (generated locally).

---

## Repository Structure

```
news_impact_nn_project/
├── src/news_impact/          # Pipeline modules
│   ├── prepare_dataset.py    # News + market merge
│   ├── train.py              # Model training CLI
│   ├── evaluate.py           # Walk-forward evaluation
│   ├── run_all.py            # Orchestrator
│   ├── backtest.py           # Threshold + cost backtests
│   ├── select_setup.py       # Final model/threshold selection
│   └── modeling.py           # MLP and sklearn models
├── data/raw/                 # User inputs (gitignored)
├── data/processed/           # ml_dataset.parquet (gitignored)
├── models/                   # Trained artefacts (gitignored)
├── reports/metrics/          # CSV metrics from last run (in repo)
├── reports/selection/        # Selected setups
└── reports/summary.md        # Auto-generated summary
```

---

## Results Summary

*From the committed `reports/metrics/` and `reports/summary.md` run (walk-forward, quick mode: start=0.7, step=20, test_window=60).*

### Forecast accuracy (walk-forward)

| Model | Avg directional accuracy | Avg RMSE | Avg R² |
|-------|-------------------------:|---------:|-------:|
| **mlp** | **0.490** | 0.0320 | −13.45 |
| rf | 0.488 | **0.0117** | −0.10 |
| knn | 0.486 | 0.0119 | −0.15 |
| ridge | 0.467 | 0.0359 | −14.21 |
| baseline_last | 0.344 | 0.0152 | −0.80 |
| baseline_zero | 0.008 | 0.0114 | −0.03 |

**Honest reading:** average directional accuracy for the MLP is **~49%**, close to a coin flip. The zero-return baseline still wins on **global RMSE** for all 15 targets because many returns are small. The MLP ranks highest on **directional accuracy** among trained models, but the edge is modest and not economically strong on its own.

### Backtest (1 bps cost, long-flat)

| Scope | Setup | Sharpe (reported) |
|-------|--------|------------------:|
| Global | mlp, threshold=0 | 0.37 |
| Per-target (avg of best) | mixed models | 1.18 |

Backtest Sharpe values come from a **simple rule-based strategy** on model predictions (not a live fund). They are useful for **comparing models and thresholds**, not as proof of deployable alpha.

### Leakage controls (documented in code)

- Walk-forward folds retrain from scratch.
- Scaling/imputation fit on **train fold only**.
- No shuffle on the time index.

---

## How to Run

### Install

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .
```

### Raw data (not included)

Place your files under `data/raw/`:

| File | Description |
|------|-------------|
| `dataset.csv` | Filtered GDELT / news export |
| `market_data.csv` | Daily prices from yfinance |

Then build the dataset and run the pipeline (commands above).

### Useful CLIs

```bash
python -m news_impact.train --data data/processed/ml_dataset.parquet --model mlp --outdir models/mlp
python -m news_impact.evaluate --data data/processed/ml_dataset.parquet --modeldir models/mlp --outdir reports
python -m news_impact.backtest --reports-dir reports --cost-bps 1.0 --thresholds 0,0.0005,0.001,0.002,0.003,0.005
python -m news_impact.select_setup --reports-dir reports --objective sharpe --cost-bps 1 --scope per-target
```

---

## Limitations

- **Weak predictive power** on average; directional accuracy ≈ 50% globally.
- **Negative R²** for several models — forecasts often worse than the mean in squared-error terms.
- **Simplified backtest** — no slippage model beyond a flat bps cost; no portfolio constraints.
- **News coverage** depends on the GDELT filter and date range of the export.
- **Regime change** — macro relationships shift; walk-forward helps but does not remove structural break risk.
- **Not production-ready** — no API, monitoring, or execution layer.

---

## Future Improvements

- Richer news representations (embeddings, entity graphs, event types).
- Proper **purged / embargoed** cross-validation for overlapping features.
- Probabilistic or quantile forecasts instead of point returns only.
- Joint multi-task architecture sharing news encoder across targets.
- Live data ingestion and experiment tracking (MLflow / Weights & Biases).
- Stronger economic evaluation (turnover, capacity, subperiod stability).

---

## Note on Version History

Most development was done **locally** and in a university Git workflow before this repository was prepared for public portfolio use. The GitHub history here shows a **clean, organised export** of the project state (source, configs, and evaluation tables), not every intermediate experiment. Further work on this repo will use normal incremental commits.

---

## Author

**Adrián** — [github.com/adro0303](https://github.com/adro0303)

Research project in applied machine learning for macro news and US market returns.
