# News → Markets (Macro USA) – Starter Project

Inputs (place them here):
- `data/raw/dataset.csv`  (your GDELT filtered news export)
- `data/raw/market_data.csv` (your yfinance export)

Pipeline:
1) Build merged ML dataset (daily features + next-day returns).
2) Train a simple neural net baseline (PyTorch) you can tune.

## Install
```bash
pip install -r requirements.txt
```

## Build dataset
```bash
python -m news_impact.prepare_dataset --news data/raw/dataset.csv --market data/raw/market_data.csv --out data/processed/ml_dataset.parquet
```

## Train baseline NN
```bash
python -m news_impact.train --data data/processed/ml_dataset.parquet --targets UUP TLT GLD BTC-USD XLK XLF XLE XLV XLI XLY XLP XLB XLU XLRE XLC
```

Outputs are saved to `runs/<timestamp>/`.

## Extended training/evaluation pipeline

The final dataset should be at:
- `data/processed/ml_dataset.parquet`

### Train one model (CLI)
```bash
python -m news_impact.train --data data/processed/ml_dataset.parquet --model baseline_zero --outdir models/baseline_zero
python -m news_impact.train --data data/processed/ml_dataset.parquet --model baseline_last --outdir models/baseline_last
python -m news_impact.train --data data/processed/ml_dataset.parquet --model knn --outdir models/knn
python -m news_impact.train --data data/processed/ml_dataset.parquet --model ridge --outdir models/ridge
python -m news_impact.train --data data/processed/ml_dataset.parquet --model rf --outdir models/rf
python -m news_impact.train --data data/processed/ml_dataset.parquet --model mlp --outdir models/mlp
```

### Evaluate a trained model
```bash
python -m news_impact.evaluate --data data/processed/ml_dataset.parquet --modeldir models/knn --outdir reports
```

### Full runner (train + eval + leaderboard)
```bash
# quick (default)
python -m news_impact.run_all --data data/processed/ml_dataset.parquet

# full
python -m news_impact.run_all --data data/processed/ml_dataset.parquet --mode full
```

### Prediction (inference)
```bash
# explicit model
python -m news_impact.predict --data data/processed/ml_dataset.parquet --modeldir models/rf --out reports/predictions/preds.csv

# best global model by metric
python -m news_impact.predict --data data/processed/ml_dataset.parquet --best --best-by directional_accuracy --best-scope global --out reports/predictions/preds.csv

# best model per target (mixed models)
python -m news_impact.predict --data data/processed/ml_dataset.parquet --best --best-by directional_accuracy --best-scope per-target --out reports/predictions/preds.csv

# simple backtest (long-flat / long-short)
python -m news_impact.backtest --reports-dir reports --cost-bps 1.0

# backtest with confidence filter + sweep
python -m news_impact.backtest --reports-dir reports --cost-bps 1.0 --thresholds 0,0.0005,0.001,0.002,0.003,0.005

# final operational selection (global / per-target)
python -m news_impact.select_setup --reports-dir reports --objective sharpe --cost-bps 1 --scope global
python -m news_impact.select_setup --reports-dir reports --objective sharpe --cost-bps 1 --scope per-target

# inference using final selection
python -m news_impact.predict --data data/processed/ml_dataset.parquet --selection-file reports/selection/final_selection_per_target.csv --out reports/predictions/preds.csv
```

Artifacts:
- Models: `models/<model>/`
- Metrics: `reports/metrics/`
- Figures: `reports/plots/<model>/`
- Summary: `reports/summary.md`
- Predictions: `reports/predictions/preds.csv`

The pipeline includes:
- multi-target regression (15 `y_*` targets)
- time-ordered split (no shuffle) + configurable expanding walk-forward
- per-target and average metrics: MAE, RMSE, R², directional accuracy / hit rate, Pearson, Spearman
- required baselines: `baseline_zero`, `baseline_last`
- kNN “similar days” baseline
