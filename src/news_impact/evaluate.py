from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from news_impact.utils.data import build_walk_forward_folds, load_dataset
from news_impact.utils.metrics import leaderboard_row_from_summary, summarize_fold_metrics
from news_impact.utils.model_ops import (
    predict_target_model,
    safe_spearman,
    train_target_model,
)
from news_impact.utils.plotting import plot_target_diagnostics


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained news_impact model.")
    p.add_argument("--data", required=True, help="Path to final parquet/csv dataset")
    p.add_argument("--modeldir", required=True, help="Directory with trained model")
    p.add_argument("--outdir", required=True, help="Reports base directory (e.g. reports)")
    p.add_argument("--wf_start", type=float, default=0.70)
    p.add_argument("--wf_step", type=int, default=20)
    p.add_argument("--wf_test_window", type=int, default=60)
    return p


def _configure_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("news_impact.evaluate")


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))
    da = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    spearman = safe_spearman(y_true, y_pred)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": da,
        "hit_rate": da,
        "pearson": pearson,
        "spearman": spearman,
    }


def main() -> None:
    args = _build_parser().parse_args()
    logger = _configure_logger()

    modeldir = Path(args.modeldir)
    outdir = Path(args.outdir)
    metrics_dir = outdir / "metrics"

    manifest = json.loads((modeldir / "manifest.json").read_text(encoding="utf-8"))
    model_name = manifest["model_name"]
    feature_cols = manifest["feature_cols"]
    target_cols = manifest["target_cols"]
    model_params = manifest.get("model_params", {})

    df = load_dataset(args.data, logger)
    folds = build_walk_forward_folds(len(df), args.wf_start, args.wf_step, args.wf_test_window)
    logger.info("Walk-forward folds=%d wf_start=%s wf_step=%d wf_test_window=%d", len(folds), args.wf_start, args.wf_step, args.wf_test_window)

    fold_rows: list[dict] = []
    pred_rows: list[dict] = []

    for target in target_cols:
        target_series = pd.to_numeric(df[target], errors="coerce")
        prev_series = target_series.shift(1)

        for fold in folds:
            tr = df.iloc[fold.train_start : fold.train_end].copy()
            te = df.iloc[fold.test_start : fold.test_end].copy()

            tr_mask = pd.to_numeric(tr[target], errors="coerce").notna().to_numpy()
            te_mask = pd.to_numeric(te[target], errors="coerce").notna().to_numpy()
            if int(np.sum(tr_mask)) < 20 or int(np.sum(te_mask)) < 5:
                continue

            X_train = tr.loc[tr_mask, feature_cols].to_numpy(dtype=float)
            y_train = tr.loc[tr_mask, target].to_numpy(dtype=float)
            X_test = te.loc[te_mask, feature_cols].to_numpy(dtype=float)
            y_test = te.loc[te_mask, target].to_numpy(dtype=float)

            if model_name in {"baseline_zero", "baseline_last"}:
                y_prev = prev_series.iloc[fold.test_start : fold.test_end].to_numpy(dtype=float)[te_mask]
                y_pred = predict_target_model(model_name, {"kind": "baseline"}, X_test, y_prev=y_prev)
            else:
                # Walk-forward: retrain with expanding window each fold.
                val_size = max(10, int(0.15 * len(X_train)))
                if len(X_train) <= val_size + 5:
                    val_size = max(5, len(X_train) // 5)
                X_tr = X_train[:-val_size]
                y_tr = y_train[:-val_size]
                X_va = X_train[-val_size:]
                y_va = y_train[-val_size:]
                if len(X_tr) < 20:
                    X_tr, y_tr = X_train, y_train
                    X_va, y_va = X_train[-min(20, len(X_train)) :], y_train[-min(20, len(y_train)) :]
                trained = train_target_model(model_name, X_tr, y_tr, X_va, y_va, model_params)
                y_pred = predict_target_model(model_name, trained, X_test)

            m = _fold_metrics(y_test, y_pred)
            fold_rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "fold_id": fold.fold_id,
                    "n_samples": int(len(y_test)),
                    **m,
                }
            )
            for d, yt, yp in zip(te.loc[te_mask, "date"], y_test, y_pred):
                pred_rows.append(
                    {
                        "model": model_name,
                        "target": target,
                        "fold_id": fold.fold_id,
                        "date": d,
                        "y_true": float(yt),
                        "y_pred": float(yp),
                    }
                )

    fold_metrics_df = pd.DataFrame(fold_rows)
    if fold_metrics_df.empty:
        raise RuntimeError("No valid folds were produced for evaluation.")
    metrics_df = summarize_fold_metrics(fold_metrics_df)
    leaderboard = pd.DataFrame([leaderboard_row_from_summary(model_name, metrics_df)])
    preds_long_df = pd.DataFrame(pred_rows).sort_values(["target", "date"]).reset_index(drop=True)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{model_name}_metrics.csv"
    leaderboard_path = metrics_dir / f"{model_name}_leaderboard_row.csv"
    fold_path = metrics_dir / f"{model_name}_fold_metrics.csv"
    preds_path = metrics_dir / f"{model_name}_preds_long.csv"
    metrics_df.to_csv(metrics_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)
    fold_metrics_df.to_csv(fold_path, index=False)
    preds_long_df.to_csv(preds_path, index=False)

    plot_dir = outdir / "plots" / model_name
    for target in target_cols:
        tg = preds_long_df[preds_long_df["target"] == target].sort_values("date")
        plot_target_diagnostics(
            dates=tg["date"],
            y_true=tg["y_true"].to_numpy(dtype=float),
            y_pred=tg["y_pred"].to_numpy(dtype=float),
            target_name=target,
            outdir=plot_dir,
        )

    logger.info("Per-target metrics written to: %s", metrics_path)
    logger.info("Per-fold metrics written to: %s", fold_path)
    logger.info("OOF predictions written to: %s", preds_path)
    logger.info("Per-target plots written to: %s", plot_dir)
    print("\nPer-target metrics summary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
