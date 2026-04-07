from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from news_impact.utils.model_ops import safe_spearman


def _pearson_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    return float(np.mean(true_sign == pred_sign))


def compute_metrics_df(y_true: np.ndarray, y_pred: np.ndarray, target_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for i, target in enumerate(target_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        r2 = float(r2_score(yt, yp))
        dacc = _directional_accuracy(yt, yp)
        pearson = _pearson_safe(yt, yp)
        spearman = safe_spearman(yt, yp)
        rows.append(
            {
                "target": target,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "directional_accuracy": dacc,
                "hit_rate": dacc,
                "pearson": pearson,
                "spearman": spearman,
                "n_samples": int(len(yt)),
            }
        )

    df = pd.DataFrame(rows)
    avg = {
        "target": "AVG",
        "mae": float(df["mae"].mean()),
        "rmse": float(df["rmse"].mean()),
        "r2": float(df["r2"].mean()),
        "directional_accuracy": float(df["directional_accuracy"].mean()),
        "hit_rate": float(df["hit_rate"].mean()),
        "pearson": float(df["pearson"].mean(skipna=True)),
        "spearman": float(df["spearman"].mean(skipna=True)),
        "n_samples": int(df["n_samples"].sum()),
    }
    return pd.concat([df, pd.DataFrame([avg])], ignore_index=True)


def leaderboard_row(model_name: str, metrics_df: pd.DataFrame) -> dict:
    base = metrics_df[metrics_df["target"] != "AVG"]
    return {
        "model": model_name,
        "avg_mae": float(base["mae"].mean()),
        "avg_rmse": float(base["rmse"].mean()),
        "avg_r2": float(base["r2"].mean()),
        "avg_directional_accuracy": float(base["directional_accuracy"].mean()),
        "avg_hit_rate": float(base["hit_rate"].mean()),
        "avg_pearson": float(base["pearson"].mean(skipna=True)),
        "avg_spearman": float(base["spearman"].mean(skipna=True)),
        "total_valid_samples": int(base["n_samples"].sum()),
    }


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["mae", "rmse", "r2", "directional_accuracy", "hit_rate", "pearson", "spearman"]
    out_rows: list[dict] = []
    for target, g in fold_metrics.groupby("target"):
        row = {"target": target}
        for m in metric_cols:
            row[f"{m}_mean"] = float(g[m].mean()) if len(g) else float("nan")
            row[f"{m}_std"] = float(g[m].std(ddof=0)) if len(g) > 1 else 0.0
        row["n_folds"] = int(g["fold_id"].nunique())
        row["n_samples"] = int(g["n_samples"].sum())
        out_rows.append(row)

    df = pd.DataFrame(out_rows).sort_values("target").reset_index(drop=True)
    avg = {"target": "AVG"}
    for m in metric_cols:
        avg[f"{m}_mean"] = float(df[f"{m}_mean"].mean(skipna=True))
        avg[f"{m}_std"] = float(df[f"{m}_std"].mean(skipna=True))
    avg["n_folds"] = int(df["n_folds"].max()) if len(df) else 0
    avg["n_samples"] = int(df["n_samples"].sum()) if len(df) else 0
    return pd.concat([df, pd.DataFrame([avg])], ignore_index=True)


def leaderboard_row_from_summary(model_name: str, summary_df: pd.DataFrame) -> dict:
    base = summary_df[summary_df["target"] != "AVG"]
    return {
        "model": model_name,
        "avg_mae": float(base["mae_mean"].mean()),
        "avg_rmse": float(base["rmse_mean"].mean()),
        "avg_r2": float(base["r2_mean"].mean()),
        "avg_directional_accuracy": float(base["directional_accuracy_mean"].mean()),
        "avg_hit_rate": float(base["hit_rate_mean"].mean()),
        "avg_pearson": float(base["pearson_mean"].mean(skipna=True)),
        "avg_spearman": float(base["spearman_mean"].mean(skipna=True)),
        "total_valid_samples": int(base["n_samples"].sum()),
    }
