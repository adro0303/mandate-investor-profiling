from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def plot_target_diagnostics(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if len(y_true) == 0:
        return

    # 1) Time series y_true vs y_pred
    plt.figure(figsize=(11, 4))
    plt.plot(dates, y_true, label="y_true", linewidth=1.5)
    plt.plot(dates, y_pred, label="y_pred", linewidth=1.2, alpha=0.85)
    plt.title(f"{target_name} - Test: y_true vs y_pred")
    plt.xlabel("date")
    plt.ylabel("return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{target_name}_timeseries.png", dpi=130)
    plt.close()

    # 2) Scatter y_true vs y_pred with diagonal reference
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.45, s=14)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
    plt.title(f"{target_name} - Scatter")
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.tight_layout()
    plt.savefig(outdir / f"{target_name}_scatter.png", dpi=130)
    plt.close()

    # 3) Residual histogram
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=40, alpha=0.85)
    plt.title(f"{target_name} - Residuals")
    plt.xlabel("y_true - y_pred")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outdir / f"{target_name}_residuals.png", dpi=130)
    plt.close()


def plot_leaderboard(leaderboard_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = [
        "avg_mae",
        "avg_rmse",
        "avg_r2",
        "avg_directional_accuracy",
        "avg_hit_rate",
        "avg_pearson",
        "avg_spearman",
    ]
    for metric in metrics:
        if metric not in leaderboard_df.columns:
            continue
        plt.figure(figsize=(8, 4))
        plt.bar(leaderboard_df["model"], leaderboard_df[metric])
        plt.title(f"Leaderboard - {metric}")
        plt.xlabel("model")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(outdir / f"leaderboard_{metric}.png", dpi=130)
        plt.close()


def plot_target_metric_comparison(metrics_by_model: pd.DataFrame, metric: str, outpath: Path) -> None:
    if metrics_by_model.empty:
        return
    pivot = metrics_by_model.pivot(index="target", columns="model", values=metric)
    if pivot.empty:
        return
    ax = pivot.plot(kind="bar", figsize=(14, 5))
    ax.set_title(f"Per-target comparison — {metric}")
    ax.set_xlabel("target")
    ax.set_ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=130)
    plt.close()


def plot_rolling_directional_accuracy(
    pred_long_df: pd.DataFrame,
    outdir: Path,
    window: int = 30,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if pred_long_df.empty:
        return
    data = pred_long_df.copy()
    data["hit"] = (np.sign(data["y_true"]) == np.sign(data["y_pred"])).astype(float)
    for target, g in data.groupby("target"):
        g = g.sort_values("date")
        if len(g) < 3:
            continue
        roll = g["hit"].rolling(window=min(window, max(3, len(g) // 3))).mean()
        plt.figure(figsize=(10, 4))
        plt.plot(g["date"], roll)
        plt.ylim(0, 1)
        plt.title(f"Rolling directional accuracy - {target}")
        plt.xlabel("date")
        plt.ylabel("rolling_hit_rate")
        plt.tight_layout()
        plt.savefig(outdir / f"{target}_rolling_da.png", dpi=130)
        plt.close()
