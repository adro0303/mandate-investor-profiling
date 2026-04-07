"""
Build report figures for course deliverables using only artifacts from this repository.

From the project root:
    python -m news_impact.generate_report_figures
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS / "figures_for_report"
DPI = 300
MODEL_ORDER = ["baseline_zero", "baseline_last", "ridge", "knn", "rf", "mlp"]
MODEL_LABELS = {
    "baseline_zero": "baseline_zero",
    "baseline_last": "baseline_last",
    "ridge": "ridge",
    "knn": "knn",
    "rf": "rf",
    "mlp": "mlp",
}


def _mkdir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def fig_01_target_distribution() -> tuple[bool, str, str]:
    path = REPORTS / "metrics" / "target_distribution.csv"
    df = _read_csv(path)
    if df is None or df.empty:
        return False, "fig_01_target_distribution.png", str(path)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    targets = [t.replace("y_", "") for t in df["target"].astype(str)]
    x = np.arange(len(targets))
    n_valid = df["n_valid"].to_numpy(float)
    pos = df["pos_ratio"].to_numpy(float) * n_valid
    neg = df["neg_ratio"].to_numpy(float) * n_valid
    zero = df["zero_ratio"].to_numpy(float) * n_valid
    w = 0.65
    ax1.bar(x, pos, w, label="Positive", color="#2ca02c", edgecolor="white", linewidth=0.5)
    ax1.bar(x, neg, w, bottom=pos, label="Negative", color="#d62728", edgecolor="white", linewidth=0.5)
    ax1.bar(x, zero, w, bottom=pos + neg, label="Zero", color="#7f7f7f", edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Valid sample count")
    ax1.set_xlabel("Target")
    ax1.set_title("Figure 1 — Valid samples per target (sign mix)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_01_target_distribution.png")
    return True, "fig_01_target_distribution.png", str(path)


def fig_02_04_leaderboard_metrics() -> list[tuple[bool, str, str]]:
    path = REPORTS / "metrics" / "leaderboard.csv"
    df = _read_csv(path)
    out: list[tuple[bool, str, str]] = []
    if df is None or df.empty:
        for fn in (
            "fig_02_global_rmse_comparison.png",
            "fig_03_global_directional_accuracy_comparison.png",
            "fig_04_global_mae_comparison.png",
        ):
            out.append((False, fn, str(path)))
        return out
    order_models = [m for m in MODEL_ORDER if m in df["model"].values]
    rest = [m for m in df["model"].unique() if m not in order_models]
    for col, fname, title, ascending, ref_line in [
        ("avg_rmse", "fig_02_global_rmse_comparison.png", "Global mean RMSE", True, None),
        (
            "avg_directional_accuracy",
            "fig_03_global_directional_accuracy_comparison.png",
            "Global mean directional accuracy",
            False,
            0.5,
        ),
        ("avg_mae", "fig_04_global_mae_comparison.png", "Global mean MAE", True, None),
    ]:
        sub = df[df["model"].isin(order_models + rest)].copy()
        sub = sub.sort_values(col, ascending=ascending)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#1f77b4" if m == "mlp" else "#aec7e8" for m in sub["model"]]
        ax.barh(sub["model"].astype(str), sub[col], color=colors, edgecolor="black", linewidth=0.4)
        if ref_line is not None:
            ax.axvline(ref_line, color="crimson", linestyle="--", linewidth=1.2, label="0.5 (random)")
            ax.legend(loc="lower right", fontsize=9)
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("Model")
        ax.set_title(f"Global comparison — {title}")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        _save(fig, fname)
        out.append((True, fname, str(path)))
    return out


def fig_05_best_rmse_by_target() -> tuple[bool, str, str]:
    bpath = REPORTS / "metrics" / "best_model_by_target_rmse.csv"
    mpath = REPORTS / "metrics" / "per_target_metrics.csv"
    best = _read_csv(bpath)
    met = _read_csv(mpath)
    if best is None or met is None or best.empty:
        return False, "fig_05_best_model_by_target_rmse.png", f"{bpath}; {mpath}"
    merged_rows = []
    for _, row in best.iterrows():
        t = row["target"]
        m = row["best_model_rmse"]
        hit = met[(met["target"] == t) & (met["model"] == m)]
        rmse = float(hit["rmse_mean"].iloc[0]) if len(hit) else float("nan")
        merged_rows.append({"target": t, "model": m, "rmse_mean": rmse})
    dd = pd.DataFrame(merged_rows).sort_values("rmse_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    models = dd["model"].astype(str)
    uniq = list(dict.fromkeys(models))
    tab_cmap = plt.colormaps["tab10"]
    model_to_c = {u: tab_cmap(i / max(len(uniq) - 1, 1)) for i, u in enumerate(uniq)}
    colors = [model_to_c[m] for m in models]
    y_pos = np.arange(len(dd))
    short_t = [t.replace("y_", "") for t in dd["target"].astype(str)]
    ax.barh(y_pos, dd["rmse_mean"], color=colors, edgecolor="black", linewidth=0.35)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{t}\n({m})" for t, m in zip(short_t, models)], fontsize=8)
    ax.set_xlabel("Mean RMSE (walk-forward)")
    ax.set_title("Figure 5 — Best model per target by RMSE (value on winning model)")
    ax.grid(axis="x", alpha=0.3)
    handles = [mpatches.Patch(color=model_to_c[u], label=u) for u in uniq]
    ax.legend(handles=handles, title="Model", loc="lower right", fontsize=8)
    plt.tight_layout()
    _save(fig, "fig_05_best_model_by_target_rmse.png")
    return True, "fig_05_best_model_by_target_rmse.png", f"{bpath} + {mpath}"


def fig_06_best_da_by_target() -> tuple[bool, str, str]:
    bpath = REPORTS / "metrics" / "best_model_by_target_directional_accuracy.csv"
    summ_path = REPORTS / "metrics" / "directional_accuracy_target_summary.csv"
    best = _read_csv(bpath)
    summ = _read_csv(summ_path)
    if best is None or summ is None or best.empty:
        return False, "fig_06_best_model_by_target_directional_accuracy.png", f"{bpath}; {summ_path}"
    merged = best.merge(summ[["target", "da_mean"]], on="target", how="left")
    merged = merged.sort_values("da_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    models = merged["best_model_directional_accuracy"].astype(str)
    uniq = list(dict.fromkeys(models))
    tab_cmap = plt.colormaps["tab10"]
    model_to_c = {u: tab_cmap(i / max(len(uniq) - 1, 1)) for i, u in enumerate(uniq)}
    colors = [model_to_c[m] for m in models]
    y_pos = np.arange(len(merged))
    short_t = [t.replace("y_", "") for t in merged["target"].astype(str)]
    ax.barh(y_pos, merged["da_mean"], color=colors, edgecolor="black", linewidth=0.35)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{t}\n({m})" for t, m in zip(short_t, models)], fontsize=8)
    ax.set_xlabel("Directional accuracy (best model)")
    ax.set_title("Figure 6 — Best model per target by directional accuracy")
    ax.axvline(0.5, color="crimson", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=0.3)
    handles = [mpatches.Patch(color=model_to_c[u], label=u) for u in uniq]
    ax.legend(handles=handles, title="Model", loc="lower right", fontsize=8)
    plt.tight_layout()
    _save(fig, "fig_06_best_model_by_target_directional_accuracy.png")
    return True, "fig_06_best_model_by_target_directional_accuracy.png", f"{bpath} + {summ_path}"


def fig_07_da_gaps() -> tuple[bool, str, str]:
    path = REPORTS / "metrics" / "directional_accuracy_target_summary.csv"
    df = _read_csv(path)
    if df is None or df.empty:
        return False, "fig_07_directional_accuracy_gaps_vs_baselines.png", str(path)
    df = df.sort_values("target")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["da_mean"], w, label="Best-model DA", color="#1f77b4")
    ax.bar(x, df["gap_vs_baseline_last"], w, label="Gap vs baseline_last", color="#ff7f0e")
    ax.bar(x + w, df["gap_vs_baseline_zero"], w, label="Gap vs baseline_zero", color="#2ca02c")
    labels = [t.replace("y_", "") for t in df["target"].astype(str)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.5, color="crimson", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_ylabel("Share")
    ax.set_xlabel("Target")
    ax.set_title("Figure 7 — Best-model directional accuracy and gaps vs baselines")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_07_directional_accuracy_gaps_vs_baselines.png")
    return True, "fig_07_directional_accuracy_gaps_vs_baselines.png", str(path)


def fig_08_heatmap() -> tuple[bool, str, str]:
    path = REPORTS / "metrics" / "per_target_metrics.csv"
    df = _read_csv(path)
    if df is None or df.empty:
        return False, "fig_08_per_target_model_heatmap.png", str(path)
    p_da = df.pivot_table(index="target", columns="model", values="directional_accuracy_mean", aggfunc="mean")
    p_rmse = df.pivot_table(index="target", columns="model", values="rmse_mean", aggfunc="mean")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title, vmin, vmax in [
        (axes[0], p_da, "Directional accuracy", 0.0, 1.0),
        (axes[1], p_rmse, "RMSE", None, None),
    ]:
        short_idx = [i.replace("y_", "") for i in mat.index.astype(str)]
        im = ax.imshow(mat.values, aspect="auto", cmap="viridis" if "DA" in title else "viridis_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(short_idx, fontsize=8)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Figure 8 — Target × model heatmaps (walk-forward metrics)", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, "fig_08_per_target_model_heatmap.png")
    return True, "fig_08_per_target_model_heatmap.png", str(path)


def _backtest_long_flat_cost(df: pd.DataFrame, cost: float) -> pd.DataFrame:
    return df[(df["strategy"] == "long_flat") & (df["cost_bps"] == cost)].copy()


def fig_09_backtest_cumret() -> tuple[bool, str, str]:
    path = REPORTS / "metrics" / "backtest_global.csv"
    df = _read_csv(path)
    if df is None or df.empty:
        return False, "fig_09_backtest_long_flat_cum_return_cost1.png", str(path)
    sub = _backtest_long_flat_cost(df, 1.0)
    if sub.empty:
        return False, "fig_09_backtest_long_flat_cum_return_cost1.png", str(path)
    models = [m for m in MODEL_ORDER if m in sub["model"].values]
    thresholds = sorted(sub["threshold"].unique())
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(thresholds))
    n = len(models)
    width = 0.12 / max(n, 1) * 6
    for i, m in enumerate(models):
        vals = []
        for th in thresholds:
            r = sub[(sub["model"] == m) & (sub["threshold"] == th)]
            vals.append(float(r["avg_cum_return"].iloc[0]) if len(r) else np.nan)
        ax.bar(x + (i - n / 2) * width, vals, width, label=m, edgecolor="black", linewidth=0.2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean cumulative return")
    ax.set_title("Figure 9 — Global backtest long_flat at 1 bps: cumulative return by model and threshold")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_09_backtest_long_flat_cum_return_cost1.png")
    return True, "fig_09_backtest_long_flat_cum_return_cost1.png", str(path)


def fig_10_backtest_sharpe() -> tuple[bool, str, str]:
    path = REPORTS / "metrics" / "backtest_global.csv"
    df = _read_csv(path)
    if df is None or df.empty:
        return False, "fig_10_backtest_long_flat_sharpe_cost1.png", str(path)
    sub = _backtest_long_flat_cost(df, 1.0)
    models = [m for m in MODEL_ORDER if m in sub["model"].values]
    thresholds = sorted(sub["threshold"].unique())
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(thresholds))
    n = len(models)
    width = 0.12 / max(n, 1) * 6
    for i, m in enumerate(models):
        vals = []
        for th in thresholds:
            r = sub[(sub["model"] == m) & (sub["threshold"] == th)]
            v = float(r["avg_sharpe"].iloc[0]) if len(r) and pd.notna(r["avg_sharpe"].iloc[0]) else np.nan
            vals.append(v)
        ax.bar(x + (i - n / 2) * width, vals, width, label=m, edgecolor="black", linewidth=0.2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean Sharpe")
    ax.set_title("Figure 10 — Global backtest long_flat at 1 bps: Sharpe by model and threshold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_10_backtest_long_flat_sharpe_cost1.png")
    return True, "fig_10_backtest_long_flat_sharpe_cost1.png", str(path)


def fig_11_cost_sensitivity() -> tuple[bool, str, str]:
    """Highlighted setups: MLP thr=0 and RF thr=0 at 1 bps (long_flat); kNN thr=0.005 (strong at 5 bps Sharpe)."""
    path = REPORTS / "metrics" / "backtest_global.csv"
    df = _read_csv(path)
    if df is None or df.empty:
        return False, "fig_11_cost_sensitivity_best_setups.png", str(path)
    setups = [
        ("mlp", "long_flat", 0.0, "MLP thr=0 (max Sharpe @1bps)"),
        ("rf", "long_flat", 0.0, "RF thr=0 (max CumRet @1bps)"),
        ("knn", "long_flat", 0.005, "kNN thr=0.005 (max Sharpe @5bps)"),
    ]
    costs = [1.0, 3.0, 5.0]
    rows = []
    for cost in costs:
        for model, strat, thr, label in setups:
            r = df[(df["cost_bps"] == cost) & (df["strategy"] == strat) & (df["model"] == model) & (df["threshold"] == thr)]
            if len(r):
                rows.append(
                    {
                        "cost_bps": cost,
                        "label": label,
                        "avg_cum_return": float(r["avg_cum_return"].iloc[0]),
                        "avg_sharpe": float(r["avg_sharpe"].iloc[0]) if pd.notna(r["avg_sharpe"].iloc[0]) else np.nan,
                    }
                )
    if not rows:
        return False, "fig_11_cost_sensitivity_best_setups.png", str(path)
    d = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for lbl, g in d.groupby("label"):
        ax1.plot(g["cost_bps"], g["avg_cum_return"], marker="o", label=lbl)
        ax2.plot(g["cost_bps"], g["avg_sharpe"], marker="s", label=lbl)
    ax1.set_xlabel("Cost (bps)")
    ax1.set_ylabel("avg_cum_return")
    ax1.set_title("Cumulative return vs cost")
    ax1.set_xticks(costs)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, loc="best")
    ax2.set_xlabel("Cost (bps)")
    ax2.set_ylabel("avg_sharpe")
    ax2.set_title("Sharpe vs cost")
    ax2.set_xticks(costs)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7, loc="best")
    fig.suptitle("Figure 11 — Cost sweep 1/3/5 bps for highlighted setups (backtest_global)", fontsize=11)
    plt.tight_layout()
    _save(fig, "fig_11_cost_sensitivity_best_setups.png")
    return True, "fig_11_cost_sensitivity_best_setups.png", str(path)


def fig_12_best_thresholds() -> tuple[bool, str, str]:
    path = REPORTS / "metrics" / "backtest_threshold_best.csv"
    df = _read_csv(path)
    if df is None or df.empty:
        return False, "fig_12_best_thresholds_by_model.png", str(path)
    sub = df[
        (df["scope"] == "global")
        & (df["target"] == "ALL")
        & (df["strategy"] == "long_flat")
        & (df["model"].isin(MODEL_ORDER))
        & (df["best_by"].isin(["sharpe", "cum_return"]))
    ].copy()
    if sub.empty:
        return False, "fig_12_best_thresholds_by_model.png", str(path)
    costs = sorted(sub["cost_bps"].unique())
    fig, axes = plt.subplots(len(costs), 2, figsize=(11, 3.2 * max(len(costs), 1)))
    for i, c in enumerate(costs):
        for j, (by, title) in enumerate([("sharpe", "Best by Sharpe"), ("cum_return", "Best by cum. return")]):
            ax = axes[i, j]
            part = sub[(sub["cost_bps"] == c) & (sub["best_by"] == by)].sort_values("model")
            if part.empty:
                ax.set_visible(False)
                continue
            ax.barh(part["model"].astype(str), part["best_threshold"].astype(float), color="#17becf", edgecolor="black")
            ax.set_xlabel("best_threshold")
            ax.set_title(f"{title} — cost_bps={c}")
            ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Figure 12 — Global best thresholds (backtest_threshold_best, long_flat)", fontsize=11, y=1.02)
    plt.tight_layout()
    _save(fig, "fig_12_best_thresholds_by_model.png")
    return True, "fig_12_best_thresholds_by_model.png", str(path)


def _selection_global_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        (REPORTS / "selection" / "final_selection_global.csv", "default_root"),
        (REPORTS / "selection" / "sharpe_1" / "final_selection_global.csv", "sharpe_1bps"),
        (REPORTS / "selection" / "sharpe_5" / "final_selection_global.csv", "sharpe_5bps"),
        (REPORTS / "selection" / "cumret_1" / "final_selection_global.csv", "cumret_1bps"),
    ]
    for path, tag in specs:
        df = _read_csv(path)
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            rows.append(
                {
                    "fuente": tag,
                    "archivo": str(path.relative_to(PROJECT_ROOT)),
                    "objective_metric": r.get("objective_metric", ""),
                    "reference_cost_bps": r.get("reference_cost_bps", ""),
                    "selected_model": r.get("selected_model", ""),
                    "selected_threshold": r.get("selected_threshold", ""),
                    "selected_strategy": r.get("selected_strategy", ""),
                    "metric_value": r.get("metric_value", ""),
                }
            )
    return rows


def fig_13_global_selection() -> tuple[bool, str, str]:
    rows = _selection_global_rows()
    if not rows:
        return False, "fig_13_global_selection_summary.png", "final_selection_global.csv (multiple paths)"
    fig, ax = plt.subplots(figsize=(14, 2 + 0.45 * len(rows)))
    ax.axis("off")
    colnames = [
        "Source",
        "Objective",
        "cost_bps",
        "Model",
        "Threshold",
        "Strategy",
        "Metric",
    ]
    table_data = []
    sources = []
    for r in rows:
        table_data.append(
            [
                r["fuente"],
                str(r["objective_metric"]),
                str(r["reference_cost_bps"]),
                str(r["selected_model"]),
                str(r["selected_threshold"]),
                str(r["selected_strategy"]),
                f"{r['metric_value']:.6g}" if isinstance(r["metric_value"], (int, float)) else str(r["metric_value"]),
            ]
        )
        sources.append(r["archivo"])
    tab = ax.table(
        cellText=table_data,
        colLabels=colnames,
        loc="center",
        cellLoc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.05, 1.35)
    ax.set_title("Figure 13 — Global selection summary (CSV under reports/selection)", fontsize=11, pad=12)
    plt.tight_layout()
    _save(fig, "fig_13_global_selection_summary.png")
    return True, "fig_13_global_selection_summary.png", "; ".join(sorted(set(sources)))


def fig_14_per_target_selection() -> tuple[bool, str, str]:
    path = REPORTS / "selection" / "final_selection_per_target.csv"
    df = _read_csv(path)
    if df is None or df.empty:
        return False, "fig_14_per_target_selection_summary.png", str(path)
    models = list(dict.fromkeys(df["selected_model"].astype(str)))
    t_short = [t.replace("y_", "") for t in df["target"].astype(str)]
    model_idx = {m: i for i, m in enumerate(models)}
    mat = np.zeros((len(df), len(models)))
    mat[:] = np.nan
    for ti, (_, row) in enumerate(df.iterrows()):
        mi = model_idx[str(row["selected_model"])]
        mat[ti, mi] = 1.0
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(len(t_short)))
    ax.set_yticklabels(t_short, fontsize=9)
    ax.set_xlabel("Selected model")
    ax.set_ylabel("Target")
    ann_thr = df["selected_threshold"].astype(str).tolist()
    ann_st = df["selected_strategy"].astype(str).tolist()
    for i in range(len(t_short)):
        ax.text(
            model_idx[df.iloc[i]["selected_model"]],
            i,
            f"thr={ann_thr[i]}\n{ann_st[i]}",
            ha="center",
            va="center",
            fontsize=5,
            color="darkred" if mat[i, model_idx[df.iloc[i]["selected_model"]]] > 0 else "black",
        )
    ax.set_title("Figure 14 — Per-target selection (final_selection_per_target.csv)")
    plt.colorbar(im, ax=ax, fraction=0.03, label="Selected (1)")
    plt.tight_layout()
    _save(fig, "fig_14_per_target_selection_summary.png")
    return True, "fig_14_per_target_selection_summary.png", str(path)


def fig_15_preference() -> tuple[bool, str, str]:
    files = [
        (REPORTS / "selection" / "sharpe_1" / "final_selection_per_target.csv", "Sharpe 1 bps"),
        (REPORTS / "selection" / "sharpe_5" / "final_selection_per_target.csv", "Sharpe 5 bps"),
        (REPORTS / "selection" / "cumret_1" / "final_selection_per_target.csv", "Cum. return 1 bps"),
    ]
    labels = []
    pref_t = []
    glob_t = []
    src = []
    for path, lab in files:
        df = _read_csv(path)
        if df is None or "prefers_per_target_over_global" not in df.columns:
            continue
        pt = int(df["prefers_per_target_over_global"].fillna(False).astype(bool).sum())
        total = len(df)
        labels.append(lab)
        pref_t.append(pt)
        glob_t.append(total - pt)
        src.append(str(path.relative_to(PROJECT_ROOT)))
    if not labels:
        return False, "fig_15_per_target_vs_global_preference.png", "sharpe_1|sharpe_5|cumret_1 final_selection_per_target"
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, pref_t, w, label="Prefer per-target", color="#2ca02c")
    ax.bar(x + w / 2, glob_t, w, label="Else (global tie or better)", color="#c7c7c7")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of targets (of 15)")
    ax.set_title("Figure 15 — Per-target vs global preference (prefers_per_target_over_global)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_15_per_target_vs_global_preference.png")
    return True, "fig_15_per_target_vs_global_preference.png", "; ".join(src)


def fig_16_mlp_architecture() -> tuple[bool, str, str]:
    mpath = PROJECT_ROOT / "models" / "mlp" / "manifest.json"
    if not mpath.exists():
        return False, "fig_16_mlp_architecture_diagram.png", str(mpath)
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    n_in = len(manifest.get("feature_cols", []))
    hidden = manifest.get("model_params", {}).get("hidden", 256)
    drop = manifest.get("model_params", {}).get("dropout", 0.1)
    n_tar = len(manifest.get("target_cols", []))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#e8f4f8"):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.2", linewidth=1.2, edgecolor="#333", facecolor=fc
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=14, linewidth=1.2, color="#333"))

    box(0.8, 7.0, 2.2, 1.6, f"Input\n{n_in} features\n(see manifest)", "#dae8fc")
    arrow(3.0, 7.8, 4.0, 7.8)
    box(4.0, 7.0, 2.8, 1.6, f"Hidden 1\nLinear({n_in}→{hidden})\nReLU + Dropout {drop}", "#fff2cc")
    arrow(6.8, 7.8, 7.5, 7.8)
    box(7.5, 7.0, 2.5, 1.6, f"Hidden 2\nLinear({hidden}→{hidden})\nReLU + Dropout {drop}", "#fff2cc")
    arrow(9.25, 7.0, 9.25, 5.5)
    box(7.5, 3.5, 2.5, 1.6, "Output\nLinear → 1 scalar\n(one model per target)", "#e1d5e7")
    arrow(8.25, 5.5, 8.25, 5.1)
    note = (
        f"Training uses {n_tar} separate MLP heads\n"
        f"(one per target; out_dim=1 in code), not one layer with {n_tar} outputs.\n"
        "Source: src/news_impact/modeling.py MLPRegressor + models/mlp/manifest.json"
    )
    ax.text(5, 1.2, note, ha="center", va="top", fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.set_title("Figure 16 — MLP architecture (as implemented)", fontsize=12)
    _save(fig, "fig_16_mlp_architecture_diagram.png")
    return True, "fig_16_mlp_architecture_diagram.png", f"{mpath} + src/news_impact/modeling.py"


def fig_17_pipeline() -> tuple[bool, str, str]:
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    boxes = [
        (1, 10.2, "GDELT / news\n(data/raw/dataset.csv)", "#d5e8d4"),
        (5.5, 10.2, "Prices / returns\n(data/raw/market_data.csv)", "#d5e8d4"),
        (2.8, 8.5, "prepare_dataset.py\n→ ml_dataset.parquet", "#fff2cc"),
        (2.8, 6.8, "train.py / evaluate.py\nrun_all.py", "#dae8fc"),
        (2.8, 5.0, "Trained models in models/", "#f8cecc"),
        (2.8, 3.2, "backtest.py\n→ reports/metrics/backtest_*.csv", "#e1d5e7"),
        (2.8, 1.4, "select_setup.py\n→ reports/selection/", "#e1d5e7"),
        (6.5, 3.2, "predict.py\n→ reports/predictions/", "#f5f5f5"),
    ]
    for x, y, txt, c in boxes:
        patch = FancyBboxPatch(
            (x, y), 3.2, 0.9, boxstyle="round,pad=0.02", facecolor=c, edgecolor="#333", linewidth=1
        )
        ax.add_patch(patch)
        ax.text(x + 1.6, y + 0.45, txt, ha="center", va="center", fontsize=8)
    arrows = [
        (3.4, 10.2, 3.8, 9.4),
        (6.2, 10.2, 5.2, 9.4),
        (4.4, 8.5, 4.4, 7.7),
        (4.4, 6.8, 4.4, 5.9),
        (4.4, 5.0, 4.4, 4.1),
        (4.4, 3.2, 4.4, 2.3),
        (5.6, 3.6, 6.4, 3.6),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12, color="#333"))
    ax.set_title("Figure 17 — Project workflow (news_impact repo layout)", fontsize=11)
    _save(fig, "fig_17_project_pipeline.png")
    return True, "fig_17_project_pipeline.png", "README.md + src/news_impact/*.py tree"


def fig_18_dashboard() -> tuple[bool, str, str]:
    lb_path = REPORTS / "metrics" / "leaderboard.csv"
    bt_path = REPORTS / "metrics" / "backtest_global.csv"
    lb = _read_csv(lb_path)
    bt = _read_csv(bt_path)
    if lb is None or lb.empty or bt is None or bt.empty:
        return False, "fig_18_results_dashboard.png", f"{lb_path}; {bt_path}"
    best_rmse = lb.loc[lb["avg_rmse"].idxmin()]
    best_da = lb.loc[lb["avg_directional_accuracy"].idxmax()]
    sub1 = bt[(bt["strategy"] == "long_flat") & (bt["cost_bps"] == 1.0)]
    sub_sh = sub1.dropna(subset=["avg_sharpe"])
    best_sh = sub_sh.loc[sub_sh["avg_sharpe"].idxmax()] if len(sub_sh) else sub1.iloc[0]
    best_cr = sub1.loc[sub1["avg_cum_return"].idxmax()]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    blocks = [
        (axes[0, 0], "Best RMSE (global)", f"Model: {best_rmse['model']}\navg_rmse: {best_rmse['avg_rmse']:.6f}"),
        (axes[0, 1], "Best directional acc. (global)", f"Model: {best_da['model']}\nDA: {best_da['avg_directional_accuracy']:.6f}"),
        (
            axes[1, 0],
            "Best Sharpe @1bps (long_flat)",
            f"Model: {best_sh['model']}\nthr: {best_sh['threshold']}\nSharpe: {best_sh['avg_sharpe']:.6f}",
        ),
        (
            axes[1, 1],
            "Best cum. return @1bps (long_flat)",
            f"Model: {best_cr['model']}\nthr: {best_cr['threshold']}\nCumRet: {best_cr['avg_cum_return']:.6f}",
        ),
    ]
    for ax, title, txt in blocks:
        ax.axis("off")
        ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=12, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.25, txt, ha="center", va="center", fontsize=10, family="monospace", transform=ax.transAxes)
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.05, 0.05), 0.9, 0.9, transform=ax.transAxes, boxstyle="round,pad=0.02", fill=False, edgecolor="#333", linewidth=1.2
            )
        )
    fig.suptitle("Figure 18 — Summary panel (leaderboard + backtest_global)")
    plt.tight_layout()
    _save(fig, "fig_18_results_dashboard.png")
    return True, "fig_18_results_dashboard.png", f"{lb_path}; {bt_path}"


def fig_optional_a_preds() -> tuple[bool, str, str]:
    path = REPORTS / "predictions" / "preds_from_selection.csv"
    df = _read_csv(path)
    if df is None or df.empty or "date" not in df.columns:
        return False, "fig_optional_A_prediction_example.png", str(path)
    target_tag = "y_TLT"
    col_pred = f"pred_{target_tag}"
    col_true = f"true_{target_tag}"
    if col_pred not in df.columns or col_true not in df.columns:
        return False, "fig_optional_A_prediction_example.png", f"Missing columns {col_pred}/{col_true}"
    d = df.dropna(subset=[col_pred, col_true]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").tail(120)
    if len(d) < 10:
        return False, "fig_optional_A_prediction_example.png", "Insufficient samples after filters"
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(d["date"], d[col_true], label="Actual return", color="black", linewidth=1)
    ax.plot(d["date"], d[col_pred], label="Prediction", color="#1f77b4", alpha=0.85, linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.set_title(f"Optional figure A — Prediction vs actual ({target_tag}, last {len(d)} days)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    _save(fig, "fig_optional_A_prediction_example.png")
    return True, "fig_optional_A_prediction_example.png", str(path)


def fig_optional_b_signals() -> tuple[bool, str, str]:
    path = REPORTS / "predictions" / "preds_from_selection.csv"
    df = _read_csv(path)
    if df is None or df.empty:
        return False, "fig_optional_B_signal_distribution.png", str(path)
    sig_cols = [c for c in df.columns if c.startswith("signal_y_")]
    if not sig_cols:
        return False, "fig_optional_B_signal_distribution.png", "No signal_y_* columns"
    vals = []
    labels = []
    for c in sorted(sig_cols):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        vals.append(s.values)
        labels.append(c.replace("signal_y_", ""))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(vals, bins=[-1.5, -0.5, 0.5, 1.5], stacked=True, label=labels, alpha=0.75, edgecolor="black")
    ax.set_xlabel("Discrete signal")
    ax.set_ylabel("Count")
    ax.set_title("Optional figure B — Stacked signal counts by target (preds_from_selection)")
    ax.legend(ncol=5, fontsize=7, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig_optional_B_signal_distribution.png")
    return True, "fig_optional_B_signal_distribution.png", str(path)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.dpi": 100,
            "savefig.dpi": DPI,
        }
    )
    _mkdir()
    missing: list[str] = []
    index: list[tuple[str, str, str, str]] = []

    def try_fig(label: str, fn, desc: str, src: str) -> None:
        try:
            ok, fname, sp = fn()
        except Exception as e:
            missing.append(f"- **{label}**: exception `{type(e).__name__}: {e}`")
            return
        if ok:
            index.append((fname, desc, src, desc))
        else:
            missing.append(f"- **{label}**: not generated; source: {sp}")

    try_fig(
        "Figure 1",
        fig_01_target_distribution,
        "Valid samples per target with fractions of positive / negative / zero returns.",
        "`reports/metrics/target_distribution.csv`",
    )

    out_lb = fig_02_04_leaderboard_metrics()
    idx_descriptions = [
        (
            "Figure 2",
            "Global mean RMSE by model (ascending).",
            "`reports/metrics/leaderboard.csv`",
        ),
        (
            "Figure 3",
            "Global directional accuracy; reference line at 0.5.",
            "`reports/metrics/leaderboard.csv`",
        ),
        (
            "Figure 4",
            "Global mean MAE by model.",
            "`reports/metrics/leaderboard.csv`",
        ),
    ]
    for i, (label, desc, src) in enumerate(idx_descriptions):
        if i < len(out_lb):
            ok, fname, sp = out_lb[i]
            if ok:
                index.append((fname, desc, src, desc))
            else:
                missing.append(f"- **{label}**: not generated; source: {sp}")

    for label, fn, desc, src in [
        (
            "Figure 5",
            fig_05_best_rmse_by_target,
            "RMSE winner per target with the winning model’s RMSE.",
            "`best_model_by_target_rmse.csv` + `per_target_metrics.csv`",
        ),
        (
            "Figure 6",
            fig_06_best_da_by_target,
            "Directional-accuracy winner per target with that model’s DA.",
            "`best_model_by_target_directional_accuracy.csv` + `directional_accuracy_target_summary.csv`",
        ),
        (
            "Figure 7",
            fig_07_da_gaps,
            "Best-model DA and gaps vs baselines.",
            "`directional_accuracy_target_summary.csv`",
        ),
        ("Figure 8", fig_08_heatmap, "Target×model heatmaps for DA and RMSE.", "`per_target_metrics.csv`"),
        (
            "Figure 9",
            fig_09_backtest_cumret,
            "Global backtest long_flat at 1 bps: cum. return by model and threshold.",
            "`backtest_global.csv`",
        ),
        (
            "Figure 10",
            fig_10_backtest_sharpe,
            "Global backtest long_flat at 1 bps: Sharpe by model and threshold.",
            "`backtest_global.csv`",
        ),
        (
            "Figure 11",
            fig_11_cost_sensitivity,
            "Metrics vs transaction cost (1/3/5 bps) for highlighted setups.",
            "`backtest_global.csv`",
        ),
        (
            "Figure 12",
            fig_12_best_thresholds,
            "Global best thresholds by model and cost.",
            "`backtest_threshold_best.csv`",
        ),
        (
            "Figure 13",
            fig_13_global_selection,
            "Table of global selections from selection CSVs.",
            "`reports/selection/**/final_selection_global.csv`",
        ),
        (
            "Figure 14",
            fig_14_per_target_selection,
            "Chosen model per target.",
            "`reports/selection/final_selection_per_target.csv`",
        ),
        (
            "Figure 15",
            fig_15_preference,
            "Counts of prefers_per_target_over_global by scenario.",
            "`reports/selection/sharpe_1|sharpe_5|cumret_1/final_selection_per_target.csv`",
        ),
        (
            "Figure 16",
            fig_16_mlp_architecture,
            "MLP architecture diagram from code and manifest.",
            "`models/mlp/manifest.json` + `src/news_impact/modeling.py`",
        ),
        ("Figure 17", fig_17_pipeline, "Repository pipeline flow diagram.", "`src/news_impact` layout + README"),
        (
            "Figure 18",
            fig_18_dashboard,
            "Summary of best global metrics and 1 bps backtest.",
            "`leaderboard.csv` + `backtest_global.csv`",
        ),
    ]:
        try_fig(label, fn, desc, src)

    for label, fn, desc, src in [
        ("Optional A", fig_optional_a_preds, "Predicted vs actual series (example y_TLT).", "`reports/predictions/preds_from_selection.csv`"),
        ("Optional B", fig_optional_b_signals, "Histogram of discrete signals per target.", "`reports/predictions/preds_from_selection.csv`"),
    ]:
        try_fig(label, fn, desc, src)

    index_path = FIG_DIR / "FIGURE_INDEX.md"
    lines = [
        "# FIGURE_INDEX — Report figures (6009CMD)",
        "",
        "Auto-generated by `python -m news_impact.generate_report_figures`.",
        "",
        "## How to regenerate figures",
        "",
        "From the repository root (`news_impact_nn_project/`):",
        "",
        "- **Windows (PowerShell):** `$env:PYTHONPATH = \"src\"; python -m news_impact.generate_report_figures`",
        "- **Linux / macOS:** `PYTHONPATH=src python -m news_impact.generate_report_figures`",
        "",
        "If the package is installed in editable mode (`pip install -e .`), you can run `python -m news_impact.generate_report_figures` without setting PYTHONPATH.",
        "",
        "---",
        "",
    ]
    for fname, desc_show, fuente, pie in index:
        lines.extend(
            [
                f"## `{fname}`",
                "",
                f"- **What it shows:** {desc_show}",
                f"- **Data source:** {fuente}",
                f"- **Suggested caption:** {pie}",
                "",
            ]
        )
    index_path.write_text("\n".join(lines), encoding="utf-8")

    miss_path = FIG_DIR / "MISSING_FIGURES.md"
    if missing:
        miss_path.write_text(
            "# Missing figures or issues\n\n" + "\n".join(missing) + "\n",
            encoding="utf-8",
        )
    else:
        miss_path.write_text(
            "# Missing figures or issues\n\n"
            "All requested figures were generated successfully (including optional ones when data was available).\n",
            encoding="utf-8",
        )

    n_ok = len(index)
    n_miss = len(missing)
    print(f"Figures generated (index entries): {n_ok}")
    print(f"Issues / not generated: {n_miss}")
    print(f"Index: {index_path}")
    print(f"Issues log: {miss_path}")


if __name__ == "__main__":
    main()
