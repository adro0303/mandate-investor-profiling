from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

OBJECTIVES = ["cum_return", "sharpe", "hit_rate", "directional_accuracy"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Final operational setup (model + threshold + strategy).")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--objective", choices=OBJECTIVES, required=True)
    p.add_argument("--cost-bps", type=float, required=True)
    p.add_argument("--scope", choices=["global", "per-target"], required=True)
    p.add_argument("--outdir", default="reports/selection")
    return p


def _configure_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("news_impact.select_setup")


def _metric_col(obj: str, scope: str) -> str:
    if obj == "directional_accuracy":
        return "avg_hit_rate" if scope == "global" else "hit_rate"
    if obj == "hit_rate":
        return "avg_hit_rate" if scope == "global" else "hit_rate"
    if obj == "cum_return":
        return "avg_cum_return" if scope == "global" else "cum_return"
    if obj == "sharpe":
        return "avg_sharpe" if scope == "global" else "sharpe"
    raise ValueError(f"Unsupported objective: {obj}")


def _ascending(obj: str) -> bool:
    return False


def _baseline_compare_global(df: pd.DataFrame, metric_col: str) -> tuple[str, float]:
    b = df[df["model"].isin(["baseline_zero", "baseline_last"])]
    if b.empty:
        return "", float("nan")
    row = b.sort_values(metric_col, ascending=False).iloc[0]
    return str(row["model"]), float(row[metric_col])


def _baseline_compare_target(df: pd.DataFrame, target: str, metric_col: str) -> tuple[str, float]:
    b = df[(df["target"] == target) & (df["model"].isin(["baseline_zero", "baseline_last"]))]
    if b.empty:
        return "", float("nan")
    row = b.sort_values(metric_col, ascending=False).iloc[0]
    return str(row["model"]), float(row[metric_col])


def main() -> None:
    args = _build_parser().parse_args()
    logger = _configure_logger()

    reports_dir = Path(args.reports_dir)
    metrics_dir = reports_dir / "metrics"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    global_path = metrics_dir / "backtest_global.csv"
    target_path = metrics_dir / "backtest_by_target.csv"
    if not global_path.exists() or not target_path.exists():
        raise FileNotFoundError("Missing backtest files; run `python -m news_impact.backtest` first.")

    g = pd.read_csv(global_path)
    t = pd.read_csv(target_path)
    cost = float(args.cost_bps)
    g = g[g["cost_bps"] == cost].copy()
    t = t[t["cost_bps"] == cost].copy()
    if g.empty or t.empty:
        raise ValueError(f"No backtest data for cost_bps={cost}")

    if args.scope == "global":
        metric_col = _metric_col(args.objective, "global")
        row = g.sort_values(metric_col, ascending=_ascending(args.objective)).iloc[0]
        baseline_model, baseline_value = _baseline_compare_global(g, metric_col)
        out = pd.DataFrame(
            [
                {
                    "target": "ALL",
                    "selected_model": str(row["model"]),
                    "selected_threshold": float(row["threshold"]),
                    "selected_strategy": str(row["strategy"]),
                    "reference_cost_bps": cost,
                    "objective_metric": args.objective,
                    "metric_value": float(row[metric_col]),
                    "baseline_model": baseline_model,
                    "baseline_metric_value": baseline_value,
                    "delta_vs_baseline": float(row[metric_col]) - baseline_value if np.isfinite(baseline_value) else np.nan,
                }
            ]
        )
        out_path = outdir / "final_selection_global.csv"
    else:
        metric_col = _metric_col(args.objective, "per-target")
        rows = []
        global_best = g.sort_values(_metric_col(args.objective, "global"), ascending=_ascending(args.objective)).iloc[0]
        global_tuple = (str(global_best["model"]), float(global_best["threshold"]), str(global_best["strategy"]))
        for target, tg in t.groupby("target"):
            best = tg.sort_values(metric_col, ascending=_ascending(args.objective)).iloc[0]
            baseline_model, baseline_value = _baseline_compare_target(t, target, metric_col)
            selected_tuple = (str(best["model"]), float(best["threshold"]), str(best["strategy"]))
            rows.append(
                {
                    "target": target,
                    "selected_model": str(best["model"]),
                    "selected_threshold": float(best["threshold"]),
                    "selected_strategy": str(best["strategy"]),
                    "reference_cost_bps": cost,
                    "objective_metric": args.objective,
                    "metric_value": float(best[metric_col]),
                    "baseline_model": baseline_model,
                    "baseline_metric_value": baseline_value,
                    "delta_vs_baseline": float(best[metric_col]) - baseline_value if np.isfinite(baseline_value) else np.nan,
                    "prefers_per_target_over_global": selected_tuple != global_tuple,
                }
            )
        out = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)
        out_path = outdir / "final_selection_per_target.csv"

    out.to_csv(out_path, index=False)
    logger.info("Final selection written to: %s", out_path)
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
