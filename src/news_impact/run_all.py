from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from news_impact.utils.data import get_feature_target_cols, load_dataset
from news_impact.utils.plotting import plot_leaderboard, plot_rolling_directional_accuracy, plot_target_metric_comparison

BEST_BY_CHOICES = ["rmse", "mae", "r2", "pearson", "spearman", "directional_accuracy", "hit_rate"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and evaluate all news_impact models.")
    p.add_argument("--data", required=True, help="Path to data/processed/ml_dataset.parquet")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--outdir", default="reports")
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--rank-by", choices=BEST_BY_CHOICES, default="directional_accuracy")
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--wf_start", type=float, default=0.70)
    p.add_argument("--wf_step", type=int, default=20)
    p.add_argument("--wf_test_window", type=int, default=60)
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--force", action="store_true", help="Force retraining and re-evaluation.")
    return p


def _configure_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("news_impact.run_all")


def _run(cmd: list[str], logger: logging.Logger) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, text=True)


def _metric_col_global(best_by: str) -> str:
    return f"avg_{best_by}"


def _metric_col_target(best_by: str) -> str:
    return f"{best_by}_mean"


def _ascending(best_by: str) -> bool:
    return best_by in {"rmse", "mae"}


def _pick_best_model_global(leaderboard: pd.DataFrame, best_by: str) -> str:
    col = _metric_col_global(best_by)
    asc = _ascending(best_by)
    return str(leaderboard.sort_values(col, ascending=asc).iloc[0]["model"])


def _make_md_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["| empty |", "| --- |"]
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join([str(row[c]) for c in cols]) + " |")
    return [header, sep, *rows]


def main() -> None:
    args = _build_parser().parse_args()
    logger = _configure_logger()
    py = sys.executable

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    models = ["baseline_zero", "baseline_last", "knn", "ridge", "rf", "mlp"]
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.outdir)
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_rows: list[pd.DataFrame] = []

    if args.mode == "quick":
        rf_estimators = 120
        mlp_epochs = 25
    else:
        rf_estimators = 400
        mlp_epochs = 120

    for model in models:
        model_dir = models_dir / model
        metric_file = metrics_dir / f"{model}_metrics.csv"
        row_file = metrics_dir / f"{model}_leaderboard_row.csv"

        if args.force or not (model_dir / "manifest.json").exists():
            train_cmd = [
                py,
                "-m",
                "news_impact.train",
                "--data",
                str(data_path),
                "--model",
                model,
                "--outdir",
                str(model_dir),
                "--rf-estimators",
                str(rf_estimators),
                "--epochs",
                str(mlp_epochs),
            ]
            if model == "knn":
                train_cmd += ["--knn-k", str(args.knn_k)]
            _run(train_cmd, logger)
        else:
            logger.info("Reusing existing model: %s", model_dir)

        if args.force or not (metric_file.exists() and row_file.exists()):
            eval_cmd = [
                py,
                "-m",
                "news_impact.evaluate",
                "--data",
                str(data_path),
                "--modeldir",
                str(model_dir),
                "--outdir",
                str(reports_dir),
                "--wf_start",
                str(args.wf_start),
                "--wf_step",
                str(args.wf_step),
                "--wf_test_window",
                str(args.wf_test_window),
            ]
            _run(eval_cmd, logger)
        else:
            logger.info("Reusing existing metrics for: %s", model)

        row_path = metrics_dir / f"{model}_leaderboard_row.csv"
        leaderboard_rows.append(pd.read_csv(row_path))

    rank_col = _metric_col_global(args.rank_by)
    leaderboard = pd.concat(leaderboard_rows, ignore_index=True).sort_values(rank_col, ascending=_ascending(args.rank_by))
    leaderboard_path = metrics_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    plot_leaderboard(leaderboard, reports_dir / "plots")
    metrics_long = []
    for model in models:
        dfm = pd.read_csv(metrics_dir / f"{model}_metrics.csv")
        dfm = dfm[dfm["target"] != "AVG"].copy()
        dfm["model"] = model
        metrics_long.append(dfm)
    metrics_long_df = pd.concat(metrics_long, ignore_index=True)
    metrics_long_df.to_csv(metrics_dir / "per_target_metrics.csv", index=False)
    plot_target_metric_comparison(metrics_long_df, "r2_mean", reports_dir / "plots" / "targets_r2_comparison.png")
    plot_target_metric_comparison(
        metrics_long_df,
        "directional_accuracy_mean",
        reports_dir / "plots" / "targets_directional_accuracy_comparison.png",
    )
    fold_rows = []
    for model in models:
        fpath = metrics_dir / f"{model}_fold_metrics.csv"
        if not fpath.exists():
            continue
        fdf = pd.read_csv(fpath)
        fdf["model"] = model
        fold_rows.append(fdf)
    fold_metrics_long_df = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    if not fold_metrics_long_df.empty:
        fold_metrics_long_df.to_csv(metrics_dir / "per_target_fold_metrics.csv", index=False)

    best_model = _pick_best_model_global(leaderboard, args.rank_by)
    best_preds = pd.read_csv(metrics_dir / f"{best_model}_preds_long.csv")
    best_preds["date"] = pd.to_datetime(best_preds["date"])
    plot_rolling_directional_accuracy(best_preds, reports_dir / "plots" / best_model)

    # Dataset / targets diagnostics
    ds_logger = logging.getLogger("news_impact.run_all.dataset")
    df = load_dataset(str(data_path), ds_logger)
    _, target_cols = get_feature_target_cols(df)
    target_stats_rows = []
    for t in target_cols:
        y = pd.to_numeric(df[t], errors="coerce")
        valid = y.dropna()
        pos_ratio = float((valid > 0).mean()) if len(valid) else np.nan
        neg_ratio = float((valid < 0).mean()) if len(valid) else np.nan
        zero_ratio = float((valid == 0).mean()) if len(valid) else np.nan
        target_stats_rows.append(
            {
                "target": t,
                "n_valid": int(len(valid)),
                "mean": float(valid.mean()) if len(valid) else np.nan,
                "std": float(valid.std(ddof=0)) if len(valid) else np.nan,
                "pos_ratio": pos_ratio,
                "neg_ratio": neg_ratio,
                "zero_ratio": zero_ratio,
            }
        )
    target_stats_df = pd.DataFrame(target_stats_rows).sort_values("target")
    target_stats_df.to_csv(metrics_dir / "target_distribution.csv", index=False)

    # Per-target rankings (RMSE and directional_accuracy)
    rmse_rank = metrics_long_df.pivot(index="target", columns="model", values="rmse_mean")
    da_rank = metrics_long_df.pivot(index="target", columns="model", values="directional_accuracy_mean")
    best_rmse_df = rmse_rank.idxmin(axis=1).rename("best_model_rmse").reset_index()
    best_da_df = da_rank.idxmax(axis=1).rename("best_model_directional_accuracy").reset_index()
    best_rmse_df.to_csv(metrics_dir / "best_model_by_target_rmse.csv", index=False)
    best_da_df.to_csv(metrics_dir / "best_model_by_target_directional_accuracy.csv", index=False)

    ranking_target = metrics_long_df[["target", "model", "rmse_mean", "directional_accuracy_mean", "n_samples"]].copy()
    ranking_target = ranking_target.sort_values(["target", "rmse_mean"], ascending=[True, True])
    ranking_target.to_csv(metrics_dir / "target_model_ranking.csv", index=False)
    da_rank_target = metrics_long_df[["target", "model", "directional_accuracy_mean", "directional_accuracy_std"]].copy()
    da_rank_target = da_rank_target.sort_values(["target", "directional_accuracy_mean"], ascending=[True, False])
    da_rank_target.to_csv(metrics_dir / "directional_accuracy_ranking_by_target.csv", index=False)
    if not fold_metrics_long_df.empty:
        da_rank_fold = fold_metrics_long_df[["target", "fold_id", "model", "directional_accuracy"]].copy()
        da_rank_fold = da_rank_fold.sort_values(["target", "fold_id", "directional_accuracy"], ascending=[True, True, False])
        da_rank_fold.to_csv(metrics_dir / "directional_accuracy_ranking_by_target_fold.csv", index=False)

    # Practical significance of directional accuracy (mean/std/excess over 0.5)
    da_sig_rows = []
    for model in models:
        by_target = metrics_long_df[metrics_long_df["model"] == model]
        da_mean = float(by_target["directional_accuracy_mean"].mean())
        da_std_target = float(by_target["directional_accuracy_mean"].std(ddof=0))
        if not fold_metrics_long_df.empty:
            by_fold = (
                fold_metrics_long_df[fold_metrics_long_df["model"] == model]
                .groupby("fold_id", as_index=False)["directional_accuracy"]
                .mean()
            )
            da_std_fold = float(by_fold["directional_accuracy"].std(ddof=0)) if len(by_fold) else np.nan
        else:
            da_std_fold = np.nan
        da_sig_rows.append(
            {
                "model": model,
                "directional_accuracy_mean": da_mean,
                "directional_accuracy_std_target": da_std_target,
                "directional_accuracy_std_fold": da_std_fold,
                "excess_over_50": da_mean - 0.5,
            }
        )
    da_sig_df = pd.DataFrame(da_sig_rows).sort_values("directional_accuracy_mean", ascending=False)
    da_sig_df.to_csv(metrics_dir / "directional_accuracy_significance.csv", index=False)

    # Per-target summary table with gaps vs baselines
    base_zero = metrics_long_df[metrics_long_df["model"] == "baseline_zero"][["target", "directional_accuracy_mean"]].rename(
        columns={"directional_accuracy_mean": "da_baseline_zero"}
    )
    base_last = metrics_long_df[metrics_long_df["model"] == "baseline_last"][["target", "directional_accuracy_mean"]].rename(
        columns={"directional_accuracy_mean": "da_baseline_last"}
    )
    da_best = (
        metrics_long_df.sort_values("directional_accuracy_mean", ascending=False)
        .groupby("target", as_index=False)
        .first()[["target", "model", "directional_accuracy_mean", "directional_accuracy_std"]]
        .rename(
            columns={
                "model": "best_model_directional_accuracy",
                "directional_accuracy_mean": "da_mean",
                "directional_accuracy_std": "da_std",
            }
        )
    )
    da_target_summary = da_best.merge(base_last, on="target", how="left").merge(base_zero, on="target", how="left")
    da_target_summary["gap_vs_baseline_last"] = da_target_summary["da_mean"] - da_target_summary["da_baseline_last"]
    da_target_summary["gap_vs_baseline_zero"] = da_target_summary["da_mean"] - da_target_summary["da_baseline_zero"]
    da_target_summary = da_target_summary.sort_values("target")
    da_target_summary.to_csv(metrics_dir / "directional_accuracy_target_summary.csv", index=False)

    best_rmse_global = _pick_best_model_global(leaderboard, "rmse")
    best_da_global = _pick_best_model_global(leaderboard, "directional_accuracy")
    baseline_models = {"baseline_zero", "baseline_last"}
    baseline_count_rmse = int(best_rmse_df["best_model_rmse"].isin(baseline_models).sum())
    trained_count_rmse = int(len(best_rmse_df) - baseline_count_rmse)
    baseline_count_da = int(best_da_df["best_model_directional_accuracy"].isin(baseline_models).sum())
    trained_count_da = int(len(best_da_df) - baseline_count_da)
    comment = ""
    if best_rmse_global == "baseline_zero" and best_da_global != "baseline_zero":
        comment = (
            "baseline_zero leads on global RMSE but not on directional_accuracy; "
            "this suggests returns concentrated near zero where predicting zero lowers average error without improving direction."
        )
    elif best_rmse_global == "baseline_zero" and best_da_global == "baseline_zero":
        comment = "baseline_zero leads on both global RMSE and directional_accuracy."
    else:
        comment = "baseline_zero does not lead on global RMSE."

    # Minimal economic backtest
    bt_cmd = [
        py,
        "-m",
        "news_impact.backtest",
        "--reports-dir",
        str(reports_dir),
        "--cost-bps",
        str(args.cost_bps),
    ]
    _run(bt_cmd, logger)
    backtest_global_path = metrics_dir / "backtest_global.csv"
    backtest_global_df = pd.read_csv(backtest_global_path) if backtest_global_path.exists() else pd.DataFrame()
    backtest_best_path = metrics_dir / "backtest_threshold_best.csv"
    backtest_best_df = pd.read_csv(backtest_best_path) if backtest_best_path.exists() else pd.DataFrame()
    backtest_sweep_path = metrics_dir / "backtest_threshold_sweep.csv"
    backtest_sweep_df = pd.read_csv(backtest_sweep_path) if backtest_sweep_path.exists() else pd.DataFrame()

    bt_global_best_table = pd.DataFrame()
    bt_target_best_table = pd.DataFrame()
    bt_improve_row = {}
    if not backtest_best_df.empty:
        bt_global_best_table = backtest_best_df[
            (backtest_best_df["scope"] == "global") & (backtest_best_df["cost_bps"] == float(args.cost_bps))
        ].copy()
        bt_target_best_table = backtest_best_df[
            (backtest_best_df["scope"] == "per_target") & (backtest_best_df["cost_bps"] == float(args.cost_bps))
        ].copy()
        if not bt_target_best_table.empty:
            bt_target_best_table["improves_vs_t0"] = bt_target_best_table["metric_value"] > bt_target_best_table["metric_at_threshold0"]
            imp = bt_target_best_table.groupby("best_by")["improves_vs_t0"].sum()
            cnt = bt_target_best_table.groupby("best_by")["improves_vs_t0"].count()
            bt_improve_row = {
                "targets_improve_cum_return_vs_t0": f"{int(imp.get('cum_return', 0))}/{int(cnt.get('cum_return', 0))}",
                "targets_improve_sharpe_vs_t0": f"{int(imp.get('sharpe', 0))}/{int(cnt.get('sharpe', 0))}",
            }

    # Setup recommendations by cost (1/3/5 bps) for Sharpe objective
    recommendations_rows = []
    backtest_by_target_path = metrics_dir / "backtest_by_target.csv"
    backtest_by_target_df = pd.read_csv(backtest_by_target_path) if backtest_by_target_path.exists() else pd.DataFrame()
    for c in [1.0, 3.0, 5.0]:
        g = backtest_global_df[(backtest_global_df["cost_bps"] == c) & (backtest_global_df["strategy"] == "long_flat")]
        tdf = backtest_by_target_df[(backtest_by_target_df["cost_bps"] == c) & (backtest_by_target_df["strategy"] == "long_flat")]
        if g.empty or tdf.empty:
            recommendations_rows.append(
                {
                    "cost_bps": c,
                    "recommended_scope": "N/A",
                    "global_setup": "N/A",
                    "global_sharpe": np.nan,
                    "per_target_avg_best_sharpe": np.nan,
                    "rationale": "Insufficient data for this cost level.",
                }
            )
            continue
        best_g = g.sort_values("avg_sharpe", ascending=False).iloc[0]
        per_t = tdf.sort_values("sharpe", ascending=False).groupby("target", as_index=False).first()
        per_target_avg = float(per_t["sharpe"].mean())
        scope = "per-target" if per_target_avg > float(best_g["avg_sharpe"]) else "global"
        rationale = (
            "Per-target beats average global Sharpe."
            if scope == "per-target"
            else "Global meets or beats average per-target Sharpe."
        )
        recommendations_rows.append(
            {
                "cost_bps": c,
                "recommended_scope": scope,
                "global_setup": f"{best_g['model']} | thr={best_g['threshold']} | {best_g['strategy']}",
                "global_sharpe": float(best_g["avg_sharpe"]),
                "per_target_avg_best_sharpe": per_target_avg,
                "rationale": rationale,
            }
        )
    recommendations_df = pd.DataFrame(recommendations_rows)
    selection_dir = reports_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    recommendations_df.to_csv(selection_dir / "recommended_setups.csv", index=False)

    summary_path = reports_dir / "summary.md"
    top_cols = [
        "model",
        "avg_mae",
        "avg_rmse",
        "avg_r2",
        "avg_directional_accuracy",
        "avg_hit_rate",
        "avg_pearson",
        "avg_spearman",
        "total_valid_samples",
    ]
    global_table = _make_md_table(leaderboard[top_cols])
    target_table = _make_md_table(ranking_target.head(90))
    stats_table = _make_md_table(target_stats_df)
    rmse_best_table = _make_md_table(best_rmse_df)
    da_best_table = _make_md_table(best_da_df)
    da_sig_table = _make_md_table(da_sig_df)
    da_target_summary_table = _make_md_table(da_target_summary)
    backtest_table = _make_md_table(backtest_global_df)
    bt_global_best_md = _make_md_table(bt_global_best_table)
    bt_target_best_md = _make_md_table(bt_target_best_table.head(120))
    bt_improve_md = _make_md_table(pd.DataFrame([bt_improve_row])) if bt_improve_row else ["| n/a |", "| --- |"]
    reco_md = _make_md_table(recommendations_df)
    summary_lines = [
        "# Summary",
        "",
        f"- Dataset: `{data_path}`",
        f"- Mode: `{args.mode}`",
        f"- Walk-forward: start={args.wf_start}, step={args.wf_step}, test_window={args.wf_test_window}",
        f"- Best model ({args.rank_by}): `{best_model}`",
        f"- Best global by RMSE: `{best_rmse_global}`",
        f"- Best global by directional_accuracy: `{best_da_global}`",
        f"- Targets where baseline is best by RMSE: {baseline_count_rmse}; trained model best: {trained_count_rmse}",
        f"- Targets where baseline is best by directional_accuracy: {baseline_count_da}; trained model best: {trained_count_da}",
        "",
        "## Leakage checks",
        "",
        "- In walk-forward evaluation, each fold retrains from scratch per target.",
        "- Imputation/scaling is fit on the fold train split only (never on test).",
        "- Trained artifacts are not reused across folds inside evaluate.",
        "- run_all without `--force` skips recomputation using existing CSVs/models; it does not mix data across time.",
        "",
        "## Global model table",
        "",
        *global_table,
        "",
        "## Practical significance of directional accuracy",
        "",
        *da_sig_table,
        "",
        "## Target distribution diagnostics",
        "",
        *stats_table,
        "",
        "## Per-target table (model vs metrics)",
        "",
        *target_table,
        "",
        "## Best model per target (RMSE)",
        "",
        *rmse_best_table,
        "",
        "## Best model per target (directional_accuracy)",
        "",
        *da_best_table,
        "",
        "## Directional accuracy summary by target (gaps vs baselines)",
        "",
        *da_target_summary_table,
        "",
        "## Automatic comment",
        "",
        f"- {comment}",
        "",
        "## Diagnostic output files",
        "",
        "- `reports/metrics/target_distribution.csv`",
        "- `reports/metrics/target_model_ranking.csv`",
        "- `reports/metrics/best_model_by_target_rmse.csv`",
        "- `reports/metrics/best_model_by_target_directional_accuracy.csv`",
        "- `reports/metrics/directional_accuracy_significance.csv`",
        "- `reports/metrics/directional_accuracy_ranking_by_target.csv`",
        "- `reports/metrics/directional_accuracy_ranking_by_target_fold.csv`",
        "- `reports/metrics/directional_accuracy_target_summary.csv`",
        "- `reports/metrics/backtest_by_target.csv`",
        "- `reports/metrics/backtest_global.csv`",
        "",
        "## Plots",
        "",
        "- `reports/plots/leaderboard_*.png`",
        "- `reports/plots/targets_r2_comparison.png`",
        "- `reports/plots/targets_directional_accuracy_comparison.png`",
        f"- `reports/plots/{best_model}/*_rolling_da.png`",
        "",
        "## Minimal economic validation",
        "",
        f"- Transaction cost used: {args.cost_bps} bps per position change.",
        "",
        "## Recommended final setup (Sharpe)",
        "",
        *reco_md,
        "",
        "### Recommendation by cost",
        "",
        f"- 1 bps: {recommendations_df.loc[recommendations_df['cost_bps']==1.0, 'recommended_scope'].iloc[0] if (recommendations_df['cost_bps']==1.0).any() else 'N/A'}",
        f"- 3 bps: {recommendations_df.loc[recommendations_df['cost_bps']==3.0, 'recommended_scope'].iloc[0] if (recommendations_df['cost_bps']==3.0).any() else 'N/A'}",
        f"- 5 bps: {recommendations_df.loc[recommendations_df['cost_bps']==5.0, 'recommended_scope'].iloc[0] if (recommendations_df['cost_bps']==5.0).any() else 'N/A'}",
        "",
        "### Best global threshold by model",
        "",
        *bt_global_best_md,
        "",
        "### Best threshold per target",
        "",
        *bt_target_best_md,
        "",
        "### Comparison vs threshold=0",
        "",
        *bt_improve_md,
        "",
        "### Global backtest table",
        "",
        *backtest_table,
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nFinal leaderboard (sorted by {args.rank_by}):")
    print(leaderboard.to_string(index=False))
    print(f"\nBest global by RMSE: {best_rmse_global}")
    print(f"Best global by directional_accuracy: {best_da_global}")
    print(f"Targets where baseline is best by RMSE: {baseline_count_rmse} | trained best: {trained_count_rmse}")
    print(f"Targets where baseline is best by directional_accuracy: {baseline_count_da} | trained best: {trained_count_da}")
    logger.info("Leaderboard written to: %s", leaderboard_path)
    logger.info("Comparison plots in: %s", reports_dir / "plots")
    logger.info("Summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
