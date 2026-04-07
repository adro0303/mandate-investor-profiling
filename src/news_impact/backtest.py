from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple backtest on OOF predictions per model.")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--cost-bps", type=float, default=1.0, help="Transaction cost per position change, in bps.")
    p.add_argument("--confidence-threshold", type=float, default=0.0)
    p.add_argument(
        "--thresholds",
        default="0,0.0005,0.001,0.002,0.003,0.005",
        help="Comma-separated threshold list for sweep.",
    )
    p.add_argument("--append-results", action="store_true", help="Append to existing CSVs instead of overwriting.")
    return p


def _configure_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("news_impact.backtest")


def _strategy_returns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mode: str,
    cost: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "long_flat":
        signal = (y_pred > threshold).astype(float)
    elif mode == "long_short":
        signal = np.where(y_pred > threshold, 1.0, np.where(y_pred < -threshold, -1.0, 0.0))
    else:
        raise ValueError(f"Unsupported strategy: {mode}")

    gross = signal * y_true
    prev = np.roll(signal, 1)
    prev[0] = 0.0
    turnover = np.abs(signal - prev)
    net = gross - cost * turnover
    return net, signal


def _perf_stats(ret: np.ndarray) -> dict:
    if len(ret) == 0:
        return {
            "cum_return": np.nan,
            "mean_ret": np.nan,
            "std_ret": np.nan,
            "sharpe": np.nan,
        }
    mean_ret = float(np.mean(ret))
    std_ret = float(np.std(ret, ddof=0))
    sharpe = float((mean_ret / std_ret) * np.sqrt(252.0)) if std_ret > 0 else np.nan
    cum_return = float(np.prod(1.0 + ret) - 1.0)
    return {
        "cum_return": cum_return,
        "mean_ret": mean_ret,
        "std_ret": std_ret,
        "sharpe": sharpe,
    }


def _parse_thresholds(raw: str) -> list[float]:
    vals = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    vals = sorted(set(vals))
    if 0.0 not in vals:
        vals = [0.0] + vals
    return vals


def _plot_threshold_curves(sweep_df: pd.DataFrame, outdir: Path, cost_bps: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if sweep_df.empty:
        return
    global_df = (
        sweep_df.groupby(["model", "strategy", "threshold"], as_index=False)
        .agg(
            avg_cum_return=("cum_return", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_hit_rate=("hit_rate", "mean"),
        )
        .sort_values(["strategy", "model", "threshold"])
    )
    for strategy in sorted(global_df["strategy"].unique()):
        sg = global_df[global_df["strategy"] == strategy]
        for metric in ["avg_cum_return", "avg_sharpe", "avg_hit_rate"]:
            plt.figure(figsize=(9, 4))
            for model, mg in sg.groupby("model"):
                plt.plot(mg["threshold"], mg[metric], marker="o", label=model)
            plt.title(f"{metric} vs threshold ({strategy}, cost={cost_bps}bps)")
            plt.xlabel("confidence_threshold")
            plt.ylabel(metric)
            plt.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(outdir / f"threshold_{strategy}_{metric}_cost{int(cost_bps)}bps.png", dpi=130)
            plt.close()


def _upsert_csv(path: Path, df: pd.DataFrame, append: bool, keys: list[str]) -> None:
    if append and path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=keys, keep="last")
        merged.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    args = _build_parser().parse_args()
    logger = _configure_logger()
    reports_dir = Path(args.reports_dir)
    metrics_dir = reports_dir / "metrics"
    cost = float(args.cost_bps) / 10000.0
    thresholds = _parse_thresholds(args.thresholds)

    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics folder does not exist: {metrics_dir}")

    pred_files = sorted(metrics_dir.glob("*_preds_long.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No *_preds_long.csv files found in {metrics_dir}")

    rows: list[dict] = []
    for pf in pred_files:
        model = pf.name.replace("_preds_long.csv", "")
        df = pd.read_csv(pf)
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        for target, tg in df.groupby("target"):
            tg = tg.sort_values(["fold_id", "date"]).reset_index(drop=True)
            y_true = tg["y_true"].to_numpy(dtype=float)
            y_pred = tg["y_pred"].to_numpy(dtype=float)
            for threshold in thresholds:
                for strategy in ("long_flat", "long_short"):
                    ret, signal = _strategy_returns(y_true, y_pred, strategy, cost, threshold)
                    perf = _perf_stats(ret)
                    # Directional hit rate: compare sign of active position vs realized return
                    hit = np.sign(signal) == np.sign(y_true)
                    hit_rate = float(np.mean(hit)) if len(hit) else np.nan
                    active_rate = float(np.mean(signal != 0.0)) if len(signal) else np.nan
                    rows.append(
                        {
                            "model": model,
                            "target": target,
                            "strategy": strategy,
                            "threshold": float(threshold),
                            "cost_bps": float(args.cost_bps),
                            "n_obs": int(len(ret)),
                            "active_rate": active_rate,
                            "hit_rate": hit_rate,
                            **perf,
                        }
                    )

    detail_df = pd.DataFrame(rows)
    detail_path = metrics_dir / "backtest_by_target.csv"
    _upsert_csv(
        detail_path,
        detail_df,
        append=args.append_results,
        keys=["model", "target", "strategy", "threshold", "cost_bps"],
    )

    global_df = (
        detail_df.groupby(["model", "strategy", "threshold", "cost_bps"], as_index=False)
        .agg(
            n_targets=("target", "nunique"),
            avg_cum_return=("cum_return", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_hit_rate=("hit_rate", "mean"),
            avg_active_rate=("active_rate", "mean"),
            avg_mean_ret=("mean_ret", "mean"),
            avg_std_ret=("std_ret", "mean"),
        )
        .sort_values(["cost_bps", "strategy", "avg_sharpe"], ascending=[True, True, False])
    )
    global_path = metrics_dir / "backtest_global.csv"
    _upsert_csv(
        global_path,
        global_df,
        append=args.append_results,
        keys=["model", "strategy", "threshold", "cost_bps"],
    )

    sweep_path = metrics_dir / "backtest_threshold_sweep.csv"
    _upsert_csv(
        sweep_path,
        detail_df,
        append=args.append_results,
        keys=["model", "target", "strategy", "threshold", "cost_bps"],
    )

    best_global_rows = []
    for (model, strategy, cost_bps), gg in global_df.groupby(["model", "strategy", "cost_bps"]):
        by_cum = gg.sort_values("avg_cum_return", ascending=False).iloc[0]
        by_sharpe = gg.sort_values("avg_sharpe", ascending=False).iloc[0]
        best_global_rows.append(
            {
                "scope": "global",
                "model": model,
                "target": "ALL",
                "strategy": strategy,
                "cost_bps": float(cost_bps),
                "best_by": "cum_return",
                "best_threshold": float(by_cum["threshold"]),
                "metric_value": float(by_cum["avg_cum_return"]),
                "metric_at_threshold0": float(gg.loc[gg["threshold"] == 0, "avg_cum_return"].iloc[0]) if (gg["threshold"] == 0).any() else np.nan,
            }
        )
        best_global_rows.append(
            {
                "scope": "global",
                "model": model,
                "target": "ALL",
                "strategy": strategy,
                "cost_bps": float(cost_bps),
                "best_by": "sharpe",
                "best_threshold": float(by_sharpe["threshold"]),
                "metric_value": float(by_sharpe["avg_sharpe"]),
                "metric_at_threshold0": float(gg.loc[gg["threshold"] == 0, "avg_sharpe"].iloc[0]) if (gg["threshold"] == 0).any() else np.nan,
            }
        )

    best_target_rows = []
    for (model, target, strategy, cost_bps), tg in detail_df.groupby(["model", "target", "strategy", "cost_bps"]):
        by_cum = tg.sort_values("cum_return", ascending=False).iloc[0]
        by_sharpe = tg.sort_values("sharpe", ascending=False).iloc[0]
        best_target_rows.append(
            {
                "scope": "per_target",
                "model": model,
                "target": target,
                "strategy": strategy,
                "cost_bps": float(cost_bps),
                "best_by": "cum_return",
                "best_threshold": float(by_cum["threshold"]),
                "metric_value": float(by_cum["cum_return"]),
                "metric_at_threshold0": float(tg.loc[tg["threshold"] == 0, "cum_return"].iloc[0]) if (tg["threshold"] == 0).any() else np.nan,
            }
        )
        best_target_rows.append(
            {
                "scope": "per_target",
                "model": model,
                "target": target,
                "strategy": strategy,
                "cost_bps": float(cost_bps),
                "best_by": "sharpe",
                "best_threshold": float(by_sharpe["threshold"]),
                "metric_value": float(by_sharpe["sharpe"]),
                "metric_at_threshold0": float(tg.loc[tg["threshold"] == 0, "sharpe"].iloc[0]) if (tg["threshold"] == 0).any() else np.nan,
            }
        )

    best_df = pd.DataFrame(best_global_rows + best_target_rows)
    best_path = metrics_dir / "backtest_threshold_best.csv"
    _upsert_csv(
        best_path,
        best_df,
        append=args.append_results,
        keys=["scope", "model", "target", "strategy", "cost_bps", "best_by"],
    )

    _plot_threshold_curves(detail_df, reports_dir / "plots" / "backtest", float(args.cost_bps))

    logger.info("Per-target backtest written to: %s", detail_path)
    logger.info("Global backtest written to: %s", global_path)
    logger.info("Threshold sweep written to: %s", sweep_path)
    logger.info("Best thresholds written to: %s", best_path)
    print("\nBacktest global:")
    print(global_df.sort_values(["strategy", "avg_sharpe"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
