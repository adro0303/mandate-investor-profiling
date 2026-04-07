from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from news_impact.utils.data import get_feature_target_cols, load_dataset, split_time
from news_impact.utils.model_ops import load_target_artifact, predict_target_model

BEST_BY_CHOICES = ["rmse", "mae", "r2", "pearson", "spearman", "directional_accuracy", "hit_rate"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run inference for news_impact models.")
    p.add_argument("--data", required=True)
    p.add_argument("--modeldir", default=None, help="Path to models/<model>")
    p.add_argument("--selection-file", default=None, help="Final selection CSV (global or per-target).")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--best", action="store_true", help="Pick best model from reports/metrics/leaderboard.csv")
    p.add_argument("--best-by", choices=BEST_BY_CHOICES, default="directional_accuracy")
    p.add_argument("--best-scope", choices=["global", "per-target"], default="global")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    return p


def _configure_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("news_impact.predict")


def _ascending(metric: str) -> bool:
    return metric in {"rmse", "mae"}


def _resolve_best_global(best_by: str) -> str:
    lb = pd.read_csv("reports/metrics/leaderboard.csv")
    col = f"avg_{best_by}"
    return str(lb.sort_values(col, ascending=_ascending(best_by)).iloc[0]["model"])


def _resolve_best_per_target(target_cols: list[str], best_by: str) -> dict[str, str]:
    metric_col = f"{best_by}_mean"
    rows = []
    models = ["baseline_zero", "baseline_last", "knn", "ridge", "rf", "mlp"]
    for m in models:
        p = Path("reports/metrics") / f"{m}_metrics.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d["target"] != "AVG"][["target", metric_col]].copy()
        d["model"] = m
        rows.append(d)
    if not rows:
        raise FileNotFoundError("No per-target metrics found to select --best-scope per-target.")
    allm = pd.concat(rows, ignore_index=True)
    out: dict[str, str] = {}
    for t in target_cols:
        dt = allm[allm["target"] == t].copy()
        if dt.empty:
            continue
        dt = dt.sort_values(metric_col, ascending=_ascending(best_by))
        out[t] = str(dt.iloc[0]["model"])
    return out


def main() -> None:
    args = _build_parser().parse_args()
    logger = _configure_logger()

    df = load_dataset(args.data, logger)
    feature_cols, target_cols = get_feature_target_cols(df)
    selected_cfg: dict[str, dict] = {}
    if args.selection_file:
        sel = pd.read_csv(args.selection_file)
        if "target" not in sel.columns:
            raise ValueError("selection-file debe contener columna 'target'.")
        global_row = sel[sel["target"].astype(str).str.upper() == "ALL"]
        for target in target_cols:
            tr = sel[sel["target"] == target]
            row = tr.iloc[0] if not tr.empty else (global_row.iloc[0] if not global_row.empty else None)
            if row is None:
                raise ValueError(f"selection-file no define target={target} ni fila global ALL.")
            selected_cfg[target] = {
                "model": str(row["selected_model"]),
                "threshold": float(row.get("selected_threshold", 0.0)),
                "strategy": str(row.get("selected_strategy", "long_flat")),
            }
        selected_by_target = {t: selected_cfg[t]["model"] for t in target_cols}
    elif args.best:
        if args.best_scope == "global":
            best_model = _resolve_best_global(args.best_by)
            selected_by_target = {t: best_model for t in target_cols}
        else:
            selected_by_target = _resolve_best_per_target(target_cols, args.best_by)
    else:
        if args.modeldir is None:
            raise ValueError("Debes pasar --modeldir o --best.")
        mname = Path(args.modeldir).name
        selected_by_target = {t: mname for t in target_cols}

    first_model = next(iter(selected_by_target.values()))
    split_manifest = json.loads((Path("models") / first_model / "manifest.json").read_text(encoding="utf-8"))
    split_cfg = split_manifest.get("split", {"train_ratio": 0.70, "val_ratio": 0.15})
    split = split_time(df, split_cfg["train_ratio"], split_cfg["val_ratio"])
    pred_df = split.test.copy()
    pred_df = pred_df.sort_values("date").reset_index(drop=True)

    if args.start_date:
        pred_df = pred_df[pred_df["date"] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        pred_df = pred_df[pred_df["date"] <= pd.to_datetime(args.end_date)]
    pred_df = pred_df.reset_index(drop=True)

    out = pd.DataFrame({"date": pred_df["date"]})
    for target in target_cols:
        X = pred_df[feature_cols].to_numpy(dtype=float)
        tmp = df[["date", target]].copy().sort_values("date").reset_index(drop=True)
        tmp["prev"] = pd.to_numeric(tmp[target], errors="coerce").shift(1)
        prev_map = tmp[["date", "prev"]]
        prev_aligned = pred_df[["date"]].merge(prev_map, on="date", how="left")["prev"].to_numpy(dtype=float)

        model_name = selected_by_target.get(target, first_model)
        threshold = float(selected_cfg.get(target, {}).get("threshold", 0.0))
        strategy = str(selected_cfg.get(target, {}).get("strategy", "long_flat"))
        modeldir = Path("models") / model_name
        manifest = json.loads((modeldir / "manifest.json").read_text(encoding="utf-8"))
        if model_name in {"baseline_zero", "baseline_last"}:
            y_pred = predict_target_model(model_name, {"kind": "baseline"}, X, y_prev=prev_aligned)
        else:
            artifact = load_target_artifact(modeldir / "targets" / target, model_name, manifest.get("model_params", {}), len(feature_cols))
            y_pred = predict_target_model(model_name, artifact, X)
        if strategy == "long_flat":
            signal = (y_pred > threshold).astype(float)
        else:
            signal = np.where(y_pred > threshold, 1.0, np.where(y_pred < -threshold, -1.0, 0.0))
        pred_exec = y_pred * np.abs(signal)
        out[f"pred_{target}"] = y_pred
        out[f"pred_exec_{target}"] = pred_exec
        out[f"true_{target}"] = pd.to_numeric(pred_df[target], errors="coerce").to_numpy(dtype=float)
        out[f"model_{target}"] = model_name
        out[f"threshold_{target}"] = threshold
        out[f"strategy_{target}"] = strategy
        out[f"signal_{target}"] = signal

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Predictions written to: %s", out_path)
    logger.info("Model selection per target: %s", selected_by_target)
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
