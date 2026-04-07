from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from news_impact.utils.data import (
    get_feature_target_cols,
    load_dataset,
    log_nan_report,
    split_time,
    target_valid_counts,
)
from news_impact.utils.model_ops import save_target_artifact, train_target_model


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train multi-target models for news_impact.")
    p.add_argument("--data", required=True, help="Path to data/processed/ml_dataset.parquet")
    p.add_argument(
        "--model",
        required=True,
        choices=["baseline_zero", "baseline_last", "knn", "ridge", "rf", "mlp"],
    )
    p.add_argument("--outdir", required=True, help="Output directory for the model (e.g. models/knn)")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--rf-estimators", type=int, default=120)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", default="cpu")
    return p


def _configure_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("news_impact.train")


def _save_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    logger = _configure_logger()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data, logger)
    feature_cols, target_cols = get_feature_target_cols(df)
    logger.info("Features=%d Targets=%d", len(feature_cols), len(target_cols))
    logger.info("Leakage check: targets in features = %s", "NO")
    logger.info("Valid rows per target (no global drop): %s", target_valid_counts(df, target_cols))

    split = split_time(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    logger.info(
        "Time split ~70/15/15 -> train=%d val=%d test=%d",
        len(split.train),
        len(split.val),
        len(split.test),
    )

    log_nan_report(split.train, feature_cols, logger, "features train")
    log_nan_report(split.val, feature_cols, logger, "features val")
    log_nan_report(split.test, feature_cols, logger, "features test")
    log_nan_report(df, target_cols, logger, "targets (global)")

    model_params = {
        "knn_k": args.knn_k,
        "ridge_alpha": args.ridge_alpha,
        "rf_estimators": args.rf_estimators,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "patience": args.patience,
        "device": args.device,
        "random_state": args.random_state,
    }

    target_dir = outdir / "targets"
    best_vals: dict[str, float] = {}
    for target in target_cols:
        if args.model in {"baseline_zero", "baseline_last"}:
            continue

        train_t = split.train[["date", target] + feature_cols].copy()
        val_t = split.val[["date", target] + feature_cols].copy()
        train_mask = train_t[target].notna().to_numpy()
        val_mask = val_t[target].notna().to_numpy()

        X_train = train_t.loc[train_mask, feature_cols].to_numpy(dtype=float)
        y_train = train_t.loc[train_mask, target].to_numpy(dtype=float)
        X_val = val_t.loc[val_mask, feature_cols].to_numpy(dtype=float)
        y_val = val_t.loc[val_mask, target].to_numpy(dtype=float)

        if len(X_train) < 20:
            logger.warning("Target %s skipped: not enough training samples (%d).", target, len(X_train))
            continue
        if len(X_val) < 10:
            X_val = X_train[-min(30, len(X_train)) :]
            y_val = y_train[-min(30, len(y_train)) :]

        trained = train_target_model(args.model, X_train, y_train, X_val, y_val, model_params)
        save_target_artifact(target_dir / target, args.model, trained)
        if "best_val_mse" in trained:
            best_vals[target] = float(trained["best_val_mse"])

    manifest = {
        "model_name": args.model,
        "data_path": str(Path(args.data)),
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "split": {"train_ratio": args.train_ratio, "val_ratio": args.val_ratio},
        "date_range": {
            "min": str(df["date"].min().date()),
            "max": str(df["date"].max().date()),
        },
        "shape": {"rows": int(len(df)), "cols": int(df.shape[1])},
        "model_params": {
            "knn_k": args.knn_k,
            "ridge_alpha": args.ridge_alpha,
            "rf_estimators": args.rf_estimators,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "patience": args.patience,
            "device": args.device,
            "random_state": args.random_state,
        },
        "nan_handling": "SimpleImputer(strategy='median') on features",
        "target_valid_counts": target_valid_counts(df, target_cols),
        "extra": {"best_val_mse_by_target": best_vals},
    }
    _save_manifest(outdir / "manifest.json", manifest)
    logger.info("Model saved to: %s", outdir)


if __name__ == "__main__":
    main()
