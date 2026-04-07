from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TARGET_COLS = [
    "y_UUP",
    "y_TLT",
    "y_GLD",
    "y_BTC-USD",
    "y_XLK",
    "y_XLF",
    "y_XLE",
    "y_XLV",
    "y_XLI",
    "y_XLY",
    "y_XLP",
    "y_XLB",
    "y_XLU",
    "y_XLRE",
    "y_XLC",
]


@dataclass
class TimeSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: int
    val_end: int


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def load_dataset(path: str, logger: logging.Logger) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    if "date" not in df.columns:
        raise ValueError("Dataset must contain a 'date' column.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    logger.info(
        "Dataset loaded: rows=%d columns=%d range=%s -> %s",
        len(df),
        df.shape[1],
        df["date"].min().date() if len(df) else "NA",
        df["date"].max().date() if len(df) else "NA",
    )
    return df


def get_feature_target_cols(df: pd.DataFrame, target_cols: list[str] | None = None) -> tuple[list[str], list[str]]:
    target_cols = target_cols or DEFAULT_TARGET_COLS
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns in dataset: {missing}")

    feature_cols = [c for c in df.columns if c not in (["date"] + target_cols)]
    overlap = sorted(set(feature_cols).intersection(set(target_cols)))
    if overlap:
        raise ValueError(f"Leakage detected: target columns appear in features: {overlap}")

    if not feature_cols:
        raise ValueError("No feature columns available.")

    return feature_cols, target_cols


def split_time(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> TimeSplit:
    if len(df) < 10:
        raise ValueError("Dataset too small for a robust time split.")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid ratios: need train_ratio>0, val_ratio>0, and train+val<1.")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    if train_end < 1 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid time split for the configured ratios.")

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return TimeSplit(train=train, val=val, test=test, train_end=train_end, val_end=val_end)


def log_nan_report(df: pd.DataFrame, cols: list[str], logger: logging.Logger, section: str) -> None:
    nan_count = int(df[cols].isna().sum().sum())
    logger.info("NaNs en %s: %d", section, nan_count)


def filter_valid_targets(df: pd.DataFrame, target_cols: list[str], logger: logging.Logger) -> pd.DataFrame:
    before = len(df)
    target_block = df[target_cols].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(target_block.to_numpy(dtype=float)).all(axis=1)
    out = df.loc[mask].copy().reset_index(drop=True)
    removed = before - len(out)
    logger.info("Filtered invalid targets: removed %d rows (%d remaining).", removed, len(out))
    return out


def target_valid_counts(df: pd.DataFrame, target_cols: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in target_cols:
        counts[target] = int(pd.to_numeric(df[target], errors="coerce").notna().sum())
    return counts


def build_walk_forward_folds(
    n_rows: int,
    wf_start: float,
    wf_step: int,
    wf_test_window: int,
) -> list[WalkForwardFold]:
    if n_rows <= 0:
        raise ValueError("n_rows must be > 0.")
    if wf_step <= 0 or wf_test_window <= 0:
        raise ValueError("wf_step and wf_test_window must be > 0.")

    if 0 < wf_start < 1:
        start_idx = int(n_rows * wf_start)
    else:
        start_idx = int(wf_start)
    start_idx = max(1, min(start_idx, n_rows - wf_test_window))

    folds: list[WalkForwardFold] = []
    fold_id = 0
    train_end = start_idx
    while train_end + wf_test_window <= n_rows:
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=0,
                train_end=train_end,
                test_start=train_end,
                test_end=train_end + wf_test_window,
            )
        )
        fold_id += 1
        train_end += wf_step

    if not folds:
        raise ValueError("Could not build walk-forward folds with the current configuration.")
    return folds
