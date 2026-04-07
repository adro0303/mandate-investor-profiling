import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

from news_impact.utils.gdelt import (
    parse_gdelt_datetime_int,
    split_v2tone,
    extract_theme_codes,
    extract_org_names,
)

DEFAULT_TICKERS = [
    "UUP", "TLT", "GLD", "BTC-USD",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE", "XLC"
]

def load_news(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = df["DATE"].apply(parse_gdelt_datetime_int)
    df["date"] = df["ts"].dt.normalize()

    tone_cols = df["V2Tone"].apply(split_v2tone).apply(pd.Series)
    for c in tone_cols.columns:
        df[c] = tone_cols[c]

    df["theme_codes"] = df["V2Themes"].apply(extract_theme_codes)
    df["org_names"] = df["V2Organizations"].apply(extract_org_names)
    df["url"] = df["DocumentIdentifier"].astype(str)

    return df

def aggregate_news_daily(df: pd.DataFrame, hash_dim: int = 256) -> tuple[pd.DataFrame, dict]:
    numeric_cols = ["tone", "pos", "neg", "polarity", "activity_density", "self_group_ref", "word_count"]

    agg = df.groupby("date").agg(
        n_articles=("url", "count"),
        **{f"{c}_mean": (c, "mean") for c in numeric_cols},
        **{f"{c}_std": (c, "std") for c in numeric_cols},
    ).reset_index()

    theme_hasher = FeatureHasher(n_features=hash_dim, input_type="string")
    org_hasher = FeatureHasher(n_features=hash_dim, input_type="string")

    theme_sparse = theme_hasher.transform(df["theme_codes"])
    org_sparse = org_hasher.transform(df["org_names"])

    theme_df = pd.DataFrame(theme_sparse.toarray(), columns=[f"th_{i}" for i in range(hash_dim)])
    org_df = pd.DataFrame(org_sparse.toarray(), columns=[f"org_{i}" for i in range(hash_dim)])

    tmp = pd.concat([df[["date"]].reset_index(drop=True), theme_df, org_df], axis=1)
    hashed_daily = tmp.groupby("date").sum(numeric_only=True).reset_index()

    out = agg.merge(hashed_daily, on="date", how="left").fillna(0)

    schema = {
        "hash_dim": hash_dim,
        "numeric_cols": numeric_cols,
        "hashed_theme_cols": [f"th_{i}" for i in range(hash_dim)],
        "hashed_org_cols": [f"org_{i}" for i in range(hash_dim)],
    }
    return out, schema

def load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"]).dt.normalize()
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    else:
        raise ValueError("market CSV must have a Date column")

    # Normalize column names to TICKER_Field
    cols = [c for c in df.columns if c not in ("Date", "date")]
    norm = {}
    for c in cols:
        cc = str(c).strip()
        cc = cc.replace("(", "").replace(")", "").replace("'", "").replace('"', "")
        cc = cc.replace(",", " ")
        cc = re.sub(r"\s+", "_", cc)
        cc = cc.replace("__", "_")
        norm[c] = cc
    df = df.rename(columns=norm)

    return df

def build_labels(market_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    out = market_df[["date"]].copy()

    for t in tickers:
        candidates = [
            f"{t}_Close", f"{t}_close",
            f"{t}_Adj_Close", f"{t}_AdjClose", f"{t}_Adj_Close"
        ]
        close_col = None
        for cand in candidates:
            if cand in market_df.columns:
                close_col = cand
                break
        if close_col is None:
            # try any column that starts with ticker and ends with _Close
            for c in market_df.columns:
                if c.upper().startswith(t.upper() + "_") and c.lower().endswith("_close"):
                    close_col = c
                    break
        if close_col is None:
            raise ValueError(f"Could not find Close column for ticker {t}.")

        close = pd.to_numeric(market_df[close_col], errors="coerce")
        out[f"y_{t}"] = close.shift(-1) / close - 1.0

    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--news", required=True)
    p.add_argument("--market", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hash-dim", type=int, default=256)
    p.add_argument("--targets", nargs="*", default=DEFAULT_TICKERS)
    args = p.parse_args()

    news = load_news(args.news)
    news_daily, schema = aggregate_news_daily(news, hash_dim=args.hash_dim)

    market = load_market(args.market)
    labels = build_labels(market, tickers=args.targets)

    merged = news_daily.merge(labels, on="date", how="inner").sort_values("date")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        merged.to_parquet(out_path, index=False)
    except Exception:
        merged.to_csv(out_path.with_suffix(".csv"), index=False)

    schema_path = out_path.with_suffix(".schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"Saved merged dataset: {out_path}")
    print(f"Rows: {len(merged):,} | Columns: {merged.shape[1]:,}")

if __name__ == "__main__":
    main()
