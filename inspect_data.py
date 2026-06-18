"""
inspect_data.py — load and summarize international_results CSVs
===============================================================

Reads the sibling international_results repo and prints a quick overview
of each file. Run from world_cup_predictions with the poet env active:

    conda activate poet
    python inspect_data.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "international_results"

FILES = [
    "results.csv",
    "goalscorers.csv",
    "shootouts.csv",
    "former_names.csv",
]


def summarize_df(name: str, df: pd.DataFrame) -> None:
    print("=" * 72)
    print(name)
    print("=" * 72)
    print(f"rows: {len(df):,}   columns: {len(df.columns)}")
    print(f"columns: {', '.join(df.columns)}")
    print()

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        valid = dates.dropna()
        if not valid.empty:
            print(f"date range: {valid.min().date()} -> {valid.max().date()}")
            print()

    print("dtypes:")
    print(df.dtypes.to_string())
    print()
    print("sample rows:")
    print(df.head(3).to_string(index=False))
    print()
    print("missing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  none")
    else:
        for col, count in missing.items():
            print(f"  {col}: {count:,}")
    print()


def main() -> None:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected international_results at {DATA_DIR}. "
            "Clone it next to world_cup_predictions."
        )

    print(f"data directory: {DATA_DIR}\n")

    for filename in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"warning: missing file -> {path}\n")
            continue
        df = pd.read_csv(path)
        summarize_df(filename, df)

    results = pd.read_csv(DATA_DIR / "results.csv")
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    print("=" * 72)
    print("results.csv — quick stats")
    print("=" * 72)
    print(f"unique home teams: {results['home_team'].nunique():,}")
    print(f"unique away teams: {results['away_team'].nunique():,}")
    print(f"unique tournaments: {results['tournament'].nunique():,}")
    print()
    print("top tournaments:")
    print(results["tournament"].value_counts().head(10).to_string())
    print()
    print("matches per year (last 10 years):")
    yearly = results.dropna(subset=["date"]).groupby(results["date"].dt.year).size()
    print(yearly.tail(10).to_string())
    print()


if __name__ == "__main__":
    main()
