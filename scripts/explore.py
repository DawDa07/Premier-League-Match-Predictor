"""Explore the raw Premier League results data."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_raw


def main() -> None:
    df = load_raw()

    print("=== Shape ===")
    print(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    print()

    print("=== Columns ===")
    print(", ".join(df.columns.tolist()))
    print()

    print("=== Date range ===")
    print(f"{df['DateTime'].min().date()} → {df['DateTime'].max().date()}")
    print()

    print("=== Seasons ===")
    print(f"Count: {df['Season'].nunique()}")
    print(f"First: {df['Season'].iloc[0]}  Last: {df['Season'].iloc[-1]}")
    print()

    print("=== Full-time result (H / D / A) ===")
    print(df["FTR"].value_counts().reindex(["H", "D", "A"]).to_string())
    print()

    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    print("=== Teams ===")
    print(f"Unique teams: {len(teams)}")
    print(", ".join(teams))
    print()

    print("=== Sample rows ===")
    cols = ["Season", "DateTime", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    print(df[cols].head(8).to_string(index=False))
    print()

    print("=== Missing values (top) ===")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("None")
    else:
        print(missing.head(12).to_string())


if __name__ == "__main__":
    main()
