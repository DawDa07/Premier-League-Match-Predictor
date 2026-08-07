"""Phase 1: explore the raw Premier League match data."""

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
    print(f"{df['date'].min().date()} → {df['date'].max().date()}")
    print()

    print("=== Competition ===")
    print(df["comp"].value_counts().to_string())
    print()

    print("=== Seasons ===")
    print(df["season"].value_counts().sort_index().to_string())
    print()

    print("=== Venue ===")
    print(df["venue"].value_counts().to_string())
    print()

    print("=== Result (from each team's view: W / D / L) ===")
    print(df["result"].value_counts().to_string())
    print()

    print("=== Teams ===")
    print(f"Unique teams: {df['team'].nunique()}")
    print(df["team"].value_counts().sort_index().to_string())
    print()

    print("=== Sample rows ===")
    cols = ["date", "team", "venue", "opponent", "result", "gf", "ga", "season"]
    print(df[cols].head(8).to_string(index=False))
    print()

    print("=== Missing values (top) ===")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("None")
    else:
        print(missing.to_string())
    print()

    print("=== Note ===")
    print("Each match appears twice (home row + away row).")
    print("Next: merge into one row per fixture (Phase 2).")


if __name__ == "__main__":
    main()
