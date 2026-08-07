"""Phase 2: clean raw team-rows into one row per match."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_MATCHES, build_matches, save_matches


def main() -> None:
    matches = build_matches()
    path = save_matches(matches)

    print("=== Processed matches ===")
    print(f"Saved: {path.relative_to(ROOT)}")
    print(f"Fixtures: {len(matches):,}")
    print(f"Date range: {matches['date'].min().date()} → {matches['date'].max().date()}")
    print()

    print("=== Result balance (H / D / A) ===")
    print(matches["result"].value_counts().reindex(["H", "D", "A"]).to_string())
    print()

    print("=== By season ===")
    print(matches["season"].value_counts().sort_index().to_string())
    print()

    print("=== Sample ===")
    cols = ["date", "home_team", "away_team", "home_goals", "away_goals", "result", "season"]
    print(matches[cols].head(10).to_string(index=False))
    print()

    print(f"Next: Phase 3 — baseline model (always predict home win) via scripts/train.py")
    print(f"(processed file ready at {PROCESSED_MATCHES.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
