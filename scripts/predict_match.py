"""Phase 6: predict a Premier League fixture from the CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import format_prediction, predict_match


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a Premier League match (Home / Draw / Away)."
    )
    parser.add_argument("home_team", help='Home team, e.g. "Arsenal" or "Man City"')
    parser.add_argument("away_team", help='Away team, e.g. "Chelsea" or "Spurs"')
    args = parser.parse_args()

    try:
        result = predict_match(args.home_team, args.away_team)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(format_prediction(result))


if __name__ == "__main__":
    main()
