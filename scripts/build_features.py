"""Phase 4: build pre-match features (form + Elo) and save them."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_MATCHES, build_matches, load_processed, save_matches
from src.features import FEATURE_COLUMNS, add_features, save_features


def load_or_build_matches():
    if PROCESSED_MATCHES.exists():
        return load_processed()
    matches = build_matches()
    save_matches(matches)
    return matches


def main() -> None:
    matches = load_or_build_matches()
    featured = add_features(matches)
    path = save_features(featured)

    print("=== Features built ===")
    print(f"Saved: {path.relative_to(ROOT)}")
    print(f"Rows: {len(featured):,}  |  Feature columns: {len(FEATURE_COLUMNS)}")
    print()

    print("=== Feature columns (pre-match only) ===")
    for col in FEATURE_COLUMNS:
        print(f"  - {col}")
    print()

    print("=== Sample (late-season row) ===")
    cols = ["date", "home_team", "away_team", "result"] + FEATURE_COLUMNS
    sample = featured.iloc[[len(featured) // 2]][cols]
    # Transpose for readable single-row view
    row = sample.iloc[0]
    for col in cols:
        val = row[col]
        if isinstance(val, float):
            print(f"  {col}: {val:.2f}")
        else:
            print(f"  {col}: {val}")
    print()

    print("Form/Elo use only *previous* results — no leakage from the match itself.")
    print("Next: python scripts/train.py")


if __name__ == "__main__":
    main()
