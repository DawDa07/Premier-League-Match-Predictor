"""Load, clean, and save match data."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_MATCHES = ROOT / "data" / "raw" / "matches.csv"
PROCESSED_MATCHES = ROOT / "data" / "processed" / "matches.csv"

# FBRef uses short names in the opponent column.
OPPONENT_NAME_MAP = {
    "Brighton": "Brighton and Hove Albion",
    "Manchester Utd": "Manchester United",
    "Newcastle Utd": "Newcastle United",
    "Sheffield Utd": "Sheffield United",
    "Tottenham": "Tottenham Hotspur",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}


def load_raw(path: Path = RAW_MATCHES) -> pd.DataFrame:
    """Load the raw FBRef-style team match table."""
    df = pd.read_csv(path, index_col=0)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_processed(path: Path = PROCESSED_MATCHES) -> pd.DataFrame:
    """Load the cleaned one-row-per-match table."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _result_from_goals(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def build_matches(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Turn team-level rows into one row per fixture.

    Prefer home-team rows (they already have home/away goals via gf/ga).
    Add away-only rows when the home team's row is missing from the CSV
    (e.g. incomplete Liverpool coverage).
    """
    df = (raw if raw is not None else load_raw()).copy()
    df["opponent"] = df["opponent"].replace(OPPONENT_NAME_MAP)

    home = df[df["venue"].str.lower() == "home"].copy()
    away = df[df["venue"].str.lower() == "away"].copy()

    from_home = pd.DataFrame(
        {
            "date": home["date"],
            "season": home["season"],
            "round": home["round"],
            "home_team": home["team"],
            "away_team": home["opponent"],
            "home_goals": pd.to_numeric(home["gf"], errors="coerce"),
            "away_goals": pd.to_numeric(home["ga"], errors="coerce"),
        }
    )

    # Away rows with no matching home row (missing home-team data).
    paired = away.merge(
        home,
        left_on=["date", "comp", "team", "opponent"],
        right_on=["date", "comp", "opponent", "team"],
        how="left",
        indicator=True,
        suffixes=("_away", "_home"),
    )
    orphan_away = paired[paired["_merge"] == "left_only"]

    from_away = pd.DataFrame(
        {
            "date": orphan_away["date"],
            "season": orphan_away["season_away"],
            "round": orphan_away["round_away"],
            "home_team": orphan_away["opponent_away"],
            "away_team": orphan_away["team_away"],
            "home_goals": pd.to_numeric(orphan_away["ga_away"], errors="coerce"),
            "away_goals": pd.to_numeric(orphan_away["gf_away"], errors="coerce"),
        }
    )

    matches = pd.concat([from_home, from_away], ignore_index=True)
    matches = matches.dropna(subset=["home_goals", "away_goals"])
    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)
    matches["result"] = [
        _result_from_goals(h, a)
        for h, a in zip(matches["home_goals"], matches["away_goals"])
    ]
    matches = matches.sort_values("date").reset_index(drop=True)

    dupes = matches.duplicated(subset=["date", "home_team", "away_team"], keep=False)
    if dupes.any():
        raise ValueError(f"Duplicate fixtures found:\n{matches.loc[dupes]}")

    return matches


def save_matches(
    matches: pd.DataFrame | None = None,
    path: Path = PROCESSED_MATCHES,
) -> Path:
    """Build (if needed) and write the processed match table."""
    if matches is None:
        matches = build_matches()
    path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(path, index=False)
    return path
