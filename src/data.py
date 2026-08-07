"""Load, clean, and save match data."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_MATCHES = ROOT / "data" / "raw" / "results.csv"
PROCESSED_MATCHES = ROOT / "data" / "processed" / "matches.csv"


def load_raw(path: Path = RAW_MATCHES) -> pd.DataFrame:
    """Load the football-data style results table (one row per match)."""
    df = pd.read_csv(path, encoding="latin-1", na_values=["NA"])
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
    return df


def load_processed(path: Path = PROCESSED_MATCHES) -> pd.DataFrame:
    """Load the cleaned one-row-per-match table."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df


def _result_from_goals(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def build_matches(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Normalize raw results into a simple match table.

    Raw data is already one fixture per row (HomeTeam / AwayTeam / FTHG / FTAG).
    """
    df = (raw if raw is not None else load_raw()).copy()

    matches = pd.DataFrame(
        {
            "date": df["DateTime"].dt.tz_convert(None),
            "season": df["Season"].astype(str),
            "home_team": df["HomeTeam"].astype(str).str.strip(),
            "away_team": df["AwayTeam"].astype(str).str.strip(),
            "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
            "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
            "ftr": df["FTR"].astype(str).str.strip().str.upper(),
        }
    )
    matches = matches.dropna(subset=["home_goals", "away_goals", "home_team", "away_team"])
    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)
    matches["result"] = [
        r if r in {"H", "D", "A"} else _result_from_goals(h, a)
        for r, h, a in zip(
            matches["ftr"], matches["home_goals"], matches["away_goals"]
        )
    ]
    matches = matches.drop(columns=["ftr"])

    matches = matches.sort_values("date").reset_index(drop=True)

    dupes = matches.duplicated(subset=["date", "home_team", "away_team"], keep=False)
    if dupes.any():
        raise ValueError(f"Duplicate fixtures found:\n{matches.loc[dupes].head(20)}")

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
