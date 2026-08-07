"""Pre-match feature engineering (form + Elo). No same-game leakage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FEATURES = ROOT / "data" / "processed" / "features.csv"

FORM_WINDOW = 5
ELO_K = 20
ELO_START = 1500.0
ELO_HOME_ADV = 100.0

# Columns the model is allowed to use (known before kickoff).
FEATURE_COLUMNS = [
    "home_points_l5",
    "away_points_l5",
    "points_diff_l5",
    "home_gd_l5",
    "away_gd_l5",
    "gd_diff_l5",
    "home_season_pts",
    "away_season_pts",
    "season_pts_diff",
    "home_elo",
    "away_elo",
    "elo_diff",
]


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _team_games(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (home and away views)."""
    base = matches[["date", "season", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    base = base.reset_index(names="match_id")

    home = pd.DataFrame(
        {
            "match_id": base["match_id"],
            "date": base["date"],
            "season": base["season"],
            "team": base["home_team"],
            "goals_for": base["home_goals"],
            "goals_against": base["away_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": base["match_id"],
            "date": base["date"],
            "season": base["season"],
            "team": base["away_team"],
            "goals_for": base["away_goals"],
            "goals_against": base["home_goals"],
        }
    )
    games = pd.concat([home, away], ignore_index=True)
    games["points"] = [
        _points(gf, ga) for gf, ga in zip(games["goals_for"], games["goals_against"])
    ]
    games["gd"] = games["goals_for"] - games["goals_against"]
    return games.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def _add_rolling_form(games: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Rolling points / GD over the previous `window` games (shifted = pre-match)."""
    games = games.copy()
    grouped = games.groupby("team", sort=False)

    games["points_l5"] = (
        grouped["points"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        .fillna(0.0)
    )
    games["gd_l5"] = (
        grouped["gd"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        .fillna(0.0)
    )
    # Season points before this match (resets each season).
    games["season_pts"] = (
        games.groupby(["team", "season"], sort=False)["points"]
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0.0)
    )
    return games


def _add_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Chronological Elo; store ratings *before* each match is played."""
    matches = matches.sort_values("date").reset_index(drop=True)
    elos: dict[str, float] = {}
    home_elos: list[float] = []
    away_elos: list[float] = []

    for row in matches.itertuples(index=False):
        home, away = row.home_team, row.away_team
        rh = elos.setdefault(home, ELO_START)
        ra = elos.setdefault(away, ELO_START)
        home_elos.append(rh)
        away_elos.append(ra)

        if row.home_goals > row.away_goals:
            score = 1.0
        elif row.home_goals < row.away_goals:
            score = 0.0
        else:
            score = 0.5

        expected_h = 1.0 / (1.0 + 10 ** ((ra - (rh + ELO_HOME_ADV)) / 400.0))
        expected_a = 1.0 - expected_h
        elos[home] = rh + ELO_K * (score - expected_h)
        elos[away] = ra + ELO_K * ((1.0 - score) - expected_a)

    out = matches.copy()
    out["home_elo"] = home_elos
    out["away_elo"] = away_elos
    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    return out


def add_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Attach pre-match features to each fixture.

    Uses only results from *previous* games (rolling stats are shifted;
    Elo is the rating entering the match).
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    games = _add_rolling_form(_team_games(matches))

    home_feats = games.rename(
        columns={
            "team": "home_team",
            "points_l5": "home_points_l5",
            "gd_l5": "home_gd_l5",
            "season_pts": "home_season_pts",
        }
    )[["match_id", "home_team", "home_points_l5", "home_gd_l5", "home_season_pts"]]

    away_feats = games.rename(
        columns={
            "team": "away_team",
            "points_l5": "away_points_l5",
            "gd_l5": "away_gd_l5",
            "season_pts": "away_season_pts",
        }
    )[["match_id", "away_team", "away_points_l5", "away_gd_l5", "away_season_pts"]]

    featured = matches.reset_index(names="match_id")
    featured = featured.merge(home_feats, on=["match_id", "home_team"], how="left")
    featured = featured.merge(away_feats, on=["match_id", "away_team"], how="left")
    featured = featured.drop(columns=["match_id"])

    featured["points_diff_l5"] = featured["home_points_l5"] - featured["away_points_l5"]
    featured["gd_diff_l5"] = featured["home_gd_l5"] - featured["away_gd_l5"]
    featured["season_pts_diff"] = featured["home_season_pts"] - featured["away_season_pts"]

    featured = _add_elo(featured)
    return featured.sort_values("date").reset_index(drop=True)


def save_features(
    featured: pd.DataFrame | None = None,
    matches: pd.DataFrame | None = None,
    path: Path = PROCESSED_FEATURES,
) -> Path:
    """Build features if needed and write CSV."""
    if featured is None:
        if matches is None:
            from src.data import load_processed

            matches = load_processed()
        featured = add_features(matches)
    path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(path, index=False)
    return path


def load_features(path: Path = PROCESSED_FEATURES) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df


def final_elos(matches: pd.DataFrame) -> dict[str, float]:
    """Elo ratings after every match in the table has been played."""
    elos: dict[str, float] = {}
    ordered = matches.sort_values("date")
    for row in ordered.itertuples(index=False):
        home, away = row.home_team, row.away_team
        rh = elos.setdefault(home, ELO_START)
        ra = elos.setdefault(away, ELO_START)

        if row.home_goals > row.away_goals:
            score = 1.0
        elif row.home_goals < row.away_goals:
            score = 0.0
        else:
            score = 0.5

        expected_h = 1.0 / (1.0 + 10 ** ((ra - (rh + ELO_HOME_ADV)) / 400.0))
        expected_a = 1.0 - expected_h
        elos[home] = rh + ELO_K * (score - expected_h)
        elos[away] = ra + ELO_K * ((1.0 - score) - expected_a)
    return elos


def current_team_states(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Snapshot of each team *after* the last match in `matches`.

    Used to build features for a not-yet-played fixture.
    """
    games = _team_games(matches)
    elos = final_elos(matches)
    # Season labels look like "2021-22" — take the latest by match date.
    latest_season = matches.loc[matches["date"].idxmax(), "season"]
    rows = []

    for team, hist in games.groupby("team", sort=False):
        hist = hist.sort_values(["date", "match_id"])
        recent = hist.tail(FORM_WINDOW)
        season_hist = hist[hist["season"] == latest_season]
        rows.append(
            {
                "team": team,
                "points_l5": float(recent["points"].sum()),
                "gd_l5": float(recent["gd"].sum()),
                "season_pts": float(season_hist["points"].sum()),
                "elo": float(elos.get(team, ELO_START)),
            }
        )

    return pd.DataFrame(rows).set_index("team")


def fixture_features(
    home_team: str,
    away_team: str,
    states: pd.DataFrame,
) -> pd.DataFrame:
    """One-row feature frame for an upcoming home vs away match."""
    if home_team not in states.index:
        raise KeyError(f"Unknown home team: {home_team}")
    if away_team not in states.index:
        raise KeyError(f"Unknown away team: {away_team}")

    home = states.loc[home_team]
    away = states.loc[away_team]
    row = {
        "home_points_l5": home["points_l5"],
        "away_points_l5": away["points_l5"],
        "points_diff_l5": home["points_l5"] - away["points_l5"],
        "home_gd_l5": home["gd_l5"],
        "away_gd_l5": away["gd_l5"],
        "gd_diff_l5": home["gd_l5"] - away["gd_l5"],
        "home_season_pts": home["season_pts"],
        "away_season_pts": away["season_pts"],
        "season_pts_diff": home["season_pts"] - away["season_pts"],
        "home_elo": home["elo"],
        "away_elo": away["elo"],
        "elo_diff": home["elo"] - away["elo"],
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
