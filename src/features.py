"""Pre-match feature engineering (form, venue form, H2H, Elo)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FEATURES = ROOT / "data" / "processed" / "features.csv"

FORM_WINDOW = 5
H2H_WINDOW = 5
DEFAULT_REST_DAYS = 7.0  # used before a team has a prior match in the data
ELO_K = 20
ELO_START = 1500.0
# Home advantage in Elo space (~50–100 common for football).
ELO_HOME_ADV = 100.0
# Bigger wins move Elo more (World Football Elo-style margins).
ELO_USE_MARGIN = True

# Columns the model is allowed to use (known before kickoff).
FEATURE_COLUMNS = [
    "home_points_l5",
    "away_points_l5",
    "points_diff_l5",
    "home_gd_l5",
    "away_gd_l5",
    "gd_diff_l5",
    "home_points_home_l5",
    "away_points_away_l5",
    "venue_points_diff",
    "h2h_points",
    "h2h_gd",
    "h2h_games",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
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
            "opponent": base["away_team"],
            "goals_for": base["home_goals"],
            "goals_against": base["away_goals"],
            "is_home": 1,
        }
    )
    away = pd.DataFrame(
        {
            "match_id": base["match_id"],
            "date": base["date"],
            "season": base["season"],
            "team": base["away_team"],
            "opponent": base["home_team"],
            "goals_for": base["away_goals"],
            "goals_against": base["home_goals"],
            "is_home": 0,
        }
    )
    games = pd.concat([home, away], ignore_index=True)
    games["points"] = [
        _points(gf, ga) for gf, ga in zip(games["goals_for"], games["goals_against"])
    ]
    games["gd"] = games["goals_for"] - games["goals_against"]
    return games.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def _add_rolling_form(games: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Overall form + venue-specific form (shifted = pre-match only)."""
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
    games["season_pts"] = (
        games.groupby(["team", "season"], sort=False)["points"]
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0.0)
    )

    # Last N home games only / last N away games only.
    venue_points = pd.Series(0.0, index=games.index)
    for is_home, col_mask in ((1, games["is_home"] == 1), (0, games["is_home"] == 0)):
        subset = games.loc[col_mask].copy()
        rolled = (
            subset.groupby("team", sort=False)["points"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
            .fillna(0.0)
        )
        venue_points.loc[subset.index] = rolled.to_numpy()

    games["venue_points_l5"] = venue_points

    # Days since this team's previous match (known before kickoff).
    games["rest_days"] = (
        games.groupby("team", sort=False)["date"]
        .diff()
        .dt.days
        .astype(float)
        .fillna(DEFAULT_REST_DAYS)
        .clip(lower=0.0)
    )
    return games


def _elo_margin_multiplier(home_goals: int, away_goals: int) -> float:
    """
    Scale Elo movement by winning margin (World Football Elo style).

    Draw or 1-goal win: 1.0
    2-goal win: 1.5
    3+ goals: (11 + goal_diff) / 8
    """
    gd = abs(int(home_goals) - int(away_goals))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def _elo_match_score(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def compute_elo_columns(
    matches: pd.DataFrame,
    *,
    k: float = ELO_K,
    home_adv: float = ELO_HOME_ADV,
    use_margin: bool = ELO_USE_MARGIN,
    start: float = ELO_START,
) -> tuple[list[float], list[float], dict[str, float]]:
    """
    Walk matches in date order.

    Returns pre-match home/away Elo lists and the final rating dict.
    """
    ordered = matches.sort_values("date").reset_index(drop=True)
    elos: dict[str, float] = {}
    home_elos: list[float] = []
    away_elos: list[float] = []

    for row in ordered.itertuples(index=False):
        home, away = row.home_team, row.away_team
        rh = elos.setdefault(home, start)
        ra = elos.setdefault(away, start)
        home_elos.append(rh)
        away_elos.append(ra)

        score = _elo_match_score(row.home_goals, row.away_goals)
        expected_h = 1.0 / (1.0 + 10 ** ((ra - (rh + home_adv)) / 400.0))
        expected_a = 1.0 - expected_h
        mov = _elo_margin_multiplier(row.home_goals, row.away_goals) if use_margin else 1.0
        k_eff = k * mov

        elos[home] = rh + k_eff * (score - expected_h)
        elos[away] = ra + k_eff * ((1.0 - score) - expected_a)

    return home_elos, away_elos, elos


def _add_elo(
    matches: pd.DataFrame,
    *,
    k: float = ELO_K,
    home_adv: float = ELO_HOME_ADV,
    use_margin: bool = ELO_USE_MARGIN,
) -> pd.DataFrame:
    """Chronological Elo; store ratings *before* each match is played."""
    matches = matches.sort_values("date").reset_index(drop=True)
    home_elos, away_elos, _ = compute_elo_columns(
        matches, k=k, home_adv=home_adv, use_margin=use_margin
    )
    out = matches.copy()
    out["home_elo"] = home_elos
    out["away_elo"] = away_elos
    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    return out


def _add_head_to_head(matches: pd.DataFrame, window: int = H2H_WINDOW) -> pd.DataFrame:
    """
    Past meetings between the two clubs (up to `window`), from the
    *current home team's* point of view. Only prior games — no leakage.
    """
    ordered = matches.sort_values("date").reset_index(drop=True)
    history: dict[frozenset[str], list[tuple[str, str, int, int]]] = defaultdict(list)

    h2h_points: list[float] = []
    h2h_gd: list[float] = []
    h2h_games: list[float] = []

    for row in ordered.itertuples(index=False):
        home, away = row.home_team, row.away_team
        key = frozenset((home, away))
        past = history[key][-window:]

        pts = 0.0
        gd = 0.0
        for prev_home, prev_away, hg, ag in past:
            if prev_home == home:
                pts += _points(hg, ag)
                gd += hg - ag
            else:
                # Current home team was away in that prior meeting.
                pts += _points(ag, hg)
                gd += ag - hg

        h2h_points.append(pts)
        h2h_gd.append(gd)
        h2h_games.append(float(len(past)))

        history[key].append((home, away, int(row.home_goals), int(row.away_goals)))

    out = ordered.copy()
    out["h2h_points"] = h2h_points
    out["h2h_gd"] = h2h_gd
    out["h2h_games"] = h2h_games
    return out


def add_features(
    matches: pd.DataFrame,
    *,
    elo_k: float = ELO_K,
    elo_home_adv: float = ELO_HOME_ADV,
    elo_use_margin: bool = ELO_USE_MARGIN,
) -> pd.DataFrame:
    """
    Attach pre-match features to each fixture.

    Uses only results from *previous* games (rolling stats are shifted;
    Elo / H2H are computed before the current match is applied).
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    games = _add_rolling_form(_team_games(matches))

    home_feats = games[games["is_home"] == 1].rename(
        columns={
            "team": "home_team",
            "points_l5": "home_points_l5",
            "gd_l5": "home_gd_l5",
            "season_pts": "home_season_pts",
            "venue_points_l5": "home_points_home_l5",
            "rest_days": "home_rest_days",
        }
    )[
        [
            "match_id",
            "home_team",
            "home_points_l5",
            "home_gd_l5",
            "home_season_pts",
            "home_points_home_l5",
            "home_rest_days",
        ]
    ]

    away_feats = games[games["is_home"] == 0].rename(
        columns={
            "team": "away_team",
            "points_l5": "away_points_l5",
            "gd_l5": "away_gd_l5",
            "season_pts": "away_season_pts",
            "venue_points_l5": "away_points_away_l5",
            "rest_days": "away_rest_days",
        }
    )[
        [
            "match_id",
            "away_team",
            "away_points_l5",
            "away_gd_l5",
            "away_season_pts",
            "away_points_away_l5",
            "away_rest_days",
        ]
    ]

    featured = matches.reset_index(names="match_id")
    featured = featured.merge(home_feats, on=["match_id", "home_team"], how="left")
    featured = featured.merge(away_feats, on=["match_id", "away_team"], how="left")
    featured = featured.drop(columns=["match_id"])

    featured["points_diff_l5"] = featured["home_points_l5"] - featured["away_points_l5"]
    featured["gd_diff_l5"] = featured["home_gd_l5"] - featured["away_gd_l5"]
    featured["season_pts_diff"] = featured["home_season_pts"] - featured["away_season_pts"]
    featured["venue_points_diff"] = (
        featured["home_points_home_l5"] - featured["away_points_away_l5"]
    )
    featured["rest_days_diff"] = featured["home_rest_days"] - featured["away_rest_days"]

    featured = _add_head_to_head(featured)
    featured = _add_elo(
        featured, k=elo_k, home_adv=elo_home_adv, use_margin=elo_use_margin
    )
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


def final_elos(
    matches: pd.DataFrame,
    *,
    k: float = ELO_K,
    home_adv: float = ELO_HOME_ADV,
    use_margin: bool = ELO_USE_MARGIN,
) -> dict[str, float]:
    """Elo ratings after every match in the table has been played."""
    _, _, elos = compute_elo_columns(
        matches, k=k, home_adv=home_adv, use_margin=use_margin
    )
    return elos


def _h2h_summary(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    window: int = H2H_WINDOW,
) -> dict[str, float]:
    """H2H stats for home_team vs away_team using completed matches only."""
    mask = (
        ((matches["home_team"] == home_team) & (matches["away_team"] == away_team))
        | ((matches["home_team"] == away_team) & (matches["away_team"] == home_team))
    )
    past = matches.loc[mask].sort_values("date").tail(window)

    pts = 0.0
    gd = 0.0
    for row in past.itertuples(index=False):
        if row.home_team == home_team:
            pts += _points(row.home_goals, row.away_goals)
            gd += row.home_goals - row.away_goals
        else:
            pts += _points(row.away_goals, row.home_goals)
            gd += row.away_goals - row.home_goals

    return {"h2h_points": pts, "h2h_gd": float(gd), "h2h_games": float(len(past))}


def current_team_states(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Snapshot of each team *after* the last match in `matches`.

    Used to build features for a not-yet-played fixture.
    """
    games = _team_games(matches)
    elos = final_elos(matches)
    latest_season = matches.loc[matches["date"].idxmax(), "season"]
    rows = []

    for team, hist in games.groupby("team", sort=False):
        hist = hist.sort_values(["date", "match_id"])
        recent = hist.tail(FORM_WINDOW)
        home_recent = hist[hist["is_home"] == 1].tail(FORM_WINDOW)
        away_recent = hist[hist["is_home"] == 0].tail(FORM_WINDOW)
        season_hist = hist[hist["season"] == latest_season]
        rows.append(
            {
                "team": team,
                "points_l5": float(recent["points"].sum()),
                "gd_l5": float(recent["gd"].sum()),
                "points_home_l5": float(home_recent["points"].sum()),
                "points_away_l5": float(away_recent["points"].sum()),
                "season_pts": float(season_hist["points"].sum()),
                "elo": float(elos.get(team, ELO_START)),
                "last_match_date": hist["date"].iloc[-1],
            }
        )

    return pd.DataFrame(rows).set_index("team")


def fixture_features(
    home_team: str,
    away_team: str,
    states: pd.DataFrame,
    matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One-row feature frame for an upcoming home vs away match."""
    if home_team not in states.index:
        raise KeyError(f"Unknown home team: {home_team}")
    if away_team not in states.index:
        raise KeyError(f"Unknown away team: {away_team}")

    home = states.loc[home_team]
    away = states.loc[away_team]
    h2h = (
        _h2h_summary(matches, home_team, away_team)
        if matches is not None
        else {"h2h_points": 0.0, "h2h_gd": 0.0, "h2h_games": 0.0}
    )

    # Assume a kickoff one week after the last date in the dataset.
    if matches is not None and len(matches):
        kickoff = pd.Timestamp(matches["date"].max()) + pd.Timedelta(days=7)
    else:
        kickoff = None

    if kickoff is not None:
        home_rest = float((kickoff - pd.Timestamp(home["last_match_date"])).days)
        away_rest = float((kickoff - pd.Timestamp(away["last_match_date"])).days)
        home_rest = max(home_rest, 0.0)
        away_rest = max(away_rest, 0.0)
    else:
        home_rest = DEFAULT_REST_DAYS
        away_rest = DEFAULT_REST_DAYS

    row = {
        "home_points_l5": home["points_l5"],
        "away_points_l5": away["points_l5"],
        "points_diff_l5": home["points_l5"] - away["points_l5"],
        "home_gd_l5": home["gd_l5"],
        "away_gd_l5": away["gd_l5"],
        "gd_diff_l5": home["gd_l5"] - away["gd_l5"],
        "home_points_home_l5": home["points_home_l5"],
        "away_points_away_l5": away["points_away_l5"],
        "venue_points_diff": home["points_home_l5"] - away["points_away_l5"],
        "h2h_points": h2h["h2h_points"],
        "h2h_gd": h2h["h2h_gd"],
        "h2h_games": h2h["h2h_games"],
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_days_diff": home_rest - away_rest,
        "home_season_pts": home["season_pts"],
        "away_season_pts": away["season_pts"],
        "season_pts_diff": home["season_pts"] - away["season_pts"],
        "home_elo": home["elo"],
        "away_elo": away["elo"],
        "elo_diff": home["elo"] - away["elo"],
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
