"""Predict a single fixture from the latest team form / Elo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import PROCESSED_MATCHES, build_matches, load_processed, save_matches
from src.features import current_team_states, fixture_features
from src.model import MODEL_PATH, load_model

# Common short names / full names → names used in results.csv
TEAM_ALIASES = {
    "brighton and hove albion": "Brighton",
    "manchester city": "Man City",
    "man city": "Man City",
    "manchester united": "Man United",
    "manchester utd": "Man United",
    "man utd": "Man United",
    "man united": "Man United",
    "newcastle united": "Newcastle",
    "newcastle utd": "Newcastle",
    "nottingham forest": "Nott'm Forest",
    "nottm forest": "Nott'm Forest",
    "forest": "Nott'm Forest",
    "sheffield wednesday": "Sheffield Weds",
    "sheff wed": "Sheffield Weds",
    "spurs": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "west bromwich albion": "West Brom",
    "west ham united": "West Ham",
    "wolverhampton wanderers": "Wolves",
    "queens park rangers": "QPR",
}


def _load_matches() -> pd.DataFrame:
    if PROCESSED_MATCHES.exists():
        return load_processed()
    matches = build_matches()
    save_matches(matches)
    return matches


def resolve_team(name: str, known_teams: list[str]) -> str:
    """Map a user-entered name to the canonical team name."""
    cleaned = " ".join(name.strip().split())
    lower = cleaned.lower()

    by_lower = {t.lower(): t for t in known_teams}
    if lower in by_lower:
        return by_lower[lower]
    if lower in TEAM_ALIASES:
        return TEAM_ALIASES[lower]

    # Unique substring match, e.g. "Liverpool" / "arsenal"
    hits = [t for t in known_teams if lower in t.lower()]
    if len(hits) == 1:
        return hits[0]

    options = ", ".join(known_teams)
    raise ValueError(
        f"Unknown team {name!r}. Use a full name or alias.\nAvailable: {options}"
    )


def predict_match(
    home_team: str,
    away_team: str,
    matches: pd.DataFrame | None = None,
    model_path: Path = MODEL_PATH,
) -> dict:
    """
    Predict H / D / A for a not-yet-played fixture.

    Features come from each team's state after the last match in our data
    (as of ~April 2022 with the current CSV).
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run: python scripts/train.py"
        )

    matches = matches if matches is not None else _load_matches()
    known = sorted(
        set(matches["home_team"]).union(matches["away_team"])
    )
    home = resolve_team(home_team, known)
    away = resolve_team(away_team, known)
    if home == away:
        raise ValueError("Home and away teams must be different.")

    bundle = load_model(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]

    states = current_team_states(matches)
    X = fixture_features(home, away, states, matches=matches)[feature_cols]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    prob_by_class = dict(zip(model.classes_, proba))

    return {
        "home_team": home,
        "away_team": away,
        "prediction": pred,
        "probabilities": {
            "H": float(prob_by_class.get("H", 0.0)),
            "D": float(prob_by_class.get("D", 0.0)),
            "A": float(prob_by_class.get("A", 0.0)),
        },
        "features": X.iloc[0].to_dict(),
        "as_of": matches["date"].max(),
    }


def format_prediction(result: dict) -> str:
    home = result["home_team"]
    away = result["away_team"]
    probs = result["probabilities"]
    label = {"H": f"{home} win", "D": "Draw", "A": f"{away} win"}[result["prediction"]]
    as_of = pd.Timestamp(result["as_of"]).date()

    lines = [
        f"Fixture: {home} vs {away}",
        f"Team strength as of: {as_of} (last date in dataset)",
        f"Predicted: {label}",
        f"  {home} win: {probs['H'] * 100:5.1f}%",
        f"  Draw:         {probs['D'] * 100:5.1f}%",
        f"  {away} win: {probs['A'] * 100:5.1f}%",
    ]
    return "\n".join(lines)
