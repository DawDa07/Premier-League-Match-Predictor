"""Train/eval helpers: baseline + candidate classifiers."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "match_model.joblib"

# Fixed label order so probabilities always align: H, D, A
LABELS = ["H", "D", "A"]


def time_based_split(
    matches: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by date order: earlier matches train, later matches test.

    Football data is time-ordered — never shuffle randomly, or you
    leak future information into training.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    ordered = matches.sort_values("date").reset_index(drop=True)
    cut = int(len(ordered) * train_ratio)
    if cut == 0 or cut == len(ordered):
        raise ValueError("Split produced an empty train or test set")

    train = ordered.iloc[:cut].copy()
    test = ordered.iloc[cut:].copy()
    return train, test


def baseline_always_home(n: int) -> list[str]:
    """Predict Home win for every match (no ML)."""
    return ["H"] * n


def accuracy(y_true: list[str] | pd.Series, y_pred: list[str] | pd.Series) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if not y_true:
        raise ValueError("Cannot score an empty set")
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def result_counts(results: pd.Series) -> pd.Series:
    return results.value_counts().reindex(LABELS).fillna(0).astype(int)


def make_model(name: str = "logistic") -> Pipeline:
    """Build a named classifier pipeline."""
    name = name.lower().strip()

    if name in {"logistic", "logreg", "lr"}:
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
        # Scaling matters for linear models (Elo vs points scales).
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if name in {"rf", "random_forest", "forest"}:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        # Trees don't need scaling; keep a no-op-free pipeline for a common API.
        return Pipeline([("clf", clf)])

    if name in {"gbm", "hgb", "hist_gradient_boosting", "boosting"}:
        clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        )
        return Pipeline([("clf", clf)])

    raise ValueError(
        f"Unknown model {name!r}. Choose: logistic, random_forest, hist_gradient_boosting"
    )


def train_model(
    train: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
    model_name: str = "logistic",
) -> Pipeline:
    X = train[feature_cols]
    y = train["result"]
    model = make_model(model_name)
    model.fit(X, y)
    return model


def evaluate_model(
    model: Pipeline,
    test: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
) -> dict:
    X = test[feature_cols]
    y_true = test["result"]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    return {
        "accuracy": accuracy(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba, labels=model.classes_),
        "y_true": y_true,
        "y_pred": y_pred,
        "report": classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS),
    }


def save_model(
    model: Pipeline,
    path: Path = MODEL_PATH,
    model_name: str = "logistic",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_name": model_name,
            "feature_columns": FEATURE_COLUMNS,
            "labels": LABELS,
        },
        path,
    )
    return path


def load_model(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)
