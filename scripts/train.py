"""Phase 5: train logistic regression and compare to the always-home baseline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_MATCHES, build_matches, load_processed, save_matches
from src.features import (
    FEATURE_COLUMNS,
    PROCESSED_FEATURES,
    add_features,
    load_features,
    save_features,
)
from src.model import (
    LABELS,
    accuracy,
    baseline_always_home,
    evaluate_model,
    result_counts,
    save_model,
    time_based_split,
    train_model,
)


def load_or_build_features():
    if PROCESSED_FEATURES.exists():
        return load_features()
    if not PROCESSED_MATCHES.exists():
        save_matches(build_matches())
    featured = add_features(load_processed())
    save_features(featured)
    return featured


def main() -> None:
    data = load_or_build_features()
    train, test = time_based_split(data, train_ratio=0.8)

    y_test = test["result"]
    baseline_pred = baseline_always_home(len(test))
    baseline_acc = accuracy(y_test, baseline_pred)

    model = train_model(train)
    metrics = evaluate_model(model, test)
    path = save_model(model)

    print("=== Time-based split (80% train / 20% test) ===")
    print(f"Train: {len(train):,}  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"Test:  {len(test):,}  ({test['date'].min().date()} → {test['date'].max().date()})")
    print(f"Features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")
    print()

    print("=== Test set result balance ===")
    print(result_counts(y_test).to_string())
    print()

    print("=== Baseline: always Home (H) ===")
    print(f"Accuracy: {baseline_acc:.1%}")
    print()

    print("=== Logistic regression ===")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"Log loss: {metrics['log_loss']:.3f}  (lower is better)")
    delta = metrics["accuracy"] - baseline_acc
    vs = "BEATS" if delta > 0 else "does not beat"
    print(f"Vs baseline: {delta:+.1%} → {vs} the always-home baseline")
    print()

    print("=== Classification report ===")
    print(metrics["report"])

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print("labels:", LABELS)
    print(metrics["confusion_matrix"])
    print()

    print(f"Saved model → {path.relative_to(ROOT)}")
    print('Predict: python scripts/predict_match.py "Arsenal" "Chelsea"')


if __name__ == "__main__":
    main()
