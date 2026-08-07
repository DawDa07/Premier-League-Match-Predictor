"""Train candidate models, compare to baseline, save the best."""

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

CANDIDATES = [
    ("logistic", "Logistic regression"),
    ("random_forest", "Random forest"),
    ("hist_gradient_boosting", "Hist gradient boosting"),
]


def load_or_build_features():
    if not PROCESSED_MATCHES.exists():
        save_matches(build_matches())

    if PROCESSED_FEATURES.exists():
        featured = load_features()
        if set(FEATURE_COLUMNS).issubset(featured.columns):
            return featured

    featured = add_features(load_processed())
    save_features(featured)
    return featured


def main() -> None:
    data = load_or_build_features()
    train, test = time_based_split(data, train_ratio=0.8)

    y_test = test["result"]
    baseline_acc = accuracy(y_test, baseline_always_home(len(test)))

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

    results = []
    for key, label in CANDIDATES:
        model = train_model(train, model_name=key)
        metrics = evaluate_model(model, test)
        results.append((key, label, model, metrics))
        delta = metrics["accuracy"] - baseline_acc
        print(f"=== {label} ===")
        print(f"Accuracy: {metrics['accuracy']:.1%}")
        print(f"Log loss: {metrics['log_loss']:.3f}  (lower is better)")
        print(f"Vs baseline: {delta:+.1%}")
        print()

    # Prefer higher accuracy; break ties with lower log loss.
    best_key, best_label, best_model, best_metrics = max(
        results, key=lambda r: (r[3]["accuracy"], -r[3]["log_loss"])
    )
    path = save_model(best_model, model_name=best_key)

    print(f"=== Winner: {best_label} ===")
    print(f"Accuracy: {best_metrics['accuracy']:.1%}")
    print(f"Log loss: {best_metrics['log_loss']:.3f}")
    print()
    print("=== Classification report (winner) ===")
    print(best_metrics["report"])
    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print("labels:", LABELS)
    print(best_metrics["confusion_matrix"])
    print()
    print(f"Saved model → {path.relative_to(ROOT)}  ({best_key})")
    print('Predict: python scripts/predict_match.py "Arsenal" "Chelsea"')


if __name__ == "__main__":
    main()
