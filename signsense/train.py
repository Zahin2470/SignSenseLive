"""SignSense.train — train the classifier from collected samples.

Usage:
    python -m signsense.train
    python -m signsense.train --test-size 0.25 --trees 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .confusion import format_confusion_text, save_confusion_heatmap
from .dataset import class_counts, load_dataset
from .model import SignClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "samples.csv"
MODEL_OUT = ROOT / "models" / "classifier.pkl"
CONFUSION_OUT = ROOT / "models" / "confusion_matrix.png"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the SignSense classifier")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=MODEL_OUT)
    parser.add_argument("--out-confusion", type=Path, default=CONFUSION_OUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    if len(y) == 0:
        print(f"No samples found at {args.data}. Run `python -m signsense.collect` first.")
        return 1

    counts = class_counts(y)
    print(f"Loaded {len(y)} samples across {len(counts)} classes:")
    for label, n in sorted(counts.items()):
        flag = "  (few samples — consider collecting more)" if n < 30 else ""
        print(f"  {label:15s} {n:4d}{flag}")

    if len(counts) < 2:
        print("\nNeed at least 2 different signs to train a classifier.")
        return 1

    clf = SignClassifier(n_estimators=args.trees)
    report = clf.fit(X, y, test_size=args.test_size)

    print(f"\nHeld-out accuracy: {report['accuracy']*100:.1f}% "
          f"({report['n_train']} train / {report['n_test']} test samples)\n")
    print(report["report"])

    cm = report["confusion_matrix"]
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(format_confusion_text(cm, report["classes"]))

    cm_path = save_confusion_heatmap(cm, report["classes"], args.out_confusion)
    if cm_path is not None:
        print(f"\nSaved confusion matrix heatmap to {cm_path}")

    clf.save(args.out)
    print(f"Saved model to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
