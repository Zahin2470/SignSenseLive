"""SignSense.train — train the classifier from collected samples.

Usage:
    python -m signsense.train
    python -m signsense.train --two-hand
    python -m signsense.train --motion
    python -m signsense.train --test-size 0.25 --trees 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .confusion import format_confusion_text, save_confusion_heatmap
from .dataset import class_counts, load_dataset
from .features import DUAL_FEATURE_DIM, FEATURE_DIM
from .model import SignClassifier
from .motion_features import MOTION_FEATURE_DIM

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "samples.csv"
DATA_PATH_2H = ROOT / "data" / "samples_2h.csv"
DATA_PATH_MOTION = ROOT / "data" / "samples_motion.csv"
MODEL_OUT = ROOT / "models" / "classifier.pkl"
MODEL_OUT_2H = ROOT / "models" / "classifier_2h.pkl"
MODEL_OUT_MOTION = ROOT / "models" / "classifier_motion.pkl"
CONFUSION_OUT = ROOT / "models" / "confusion_matrix.png"
CONFUSION_OUT_2H = ROOT / "models" / "confusion_matrix_2h.png"
CONFUSION_OUT_MOTION = ROOT / "models" / "confusion_matrix_motion.png"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the SignSense classifier")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--two-hand", action="store_true", help="Train on two-handed sign samples")
    mode_group.add_argument("--motion", action="store_true", help="Train on motion-sign samples")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--out-confusion", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args()

    if args.motion:
        feature_dim = MOTION_FEATURE_DIM
        data_path = args.data or DATA_PATH_MOTION
        out_path = args.out or MODEL_OUT_MOTION
        confusion_path = args.out_confusion or CONFUSION_OUT_MOTION
        hand_flag = " --motion"
    elif args.two_hand:
        feature_dim = DUAL_FEATURE_DIM
        data_path = args.data or DATA_PATH_2H
        out_path = args.out or MODEL_OUT_2H
        confusion_path = args.out_confusion or CONFUSION_OUT_2H
        hand_flag = " --two-hand"
    else:
        feature_dim = FEATURE_DIM
        data_path = args.data or DATA_PATH
        out_path = args.out or MODEL_OUT
        confusion_path = args.out_confusion or CONFUSION_OUT
        hand_flag = ""

    X, y = load_dataset(data_path, feature_dim=feature_dim)
    if len(y) == 0:
        collect_cmd = "collect_motion" if args.motion else "collect"
        print(f"No samples found at {data_path}. Run `python -m signsense.{collect_cmd}{hand_flag if not args.motion else ''}` first.")
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

    cm_path = save_confusion_heatmap(cm, report["classes"], confusion_path)
    if cm_path is not None:
        print(f"\nSaved confusion matrix heatmap to {cm_path}")

    clf.save(out_path)
    print(f"Saved model to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
