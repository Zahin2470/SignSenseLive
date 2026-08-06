"""SignSense.model — train, predict, and persist the sign classifier.

A RandomForest, not a deep net: hand-collected datasets here are
realistically a few hundred samples per class, not thousands — a
forest of shallow trees on 63 well-normalized features generalizes
better than a neural net would at that scale, trains in under a
second on a laptop CPU, and needs no GPU. Swap in a different
sklearn-compatible classifier later if the dataset grows large enough
to justify it; the surrounding train/predict/save/load API won't change.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


class SignClassifier:
    def __init__(self, n_estimators: int = 200, random_state: int = 42) -> None:
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, max_depth=None,
        )
        self.classes_: list[str] = []

    def fit(self, X: np.ndarray, y: list[str], *, test_size: float = 0.2) -> dict:
        """Train and hold out a slice for evaluation. Returns a small
        report dict — print report['report'] for a per-class breakdown."""
        distinct = set(y)
        if len(distinct) < 2:
            raise ValueError(
                f"Need at least 2 distinct classes to train, found {len(distinct)}: {distinct}. "
                "Collect more signs first."
            )
        counts: dict[str, int] = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        can_stratify = min(counts.values()) >= 2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if can_stratify else None,
        )
        self.clf.fit(X_train, y_train)
        self.classes_ = list(self.clf.classes_)
        y_pred = self.clf.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "report": classification_report(y_test, y_pred, zero_division=0),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "classes": self.classes_,
        }

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Returns (predicted_label, confidence 0..1)."""
        proba = self.clf.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        return self.classes_[idx], float(proba[idx])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"clf": self.clf, "classes_": self.classes_}, f)

    @classmethod
    def load(cls, path: Path) -> "SignClassifier":
        obj = cls()
        with path.open("rb") as f:
            data = pickle.load(f)
        obj.clf = data["clf"]
        obj.classes_ = data["classes_"]
        return obj

    @classmethod
    def try_load(cls, path: Path) -> Optional["SignClassifier"]:
        if not path.is_file():
            return None
        try:
            return cls.load(path)
        except Exception as exc:
            print(f"[model] could not load {path}: {exc}")
            return None
