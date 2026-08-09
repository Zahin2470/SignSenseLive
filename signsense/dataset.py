"""SignSense.dataset — plain-CSV storage for labeled landmark samples.

Deliberately not a database or a binary format: a CSV means you can
open a samples file in a spreadsheet, sanity-check it, delete a bad
row by hand, or merge datasets collected on different days just by
concatenating files. One row per captured sample.

Works for any fixed feature width — single-hand (63-dim), dual-hand
(128-dim), or motion sequences — so long as every sample in one file
uses the same `feature_dim`. Keep different feature schemes in
different files (that's why collect/train/live all take a `--data`
path) rather than mixing widths in one CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM


def _header(feature_dim: int) -> list[str]:
    return ["label"] + [f"f{i}" for i in range(feature_dim)]


def append_sample(path: Path, label: str, features: np.ndarray, *, feature_dim: int = FEATURE_DIM) -> None:
    """Add one labeled sample, creating the file (with header) if needed."""
    features = np.asarray(features, dtype=np.float32).reshape(-1)
    if features.shape[0] != feature_dim:
        raise ValueError(f"expected {feature_dim} features, got {features.shape[0]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(_header(feature_dim))
        writer.writerow([label] + [f"{v:.6f}" for v in features.tolist()])


def load_dataset(path: Path, *, feature_dim: int = FEATURE_DIM) -> tuple[np.ndarray, list[str]]:
    """Returns (X, labels): X has shape (N, feature_dim); labels has length N."""
    if not path.is_file():
        return np.empty((0, feature_dim), dtype=np.float32), []
    rows: list[list[float]] = []
    labels: list[str] = []
    expected_header = _header(feature_dim)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != expected_header:
            got_dim = max(0, len(header) - 1) if header else 0
            raise ValueError(
                f"{path} has {got_dim} feature columns, expected {feature_dim}. "
                "Wrong file for this mode? Single-hand and dual-hand samples must "
                "live in separate files (see the --data / --two-hand flags)."
            )
        for row in reader:
            if not row:
                continue
            labels.append(row[0])
            rows.append([float(v) for v in row[1 : 1 + feature_dim]])
    X = np.array(rows, dtype=np.float32) if rows else np.empty((0, feature_dim), dtype=np.float32)
    return X, labels


def class_counts(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts
