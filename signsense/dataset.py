"""SignSense.dataset — plain-CSV storage for labeled landmark samples.

Deliberately not a database or a binary format: a CSV means you can
open `data/samples.csv` in a spreadsheet, sanity-check it, delete a
bad row by hand, or merge datasets collected on different days just by
concatenating files. One row per captured sample.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM

HEADER = ["label"] + [f"f{i}" for i in range(FEATURE_DIM)]


def append_sample(path: Path, label: str, features: np.ndarray) -> None:
    """Add one labeled sample, creating the file (with header) if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(HEADER)
        writer.writerow([label] + [f"{v:.6f}" for v in features.tolist()])


def load_dataset(path: Path) -> tuple[np.ndarray, list[str]]:
    """Returns (X, labels): X has shape (N, FEATURE_DIM); labels has length N."""
    if not path.is_file():
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []
    rows: list[list[float]] = []
    labels: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != HEADER:
            raise ValueError(
                f"{path} doesn't look like a SignSense samples file "
                "(header mismatch — was it edited or from an older version?)"
            )
        for row in reader:
            if not row:
                continue
            labels.append(row[0])
            rows.append([float(v) for v in row[1 : 1 + FEATURE_DIM]])
    X = np.array(rows, dtype=np.float32) if rows else np.empty((0, FEATURE_DIM), dtype=np.float32)
    return X, labels


def class_counts(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts
