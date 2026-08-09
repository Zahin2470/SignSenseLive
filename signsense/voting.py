"""SignSense.voting — temporal majority-vote smoothing for per-frame
classifier predictions.

A classifier predicts fresh every frame, which flickers between
visually similar signs on live, slightly-noisy video even when it's
working correctly — that's normal, not a bug. This debounces it: keep
the last N predictions in a rolling window and only "commit" to a
label once it's the clear majority — the same idea as debouncing a
noisy sensor. Shared by live.py and practice.py so both modes read
signs the same way.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional

import numpy as np


class StablePredictor:
    def __init__(self, window: int = 8, min_agreement: float = 0.6, min_confidence: float = 0.55) -> None:
        self.window = window
        self.min_agreement = min_agreement
        self.min_confidence = min_confidence
        self._votes: deque[str] = deque(maxlen=window)
        self._confidences: deque[float] = deque(maxlen=window)
        self.stable_label: Optional[str] = None
        self.stable_confidence: float = 0.0

    def reset(self) -> None:
        self._votes.clear()
        self._confidences.clear()
        self.stable_label = None
        self.stable_confidence = 0.0

    def update(self, label: Optional[str], confidence: float = 0.0) -> Optional[str]:
        """Feed one frame's raw prediction (label=None if no hand is
        visible this frame). Returns the current stable label, if any."""
        if label is None:
            self.reset()
            return None

        self._votes.append(label)
        self._confidences.append(confidence)
        if len(self._votes) < self.window:
            return self.stable_label

        winner, count = Counter(self._votes).most_common(1)[0]
        agreement = count / len(self._votes)
        avg_conf = float(np.mean([c for l, c in zip(self._votes, self._confidences) if l == winner]))

        if agreement >= self.min_agreement and avg_conf >= self.min_confidence:
            self.stable_label = winner
            self.stable_confidence = avg_conf
        else:
            self.stable_label = None
            self.stable_confidence = 0.0
        return self.stable_label
