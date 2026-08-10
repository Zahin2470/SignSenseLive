"""SignSense.skeleton — rainbow hand-skeleton overlay with a subtle
fingertip pulse, ported from the same visual language as VisionPuzzle
Studio (same landmark color scheme) for a consistent look across both
projects, but reimplemented here so SignSense has no dependency on
VisionPuzzle's package.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from .landmarks import CONNECTIONS, LANDMARK_COLOR

_TIPS = {4, 8, 12, 16, 20}


def _tip_glow(frame: np.ndarray, center: tuple[int, int], radius: int, color, intensity: float = 0.22) -> None:
    """Small ROI-only glow — unlike a full-frame copy+blend, this stays
    cheap even drawn 5x per frame (one per fingertip)."""
    cx, cy = center
    h, w = frame.shape[:2]
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(w, cx + radius), min(h, cy + radius)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    glow = roi.copy()
    cv2.circle(glow, (cx - x0, cy - y0), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(glow, intensity, roi, 1.0 - intensity, 0, dst=roi)


def draw_skeleton(
    frame: np.ndarray,
    landmarks: np.ndarray,
    w: int,
    h: int,
    *,
    t: Optional[float] = None,
    pulse: bool = True,
) -> np.ndarray:
    """landmarks: (21, 3) normalized (x, y, z) MediaPipe hand landmarks.
    Draws colored bone connections plus per-finger-colored joints, with
    fingertips gently pulsing (radius + soft glow) when `pulse=True` —
    a small "this is alive" cue that a plain static dot doesn't give.
    """
    now = t if t is not None else time.perf_counter()
    beat = 0.5 + 0.5 * math.sin(now * 5.0) if pulse else 0.0

    pts = []
    for lm in landmarks:
        x = int(np.clip(lm[0], 0.0, 1.0) * (w - 1))
        y = int(np.clip(lm[1], 0.0, 1.0) * (h - 1))
        pts.append((x, y))

    for a, b, color in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        color = LANDMARK_COLOR.get(i, (200, 200, 200))
        if i in _TIPS:
            radius = 5 + (int(2 * beat) if pulse else 0)
            if pulse:
                _tip_glow(frame, (x, y), radius + 6 + int(3 * beat), color, intensity=0.2)
        elif i == 0:
            radius = 6
        else:
            radius = 3
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)

    return frame
