"""SignSense.features — turn raw MediaPipe hand landmarks into a
translation- and scale-invariant feature vector suitable for a classifier.

Why normalize at all: MediaPipe's landmark coordinates are in normalized
image space (0..1), so where your hand sits in the frame and how close
it is to the camera both leak into the raw numbers. A classifier trained
on raw coordinates would silently learn "sign A only counts near the
top-left of frame" instead of the actual hand shape. Centering on the
wrist and scaling by a reference bone length removes both of those
confounds, leaving (mostly) just the shape of the sign.
"""

from __future__ import annotations

import numpy as np

WRIST = 0
MIDDLE_MCP = 9  # base of the middle finger — a stable reference bone

FEATURE_DIM = 63  # 21 landmarks x (x, y, z)


def landmarks_to_features(landmarks: np.ndarray) -> np.ndarray:
    """landmarks: (21, 3) array of (x, y, z) in MediaPipe's normalized
    image coordinates. Returns a flat (63,) translation- and
    scale-normalized feature vector — the wrist always maps to the
    origin, and the wrist-to-middle-knuckle bone always has length 1.
    """
    pts = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
    origin = pts[WRIST].copy()
    centered = pts - origin
    scale = float(np.linalg.norm(centered[MIDDLE_MCP]))
    if scale < 1e-6:
        scale = 1e-6
    normalized = centered / scale
    return normalized.reshape(-1).astype(np.float32)
