"""SignSense.motion_features — turn a short SEQUENCE of hand landmarks
into a feature vector capturing both shape and movement, for signs
that involve motion (e.g. ASL letters like J or Z) rather than a
static pose.

Design: rather than feeding a full per-frame sequence into a recurrent
model — which needs far more training data than a webcam session
realistically produces, and which this project's whole philosophy
(small-dataset-friendly, explainable, CPU-only) argues against — this
hand-engineers a compact motion summary instead:

  - hand SHAPE at the start, middle, and end of the clip (63 dims each)
  - net wrist DISPLACEMENT over the clip (3 dims)
  - total path length the wrist traveled (1 dim)
  - a "straightness" ratio = path length / displacement magnitude
    (1.0 = a perfectly straight motion, higher = more curved/wandering)

Everything is normalized by the hand's own scale at the start frame,
so a sign performed close to the camera and one performed farther away
produce the same features — same principle as the static-pose
normalization in features.py, extended to a trajectory.
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_DIM, MIDDLE_MCP, WRIST, landmarks_to_features

MOTION_FEATURE_DIM = FEATURE_DIM * 3 + 3 + 1 + 1  # start+mid+end shape, displacement, path length, straightness


def sequence_to_motion_features(frames: list) -> np.ndarray:
    """frames: list of (21, 3) raw landmark arrays across a short
    recording window, in chronological order. Needs at least 2 frames.
    """
    if len(frames) < 2:
        raise ValueError(f"need at least 2 frames to compute motion features, got {len(frames)}")

    pts = [np.asarray(f, dtype=np.float32).reshape(21, 3) for f in frames]
    mid_idx = len(pts) // 2

    start_shape = landmarks_to_features(pts[0])
    mid_shape = landmarks_to_features(pts[mid_idx])
    end_shape = landmarks_to_features(pts[-1])

    scale = float(np.linalg.norm(pts[0][MIDDLE_MCP] - pts[0][WRIST]))
    if scale < 1e-6:
        scale = 1e-6

    wrist_path = np.array([p[WRIST] for p in pts], dtype=np.float32)  # (T, 3)
    displacement = (wrist_path[-1] - wrist_path[0]) / scale

    deltas = np.diff(wrist_path, axis=0)
    path_length = float(np.sum(np.linalg.norm(deltas, axis=1))) / scale
    disp_mag = float(np.linalg.norm(wrist_path[-1] - wrist_path[0])) / scale
    straightness = path_length / (disp_mag + 1e-6)

    return np.concatenate([
        start_shape,
        mid_shape,
        end_shape,
        displacement.astype(np.float32),
        np.array([path_length, straightness], dtype=np.float32),
    ]).astype(np.float32)
