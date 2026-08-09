"""SignSense.tracker — a lean MediaPipe Hand Landmarker wrapper.

Deliberately simpler than a dual-hand game tracker: sign classification
wants the RAW instantaneous hand shape each frame, not a temporally
smoothed one — smoothing across frames would blur the transition
between two different signs. If you want steadier predictions, do
temporal voting on the *predicted labels* over a few frames (see
live.py) rather than smoothing the landmarks themselves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HandResult:
    landmarks: np.ndarray  # (21, 3) normalized x, y, z
    handedness: str        # "Left" | "Right" (viewer / mirrored space)
    score: float


class HandTracker:
    """Tracks up to `max_hands` hands from BGR frames via MediaPipe."""

    def __init__(self, model_path: Path, max_hands: int = 1) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing hand model: {model_path}\n"
                "Grab hand_landmarker.task from MediaPipe's model zoo and "
                "place it there (see README)."
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_ms = int(time.perf_counter() * 1000)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, *, mirrored: bool = True) -> list[HandResult]:
        from mediapipe import Image as MpImage, ImageFormat
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)

        ts = int(time.perf_counter() * 1000) - self._start_ms
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self._landmarker.detect_for_video(mp_image, ts)
        hands: list[HandResult] = []
        if not result.hand_landmarks:
            return hands

        for i, lms in enumerate(result.hand_landmarks):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            handedness = "Right"
            score = 0.0
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                handedness = cat.category_name
                score = float(cat.score)
            if mirrored:
                handedness = "Left" if handedness == "Right" else "Right"
            hands.append(HandResult(landmarks=pts, handedness=handedness, score=score))
        return hands

    def close(self) -> None:
        self._landmarker.close()


def hands_by_side(hands: list[HandResult]) -> dict:
    """Organize detected hands into {'Left': landmarks|None, 'Right': landmarks|None}.
    If MediaPipe ever reports two hands with the same handedness label
    (rare, but happens on ambiguous frames), keep the higher-confidence
    one rather than silently dropping/overwriting without a rule."""
    result: dict = {"Left": None, "Right": None}
    best_score: dict = {"Left": -1.0, "Right": -1.0}
    for hr in hands:
        side = hr.handedness if hr.handedness in ("Left", "Right") else "Right"
        if hr.score > best_score[side]:
            result[side] = hr.landmarks
            best_score[side] = hr.score
    return result
