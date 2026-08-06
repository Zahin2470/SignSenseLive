"""SignSense.live — real-time sign recognition demo.

The classifier predicts fresh every single frame, which flickers
between visually similar signs when a raw per-frame prediction is
shown directly (this is normal — even a good classifier's confidence
wobbles frame to frame on live, slightly-noisy video). We smooth that
out with simple temporal voting: keep the last N predictions in a
short rolling window and only display a sign once it's the majority
vote — same idea as debouncing a noisy sensor.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import ui
from .features import landmarks_to_features
from .model import SignClassifier
from .tracker import HandTracker

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = ROOT / "models" / "classifier.pkl"

VOTE_WINDOW = 8
MIN_CONFIDENCE = 0.55


class LiveApp:
    def __init__(self, camera_index: int = 0) -> None:
        self.tracker = HandTracker(MODEL_PATH, max_hands=1)
        self.classifier = SignClassifier.try_load(CLASSIFIER_PATH)
        self.camera_index = camera_index
        self._votes: deque[str] = deque(maxlen=VOTE_WINDOW)
        self._confidences: deque[float] = deque(maxlen=VOTE_WINDOW)
        self._stable_label: Optional[str] = None
        self._stable_conf = 0.0
        self._conf_disp = 0.0

    def run(self) -> int:
        if self.classifier is None:
            print(f"No trained model found at {CLASSIFIER_PATH}.")
            print("Run  python -m signsense.collect  to record some signs, then")
            print("     python -m signsense.train    to train the classifier.")
            return 1

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("ERROR: could not open webcam.")
            return 1
        win = "SignSense — Live"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print(f"SignSense live — recognizing: {', '.join(self.classifier.classes_)}  ·  Q to quit")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                hands = self.tracker.process(frame, mirrored=True)
                self._update_prediction(hands)
                frame = self._draw(frame, hands, w, h)

                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
        finally:
            self.tracker.close()
            cap.release()
            cv2.destroyAllWindows()
        return 0

    def _update_prediction(self, hands) -> None:
        if not hands:
            self._votes.clear()
            self._confidences.clear()
            self._stable_label = None
            self._stable_conf = 0.0
            return

        feat = landmarks_to_features(hands[0].landmarks)
        label, conf = self.classifier.predict(feat)
        self._votes.append(label)
        self._confidences.append(conf)

        if len(self._votes) < VOTE_WINDOW:
            return  # not enough history yet to commit to a stable reading

        winner, count = Counter(self._votes).most_common(1)[0]
        agreement = count / len(self._votes)
        avg_conf = float(np.mean([c for l, c in zip(self._votes, self._confidences) if l == winner]))

        if agreement >= 0.6 and avg_conf >= MIN_CONFIDENCE:
            self._stable_label = winner
            self._stable_conf = avg_conf
        else:
            self._stable_label = None
            self._stable_conf = 0.0

    def _draw(self, frame: np.ndarray, hands, w: int, h: int) -> np.ndarray:
        for hr in hands:
            for x, y, _z in hr.landmarks:
                cv2.circle(frame, (int(x * w), int(y * h)), 3, ui.ACCENT_HOT, -1, cv2.LINE_AA)

        panel_w = min(w - 28, 360)
        frame = ui.glass_panel(frame, (14, 14), (14 + panel_w, 104), radius=18)
        ui.glow_dot(frame, (34, 37), 10, ui.ACCENT, intensity=0.3)
        cv2.circle(frame, (34, 37), 5, ui.ACCENT, -1, cv2.LINE_AA)
        ui.put_text(frame, "SignSense", (50, 42), scale=0.58, color=ui.ACCENT, weight=2)

        target_conf = self._stable_conf if self._stable_label else 0.0
        self._conf_disp = ui.smooth_toward(self._conf_disp, target_conf, 0.25)

        if self._stable_label:
            ui.put_text(frame, self._stable_label, (24, 82), scale=0.85, color=ui.SUCCESS, weight=3)
        elif hands:
            ui.put_text(frame, "reading...", (24, 78), scale=0.5, color=ui.TEXT_MUTED, shadow=False)
        else:
            ui.put_text(frame, "show your hand", (24, 78), scale=0.5, color=ui.TEXT_MUTED, shadow=False)

        ui.progress_bar(frame, 24, 92, panel_w - 48, 5, self._conf_disp)
        return frame


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="SignSense — live sign recognition")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    return LiveApp(camera_index=args.camera).run()


if __name__ == "__main__":
    raise SystemExit(main())
