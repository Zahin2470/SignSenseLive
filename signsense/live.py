"""SignSense.live — real-time sign recognition demo.

Prediction stability comes from `voting.StablePredictor` — see that
module for why raw per-frame predictions flicker and how debouncing
fixes it. This file is just the camera loop + HUD around it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import ui
from .features import landmarks_to_features
from .model import SignClassifier
from .speech import Speaker
from .tracker import HandTracker
from .voting import StablePredictor

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = ROOT / "models" / "classifier.pkl"


class LiveApp:
    def __init__(self, camera_index: int = 0, *, speech: bool = True) -> None:
        self.tracker = HandTracker(MODEL_PATH, max_hands=1)
        self.classifier = SignClassifier.try_load(CLASSIFIER_PATH)
        self.camera_index = camera_index
        self.predictor = StablePredictor()
        self.speaker = Speaker(enabled=speech)
        self._last_spoken: Optional[str] = None
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
        print(f"SignSense live — recognizing: {', '.join(self.classifier.classes_)}  ·  M mute · Q quit")

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
                if key in (ord("m"), ord("M")):
                    self.speaker.toggle()
        finally:
            self.tracker.close()
            self.speaker.close()
            cap.release()
            cv2.destroyAllWindows()
        return 0

    def _update_prediction(self, hands) -> None:
        if not hands:
            self.predictor.update(None)
            self._last_spoken = None
            return
        feat = landmarks_to_features(hands[0].landmarks)
        label, conf = self.classifier.predict(feat)
        stable = self.predictor.update(label, conf)
        if stable is not None and stable != self._last_spoken:
            self.speaker.say(stable)
            self._last_spoken = stable

    def _draw(self, frame: np.ndarray, hands, w: int, h: int) -> np.ndarray:
        for hr in hands:
            for x, y, _z in hr.landmarks:
                cv2.circle(frame, (int(x * w), int(y * h)), 3, ui.ACCENT_HOT, -1, cv2.LINE_AA)

        panel_w = min(w - 28, 360)
        frame = ui.glass_panel(frame, (14, 14), (14 + panel_w, 104), radius=18)
        ui.glow_dot(frame, (34, 37), 10, ui.ACCENT, intensity=0.3)
        cv2.circle(frame, (34, 37), 5, ui.ACCENT, -1, cv2.LINE_AA)
        ui.put_text(frame, "SignSense", (50, 42), scale=0.58, color=ui.ACCENT, weight=2)
        if not self.speaker.enabled:
            ui.chip(frame, "MUTED", 14 + panel_w - 78, 20, color=ui.TEXT_MUTED)

        label, conf = self.predictor.stable_label, self.predictor.stable_confidence
        target_conf = conf if label else 0.0
        self._conf_disp = ui.smooth_toward(self._conf_disp, target_conf, 0.25)

        if label:
            ui.put_text(frame, label, (24, 82), scale=0.85, color=ui.SUCCESS, weight=3)
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
    parser.add_argument("--no-speech", action="store_true", help="Disable text-to-speech output")
    args = parser.parse_args()
    return LiveApp(camera_index=args.camera, speech=not args.no_speech).run()


if __name__ == "__main__":
    raise SystemExit(main())

