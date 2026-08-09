"""SignSense.live_motion — live recognition for motion-based signs.

Unlike live.py's continuous per-frame classification, a motion sign
can't be judged mid-gesture — you need to see the whole clip. So this
uses the same start/stop recording workflow as collect_motion.py:
press SPACE, perform the sign, press SPACE again, and *then* it
predicts on the completed clip.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import ui
from .model import SignClassifier
from .motion_features import MOTION_FEATURE_DIM, sequence_to_motion_features
from .speech import Speaker
from .tracker import HandTracker

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = ROOT / "models" / "classifier_motion.pkl"
MIN_FRAMES = 6


class LiveMotionApp:
    def __init__(self, camera_index: int = 0, *, speech: bool = True, classifier_path: Optional[Path] = None) -> None:
        self.tracker = HandTracker(MODEL_PATH, max_hands=1)
        self.classifier_path = classifier_path or CLASSIFIER_PATH
        self.classifier = SignClassifier.try_load(self.classifier_path)
        self.camera_index = camera_index
        self.speaker = Speaker(enabled=speech)
        self.recording = False
        self._buffer: list[np.ndarray] = []
        self._last_hand: Optional[np.ndarray] = None
        self._result_label: Optional[str] = None
        self._result_conf = 0.0
        self._result_until = 0.0

    def run(self) -> int:
        if self.classifier is None:
            print(f"No trained motion model found at {self.classifier_path}.")
            print("Run  python -m signsense.collect_motion  then  python -m signsense.train --motion  first.")
            return 1

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("ERROR: could not open webcam.")
            return 1
        win = "SignSense — Live Motion"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print(f"SignSense live motion — recognizing: {', '.join(self.classifier.classes_)}  ·  "
              "SPACE start/stop · M mute · Q quit")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                hands = self.tracker.process(frame, mirrored=True)
                self._last_hand = hands[0].landmarks if hands else None
                if self.recording and self._last_hand is not None:
                    self._buffer.append(self._last_hand.copy())

                frame = self._draw(frame, hands, w, h)
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("m"), ord("M")):
                    self.speaker.toggle()
                if key == 32:
                    self._toggle_recording()
        finally:
            self.tracker.close()
            self.speaker.close()
            cap.release()
            cv2.destroyAllWindows()
        return 0

    def _toggle_recording(self) -> None:
        if not self.recording:
            self.recording = True
            self._buffer = []
            self._result_label = None
            return

        self.recording = False
        n = len(self._buffer)
        if n >= MIN_FRAMES:
            feat = sequence_to_motion_features(self._buffer)
            label, conf = self.classifier.predict(feat)
            self._result_label = label
            self._result_conf = conf
            self._result_until = time.perf_counter() + 3.0
            self.speaker.say(label)
        self._buffer = []

    def _draw(self, frame: np.ndarray, hands, w: int, h: int) -> np.ndarray:
        for hr in hands:
            for x, y, _z in hr.landmarks:
                cv2.circle(frame, (int(x * w), int(y * h)), 3, ui.ACCENT_HOT, -1, cv2.LINE_AA)

        panel_w = min(w - 28, 400)
        frame = ui.glass_panel(frame, (14, 14), (14 + panel_w, 110), radius=18)
        dot_color = ui.DANGER if self.recording else ui.ACCENT
        ui.glow_dot(frame, (34, 37), 10, dot_color, intensity=0.32)
        cv2.circle(frame, (34, 37), 5, dot_color, -1, cv2.LINE_AA)
        ui.put_text(frame, "SignSense", (50, 42), scale=0.58, color=ui.ACCENT, weight=2)
        ui.chip(frame, "MOTION", 14 + panel_w - 92, 20, color=ui.ACCENT_HOT)

        if self.recording:
            ui.put_text(frame, f"\u25cf recording... {len(self._buffer)} frames", (24, 78), scale=0.5, color=ui.DANGER, shadow=False)
        elif self._result_label and time.perf_counter() < self._result_until:
            ui.put_text(frame, self._result_label, (24, 82), scale=0.8, color=ui.SUCCESS, weight=3)
            ui.progress_bar(frame, 24, 96, panel_w - 48, 5, self._result_conf)
        else:
            ui.put_text(frame, "SPACE to record a sign", (24, 78), scale=0.5, color=ui.TEXT_MUTED, shadow=False)

        return frame


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="SignSense — live motion-sign recognition")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()
    return LiveMotionApp(camera_index=args.camera, speech=not args.no_speech, classifier_path=args.model).run()


if __name__ == "__main__":
    raise SystemExit(main())
