"""SignSense.collect_motion — record labeled samples for signs that
involve movement (e.g. ASL letters like J or Z), not a static pose.

Unlike collect.py's continuous per-frame capture, a motion sign is ONE
sample per performance: press SPACE to start recording a short clip,
perform the sign, press SPACE again to stop — the whole clip becomes
one labeled sample (via motion_features.sequence_to_motion_features).
Single-hand only for now; combining motion + two-hand signs is a
natural extension but adds real complexity and isn't implemented here
(see README).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import ui
from .dataset import append_sample, class_counts, load_dataset
from .features import WRIST
from .motion_features import MOTION_FEATURE_DIM, sequence_to_motion_features
from .skeleton import draw_skeleton
from .tracker import HandTracker

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
DATA_PATH = ROOT / "data" / "samples_motion.csv"

MIN_FRAMES = 6  # discard a recording this short — not enough to summarize meaningfully


class CollectMotionApp:
    def __init__(self, camera_index: int = 0, *, data_path: Optional[Path] = None) -> None:
        self.tracker = HandTracker(MODEL_PATH, max_hands=1)
        self.camera_index = camera_index
        self.data_path = data_path or DATA_PATH
        self.label = "sign_1"
        self._editing_label = False
        self._edit_buffer = ""
        self.recording = False
        self._buffer: list[np.ndarray] = []
        self._last_hand: Optional[np.ndarray] = None
        self._last_status = ""
        self._status_until = 0.0

    def run(self) -> int:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("ERROR: could not open webcam.")
            return 1
        win = "SignSense — Collect Motion"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print("SignSense motion collector — L to name a label, SPACE to start/stop a recording, Q to quit")

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
                if not self._handle_key(key):
                    break
        finally:
            self.tracker.close()
            cap.release()
            cv2.destroyAllWindows()
        return 0

    def _toggle_recording(self) -> None:
        if not self.recording:
            self.recording = True
            self._buffer = []
        else:
            self.recording = False
            n = len(self._buffer)
            if n < MIN_FRAMES:
                self._set_status(f"too short ({n} frames) — discarded, try a slower/longer motion", time.perf_counter())
                self._buffer = []
                return
            try:
                feat = sequence_to_motion_features(self._buffer)
                append_sample(self.data_path, self.label, feat, feature_dim=MOTION_FEATURE_DIM)
                self._set_status(f"saved ({n} frames)", time.perf_counter())
            except Exception as exc:
                self._set_status(f"error: {exc}", time.perf_counter())
            self._buffer = []

    def _set_status(self, text: str, now: float) -> None:
        self._last_status = text
        self._status_until = now + 2.0

    def _draw(self, frame: np.ndarray, hands, w: int, h: int) -> np.ndarray:
        if self.recording and len(self._buffer) >= 2:
            trail_pts = [
                (int(np.clip(f[WRIST][0], 0.0, 1.0) * (w - 1)), int(np.clip(f[WRIST][1], 0.0, 1.0) * (h - 1)))
                for f in self._buffer
            ]
            ui.draw_trail(frame, trail_pts, ui.DANGER)

        for hr in hands:
            draw_skeleton(frame, hr.landmarks, w, h)

        frame = ui.glass_panel(frame, (14, 14), (min(w - 14, 440), 104), radius=16)
        dot_color = ui.DANGER if self.recording else ui.ACCENT
        ui.glow_dot(frame, (34, 37), 10, dot_color, intensity=0.35)
        cv2.circle(frame, (34, 37), 5, dot_color, -1, cv2.LINE_AA)
        label_text = self._edit_buffer + "_" if self._editing_label else self.label
        ui.put_text(frame, label_text, (50, 42), scale=0.6, color=ui.ACCENT, weight=2)
        ui.chip(frame, "MOTION", min(w - 14, 440) - 96, 20, color=ui.ACCENT_HOT)

        if self.recording:
            ui.put_text(frame, f"\u25cf recording... {len(self._buffer)} frames", (24, 72), scale=0.42, color=ui.DANGER, shadow=False)
        elif time.perf_counter() < self._status_until:
            ui.put_text(frame, self._last_status, (24, 72), scale=0.42, color=ui.SUCCESS, shadow=False)
        else:
            hint = "hand visible" if self._last_hand is not None else "show your hand"
            ui.put_text(frame, hint, (24, 72), scale=0.42, color=ui.TEXT_MUTED, shadow=False)

        X, labels = load_dataset(self.data_path, feature_dim=MOTION_FEATURE_DIM)
        counts = class_counts(labels)
        y = 130
        ui.put_text(frame, f"DATASET  ({len(labels)} clips total)", (14, y), scale=0.44, color=ui.TEXT_MUTED, shadow=False)
        for i, (lbl, n) in enumerate(sorted(counts.items())):
            color = ui.SUCCESS if lbl == self.label else ui.TEXT
            ui.put_text(frame, f"{lbl}: {n}", (14, y + 26 + i * 22), scale=0.42, color=color)

        hint_y = h - 16
        if self._editing_label:
            hint = "Type label, ENTER to confirm, Esc to cancel"
        else:
            hint = "L rename label \u00b7 SPACE start/stop recording \u00b7 Q quit"
        ui.put_text(frame, hint, (14, hint_y), scale=0.42, color=ui.TEXT_MUTED, shadow=False)
        return frame

    def _handle_key(self, key: int) -> bool:
        if self._editing_label:
            if key in (13, 10):
                if self._edit_buffer:
                    self.label = self._edit_buffer
                self._editing_label = False
                self._edit_buffer = ""
            elif key == 27:
                self._editing_label = False
                self._edit_buffer = ""
            elif key == 8:
                self._edit_buffer = self._edit_buffer[:-1]
            elif 32 <= key < 127:
                self._edit_buffer += chr(key)
            return True

        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("l"), ord("L")) and not self.recording:
            self._editing_label = True
            self._edit_buffer = ""
        if key == 32:
            self._toggle_recording()
        return True


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="SignSense — collect motion-based sign samples")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    return CollectMotionApp(camera_index=args.camera, data_path=args.data).run()


if __name__ == "__main__":
    raise SystemExit(main())
