"""SignSense.collect — record labeled hand-landmark samples via webcam.

Workflow: type a label once, then hold that sign in front of the
camera and press SPACE to start capturing — it keeps recording samples
every couple of frames (so you naturally get slight variation in angle
and distance, which is what makes the classifier robust instead of
memorizing one exact pose) until you press SPACE again to stop.
Switch labels anytime by typing a new one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import ui
from .audio import AudioManager
from .dataset import append_sample, class_counts, load_dataset
from .features import DUAL_FEATURE_DIM, FEATURE_DIM, dual_hand_features, landmarks_to_features
from .skeleton import draw_skeleton
from .tracker import HandTracker, hands_by_side

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
DATA_PATH = ROOT / "data" / "samples.csv"
DATA_PATH_2H = ROOT / "data" / "samples_2h.csv"


class CollectApp:
    def __init__(self, camera_index: int = 0, *, two_hand: bool = False, data_path: Optional[Path] = None) -> None:
        self.two_hand = two_hand
        self.tracker = HandTracker(MODEL_PATH, max_hands=2 if two_hand else 1)
        self.camera_index = camera_index
        self.data_path = data_path or (DATA_PATH_2H if two_hand else DATA_PATH)
        self.feature_dim = DUAL_FEATURE_DIM if two_hand else FEATURE_DIM
        self.label = "sign_1"
        self._editing_label = False
        self._edit_buffer = ""
        self.capturing = False
        self._last_hands: list = []
        self._frame_counter = 0
        self.audio = AudioManager()

    def run(self) -> int:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("ERROR: could not open webcam.")
            return 1
        win = "SignSense — Collect" + (" (2-hand)" if self.two_hand else "")
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print("SignSense collector — L to name a label, SPACE to toggle capture, T theme, M mute, Q to quit"
              + (" [two-hand mode]" if self.two_hand else ""))
        self.audio.play_music("ambient")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                self._frame_counter += 1

                hands = self.tracker.process(frame, mirrored=True)
                self._last_hands = hands

                if self.capturing and hands and self._frame_counter % 2 == 0:
                    feat = self._extract_features(hands)
                    append_sample(self.data_path, self.label, feat, feature_dim=self.feature_dim)

                frame = self._draw(frame, hands, w, h)
                frame = ui.vignette(frame, strength=0.2)
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break
        finally:
            self.tracker.close()
            self.audio.close()
            cap.release()
            cv2.destroyAllWindows()
        return 0

    def _extract_features(self, hands: list) -> np.ndarray:
        if self.two_hand:
            return dual_hand_features(hands_by_side(hands))
        return landmarks_to_features(hands[0].landmarks)

    def _draw(self, frame: np.ndarray, hands, w: int, h: int) -> np.ndarray:
        for hr in hands:
            draw_skeleton(frame, hr.landmarks, w, h)

        frame = ui.glass_panel(frame, (14, 14), (min(w - 14, 420), 92), radius=16)
        ui.glow_dot(frame, (34, 37), 10, ui.ACCENT if not self.capturing else ui.SUCCESS, intensity=0.32)
        cv2.circle(frame, (34, 37), 5, ui.ACCENT if not self.capturing else ui.SUCCESS, -1, cv2.LINE_AA)
        label_text = self._edit_buffer + "_" if self._editing_label else self.label
        ui.put_text(frame, label_text, (50, 42), scale=0.6, color=ui.ACCENT, weight=2)
        if self.two_hand:
            ui.chip(frame, "2-HAND", min(w - 14, 420) - 92, 20, color=ui.ACCENT_HOT)

        if self.two_hand:
            sides = hands_by_side(hands)
            present = [s for s in ("Left", "Right") if sides[s] is not None]
            hand_state = f"visible: {', '.join(present)}" if present else "show your hand(s)"
        else:
            hand_state = "hand visible" if hands else "show your hand"
        ui.put_text(frame, hand_state, (24, 72), scale=0.42, color=ui.TEXT_MUTED, shadow=False)

        X, labels = load_dataset(self.data_path, feature_dim=self.feature_dim)
        counts = class_counts(labels)
        y = 118
        ui.put_text(frame, f"DATASET  ({len(labels)} samples total)", (14, y), scale=0.44, color=ui.TEXT_MUTED, shadow=False)
        for i, (lbl, n) in enumerate(sorted(counts.items())):
            color = ui.SUCCESS if lbl == self.label else ui.TEXT
            ui.put_text(frame, f"{lbl}: {n}", (14, y + 26 + i * 22), scale=0.42, color=color)

        hint_y = h - 16
        if self._editing_label:
            hint = "Type label, ENTER to confirm, Esc to cancel"
        else:
            hint = "L rename · SPACE capture · T theme · M mute · Q quit"
        ui.put_text(frame, hint, (14, hint_y), scale=0.42, color=ui.TEXT_MUTED, shadow=False)
        return frame

    def _handle_key(self, key: int) -> bool:
        if self._editing_label:
            if key in (13, 10):  # Enter
                if self._edit_buffer:
                    self.label = self._edit_buffer
                self._editing_label = False
                self._edit_buffer = ""
            elif key == 27:  # Esc
                self._editing_label = False
                self._edit_buffer = ""
            elif key == 8:  # Backspace
                self._edit_buffer = self._edit_buffer[:-1]
            elif 32 <= key < 127:
                self._edit_buffer += chr(key)
            return True

        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("l"), ord("L")):
            self._editing_label = True
            self._edit_buffer = ""
        if key == 32:  # SPACE — toggle capture on/off (not "hold": cv2's
            # key polling doesn't reliably report a key as continuously
            # held across platforms, so a press-to-toggle "recording"
            # state is much more dependable than press-and-hold).
            self.capturing = not self.capturing
            if self.capturing:
                self.audio.play_sfx("capture")
        if key in (ord("t"), ord("T")):
            ui.set_theme(ui.next_theme_name())
        if key in (ord("m"), ord("M")):
            self.audio.toggle_mute()
        return True


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="SignSense — collect labeled sign samples")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--two-hand", action="store_true", help="Collect two-handed signs instead of single-hand")
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    return CollectApp(camera_index=args.camera, two_hand=args.two_hand, data_path=args.data).run()


if __name__ == "__main__":
    raise SystemExit(main())
