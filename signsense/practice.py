"""SignSense.practice — quiz/match mode.

The app names a sign, you make it, it tracks how fast and how often
you get it right. A "wrong" is only counted once per round (the first
time you stabilize on some other sign) so fumbling toward the right
shape doesn't rack up a wall of misses — the point is practice, not
punishment.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import ui
from .audio import AudioManager
from .features import DUAL_FEATURE_DIM, FEATURE_DIM, dual_hand_features, landmarks_to_features
from .model import SignClassifier
from .skeleton import draw_skeleton
from .speech import Speaker
from .tracker import HandTracker, hands_by_side
from .voting import StablePredictor

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = ROOT / "models" / "classifier.pkl"
CLASSIFIER_PATH_2H = ROOT / "models" / "classifier_2h.pkl"

FLASH_SECONDS = 0.9  # brief pause on a correct hit before the next prompt engages


class PracticeApp:
    def __init__(self, camera_index: int = 0, *, speech: bool = True, two_hand: bool = False,
                 classifier_path: Optional[Path] = None) -> None:
        self.two_hand = two_hand
        self.tracker = HandTracker(MODEL_PATH, max_hands=2 if two_hand else 1)
        path = classifier_path or (CLASSIFIER_PATH_2H if two_hand else CLASSIFIER_PATH)
        self.classifier_path = path
        self.classifier = SignClassifier.try_load(path)
        self.camera_index = camera_index
        self.predictor = StablePredictor()
        self.speaker = Speaker(enabled=speech)
        self.audio = AudioManager()

        self.target: Optional[str] = None
        self.correct_count = 0
        self.wrong_count = 0
        self.streak = 0
        self.best_streak = 0
        self.reaction_times: list[float] = []
        self._last_wrong: Optional[str] = None
        self._round_started_at = 0.0
        self._flash_until = 0.0
        self._session_start = 0.0
        self._conf_disp = 0.0

    def run(self) -> int:
        if self.classifier is None:
            print(f"No trained model found at {self.classifier_path}.")
            hand_flag = " --two-hand" if self.two_hand else ""
            print(f"Run  python -m signsense.collect{hand_flag}  then  python -m signsense.train{hand_flag}  first.")
            return 1
        if len(self.classifier.classes_) < 2:
            print("Need at least 2 trained signs to play Practice mode — collect and train more first.")
            return 1

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("ERROR: could not open webcam.")
            return 1
        win = "SignSense — Practice" + (" (2-hand)" if self.two_hand else "")
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print(f"SignSense practice — {len(self.classifier.classes_)} signs  ·  N skip · T theme · M mute · Q quit")
        self.audio.play_music("ambient")

        self._session_start = time.perf_counter()
        self._next_round()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                hands = self.tracker.process(frame, mirrored=True)
                self._update(hands)
                frame = self._draw(frame, hands, w, h)
                frame = ui.vignette(frame, strength=0.2)

                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("m"), ord("M")):
                    self.speaker.toggle()
                    self.audio.toggle_mute()
                if key in (ord("t"), ord("T")):
                    ui.set_theme(ui.next_theme_name())
                if key in (ord("n"), ord("N")):
                    self._next_round()
        finally:
            self.tracker.close()
            self.speaker.close()
            self.audio.close()
            cap.release()
            cv2.destroyAllWindows()
            self._print_summary()
        return 0

    def _extract_features(self, hands: list) -> np.ndarray:
        if self.two_hand:
            return dual_hand_features(hands_by_side(hands))
        return landmarks_to_features(hands[0].landmarks)

    def _next_round(self) -> None:
        choices = [c for c in self.classifier.classes_ if c != self.target] or list(self.classifier.classes_)
        self.target = random.choice(choices)
        self.predictor.reset()
        self._last_wrong = None
        self._round_started_at = time.perf_counter()
        self.speaker.say(f"Show me {self.target}")

    def _update(self, hands) -> None:
        now = time.perf_counter()
        if now < self._flash_until:
            return
        if not hands:
            self.predictor.update(None)
            return
        feat = self._extract_features(hands)
        label, conf = self.classifier.predict(feat)
        stable = self.predictor.update(label, conf)
        if stable is None:
            return
        if stable == self.target:
            self._register_hit(now)
        elif stable != self._last_wrong:
            self._last_wrong = stable
            self.wrong_count += 1
            self.streak = 0
            self.speaker.say("try again")
            self.audio.play_sfx("wrong")

    def _register_hit(self, now: float) -> None:
        self.reaction_times.append(now - self._round_started_at)
        self.correct_count += 1
        self.streak += 1
        self.best_streak = max(self.best_streak, self.streak)
        self.speaker.say("Correct")
        self.audio.play_sfx("correct")
        self._flash_until = now + FLASH_SECONDS
        self._next_round()

    def _draw(self, frame: np.ndarray, hands, w: int, h: int) -> np.ndarray:
        for hr in hands:
            draw_skeleton(frame, hr.landmarks, w, h)

        panel_w = min(w - 28, 420)
        frame = ui.glass_panel(frame, (14, 14), (14 + panel_w, 150), radius=18)
        ui.glow_dot(frame, (34, 37), 10, ui.ACCENT, intensity=0.3)
        cv2.circle(frame, (34, 37), 5, ui.ACCENT, -1, cv2.LINE_AA)
        ui.put_text(frame, "PRACTICE", (50, 42), scale=0.5, color=ui.ACCENT, weight=2)
        badge_x = 14 + panel_w - 78
        if self.two_hand:
            ui.chip(frame, "2-HAND", badge_x - 84, 20, color=ui.ACCENT_HOT)
        if not self.speaker.enabled:
            ui.chip(frame, "MUTED", badge_x, 20, color=ui.TEXT_MUTED)

        flashing = time.perf_counter() < self._flash_until
        prompt_color = ui.SUCCESS if flashing else ui.TEXT
        prompt_text = "Correct!" if flashing else f"Show me: {self.target}"
        ui.put_text(frame, prompt_text, (24, 82), scale=0.75, color=prompt_color, weight=3)

        stats = f"Streak {self.streak}  (best {self.best_streak})   \u00b7   {self.correct_count} correct"
        ui.put_text(frame, stats, (24, 112), scale=0.44, color=ui.TEXT_MUTED, shadow=False)

        if self.reaction_times:
            avg = sum(self.reaction_times[-5:]) / len(self.reaction_times[-5:])
            ui.put_text(frame, f"avg reaction {avg:.1f}s", (24, 136), scale=0.4, color=ui.TEXT_MUTED, shadow=False)

        label = self.predictor.stable_label
        conf = self.predictor.stable_confidence if label else 0.0
        self._conf_disp = ui.smooth_toward(self._conf_disp, conf, 0.25)
        ui.progress_bar(frame, 24, 144, panel_w - 48, 4, self._conf_disp)

        ui.put_text(frame, "N skip \u00b7 T theme \u00b7 M mute \u00b7 Q quit", (14, h - 16), scale=0.4, color=ui.TEXT_MUTED, shadow=False)
        return frame

    def _print_summary(self) -> None:
        elapsed = time.perf_counter() - self._session_start
        total = self.correct_count + self.wrong_count
        accuracy = (self.correct_count / total * 100) if total else 0.0
        avg_reaction = sum(self.reaction_times) / len(self.reaction_times) if self.reaction_times else 0.0
        print("\n=== Practice session ===")
        print(f"Correct: {self.correct_count}   Wrong: {self.wrong_count}   Accuracy: {accuracy:.0f}%")
        print(f"Best streak: {self.best_streak}   Avg reaction: {avg_reaction:.2f}s   Session: {elapsed:.0f}s")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="SignSense — Practice/Match mode")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--two-hand", action="store_true", help="Practice two-handed signs")
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()
    return PracticeApp(
        camera_index=args.camera, speech=not args.no_speech,
        two_hand=args.two_hand, classifier_path=args.model,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
