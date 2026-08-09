#!/usr/bin/env python3
"""SignSense — webcam sign-language recognizer with a trainable classifier.

Modes:
    python main.py collect          # record labeled hand-sign samples (static)
    python main.py collect_motion   # record motion-sign clips (e.g. swipes, J/Z-style letters)
    python main.py train            # train the classifier from collected samples
    python main.py live             # run the live recognizer (static)
    python main.py live_motion      # run the live recognizer (motion signs)
    python main.py practice         # quiz mode: match the prompted sign

Add --two-hand to collect/train/live/practice for two-handed signs
(separate dataset and model file from the single-hand default).
Add --motion to train for motion-sign models (collect_motion/live_motion
are already motion-only, so they don't need the flag).

Each also runs standalone: `python -m signsense.collect`, etc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="SignSense")
    parser.add_argument(
        "mode",
        choices=["collect", "collect_motion", "train", "live", "live_motion", "practice"],
        help="What to run",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (camera-based modes only)")
    parser.add_argument("--two-hand", action="store_true", help="Use two-handed signs (collect/train/live/practice)")
    parser.add_argument("--motion", action="store_true", help="Train a motion-sign model (train only)")
    args, remaining = parser.parse_known_args()

    if args.mode == "collect":
        from signsense.collect import CollectApp
        return CollectApp(camera_index=args.camera, two_hand=args.two_hand).run()
    if args.mode == "collect_motion":
        from signsense.collect_motion import CollectMotionApp
        return CollectMotionApp(camera_index=args.camera).run()
    if args.mode == "live":
        from signsense.live import LiveApp
        return LiveApp(camera_index=args.camera, two_hand=args.two_hand).run()
    if args.mode == "live_motion":
        from signsense.live_motion import LiveMotionApp
        return LiveMotionApp(camera_index=args.camera).run()
    if args.mode == "practice":
        from signsense.practice import PracticeApp
        return PracticeApp(camera_index=args.camera, two_hand=args.two_hand).run()
    if args.mode == "train":
        from signsense.train import main as train_main
        # --two-hand / --motion were consumed by the parser above (shared
        # top-level flags) — put back whichever was set for train's own parser.
        forwarded = list(remaining)
        if args.two_hand:
            forwarded.append("--two-hand")
        if args.motion:
            forwarded.append("--motion")
        sys.argv = [sys.argv[0]] + forwarded
        return train_main()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
