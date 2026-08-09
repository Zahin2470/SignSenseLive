#!/usr/bin/env python3
"""SignSense — webcam sign-language recognizer with a trainable classifier.

Three modes, one entry point:
    python main.py collect     # record labeled hand-sign samples
    python main.py train       # train the classifier from collected samples
    python main.py live        # run the live recognizer
    python main.py practice    # quiz mode: match the prompted sign

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
    parser.add_argument("mode", choices=["collect", "train", "live", "practice"], help="What to run")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (collect/live/practice only)")
    args, remaining = parser.parse_known_args()

    if args.mode == "collect":
        from signsense.collect import CollectApp
        return CollectApp(camera_index=args.camera).run()
    if args.mode == "live":
        from signsense.live import LiveApp
        return LiveApp(camera_index=args.camera).run()
    if args.mode == "practice":
        from signsense.practice import PracticeApp
        return PracticeApp(camera_index=args.camera).run()
    if args.mode == "train":
        from signsense.train import main as train_main
        sys.argv = [sys.argv[0]] + remaining
        return train_main()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
