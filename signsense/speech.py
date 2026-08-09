"""SignSense.speech — text-to-speech output, fail-soft and non-blocking.

Wraps pyttsx3. Two things matter here:

1. Fail-soft: if pyttsx3 isn't installed, or no TTS voice/engine is
   available on this machine (a genuinely common situation — TTS
   engines are OS-dependent and occasionally just missing), every
   method silently no-ops. Sign recognition still works; it's quiet.

2. Non-blocking: pyttsx3's `runAndWait()` blocks the calling thread
   until speech finishes. Calling it directly from the camera loop
   would freeze the video feed for the length of each spoken word —
   a jarring stutter in something that's supposed to feel live. A
   single dedicated worker thread owns the engine and speaks from a
   queue instead, so the camera loop is never blocked.

Known platform quirk (documented here rather than hidden): pyttsx3's
behavior around repeated `say()`/`runAndWait()` calls in one process
varies by OS and installed voice backend — it's solid on Windows,
generally fine on Linux (espeak), and has had intermittent
only-works-once reports on some macOS setups depending on the pyobjc
version. If speech stops working after the first phrase on your
machine, that's a known pyttsx3/macOS interaction, not a SignSense bug
— toggling mute (M) off and back on works around it in most reports.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional


class Speaker:
    def __init__(self, *, enabled: bool = True, rate: int = 175) -> None:
        self.enabled = enabled
        self._engine = None
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

        if not enabled:
            return
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
        except Exception as exc:
            print(f"[speech] disabled — no TTS engine available ({exc})")
            self._engine = None
            return

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:  # shutdown signal
                break
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                print(f"[speech] error speaking '{text}': {exc}")

    def say(self, text: str) -> None:
        if not self.enabled or self._engine is None or not text:
            return
        # Drop anything still queued but unspoken — otherwise a burst of
        # sign changes queues up and narrates several seconds late.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text)

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def close(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=1.0)
