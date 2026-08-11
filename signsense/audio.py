"""SignSense.audio — ambient background music + one-shot SFX.

Same fail-soft philosophy as VisionPuzzle Studio's audio module: built
on `pygame.mixer`, and if pygame isn't installed, no audio device is
available, or a sound file is simply missing, every method silently
no-ops instead of crashing the app. None of the recognition/training
logic depends on sound working.

Expected asset layout (relative to assets/audio/):

    music/ambient.ogg     looping background bed for any camera app
    sfx/stable.wav        a sign just became a stable, confident reading
    sfx/correct.wav       Practice mode: correct match
    sfx/wrong.wav         Practice mode: wrong sign
    sfx/record_start.wav  motion collect/live: recording started
    sfx/record_stop.wav   motion collect/live: recording stopped/saved
    sfx/capture.wav       collect mode: capture toggled on

Any file you haven't added yet is simply skipped.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        print(f"[audio] {msg}")
        _WARNED.add(msg)


class AudioManager:
    SFX_FILES = {
        "stable": "stable.wav",
        "correct": "correct.wav",
        "wrong": "wrong.wav",
        "record_start": "record_start.wav",
        "record_stop": "record_stop.wav",
        "capture": "capture.wav",
    }
    MUSIC_FILES = {
        "ambient": "ambient.ogg",
    }

    def __init__(self, assets_dir: Optional[Path] = None, *, enabled: bool = True) -> None:
        if assets_dir is None:
            assets_dir = Path(__file__).resolve().parent.parent / "assets" / "audio"
        self.assets_dir = Path(assets_dir)
        self.music_volume = 0.30
        self.sfx_volume = 0.85
        self.muted = False

        self._ok = False
        self._pygame = None
        self._sfx: dict[str, object] = {}
        self._current_track: Optional[str] = None
        self._last_played: dict[str, float] = {}

        if enabled:
            self._init_mixer()
            if self._ok:
                self._load_sfx()

    def _init_mixer(self) -> None:
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._pygame = pygame
            self._ok = True
        except Exception as exc:  # pragma: no cover - environment dependent
            _warn_once(f"disabled — no audio backend available ({exc})")
            self._ok = False

    def _load_sfx(self) -> None:
        sfx_dir = self.assets_dir / "sfx"
        for name, fname in self.SFX_FILES.items():
            path = sfx_dir / fname
            if not path.is_file():
                continue
            try:
                snd = self._pygame.mixer.Sound(str(path))
                snd.set_volume(self.sfx_volume)
                self._sfx[name] = snd
            except Exception as exc:
                _warn_once(f"could not load sfx '{name}': {exc}")

    def play_music(self, track: str = "ambient", *, loop: bool = True, fade_ms: int = 700) -> None:
        if not self._ok or track == self._current_track:
            return
        fname = self.MUSIC_FILES.get(track)
        if not fname:
            _warn_once(f"unknown music track '{track}'")
            return
        path = self.assets_dir / "music" / fname
        if not path.is_file():
            self._current_track = track
            return
        try:
            if self._current_track is not None:
                self._pygame.mixer.music.fadeout(fade_ms)
            self._pygame.mixer.music.load(str(path))
            self._pygame.mixer.music.set_volume(0.0 if self.muted else self.music_volume)
            self._pygame.mixer.music.play(loops=-1 if loop else 0, fade_ms=fade_ms)
            self._current_track = track
        except Exception as exc:
            _warn_once(f"could not play music '{track}': {exc}")

    def stop_music(self, *, fade_ms: int = 500) -> None:
        if not self._ok:
            return
        try:
            self._pygame.mixer.music.fadeout(fade_ms)
        except Exception:
            pass
        self._current_track = None

    def play_sfx(self, name: str, *, cooldown: float = 0.06) -> None:
        if not self._ok or self.muted:
            return
        snd = self._sfx.get(name)
        if snd is None:
            return
        now = time.perf_counter()
        if now - self._last_played.get(name, 0.0) < cooldown:
            return
        self._last_played[name] = now
        try:
            snd.play()
        except Exception:
            pass

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        if self._ok:
            try:
                self._pygame.mixer.music.set_volume(0.0 if self.muted else self.music_volume)
            except Exception:
                pass
        return self.muted

    def close(self) -> None:
        if not self._ok:
            return
        try:
            self._pygame.mixer.music.stop()
            self._pygame.mixer.quit()
        except Exception:
            pass
