"""SignSense.ui — shared visual design: palette, glass panels, custom
typography. A trimmed-down sibling of the design system built for
VisionPuzzle Studio (same techniques: PIL-rendered Poppins with glyph
caching for speed, cheap layered drop shadows, concentric-circle glow),
now including the same switchable-theme approach.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False

# ── Palette (BGR) ───────────────────────────────────────────────────────────
# Plain module globals (not a class/dict lookup) so every file keeps
# writing `ui.ACCENT`, `ui.BG`, etc. exactly as before — set_theme()
# reassigns them in place, and since Python resolves `ui.X` by
# attribute lookup at call time (not import time), every call site
# picks up a new theme automatically. (Default *parameter values* like
# `def f(color=ACCENT)` would NOT update this way — this module avoids
# that pattern throughout.)

THEMES: dict[str, dict[str, tuple[int, int, int]]] = {
    "dark": {
        "BG": (18, 16, 20), "BG_ELEVATED": (30, 26, 34),
        "STROKE": (90, 80, 100), "TEXT": (240, 238, 242), "TEXT_MUTED": (170, 164, 176),
        "ACCENT": (210, 130, 255), "ACCENT_HOT": (140, 220, 255),
        "SUCCESS": (120, 210, 150), "DANGER": (90, 90, 220),
    },
    "light": {
        "BG": (238, 240, 242), "BG_ELEVATED": (220, 222, 226),
        "STROKE": (180, 175, 190), "TEXT": (30, 28, 34), "TEXT_MUTED": (105, 100, 115),
        "ACCENT": (190, 70, 160), "ACCENT_HOT": (190, 130, 30),
        "SUCCESS": (80, 155, 80), "DANGER": (55, 55, 195),
    },
    "neon": {
        "BG": (12, 6, 18), "BG_ELEVATED": (30, 12, 42),
        "STROKE": (160, 50, 190), "TEXT": (245, 245, 255), "TEXT_MUTED": (175, 145, 205),
        "ACCENT": (255, 60, 220), "ACCENT_HOT": (60, 255, 230),
        "SUCCESS": (140, 255, 60), "DANGER": (60, 40, 255),
    },
    "mono": {
        "BG": (15, 15, 15), "BG_ELEVATED": (36, 36, 36),
        "STROKE": (120, 120, 120), "TEXT": (245, 245, 245), "TEXT_MUTED": (170, 170, 170),
        "ACCENT": (255, 255, 255), "ACCENT_HOT": (205, 205, 205),
        "SUCCESS": (235, 235, 235), "DANGER": (150, 150, 150),
    },
}

_CURRENT_THEME = "dark"
BG = THEMES["dark"]["BG"]
BG_ELEVATED = THEMES["dark"]["BG_ELEVATED"]
STROKE = THEMES["dark"]["STROKE"]
TEXT = THEMES["dark"]["TEXT"]
TEXT_MUTED = THEMES["dark"]["TEXT_MUTED"]
ACCENT = THEMES["dark"]["ACCENT"]
ACCENT_HOT = THEMES["dark"]["ACCENT_HOT"]
SUCCESS = THEMES["dark"]["SUCCESS"]
DANGER = THEMES["dark"]["DANGER"]


def set_theme(name: str) -> bool:
    global _CURRENT_THEME, BG, BG_ELEVATED, STROKE, TEXT, TEXT_MUTED, ACCENT, ACCENT_HOT, SUCCESS, DANGER
    theme = THEMES.get(name)
    if theme is None:
        return False
    BG = theme["BG"]
    BG_ELEVATED = theme["BG_ELEVATED"]
    STROKE = theme["STROKE"]
    TEXT = theme["TEXT"]
    TEXT_MUTED = theme["TEXT_MUTED"]
    ACCENT = theme["ACCENT"]
    ACCENT_HOT = theme["ACCENT_HOT"]
    SUCCESS = theme["SUCCESS"]
    DANGER = theme["DANGER"]
    _CURRENT_THEME = name
    return True


def get_theme_name() -> str:
    return _CURRENT_THEME


def theme_names() -> list[str]:
    return list(THEMES.keys())


def next_theme_name() -> str:
    names = theme_names()
    idx = names.index(_CURRENT_THEME) if _CURRENT_THEME in names else -1
    return names[(idx + 1) % len(names)]

# ── Typography ──────────────────────────────────────────────────────────────
_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_FONT_FILES = {1: "Poppins-Medium.ttf", 2: "Poppins-SemiBold.ttf", 3: "Poppins-ExtraBold.ttf"}
_FONT_CACHE: dict[tuple[int, int], "ImageFont.FreeTypeFont"] = {}
_GLYPH_CACHE: dict[tuple, np.ndarray] = {}
_SCALE_TO_PX = 31


def _get_font(weight: int, px_size: int):
    key = (weight, px_size)
    font = _FONT_CACHE.get(key)
    if font is not None:
        return font
    fname = _FONT_FILES.get(weight, _FONT_FILES[1])
    try:
        font = ImageFont.truetype(str(_FONT_DIR / fname), px_size)
    except Exception:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def text_size(text: str, *, scale: float = 0.55, weight: int = 1) -> tuple[int, int]:
    if not _PIL_OK or not text:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
        return tw, th
    px_size = max(10, int(round(scale * _SCALE_TO_PX)))
    font = _get_font(weight, px_size)
    bbox = font.getbbox(text)
    ascent, descent = font.getmetrics()
    return bbox[2], ascent + descent


def _render_glyph(text: str, font, color: tuple[int, int, int], shadow: bool) -> np.ndarray:
    key = (text, id(font), color, shadow)
    cached = _GLYPH_CACHE.get(key)
    if cached is not None:
        return cached
    ascent, descent = font.getmetrics()
    bbox = font.getbbox(text)
    canvas_w = max(1, bbox[2]) + 4
    canvas_h = ascent + descent + 4
    patch = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    rgb = (color[2], color[1], color[0])
    if shadow:
        draw.text((3, 3), text, font=font, fill=(0, 0, 0, 255))
    draw.text((2, 2), text, font=font, fill=(rgb[0], rgb[1], rgb[2], 255))
    arr = np.array(patch)
    if len(_GLYPH_CACHE) > 400:
        _GLYPH_CACHE.clear()
    _GLYPH_CACHE[key] = arr
    return arr


def _blend_rgba(img: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    h, w = img.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + pw), min(h, y + ph)
    if x1 <= x0 or y1 <= y0:
        return
    px0, py0 = x0 - x, y0 - y
    px1, py1 = px0 + (x1 - x0), py0 + (y1 - y0)
    region = img[y0:y1, x0:x1].astype(np.float32)
    src = patch[py0:py1, px0:px1]
    alpha = src[..., 3:4].astype(np.float32) / 255.0
    rgb_bgr = src[..., (2, 1, 0)].astype(np.float32)
    img[y0:y1, x0:x1] = (region * (1.0 - alpha) + rgb_bgr * alpha).astype(np.uint8)


def put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.55,
    color: Optional[tuple[int, int, int]] = None,
    weight: int = 1,
    shadow: bool = True,
) -> None:
    if color is None:
        color = TEXT
    if not _PIL_OK or not text:
        if shadow:
            cv2.putText(img, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), weight + 1, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)
        return
    px_size = max(10, int(round(scale * _SCALE_TO_PX)))
    font = _get_font(weight, px_size)
    ascent, _descent = font.getmetrics()
    patch = _render_glyph(text, font, color, shadow)
    x, y = org
    _blend_rgba(img, patch, x - 2, y - ascent - 2)


# ── Shapes / panels ─────────────────────────────────────────────────────────

def rounded_rect(img, pt1, pt2, color, *, radius: int = 12, thickness: int = -1) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 <= x1 or y2 <= y1:
        return
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def _lighten(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(c + (255 - c) * amount) for c in color)  # type: ignore[return-value]


def _vertical_gradient(h: int, w: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(1, h), dtype=np.float32).reshape(-1, 1, 1)
    top_a = np.array(top, dtype=np.float32).reshape(1, 1, 3)
    bot_a = np.array(bottom, dtype=np.float32).reshape(1, 1, 3)
    grad = top_a * (1.0 - t) + bot_a * t
    return np.broadcast_to(grad, (max(1, h), max(1, w), 3)).astype(np.uint8)


def drop_shadow(frame, pt1, pt2, *, radius: int = 14, offset: int = 5, alpha: float = 0.30) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = frame.shape[:2]
    for spread, a in ((0, alpha), (2, alpha * 0.55), (4, alpha * 0.3)):
        sx1, sy1 = max(0, x1 - spread + offset), max(0, y1 - spread + offset)
        sx2, sy2 = min(w, x2 + spread + offset), min(h, y2 + spread + offset)
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        roi = frame[sy1:sy2, sx1:sx2]
        dark = np.zeros_like(roi)
        cv2.addWeighted(dark, a, roi, 1.0 - a, 0, dst=roi)


def glow_dot(frame, center, radius: int, color, *, intensity: float = 0.35, layers: int = 4) -> None:
    cx, cy = center
    h, w = frame.shape[:2]
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(w, cx + radius), min(h, cy + radius)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    glow = roi.copy()
    for i in range(layers, 0, -1):
        r = int(radius * i / layers)
        a = intensity * (1.0 - i / (layers + 1))
        cv2.circle(glow, (cx - x0, cy - y0), r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(glow, a, roi, 1.0 - a, 0, dst=roi)
        glow[:] = roi


def glass_panel(frame, pt1, pt2, *, alpha: float = 0.6, radius: int = 14, border: bool = True,
                 gradient: bool = True, shadow: bool = True) -> np.ndarray:
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    if shadow:
        drop_shadow(frame, (x1, y1), (x2, y2), radius=radius)
    roi = frame[y1:y2, x1:x2]
    if gradient:
        dark = _vertical_gradient(y2 - y1, x2 - x1, _lighten(BG_ELEVATED, 0.10), BG_ELEVATED)
    else:
        dark = np.empty_like(roi)
        dark[:] = BG_ELEVATED
    cv2.addWeighted(dark, alpha, roi, 1.0 - alpha, 0, dst=roi)
    if border:
        rounded_rect(frame, (x1, y1), (x2, y2), STROKE, radius=radius, thickness=1)
    return frame


def chip(frame, text: str, x: int, y: int, *, color: Optional[tuple[int, int, int]] = None, filled: bool = False) -> None:
    if color is None:
        color = ACCENT
    pad_x, pad_y = 14, 8
    tw, th = text_size(text, scale=0.46, weight=1)
    w, h = tw + pad_x * 2, th + pad_y * 2
    if filled:
        rounded_rect(frame, (x, y), (x + w, y + h), color, radius=h // 2)
        put_text(frame, text, (x + pad_x, y + h - pad_y - 2), scale=0.46, color=BG, weight=1, shadow=False)
    else:
        rounded_rect(frame, (x, y), (x + w, y + h), BG_ELEVATED, radius=h // 2)
        rounded_rect(frame, (x, y), (x + w, y + h), color, radius=h // 2, thickness=1)
        put_text(frame, text, (x + pad_x, y + h - pad_y - 2), scale=0.46, color=color, weight=1)


def progress_bar(frame, x: int, y: int, width: int, height: int, value: float) -> None:
    v = max(0.0, min(1.0, value))
    cv2.rectangle(frame, (x, y), (x + width, y + height), STROKE, -1)
    fill = int(width * v)
    if fill > 2:
        color = SUCCESS if v > 0.6 else (ACCENT_HOT if v > 0.3 else DANGER)
        cv2.rectangle(frame, (x, y), (x + fill, y + height), color, -1)


def draw_trail(
    frame: np.ndarray,
    points: list,
    color: tuple[int, int, int],
    *,
    dim_color: Optional[tuple[int, int, int]] = None,
    max_alpha: float = 0.85,
    base_thickness: int = 2,
    head_radius: int = 9,
) -> None:
    """Fading motion trail through a list of (x, y) pixel points, oldest
    first. One overlay + one blend regardless of trail length (cheap
    even with 30+ points), tail dimmer/thinner, head brighter/thicker,
    with a glow at the most recent point — used to show the path a
    hand traveled while recording a motion sign.
    """
    n = len(points)
    if n == 0:
        return
    if dim_color is None:
        dim_color = tuple(int(c * 0.32) for c in color)
    if n == 1:
        glow_dot(frame, points[0], head_radius, color, intensity=0.3)
        return

    overlay = frame.copy()
    for i in range(1, n):
        t = i / (n - 1)
        thickness = max(1, int(round(base_thickness + t * 3)))
        seg_color = tuple(int(dim_color[k] + (color[k] - dim_color[k]) * t) for k in range(3))
        cv2.line(overlay, points[i - 1], points[i], seg_color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, max_alpha, frame, 1.0 - max_alpha, 0, dst=frame)
    glow_dot(frame, points[-1], head_radius, color, intensity=0.35)


_VIGNETTE_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def vignette(frame: np.ndarray, strength: float = 0.22) -> np.ndarray:
    """Cached radial edge-darkening — cheap after the first frame at a
    given resolution/strength, since the multiplicative mask is reused
    rather than recomputed. Applied once per frame, after everything
    else is drawn, for a subtle cinematic camera feel."""
    h, w = frame.shape[:2]
    key = (h, w, int(strength * 100))
    mask = _VIGNETTE_CACHE.get(key)
    if mask is None:
        ys = np.linspace(-1, 1, h, dtype=np.float32)
        xs = np.linspace(-1, 1, w, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        r = np.sqrt(xv * xv + yv * yv)
        mask = (1.0 - np.clip((r - 0.55) / 0.75, 0.0, 1.0) * strength).astype(np.float32)
        _VIGNETTE_CACHE.clear()
        _VIGNETTE_CACHE[key] = mask
    out = frame.astype(np.float32)
    out *= mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def smooth_toward(current: float, target: float, alpha: float = 0.35) -> float:
    return current * (1.0 - alpha) + target * alpha
