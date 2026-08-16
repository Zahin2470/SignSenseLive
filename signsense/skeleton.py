"""SignSense.skeleton — "neural scan" hand visualization.

Deliberately a different visual language from VisionPuzzle Studio's
rainbow per-finger skeleton — that design fits a playful game; this
one is meant to read as an analysis tool actively *reading* your
hand, not a cartoon puppet:

  - bones and joints follow a single violet→cyan GRADIENT (theme
    ACCENT at the wrist fading to ACCENT_HOT at the fingertips)
    instead of five separate rainbow hues
  - a thin "web" connects the five fingertips to each other — not an
    anatomical connection, just a scan-mesh overlay
  - a dashed ring slowly rotates around the wrist, like a sensor
    actively tracking a target
  - everything can react to prediction CONFIDENCE (0..1): dim and
    slow when uncertain, bright and lively when the classifier is
    sure — ties the visual straight to what the model is doing,
    which a fixed rainbow skeleton can't express

Uses whatever the active theme's ACCENT/ACCENT_HOT happen to be, so
switching themes (`T`) re-colors the hand too, not just the panels.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from . import ui

# Kinematic finger chains for anatomical depth mapping
_FINGER_CHAINS = (
    (0, 1, 2, 3, 4),     # Thumb
    (0, 5, 6, 7, 8),     # Index
    (5, 9, 10, 11, 12),  # Middle
    (9, 13, 14, 15, 16), # Ring
    (13, 17, 18, 19, 20) # Pinky
)

_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
_TIPS_RING = (4, 8, 12, 16, 20)
_TIP_SET = set(_TIPS_RING)
_PALM_POLY = (0, 1, 5, 9, 13, 17)


def _lerp_color(c0: tuple[int, int, int], c1: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(c0[0] + (c1[0] - c0[0]) * t),
        int(c0[1] + (c1[1] - c0[1]) * t),
        int(c0[2] + (c1[2] - c0[2]) * t),
    )


def _draw_neon_bone(
    frame: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    confidence: float = 1.0,
) -> None:
    """Multi-stage volumetric neon bloom line with White+Black mixed core."""
    glow = frame.copy()

    # 1. Outer ambient diffuse bloom
    cv2.line(glow, p0, p1, color, thickness * 5, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.15 * confidence, frame, 1.0 - (0.15 * confidence), 0, dst=frame)

    # 2. Mid intense neon beam
    cv2.line(glow, p0, p1, color, thickness * 2, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.35 * confidence, frame, 1.0 - (0.35 * confidence), 0, dst=frame)

    # 3. Dual White + Black mixed core line
    cv2.line(frame, p0, p1, (255, 255, 255), thickness + 1, cv2.LINE_AA)
    cv2.line(frame, p0, p1, (10, 10, 10), max(1, thickness - 1), cv2.LINE_AA)


def _draw_cyber_node(
    frame: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    is_tip: bool = False,
    beat: float = 0.0,
    confidence: float = 1.0,
    angle: float = 0.0,
) -> None:
    """Cybernetic node with glowing core, target reticle ring, and Black+White mixed center."""
    cx, cy = center
    h, w = frame.shape[:2]

    if is_tip:
        # Dynamic reticle ring
        r_outer = radius + 7 + int(3 * beat)
        cv2.circle(frame, center, r_outer, color, 1, cv2.LINE_AA)

        # Rotating crosshair tick marks (Black border stroke + White core line)
        for d in (0, 90, 180, 270):
            rad = math.radians(angle + d)
            p_in = (int(cx + (r_outer - 2) * math.cos(rad)), int(cy + (r_outer - 2) * math.sin(rad)))
            p_out = (int(cx + (r_outer + 3) * math.cos(rad)), int(cy + (r_outer + 3) * math.sin(rad)))
            cv2.line(frame, p_in, p_out, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.line(frame, p_in, p_out, (255, 255, 255), 1, cv2.LINE_AA)

    # Outer ambient radial glow
    bloom_radius = radius + (8 if is_tip else 4)
    x0, y0 = max(0, cx - bloom_radius), max(0, cy - bloom_radius)
    x1, y1 = min(w, cx + bloom_radius), min(h, cy + bloom_radius)

    if x1 > x0 and y1 > y0:
        roi = frame[y0:y1, x0:x1]
        glow = roi.copy()
        cv2.circle(glow, (cx - x0, cy - y0), bloom_radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.25 * confidence, roi, 1.0 - (0.25 * confidence), 0, dst=roi)

    # Solid node body
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)

    # Black + White mixed joint point (White halo ring with ink-black center dot)
    core_r = max(2, radius // 2 + 1)
    cv2.circle(frame, center, core_r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, center, max(1, core_r - 1), (10, 10, 10), -1, cv2.LINE_AA)


def draw_skeleton(
    frame: np.ndarray,
    landmarks: np.ndarray,
    w: int,
    h: int,
    *,
    t: Optional[float] = None,
    pulse: bool = True,
    confidence: float = 1.0,
) -> np.ndarray:
    now = t if t is not None else time.perf_counter()
    confidence = max(0.0, min(1.0, confidence))
    base_color = ui.ACCENT
    tip_color = ui.ACCENT_HOT

    pts = [
        (int(np.clip(lm[0], 0.0, 1.0) * (w - 1)), int(np.clip(lm[1], 0.0, 1.0) * (h - 1)))
        for lm in landmarks
    ]

    # Topological depth mapping
    depth_map = {0: 0.0}
    for chain in _FINGER_CHAINS:
        for pos, idx in enumerate(chain[1:], start=1):
            depth_map[idx] = pos / 4.0

    # Temporal energy wave running down limbs
    wave = math.sin(now * 4.0) * 0.15

    # 1. Translucent Holographic Mesh (Palm & Tip Web Membrane)
    palm_pts = np.array([pts[i] for i in _PALM_POLY], dtype=np.int32)
    mesh_overlay = frame.copy()
    cv2.fillConvexPoly(mesh_overlay, palm_pts, _lerp_color(base_color, ui.BG, 0.75), cv2.LINE_AA)
    cv2.addWeighted(mesh_overlay, 0.14 * confidence, frame, 0.86, 0, dst=frame)

    tip_pts = np.array([pts[i] for i in _TIPS_RING], dtype=np.int32)
    tip_overlay = frame.copy()
    cv2.fillPoly(tip_overlay, [tip_pts], _lerp_color(tip_color, ui.BG, 0.82), cv2.LINE_AA)
    cv2.addWeighted(tip_overlay, 0.09 * confidence, frame, 0.91, 0, dst=frame)

    # 2. Tip Mesh Perimeter Lines (White + Black dual stroke)
    for i in range(len(_TIPS_RING)):
        p0 = pts[_TIPS_RING[i]]
        p1 = pts[_TIPS_RING[(i + 1) % len(_TIPS_RING)]]
        cv2.line(frame, p0, p1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(frame, p0, p1, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Volumetric Triple-Pass Neon Bones with Black+White Mixed Cores
    for a, b in _BONES:
        raw_t = (depth_map.get(a, 0.0) + depth_map.get(b, 0.0)) / 2.0
        grad_t = max(0.0, min(1.0, raw_t + wave))
        color = _lerp_color(base_color, tip_color, grad_t)
        if confidence < 1.0:
            color = _lerp_color(ui.TEXT_MUTED, color, 0.35 + 0.65 * confidence)
        _draw_neon_bone(frame, pts[a], pts[b], color, thickness=2, confidence=confidence)

    # 4. Cybernetic Reticle Joints with Black Center Points
    pulse_speed = 3.0 + 4.0 * confidence
    beat = 0.5 + 0.5 * math.sin(now * pulse_speed) if pulse else 0.0
    rot_angle = (now * 120.0) % 360.0

    for i, pt in enumerate(pts):
        raw_t = depth_map.get(i, i / 20.0)
        grad_t = max(0.0, min(1.0, raw_t + wave))
        color = _lerp_color(base_color, tip_color, grad_t)
        if confidence < 1.0:
            color = _lerp_color(ui.TEXT_MUTED, color, 0.35 + 0.65 * confidence)

        is_tip = i in _TIP_SET
        radius = (5 + int(3 * beat)) if is_tip else (6 if i == 0 else 3)
        _draw_cyber_node(
            frame, pt, radius, color, is_tip=is_tip, beat=beat, confidence=confidence, angle=rot_angle
        )

    # 5. Sci-Fi Wrist Radar HUD (White + Black mixed indicators)
    wrist = pts[0]
    spin_deg = (now * 100.0) % 360.0

    # Outer compass-ring with alternating black and white ticks
    cv2.circle(frame, wrist, 26, _lerp_color(base_color, tip_color, 0.5), 1, cv2.LINE_AA)
    for idx, d in enumerate(range(0, 360, 45)):
        rad = math.radians(spin_deg + d)
        p_in = (int(wrist[0] + 24 * math.cos(rad)), int(wrist[1] + 24 * math.sin(rad)))
        p_out = (int(wrist[0] + 28 * math.cos(rad)), int(wrist[1] + 28 * math.sin(rad)))
        tick_col = (0, 0, 0) if idx % 2 == 0 else (255, 255, 255)
        cv2.line(frame, p_in, p_out, tick_col, 1, cv2.LINE_AA)

    # Inner counter-rotating dashed sensor arc in crisp contrast line
    step = 360.0 / 8
    for i in range(8):
        a0 = math.radians(-spin_deg * 1.5 + i * step)
        a1 = math.radians(-spin_deg * 1.5 + i * step + step * 0.4)
        p0 = (int(wrist[0] + 18 * math.cos(a0)), int(wrist[1] + 18 * math.sin(a0)))
        p1 = (int(wrist[0] + 18 * math.cos(a1)), int(wrist[1] + 18 * math.sin(a1)))
        arc_col = (0, 0, 0) if i % 2 == 0 else (255, 255, 255)
        cv2.line(frame, p0, p1, arc_col, 1, cv2.LINE_AA)

    return frame