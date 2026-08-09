"""SignSense.confusion — render a confusion matrix as a heatmap image.

Accuracy alone hides *which* signs get mixed up with which — two signs
that look similar in landmark-space will confuse a classifier no
matter how much data you throw at it, and the fix is usually "collect
more contrastive examples of these two" rather than "collect more of
everything." This makes that pattern visible instead of buried in a
percentage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import ui


def render_confusion_heatmap(cm: np.ndarray, labels: list[str], *, cell_size: int = 60) -> np.ndarray:
    """cm: (n, n) array where cm[i, j] = count of true class i predicted
    as class j (sklearn's confusion_matrix convention). Diagonal cells
    (correct predictions) tint toward SUCCESS; off-diagonal (confusions)
    tint toward DANGER, both scaled by how large that count is relative
    to the matrix's busiest cell.
    """
    n = len(labels)
    short = [lbl[:6] for lbl in labels]
    label_w = max(90, max((ui.text_size(lbl, scale=0.4, weight=1)[0] for lbl in labels), default=90) + 24)
    top_h = 74
    margin = 20
    grid_w = cell_size * n
    grid_h = cell_size * n
    W = label_w + grid_w + margin * 2
    H = top_h + grid_h + margin

    canvas = np.full((H, W, 3), ui.BG, dtype=np.uint8)
    ui.put_text(canvas, "Confusion Matrix", (margin, 30), scale=0.6, color=ui.ACCENT, weight=2)
    ui.put_text(
        canvas, "rows = actual sign  \u00b7  columns = predicted sign",
        (margin, 54), scale=0.36, color=ui.TEXT_MUTED, shadow=False,
    )

    max_val = max(1, int(cm.max()))
    ox, oy = label_w, top_h

    for j, lbl in enumerate(short):
        cx = ox + j * cell_size + cell_size // 2
        tw, _ = ui.text_size(lbl, scale=0.32, weight=1)
        ui.put_text(canvas, lbl, (cx - tw // 2, oy - 8), scale=0.32, color=ui.TEXT_MUTED, shadow=False)

    for i, lbl in enumerate(labels):
        ry = oy + i * cell_size
        tw, th = ui.text_size(lbl, scale=0.38, weight=1)
        ui.put_text(canvas, lbl, (ox - tw - 10, ry + cell_size // 2 + th // 2), scale=0.38, color=ui.TEXT)

        for j in range(n):
            val = int(cm[i, j])
            t = val / max_val
            cx0, cy0 = ox + j * cell_size, ry
            cx1, cy1 = cx0 + cell_size - 2, cy0 + cell_size - 2

            base = np.array(ui.SUCCESS if i == j else ui.DANGER, dtype=np.float32)
            bg = np.array(ui.BG_ELEVATED, dtype=np.float32)
            color = tuple(int(v) for v in (bg * (1.0 - t) + base * t))
            cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), color, -1)
            cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), ui.STROKE, 1, cv2.LINE_AA)

            if val > 0:
                vs = str(val)
                vw, vh = ui.text_size(vs, scale=0.4, weight=2)
                text_color = ui.BG if t > 0.5 else ui.TEXT_MUTED
                ui.put_text(
                    canvas, vs,
                    (cx0 + (cell_size - vw) // 2, cy0 + (cell_size + vh) // 2),
                    scale=0.4, color=text_color, weight=2, shadow=False,
                )
    return canvas


def save_confusion_heatmap(cm: np.ndarray, labels: list[str], path: Path) -> Optional[Path]:
    img = render_confusion_heatmap(cm, labels)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(path), img):
            return path
        print(f"[confusion] cv2.imwrite failed for {path}")
    except Exception as exc:
        print(f"[confusion] could not save heatmap: {exc}")
    return None


def format_confusion_text(cm: np.ndarray, labels: list[str]) -> str:
    """Plain-text fallback table — useful over SSH or if you just want
    a quick terminal glance without opening the saved PNG."""
    short = [lbl[:8] for lbl in labels]
    col_w = max(6, max(len(s) for s in short) + 1)
    header = " " * (col_w + 2) + "".join(f"{s:>{col_w}}" for s in short)
    lines = [header]
    for i, lbl in enumerate(labels):
        row = "".join(f"{int(cm[i, j]):>{col_w}}" for j in range(len(labels)))
        lines.append(f"{lbl[:8]:<{col_w+2}}{row}")
    return "\n".join(lines)
