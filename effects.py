"""
effects.py
----------
Extra AR overlays layered on top of the base wireframe mesh:

- draw_visor()            : stylized red/gold wraparound sci-fi visor,
                             geometrically fitted to the eye landmarks.
- SparkleEffect           : twinkling confetti-style dots anchored to landmarks.
- draw_scan_brackets()    : camera-style corner brackets around the face bbox.
- draw_scan_line()        : animated horizontal scanning sweep.
- draw_reticles()         : small rotating crosshair targets near temples.
- draw_status_panel()     : "FACIAL RECOGNITION" readout panel.
- draw_tech_badge()       : generic circular sci-fi emblem (top-right).
"""

import math
import random
import time

import cv2
import numpy as np

import config


# --------------------------------------------------------------------------
# Visor
# --------------------------------------------------------------------------

_LENS_VERTS = [
    (-0.50, -0.15), (-0.30, -0.55), (0.40, -0.55),
    (0.62, 0.00), (0.40, 0.55), (-0.30, 0.55), (-0.50, 0.15),
]


def _eye_axes(outer_pt, inner_pt):
    dx = outer_pt[0] - inner_pt[0]
    dy = outer_pt[1] - inner_pt[1]
    length = math.hypot(dx, dy) or 1e-6
    xdir = (dx / length, dy / length)
    ydir = (-xdir[1], xdir[0])
    return xdir, ydir, length


def _to_world(center, xdir, ydir, lx, ly):
    x = center[0] + xdir[0] * lx + ydir[0] * ly
    y = center[1] + xdir[1] * lx + ydir[1] * ly
    return (int(round(x)), int(round(y)))


def _hexagon(center, xdir, ydir, w, h):
    return [_to_world(center, xdir, ydir, vx * w, vy * h) for vx, vy in _LENS_VERTS]


def _draw_one_lens(frame, outer_pt, inner_pt):
    xdir, ydir, span = _eye_axes(outer_pt, inner_pt)
    center = ((outer_pt[0] + inner_pt[0]) / 2.0, (outer_pt[1] + inner_pt[1]) / 2.0)
    w, h = span * 2.4, span * 2.4 * 0.6

    rim = np.array(_hexagon(center, xdir, ydir, w, h), dtype=np.int32)
    lens = np.array(_hexagon(center, xdir, ydir, w * 0.80, h * 0.74), dtype=np.int32)

    cv2.fillPoly(frame, [rim], config.COLOR_VISOR_RIM, cv2.LINE_AA)
    cv2.fillPoly(frame, [lens], config.COLOR_VISOR_LENS, cv2.LINE_AA)

    cv2.polylines(frame, [rim], True, config.COLOR_VISOR_GOLD, 2, cv2.LINE_AA)
    shine_a = _to_world(center, xdir, ydir, -w * 0.15, -h * 0.30)
    shine_b = _to_world(center, xdir, ydir, w * 0.25, -h * 0.05)
    cv2.line(frame, shine_a, shine_b, config.COLOR_VISOR_SHINE, 3, cv2.LINE_AA)

    tip_top = _to_world(center, xdir, ydir, w * 0.40, -h * 0.55)
    tip_bot = _to_world(center, xdir, ydir, w * 0.40, h * 0.55)
    far_pt = _to_world(center, xdir, ydir, w * 1.05, -h * 0.10)
    fin = np.array([tip_top, tip_bot, far_pt], dtype=np.int32)
    cv2.fillPoly(frame, [fin], config.COLOR_VISOR_RIM, cv2.LINE_AA)

    return center, xdir, ydir, w, h


def draw_visor(frame, points):
    """Fit and draw the stylized visor across both eyes + a bridge piece."""
    if len(points) < config.BASE_LANDMARK_COUNT:
        return

    l_outer, l_inner = points[33], points[133]
    r_inner, r_outer = points[362], points[263]

    l_center, l_x, l_y, l_w, l_h = _draw_one_lens(frame, l_outer, l_inner)
    r_center, r_x, r_y, r_w, r_h = _draw_one_lens(frame, r_outer, r_inner)

    l_top = _to_world(l_center, l_x, l_y, -l_w * 0.5, -l_h * 0.15)
    l_bot = _to_world(l_center, l_x, l_y, -l_w * 0.5, l_h * 0.15)
    r_top = _to_world(r_center, r_x, r_y, -r_w * 0.5, -r_h * 0.15)
    r_bot = _to_world(r_center, r_x, r_y, -r_w * 0.5, r_h * 0.15)
    bridge = np.array([l_top, l_bot, r_bot, r_top], dtype=np.int32)
    cv2.fillPoly(frame, [bridge], config.COLOR_VISOR_GOLD, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Sparkle / confetti particles
# --------------------------------------------------------------------------

_SPARKLE_ANCHOR_POOL = [10, 109, 67, 103, 54, 21, 251, 284, 332,
                        197, 195, 5, 4, 1, 19, 94,
                        50, 280, 234, 454, 132, 361, 207, 427, 195, 5]


class SparkleEffect:
    """Persistent twinkling particles anchored to landmark indices."""

    def __init__(self, count: int = config.NUM_SPARKLES):
        self._count = count
        self._particles = None

    def _build(self):
        rng = random.Random(42)
        particles = []
        for _ in range(self._count):
            particles.append({
                "idx": rng.choice(_SPARKLE_ANCHOR_POOL),
                "dx": rng.randint(-22, 22),
                "dy": rng.randint(-22, 22),
                "color": rng.choice(config.SPARKLE_COLORS),
                "phase": rng.uniform(0, 2 * math.pi),
                "speed": rng.uniform(2.5, 5.0),
                "size": rng.choice([1, 1, 2]),
            })
        self._particles = particles

    def draw(self, frame, points):
        if len(points) < config.BASE_LANDMARK_COUNT:
            return
        if self._particles is None:
            self._build()

        t = time.time()
        overlay = frame.copy()
        for p in self._particles:
            brightness = (math.sin(t * p["speed"] + p["phase"]) + 1) / 2
            if brightness < 0.45:
                continue
            base_x, base_y = points[p["idx"]]
            x, y = base_x + p["dx"], base_y + p["dy"]
            size = p["size"] + (1 if brightness > 0.85 else 0)
            cv2.circle(overlay, (x, y), size, p["color"], -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, dst=frame)


# --------------------------------------------------------------------------
# Scanning / facial-recognition HUD
# --------------------------------------------------------------------------

def draw_scan_brackets(frame, bbox, length=22, thickness=2, color=config.COLOR_SCAN_BRACKET):
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]
    for cx, cy, sx, sy in corners:
        cv2.line(frame, (cx, cy), (cx + sx * length, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + sy * length), color, thickness, cv2.LINE_AA)


def draw_scan_line(frame, bbox, frame_count, period=60):
    x1, y1, x2, y2 = bbox
    height = max(y2 - y1, 1)
    progress = (frame_count % period) / period
    y = int(y1 + progress * height)

    overlay = frame.copy()
    cv2.line(overlay, (x1, y), (x2, y), config.COLOR_SCAN_LINE, 2, cv2.LINE_AA)
    cv2.line(overlay, (x1, y), (x2, y), config.COLOR_SCAN_LINE, 6, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)


def draw_reticles(frame, points, frame_count):
    if len(points) < config.BASE_LANDMARK_COUNT:
        return
    targets = [points[127], points[356]]
    angle = (frame_count * 3) % 360
    for (x, y) in targets:
        cv2.circle(frame, (x, y), 14, config.COLOR_RETICLE, 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 6, config.COLOR_RETICLE, 1, cv2.LINE_AA)
        rad = math.radians(angle)
        tx = int(x + 20 * math.cos(rad))
        ty = int(y + 20 * math.sin(rad))
        cv2.line(frame, (x, y), (tx, ty), config.COLOR_RETICLE, 1, cv2.LINE_AA)


def draw_status_panel(frame, bbox, frame_count, period=60):
    x1, y1, x2, y2 = bbox
    progress = int(((frame_count % period) / period) * 100)
    label = f"FACIAL RECOGNITION  [{progress:3d}%]"

    panel_x, panel_y = x1, max(y1 - 30, 20)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 6, panel_y - th - 8),
                  (panel_x + tw + 6, panel_y + 6), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    cv2.putText(frame, label, (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                config.COLOR_SCAN_LINE, 1, cv2.LINE_AA)


def draw_tech_badge(frame, frame_count):
    h, w = frame.shape[:2]
    cx, cy, r = w - 70, 60, 38

    cv2.circle(frame, (cx, cy), r, config.COLOR_BADGE, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r - 7, config.COLOR_BADGE, 1, cv2.LINE_AA)

    for k in range(4):
        angle = math.radians(frame_count * 2 + k * 90)
        x1 = int(cx + (r - 4) * math.cos(angle))
        y1 = int(cy + (r - 4) * math.sin(angle))
        x2 = int(cx + (r + 5) * math.cos(angle))
        y2 = int(cy + (r + 5) * math.sin(angle))
        cv2.line(frame, (x1, y1), (x2, y2), config.COLOR_BADGE_ACCENT, 2, cv2.LINE_AA)

    cv2.putText(frame, "AR", (cx - 16, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                config.COLOR_BADGE, 2, cv2.LINE_AA)
