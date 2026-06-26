"""
modules/filters.py
--------------------
All AR overlay "filters":

- SunglassesFilter : alpha-blends ironman_sunglasses.png onto the face,
                     scaled and rotated to match the eye line. Falls
                     back to a drawn vector visor if the PNG is missing.
- SparkleEffect     : twinkling confetti-style particles across the face.
- Scanning HUD      : corner brackets, animated scan line, rotating
                       crosshair reticles, a live status panel, and a
                       generic rotating tech badge (top-right). These
                       are deliberately generic/sci-fi — not a
                       reproduction of any copyrighted logo or character.
"""

import math
import os
import random
import time

import cv2
import numpy as np

import config
from modules import landmarks as lm
from modules import pose


def _load_rgba_asset(path):
    """Load an image as RGBA, adding an opaque alpha channel if missing.
    Returns None if the path doesn't exist or can't be read (callers
    should degrade gracefully rather than crash)."""
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])
    return img


# --------------------------------------------------------------------------
# Sunglasses (image-based) with vector fallback
# --------------------------------------------------------------------------

def _overlay_rgba(frame, rgba, top_left, alpha_mult: float = 1.0):
    """Alpha-blend an RGBA image onto frame at top_left, clipped to bounds.
    alpha_mult scales the image's own alpha channel (1.0 = use as-is,
    <1.0 = make the whole overlay more see-through)."""
    x, y = top_left
    h, w = rgba.shape[:2]
    fh, fw = frame.shape[:2]

    src_x0, src_y0 = max(0, -x), max(0, -y)
    dst_x0, dst_y0 = max(0, x), max(0, y)
    dst_x1, dst_y1 = min(fw, x + w), min(fh, y + h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return

    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    region = rgba[src_y0:src_y1, src_x0:src_x1]
    alpha = (region[:, :, 3:4].astype(np.float32) / 255.0) * alpha_mult
    rgb = region[:, :, :3].astype(np.float32)

    dst = frame[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
    blended = rgb * alpha + dst * (1 - alpha)
    frame[dst_y0:dst_y1, dst_x0:dst_x1] = blended.astype(np.uint8)


class SunglassesFilter:
    """
    Loads ironman_sunglasses.png once and fits it to the eyes each frame.
    If the asset is missing, draws a simple vector visor instead so the
    app still works (graceful degradation, no crash).
    """

    def __init__(self, png_path=config.SUNGLASSES_PNG_PATH):
        self._png_rgba = _load_rgba_asset(png_path)

    @property
    def using_image(self) -> bool:
        return self._png_rgba is not None

    def draw(self, frame, points):
        if len(points) < 468:
            return
        if self._png_rgba is not None:
            self._draw_image(frame, points)
        else:
            self._draw_vector_fallback(frame, points)

    def _draw_image(self, frame, points):
        span, center = pose.eye_span_and_center(points)
        angle = pose.roll_angle_degrees(points)

        target_w = int(span * config.SUNGLASSES_WIDTH_SCALE)
        h0, w0 = self._png_rgba.shape[:2]
        target_h = int(h0 * (target_w / w0))
        if target_w < 4 or target_h < 4:
            return

        resized = cv2.resize(self._png_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

        rot_mat = cv2.getRotationMatrix2D((target_w / 2, target_h / 2), -angle, 1.0)
        rotated = cv2.warpAffine(
            resized, rot_mat, (target_w, target_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )

        y_offset = span * config.SUNGLASSES_Y_OFFSET_SCALE
        top_left = (int(center[0] - target_w / 2), int(center[1] - target_h / 2 + y_offset))
        _overlay_rgba(frame, rotated, top_left)

    @staticmethod
    def _draw_vector_fallback(frame, points):
        l_outer, l_inner = points[lm.LEFT_EYE_OUTER_CORNER], points[lm.LEFT_EYE_INNER_CORNER]
        r_inner, r_outer = points[lm.RIGHT_EYE_INNER_CORNER], points[lm.RIGHT_EYE_OUTER_CORNER]
        for outer, inner in ((l_outer, l_inner), (r_outer, r_inner)):
            xdir, ydir, span = pose.eye_axes(outer, inner)
            center = ((outer[0] + inner[0]) / 2.0, (outer[1] + inner[1]) / 2.0)
            w, h = span * 2.4, span * 2.4 * 0.6
            verts = [(-0.5, -0.15), (-0.3, -0.55), (0.4, -0.55),
                     (0.62, 0.0), (0.4, 0.55), (-0.3, 0.55), (-0.5, 0.15)]
            poly = np.array([pose.to_world(center, xdir, ydir, vx * w, vy * h)
                              for vx, vy in verts], dtype=np.int32)
            cv2.fillPoly(frame, [poly], config.COLOR_VISOR_RIM, cv2.LINE_AA)
            inset = np.array([pose.to_world(center, xdir, ydir, vx * w * 0.8, vy * h * 0.74)
                               for vx, vy in verts], dtype=np.int32)
            cv2.fillPoly(frame, [inset], config.COLOR_VISOR_LENS, cv2.LINE_AA)
            cv2.polylines(frame, [poly], True, config.COLOR_VISOR_GOLD, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Sparkle / confetti particles
# --------------------------------------------------------------------------

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
                "idx": rng.choice(lm.SPARKLE_ANCHOR_POOL),
                "dx": rng.randint(-22, 22),
                "dy": rng.randint(-22, 22),
                "color": rng.choice(config.SPARKLE_COLORS),
                "phase": rng.uniform(0, 2 * math.pi),
                "speed": rng.uniform(2.5, 5.0),
                "size": rng.choice([1, 1, 2]),
            })
        self._particles = particles

    def draw(self, frame, points):
        if len(points) < 468:
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


class FaceScanMask:
    """
    Overlays mask.png (a wireframe face graphic with its own corner
    brackets) on top of the detected face, scaled to the bounding box
    and rotated to the head's roll angle, blended semi-transparently
    so the real face mesh underneath still shows through.

    Since this image already includes corner brackets, it replaces the
    drawn draw_scan_brackets() — compositor falls back to the drawn
    brackets only if this asset isn't present.
    """

    def __init__(self, path=config.FACE_SCAN_MASK_PATH):
        self._img = _load_rgba_asset(path)

    @property
    def available(self) -> bool:
        return self._img is not None

    def draw(self, frame, points, bbox) -> bool:
        """Returns True if it drew the image mask, False if there was
        nothing to draw (caller should fall back to drawn brackets)."""
        if self._img is None:
            return False

        x1, y1, x2, y2 = bbox
        w = int((x2 - x1) * config.MASK_PADDING_SCALE)
        h = int((y2 - y1) * config.MASK_PADDING_SCALE)
        if w < 4 or h < 4:
            return True

        angle = pose.roll_angle_degrees(points) if len(points) >= 468 else 0.0
        resized = cv2.resize(self._img, (w, h), interpolation=cv2.INTER_AREA)
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        rotated = cv2.warpAffine(
            resized, rot_mat, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        top_left = (cx - w // 2, cy - h // 2)
        _overlay_rgba(frame, rotated, top_left, alpha_mult=config.MASK_ALPHA)
        return True


def draw_scan_line(frame, bbox, frame_count, period=60):
    x1, y1, x2, y2 = bbox
    height = max(y2 - y1, 1)
    progress = (frame_count % period) / period
    y = int(y1 + progress * height)

    overlay = frame.copy()
    cv2.line(overlay, (x1, y), (x2, y), config.COLOR_SCAN_LINE, 2, cv2.LINE_AA)
    cv2.line(overlay, (x1, y), (x2, y), config.COLOR_SCAN_LINE, 6, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)


def _draw_reticles_vector_fallback(frame, points, frame_count):
    targets = [points[lm.LEFT_TEMPLE], points[lm.RIGHT_TEMPLE]]
    angle = (frame_count * 3) % 360
    for (x, y) in targets:
        cv2.circle(frame, (x, y), 14, config.COLOR_RETICLE, 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 6, config.COLOR_RETICLE, 1, cv2.LINE_AA)
        rad = math.radians(angle)
        tx = int(x + 20 * math.cos(rad))
        ty = int(y + 20 * math.sin(rad))
        cv2.line(frame, (x, y), (tx, ty), config.COLOR_RETICLE, 1, cv2.LINE_AA)


class ReticleFilter:
    """
    Image-based crosshair/reticle placed at both temples, rotating slowly.
    Holds several styles (loaded from config.RETICLE_PATHS) and switches
    between them via set_style()/cycle() — wire these to number keys in
    main.py so the person can flip through designs live.
    Falls back to a drawn vector reticle if no images loaded at all.
    """

    def __init__(self, paths=config.RETICLE_PATHS):
        self._images = [_load_rgba_asset(p) for p in paths]
        self.index = 0

    @property
    def style_count(self) -> int:
        return len(self._images)

    @property
    def current_image_available(self) -> bool:
        return self._images[self.index] is not None

    def set_style(self, index: int):
        if self._images:
            self.index = index % len(self._images)

    def cycle(self):
        if self._images:
            self.index = (self.index + 1) % len(self._images)

    def draw(self, frame, points, frame_count):
        if len(points) < 468:
            return
        img = self._images[self.index] if self._images else None
        if img is None:
            _draw_reticles_vector_fallback(frame, points, frame_count)
            return

        span, _ = pose.eye_span_and_center(points)
        size = max(int(span * config.RETICLE_SIZE_SCALE), 12)
        angle = (frame_count * config.RETICLE_SPIN_SPEED) % 360

        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        rot_mat = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            resized, rot_mat, (size, size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )

        for idx in (lm.LEFT_TEMPLE, lm.RIGHT_TEMPLE):
            cx, cy = points[idx]
            top_left = (int(cx - size / 2), int(cy - size / 2))
            _overlay_rgba(frame, rotated, top_left)


def draw_status_panel(frame, bbox, frame_count, period=60):
    """Drawn-text fallback used only if the recognition_panel.png asset is missing."""
    x1, y1, _, _ = bbox
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


class RecognitionPanel:
    """
    Displays the recognition_panel.png HUD widget at a fixed screen
    corner. Falls back to the drawn text panel (draw_status_panel) if
    the image asset isn't present.
    """

    def __init__(self, path=config.RECOGNITION_PANEL_PATH, width=260):
        self._img = _load_rgba_asset(path)
        self._width = width

    @property
    def available(self) -> bool:
        return self._img is not None

    def draw(self, frame, bbox, frame_count, anchor="bottom-left", margin=16):
        if self._img is None:
            draw_status_panel(frame, bbox, frame_count)
            return

        h0, w0 = self._img.shape[:2]
        target_w = self._width
        target_h = int(h0 * (target_w / w0))
        resized = cv2.resize(self._img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        fh, fw = frame.shape[:2]
        if anchor == "bottom-left":
            top_left = (margin, fh - target_h - margin - 40)  # clear of bottom bar
        elif anchor == "top-right":
            top_left = (fw - target_w - margin, margin + 60)  # clear of top bar
        else:
            top_left = (margin, margin + 60)

        _overlay_rgba(frame, resized, top_left)


def draw_tech_badge(frame, frame_count):
    """Generic circular sci-fi emblem, top-right corner."""
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