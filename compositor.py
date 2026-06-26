"""
compositor.py
--------------
Combines every visual layer onto a single frame, in order:

  1. base wireframe mesh (oval, eyes, brows, lips, nose, dot cloud)
  2. AR filters (sunglasses, sparkles, scanning HUD)        [modules/filters.py]
  3. top/bottom HUD bars (FPS, status, key hints)

Kept separate from main.py so main.py only deals with the camera loop
and keyboard input — all "what gets drawn" logic lives here.
"""

import cv2

import config
from modules import filters
from modules import landmarks as lm

_PATHS = [
    (lm.FACE_OVAL, config.COLOR_FACE_OVAL),
    (lm.LEFT_EYEBROW, config.COLOR_EYEBROWS),
    (lm.RIGHT_EYEBROW, config.COLOR_EYEBROWS),
    (lm.LEFT_EYE, config.COLOR_EYES),
    (lm.RIGHT_EYE, config.COLOR_EYES),
    (lm.LIPS_OUTER, config.COLOR_LIPS),
    (lm.LIPS_INNER, config.COLOR_LIPS),
    (lm.NOSE_BRIDGE, config.COLOR_NOSE),
]


def _draw_path(frame, points, indices, color, thickness=config.LINE_THICKNESS):
    for i in range(len(indices) - 1):
        i1, i2 = indices[i], indices[i + 1]
        if i1 >= len(points) or i2 >= len(points):
            continue
        cv2.line(frame, points[i1], points[i2], color, thickness, cv2.LINE_AA)


def draw_dots(frame, points, color=config.COLOR_MESH_DOTS, radius=config.DOT_RADIUS, alpha=0.55):
    overlay = frame.copy()
    for (x, y) in points:
        cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def draw_wireframe(frame, points, show_dots: bool = True):
    if show_dots:
        draw_dots(frame, points)
    for indices, color in _PATHS:
        _draw_path(frame, points, indices, color)
    if len(points) >= 478:
        _draw_path(frame, points, lm.LEFT_IRIS, config.COLOR_IRIS, 1)
        _draw_path(frame, points, lm.RIGHT_IRIS, config.COLOR_IRIS, 1)


def get_bbox(points, frame_shape, margin: int = 10):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = max(min(xs) - margin, 0), max(min(ys) - margin, 0)
    x2, y2 = min(max(xs) + margin, frame_shape[1]), min(max(ys) + margin, frame_shape[0])
    return x1, y1, x2, y2


def draw_bounding_box(frame, points, color=config.COLOR_BBOX, margin: int = 10):
    x1, y1, x2, y2 = get_bbox(points, frame.shape, margin)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)


def draw_top_hud(frame, fps: float, num_faces: int, smoothing_on: bool, recording: bool):
    h, w = frame.shape[:2]
    bar_h = 56

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), config.COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    cv2.putText(frame, config.WINDOW_NAME, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                config.COLOR_HUD_TEXT, 2, cv2.LINE_AA)

    status = (f"FPS: {fps:5.1f}   Faces: {num_faces}   "
              f"Smoothing: {'ON' if smoothing_on else 'OFF'}"
              f"{'   ● REC' if recording else ''}")
    cv2.putText(frame, status, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                config.COLOR_HUD_TEXT, 1, cv2.LINE_AA)

    if recording:
        cv2.circle(frame, (w - 16, 42), 5, (0, 0, 255), -1, cv2.LINE_AA)


def draw_bottom_bar(frame, fps: float, hints: str):
    h, w = frame.shape[:2]
    bar_h = 34
    y0 = h - bar_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), config.COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    cv2.putText(frame, f"FPS: {fps:.0f}", (16, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2, cv2.LINE_AA)

    (tw, th), _ = cv2.getTextSize(hints, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, hints, ((w - tw) // 2, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)


class Compositor:
    """
    Owns the toggle state (dots/visor/sparkle/scan-hud/smoothing) and the
    stateful filter objects (sunglasses, sparkles), and draws one fully
    composited frame per call to render().
    """

    def __init__(self):
        self.show_wireframe = False     # green mesh lines (oval/eyes/lips/etc.)
        self.show_dots = False          # dot-cloud (only matters if wireframe is on)
        self.show_visor = True          # Iron Man sunglasses
        self.show_sparkle = False       # confetti/sparkle particles
        self.show_scan_hud = True       # face mask, scan line, reticles, panel, badge
        self.sunglasses = filters.SunglassesFilter()
        self.sparkle = filters.SparkleEffect()
        self.reticle = filters.ReticleFilter()
        self.panel = filters.RecognitionPanel()
        self.face_mask = filters.FaceScanMask()

    def render(self, frame, face_result, frame_count: int, fps: float, recording: bool):
        for points in face_result.faces:
            if self.show_wireframe:
                draw_wireframe(frame, points, show_dots=self.show_dots)

            if self.show_visor:
                self.sunglasses.draw(frame, points)
            if self.show_sparkle:
                self.sparkle.draw(frame, points)

            if self.show_scan_hud:
                bbox = get_bbox(points, frame.shape)
                if not self.face_mask.draw(frame, points, bbox):
                    filters.draw_scan_brackets(frame, bbox)
                filters.draw_scan_line(frame, bbox, frame_count)
                self.reticle.draw(frame, points, frame_count)
                self.panel.draw(frame, bbox, frame_count)
            elif self.show_wireframe:
                draw_bounding_box(frame, points)

        if self.show_scan_hud:
            filters.draw_tech_badge(frame, frame_count)

        draw_top_hud(frame, fps, face_result.num_faces, self._smoothing_on, recording)
        hints = "V visor|H hud|W wireframe|F sparkle|1-4 reticle|M dots|C smooth|S shot|R rec|Q quit"
        draw_bottom_bar(frame, fps, hints)

    # smoothing_on is tracked in main.py's loop but the HUD needs it too;
    # main.py sets this attribute each frame before calling render().
    _smoothing_on = True