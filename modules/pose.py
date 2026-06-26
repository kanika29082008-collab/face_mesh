"""
modules/pose.py
-----------------
Geometry helpers shared by anything that needs to position/rotate an
overlay relative to the face — e.g. fitting sunglasses to the eye line,
or computing a face's roll angle. Kept separate from filters.py so the
math can be unit-tested or reused (e.g. for future head-pose features)
independently of any drawing code.
"""

import math

from modules import landmarks as lm


def eye_axes(outer_pt, inner_pt):
    """
    Given two eye-corner points, return:
      - xdir: unit vector pointing from inner -> outer corner (i.e.
              "outward", toward the ear)
      - ydir: unit vector perpendicular to xdir ("down" relative to the
              face's current tilt)
      - length: pixel distance between the two points
    """
    dx = outer_pt[0] - inner_pt[0]
    dy = outer_pt[1] - inner_pt[1]
    length = math.hypot(dx, dy) or 1e-6
    xdir = (dx / length, dy / length)
    ydir = (-xdir[1], xdir[0])
    return xdir, ydir, length


def roll_angle_degrees(points):
    """Head roll (in-plane tilt) in degrees, derived from the eye line."""
    left = points[lm.LEFT_EYE_OUTER_CORNER]
    right = points[lm.RIGHT_EYE_OUTER_CORNER]
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return math.degrees(math.atan2(dy, dx))


def eye_span_and_center(points):
    """
    Distance and midpoint between the two eyes' outer corners — useful
    as a stable, rotation-robust scale reference for sizing overlays
    (sunglasses width, visor size, etc.).
    """
    left = points[lm.LEFT_EYE_OUTER_CORNER]
    right = points[lm.RIGHT_EYE_OUTER_CORNER]
    span = math.hypot(right[0] - left[0], right[1] - left[1])
    center = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    return span, center


def to_world(center, xdir, ydir, local_x, local_y):
    """Map a local (x, y) offset (in the xdir/ydir basis) to pixel coords."""
    x = center[0] + xdir[0] * local_x + ydir[0] * local_y
    y = center[1] + xdir[1] * local_x + ydir[1] * local_y
    return int(round(x)), int(round(y))


class LandmarkSmoother:
    """
    Exponential moving average smoother for a set of (x, y) points.

    Without smoothing, raw per-frame landmark detections visibly jitter
    even when the face is still. This filter blends each new point with
    its previous smoothed value:
        smoothed = alpha * new + (1 - alpha) * smoothed_prev
    Lower alpha => smoother but slightly more lag. Resets automatically
    if the number of points changes (e.g. face lost then re-detected).
    """

    def __init__(self, alpha: float = 0.55):
        self.alpha = alpha
        self._prev = None  # list[tuple[float, float]] | None

    def smooth(self, points):
        if self._prev is None or len(self._prev) != len(points):
            self._prev = [(float(x), float(y)) for x, y in points]
            return points

        a = self.alpha
        new_prev = []
        out = []
        for (nx, ny), (px, py) in zip(points, self._prev):
            sx = a * nx + (1 - a) * px
            sy = a * ny + (1 - a) * py
            new_prev.append((sx, sy))
            out.append((int(sx), int(sy)))
        self._prev = new_prev
        return out

    def reset(self):
        self._prev = None