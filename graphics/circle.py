"""
Circle and ellipse scan-conversion.

Syllabus reference: CSE 452, Week 2 -- "Implement Bresenham & Mid-Point circle
drawing algorithm".

Three distinct algorithms live here:

``bresenham_circle_points``
    Bresenham's circle algorithm.  Decision variable starts at ``3 - 2r`` and
    is updated with ``4x + 6`` or ``4(x - y) + 10``.

``midpoint_circle_points``
    The midpoint circle algorithm.  Decision variable starts at ``1 - r`` and
    is updated with ``2x + 1`` or ``2(x - y) + 1``.

``midpoint_ellipse_points``
    The two-region midpoint ellipse algorithm, needed because the holes in the
    game board are drawn in perspective and are therefore not circular.

All three exploit symmetry: only one octant (or one quadrant for the ellipse)
is actually computed, and the remaining points are mirrored.  That is the
whole point of these algorithms -- one eighth of the arithmetic.
"""

from __future__ import annotations

import math
from typing import Sequence

from .raster import Canvas, Color


# ---------------------------------------------------------------------------
# Symmetry helper
# ---------------------------------------------------------------------------


def _eight_way_symmetry(xc: int, yc: int, x: int, y: int) -> list[tuple[int, int]]:
    """Mirror one octant point into all eight octants of a circle."""
    return [
        (xc + x, yc + y),
        (xc - x, yc + y),
        (xc + x, yc - y),
        (xc - x, yc - y),
        (xc + y, yc + x),
        (xc - y, yc + x),
        (xc + y, yc - x),
        (xc - y, yc - x),
    ]


# ---------------------------------------------------------------------------
# Bresenham circle
# ---------------------------------------------------------------------------


def bresenham_circle_points(xc: float, yc: float, radius: float) -> list[tuple[int, int]]:
    """Generate circle pixels using Bresenham's circle algorithm.

    Starting at the top of the circle ``(0, r)`` the algorithm walks the second
    octant.  The decision variable ``d`` measures whether the ideal circle has
    fallen far enough for the next pixel to drop a row:

    * ``d < 0``  -> choose the horizontal neighbour, ``d += 4x + 6``
    * ``d >= 0`` -> choose the diagonal neighbour, ``d += 4(x - y) + 10``
    """
    xc, yc, radius = int(round(xc)), int(round(yc)), int(round(radius))
    if radius < 0:
        return []
    if radius == 0:
        return [(xc, yc)]

    x = 0
    y = radius
    d = 3 - 2 * radius

    points: list[tuple[int, int]] = []
    while x <= y:
        points.extend(_eight_way_symmetry(xc, yc, x, y))
        if d < 0:
            d += 4 * x + 6
        else:
            d += 4 * (x - y) + 10
            y -= 1
        x += 1
    return points


# ---------------------------------------------------------------------------
# Midpoint circle
# ---------------------------------------------------------------------------


def midpoint_circle_points(xc: float, yc: float, radius: float) -> list[tuple[int, int]]:
    """Generate circle pixels using the midpoint circle algorithm.

    The decision parameter is the circle function evaluated at the midpoint
    between the two candidate pixels,
    ``p = f(x + 1, y - 1/2) = (x+1)^2 + (y-1/2)^2 - r^2``.
    Multiplying out and dropping the constant ``1/4`` (which never changes the
    sign) gives the integer initial value ``p = 1 - r``.

    * ``p < 0``  -> midpoint is inside the circle, keep ``y``, ``p += 2x + 1``
    * ``p >= 0`` -> midpoint is outside, decrement ``y``, ``p += 2(x - y) + 1``
    """
    xc, yc, radius = int(round(xc)), int(round(yc)), int(round(radius))
    if radius < 0:
        return []
    if radius == 0:
        return [(xc, yc)]

    x = 0
    y = radius
    p = 1 - radius

    points: list[tuple[int, int]] = list(_eight_way_symmetry(xc, yc, x, y))
    while x < y:
        x += 1
        if p < 0:
            p += 2 * x + 1
        else:
            y -= 1
            p += 2 * (x - y) + 1
        points.extend(_eight_way_symmetry(xc, yc, x, y))
    return points


# ---------------------------------------------------------------------------
# Midpoint ellipse
# ---------------------------------------------------------------------------


def midpoint_ellipse_points(
    xc: float, yc: float, rx: float, ry: float
) -> list[tuple[int, int]]:
    """Generate ellipse pixels using the two-region midpoint algorithm.

    An ellipse has no eight-fold symmetry, only four-fold, and its slope passes
    through -1 partway around the quadrant.  The algorithm therefore splits the
    first quadrant into two regions:

    * **Region 1** -- slope is shallower than -1, so ``x`` is the driving axis.
      Stop when ``2 ry^2 x >= 2 rx^2 y``.
    * **Region 2** -- slope is steeper than -1, so ``y`` becomes the driving
      axis and the loop runs down to ``y = 0``.
    """
    xc, yc = int(round(xc)), int(round(yc))
    rx, ry = int(round(rx)), int(round(ry))
    if rx <= 0 or ry <= 0:
        return [(xc, yc)]

    points: list[tuple[int, int]] = []

    def plot4(x: int, y: int) -> None:
        points.extend(
            [
                (xc + x, yc + y),
                (xc - x, yc + y),
                (xc + x, yc - y),
                (xc - x, yc - y),
            ]
        )

    rx2 = rx * rx
    ry2 = ry * ry
    two_rx2 = 2 * rx2
    two_ry2 = 2 * ry2

    # ---- Region 1 -------------------------------------------------------
    x = 0
    y = ry
    px = 0
    py = two_rx2 * y
    plot4(x, y)

    p = round(ry2 - rx2 * ry + 0.25 * rx2)
    while px < py:
        x += 1
        px += two_ry2
        if p < 0:
            p += ry2 + px
        else:
            y -= 1
            py -= two_rx2
            p += ry2 + px - py
        plot4(x, y)

    # ---- Region 2 -------------------------------------------------------
    p = round(ry2 * (x + 0.5) ** 2 + rx2 * (y - 1) ** 2 - rx2 * ry2)
    while y > 0:
        y -= 1
        py -= two_rx2
        if p > 0:
            p += rx2 - py
        else:
            x += 1
            px += two_ry2
            p += rx2 - py + px
        plot4(x, y)

    return points


# ---------------------------------------------------------------------------
# Span extraction -- turns a boundary point set into fillable scan-lines
# ---------------------------------------------------------------------------


def boundary_spans(points: Sequence[tuple[int, int]]) -> dict[int, tuple[int, int]]:
    """Collapse a closed boundary into one ``(x_min, x_max)`` run per row.

    A convex outline such as a circle or ellipse touches each scan-line at
    exactly two x positions, so the interior of the shape on that row is simply
    everything between them.  This is what lets an area fill run at span speed
    while the *shape* is still decided entirely by the scan-conversion
    algorithm above.
    """
    spans: dict[int, tuple[int, int]] = {}
    for x, y in points:
        current = spans.get(y)
        if current is None:
            spans[y] = (x, x)
        elif x < current[0]:
            spans[y] = (x, current[1])
        elif x > current[1]:
            spans[y] = (current[0], x)
    return spans


# ---------------------------------------------------------------------------
# Canvas-facing helpers
# ---------------------------------------------------------------------------

CIRCLE_ALGORITHMS = ("midpoint", "bresenham")


def draw_circle(
    canvas: Canvas,
    xc: float,
    yc: float,
    radius: float,
    color: Color,
    algorithm: str = "midpoint",
    thickness: int = 1,
) -> None:
    """Draw a circle outline.

    Thickness greater than one is produced by drawing concentric circles,
    which is the standard way to thicken a scan-converted curve without
    resorting to a different primitive.
    """
    if algorithm == "midpoint":
        generator = midpoint_circle_points
    elif algorithm == "bresenham":
        generator = bresenham_circle_points
    else:
        raise ValueError(
            f"unknown circle algorithm {algorithm!r}; expected one of {CIRCLE_ALGORITHMS}"
        )

    thickness = max(1, int(thickness))
    base = int(round(radius))
    for ring in range(thickness):
        r = base - ring
        if r < 0:
            break
        for px, py in generator(xc, yc, r):
            canvas.put_pixel(px, py, color)


def fill_circle(
    canvas: Canvas,
    xc: float,
    yc: float,
    radius: float,
    color: Color,
    algorithm: str = "midpoint",
) -> None:
    """Fill a disc by span-filling between the scan-converted boundary points."""
    if algorithm == "bresenham":
        points = bresenham_circle_points(xc, yc, radius)
    else:
        points = midpoint_circle_points(xc, yc, radius)
    for y, (x_start, x_end) in boundary_spans(points).items():
        canvas.put_span(y, x_start, x_end, color)


def fill_circle_blend(
    canvas: Canvas, xc: float, yc: float, radius: float, color: Color, alpha: float
) -> None:
    """Alpha-blended disc fill -- used for glows and shadows."""
    points = midpoint_circle_points(xc, yc, radius)
    for y, (x_start, x_end) in boundary_spans(points).items():
        canvas.blend_span(y, x_start, x_end, color, alpha)


def draw_ellipse(
    canvas: Canvas,
    xc: float,
    yc: float,
    rx: float,
    ry: float,
    color: Color,
    thickness: int = 1,
) -> None:
    """Draw an ellipse outline using the midpoint ellipse algorithm."""
    thickness = max(1, int(thickness))
    for ring in range(thickness):
        cur_rx = int(round(rx)) - ring
        cur_ry = int(round(ry)) - ring
        if cur_rx <= 0 or cur_ry <= 0:
            break
        for px, py in midpoint_ellipse_points(xc, yc, cur_rx, cur_ry):
            canvas.put_pixel(px, py, color)


def fill_ellipse(
    canvas: Canvas, xc: float, yc: float, rx: float, ry: float, color: Color
) -> None:
    """Fill an ellipse by span-filling its midpoint-generated boundary."""
    points = midpoint_ellipse_points(xc, yc, rx, ry)
    for y, (x_start, x_end) in boundary_spans(points).items():
        canvas.put_span(y, x_start, x_end, color)


def fill_ellipse_blend(
    canvas: Canvas, xc: float, yc: float, rx: float, ry: float, color: Color, alpha: float
) -> None:
    points = midpoint_ellipse_points(xc, yc, rx, ry)
    for y, (x_start, x_end) in boundary_spans(points).items():
        canvas.blend_span(y, x_start, x_end, color, alpha)


def draw_arc(
    canvas: Canvas,
    xc: float,
    yc: float,
    radius: float,
    start_deg: float,
    end_deg: float,
    color: Color,
    thickness: int = 1,
) -> None:
    """Draw the portion of a circle between two angles.

    The full circle is scan-converted and each candidate pixel is then tested
    against the angular window.  Angles increase clockwise on screen because
    the y axis points down.
    """
    start = math.radians(start_deg) % (2 * math.pi)
    end = math.radians(end_deg) % (2 * math.pi)

    def in_window(angle: float) -> bool:
        angle %= 2 * math.pi
        if start <= end:
            return start <= angle <= end
        return angle >= start or angle <= end

    thickness = max(1, int(thickness))
    for ring in range(thickness):
        r = int(round(radius)) - ring
        if r <= 0:
            break
        for px, py in midpoint_circle_points(xc, yc, r):
            angle = math.atan2(py - yc, px - xc)
            if in_window(angle):
                canvas.put_pixel(px, py, color)


def fill_ring(
    canvas: Canvas,
    xc: float,
    yc: float,
    outer_radius: float,
    inner_radius: float,
    color: Color,
) -> None:
    """Fill an annulus by subtracting the inner disc's spans from the outer's."""
    outer = boundary_spans(midpoint_circle_points(xc, yc, outer_radius))
    inner = boundary_spans(midpoint_circle_points(xc, yc, inner_radius))
    for y, (ox0, ox1) in outer.items():
        hole = inner.get(y)
        if hole is None:
            canvas.put_span(y, ox0, ox1, color)
        else:
            canvas.put_span(y, ox0, hole[0] - 1, color)
            canvas.put_span(y, hole[1] + 1, ox1, color)
