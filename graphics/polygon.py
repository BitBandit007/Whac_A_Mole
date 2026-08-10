"""
Polygon outlines, scan-line area fill, and shape constructors.

Syllabus reference: CSE 452, Week 3 -- "Draw basic shapes e.g. line, rectangle,
triangle etc.", extended with the scan-line fill algorithm.

The fill algorithm here is the standard **scan-line / edge-list** method:

1.  For each horizontal scan-line crossing the polygon, find every edge it
    intersects and record the x coordinate of the crossing.
2.  Sort those x values.
3.  Fill between them in pairs (the even-odd rule): between crossings 0-1,
    2-3, and so on.

The subtle part is the vertex rule.  A scan-line that passes exactly through a
vertex would naively record *two* crossings and break the pairing.  The fix is
to treat each edge as half-open in y -- an edge contributes to scan-line ``y``
only when ``y_min <= y < y_max``.  A local minimum or maximum then contributes
zero or two crossings as it should, and a vertex where the polygon merely
passes through contributes exactly one.
"""

from __future__ import annotations

import math
from typing import Sequence

from .clipping import Window, cohen_sutherland, sutherland_hodgman
from .line import draw_line, draw_thick_line
from .raster import Canvas, Color

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Outlines
# ---------------------------------------------------------------------------


def draw_polygon(
    canvas: Canvas,
    points: Sequence[Point],
    color: Color,
    thickness: int = 1,
    closed: bool = True,
    window: Window | None = None,
) -> None:
    """Stroke a polygon outline.

    If ``window`` is supplied each edge is first clipped with Cohen-Sutherland,
    which keeps the rasteriser from walking long runs of pixels that would only
    be discarded by the canvas clip test.
    """
    count = len(points)
    if count < 2:
        return

    limit = count if closed else count - 1
    for index in range(limit):
        a = points[index]
        b = points[(index + 1) % count]

        if window is not None:
            clipped = cohen_sutherland(a, b, window)
            if clipped is None:
                continue
            a, b = clipped

        if thickness <= 1:
            draw_line(canvas, a[0], a[1], b[0], b[1], color)
        else:
            draw_thick_line(canvas, a[0], a[1], b[0], b[1], color, thickness)


# ---------------------------------------------------------------------------
# Scan-line fill
# ---------------------------------------------------------------------------


def polygon_spans(points: Sequence[Point]) -> list[tuple[int, int, int]]:
    """Compute the fill spans of a polygon as ``(y, x_start, x_end)`` triples.

    Returning spans rather than drawing them keeps the algorithm testable and
    lets callers reuse the result (for example to fill a shape and then tint
    the same spans a second time).
    """
    count = len(points)
    if count < 3:
        return []

    ys = [p[1] for p in points]
    y_start = int(math.ceil(min(ys)))
    y_end = int(math.floor(max(ys)))
    if y_end < y_start:
        return []

    # Pre-build the edge list, discarding horizontal edges: they never
    # contribute a crossing under the half-open rule, and including them would
    # mean dividing by zero.
    edges: list[tuple[float, float, float, float]] = []
    for index in range(count):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % count]
        if y0 == y1:
            continue
        edges.append((x0, y0, x1, y1))

    if not edges:
        return []

    spans: list[tuple[int, int, int]] = []
    for y in range(y_start, y_end + 1):
        scan_y = y + 0.5      # sample at pixel centres
        crossings: list[float] = []
        for x0, y0, x1, y1 in edges:
            y_low, y_high = (y0, y1) if y0 < y1 else (y1, y0)
            # Half-open in y: include the lower endpoint, exclude the upper.
            if y_low <= scan_y < y_high:
                t = (scan_y - y0) / (y1 - y0)
                crossings.append(x0 + t * (x1 - x0))

        if len(crossings) < 2:
            continue

        crossings.sort()
        for pair in range(0, len(crossings) - 1, 2):
            x_start = int(math.ceil(crossings[pair] - 0.5))
            x_end = int(math.floor(crossings[pair + 1] - 0.5))
            if x_end >= x_start:
                spans.append((y, x_start, x_end))

    return spans


def fill_polygon(canvas: Canvas, points: Sequence[Point], color: Color) -> None:
    """Fill a polygon using the scan-line algorithm."""
    for y, x_start, x_end in polygon_spans(points):
        canvas.put_span(y, x_start, x_end, color)


def fill_polygon_blend(
    canvas: Canvas, points: Sequence[Point], color: Color, alpha: float
) -> None:
    """Alpha-blended polygon fill."""
    for y, x_start, x_end in polygon_spans(points):
        canvas.blend_span(y, x_start, x_end, color, alpha)


def fill_polygon_clipped(
    canvas: Canvas, points: Sequence[Point], color: Color, window: Window
) -> list[Point]:
    """Clip a polygon to ``window`` with Sutherland-Hodgman, then fill it.

    Returns the clipped vertex list so the caller can stroke the same outline
    without clipping it twice.  This is the routine that makes a mole appear to
    rise *out of* its hole: the mole's silhouette is clipped against a window
    whose bottom edge is the hole rim.
    """
    clipped = sutherland_hodgman(points, window)
    if len(clipped) >= 3:
        fill_polygon(canvas, clipped, color)
    return clipped


# ---------------------------------------------------------------------------
# Shape constructors
#
# Each returns a list of vertices in *local* coordinates centred on the origin,
# ready to be positioned by a matrix from ``graphics.transform2d``.
# ---------------------------------------------------------------------------


def regular_polygon(radius: float, sides: int, rotation_degrees: float = 0.0) -> list[Point]:
    """Vertices of a regular n-gon inscribed in a circle of ``radius``."""
    sides = max(3, int(sides))
    offset = math.radians(rotation_degrees)
    return [
        (
            radius * math.cos(offset + 2 * math.pi * i / sides),
            radius * math.sin(offset + 2 * math.pi * i / sides),
        )
        for i in range(sides)
    ]


def star_polygon(
    outer_radius: float, inner_radius: float, points: int = 5, rotation_degrees: float = -90.0
) -> list[Point]:
    """Vertices of a star, alternating between the two radii."""
    vertices: list[Point] = []
    offset = math.radians(rotation_degrees)
    step = math.pi / points
    for i in range(points * 2):
        radius = outer_radius if i % 2 == 0 else inner_radius
        angle = offset + i * step
        vertices.append((radius * math.cos(angle), radius * math.sin(angle)))
    return vertices


def rectangle(width: float, height: float) -> list[Point]:
    """Axis-aligned rectangle centred on the origin."""
    hw, hh = width / 2.0, height / 2.0
    return [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]


def rounded_rectangle(
    width: float, height: float, radius: float, segments: int = 6
) -> list[Point]:
    """Rectangle with quarter-circle corners, approximated as a polygon."""
    hw, hh = width / 2.0, height / 2.0
    radius = min(radius, hw, hh)
    vertices: list[Point] = []
    corners = [
        (hw - radius, hh - radius, 0.0),
        (-(hw - radius), hh - radius, 90.0),
        (-(hw - radius), -(hh - radius), 180.0),
        (hw - radius, -(hh - radius), 270.0),
    ]
    for cx, cy, start in corners:
        for i in range(segments + 1):
            angle = math.radians(start + 90.0 * i / segments)
            vertices.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return vertices


def capsule(width: float, height: float, segments: int = 10) -> list[Point]:
    """A dome-topped, flat-bottomed silhouette -- the mole's body outline."""
    hw = width / 2.0
    hh = height / 2.0
    dome_radius = hw
    vertices: list[Point] = []
    for i in range(segments + 1):
        angle = math.pi + math.pi * i / segments
        vertices.append((dome_radius * math.cos(angle), -hh + dome_radius * math.sin(angle)))
    vertices.append((hw, hh))
    vertices.append((-hw, hh))
    return vertices


def heart(size: float) -> list[Point]:
    """A heart outline, used for the lives indicator."""
    vertices: list[Point] = []
    for i in range(24):
        t = 2 * math.pi * i / 24
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        vertices.append((x * size / 16.0, y * size / 16.0))
    return vertices


def arrow(length: float, width: float, head_length: float, head_width: float) -> list[Point]:
    """A rightward-pointing arrow polygon."""
    hw = width / 2.0
    hhw = head_width / 2.0
    tail = length - head_length
    return [
        (0.0, -hw),
        (tail, -hw),
        (tail, -hhw),
        (length, 0.0),
        (tail, hhw),
        (tail, hw),
        (0.0, hw),
    ]


def signed_area(points: Sequence[Point]) -> float:
    """Signed area via the shoelace formula.

    Positive means counter-clockwise in maths orientation, which appears
    *clockwise* on screen because y points down.  Used to normalise vertex
    winding before filling.
    """
    total = 0.0
    count = len(points)
    for index in range(count):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % count]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Even-odd point-in-polygon test by ray casting.

    A horizontal ray is cast to the right of the point; each edge it crosses
    toggles the inside flag.  Same parity rule that drives the scan-line fill.
    """
    x, y = point
    inside = False
    count = len(polygon)
    j = count - 1
    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside
