"""
Line scan-conversion:  DDA and Bresenham.

Syllabus reference: CSE 452, Week 1 -- "Implement DDA line & Bresenham's line
drawing algorithm".

Both algorithms are implemented as *point generators* first
(:func:`dda_points`, :func:`bresenham_points`) and only then wrapped in
canvas-drawing helpers.  Separating generation from plotting means the unit
tests in ``tests/test_line.py`` can compare the two algorithms pixel-for-pixel
without a framebuffer, and the same generators can be reused by the stroke
font in ``graphics.text``.

Why two algorithms?
-------------------
DDA is the straightforward one: compute the slope, step along the major axis
in unit increments and round.  It needs floating-point arithmetic and one
round per pixel.

Bresenham removes the floating point entirely.  It keeps an integer *decision
variable* that tracks twice the difference between the true line and the
midpoint between the two candidate pixels; the sign of that variable chooses
the next pixel.  Same result, integer-only arithmetic, which is why hardware
rasterisers use it.
"""

from __future__ import annotations

from typing import Iterator, Sequence

from .raster import Canvas, Color


# ---------------------------------------------------------------------------
# DDA -- Digital Differential Analyzer
# ---------------------------------------------------------------------------


def dda_points(x0: float, y0: float, x1: float, y1: float) -> list[tuple[int, int]]:
    """Generate the pixels of a line with the DDA algorithm.

    The number of steps is the larger of ``|dx|`` and ``|dy|`` so that the line
    is sampled along its *major* axis; sampling along the minor axis would
    leave gaps.  Each step advances by the fractional increments
    ``dx/steps`` and ``dy/steps`` and rounds to the nearest pixel.
    """
    x0f, y0f, x1f, y1f = float(x0), float(y0), float(x1), float(y1)
    dx = x1f - x0f
    dy = y1f - y0f

    steps = int(round(max(abs(dx), abs(dy))))
    if steps == 0:
        return [(int(round(x0f)), int(round(y0f)))]

    x_increment = dx / steps
    y_increment = dy / steps

    points: list[tuple[int, int]] = []
    x, y = x0f, y0f
    for _ in range(steps + 1):
        points.append((int(round(x)), int(round(y))))
        x += x_increment
        y += y_increment
    return points


# ---------------------------------------------------------------------------
# Bresenham
# ---------------------------------------------------------------------------


def bresenham_points(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Generate the pixels of a line with Bresenham's algorithm.

    This is the generalised integer form that handles all eight octants
    without any special-casing of the slope:

    * ``dx`` and ``dy`` are the absolute deltas,
    * ``sx`` and ``sy`` are the step directions (+1 or -1),
    * ``err`` is the running decision variable, initialised to ``dx - dy``.

    At every pixel the algorithm asks whether doubling the error crosses the
    ``-dy`` / ``+dx`` thresholds; each crossing commits a step along that axis.
    No division, no floating point, no rounding.
    """
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    points: list[tuple[int, int]] = []
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        err2 = 2 * err
        if err2 > -dy:
            err -= dy
            x += sx
        if err2 < dx:
            err += dx
            y += sy
    return points


def bresenham_iter(x0: int, y0: int, x1: int, y1: int) -> Iterator[tuple[int, int]]:
    """Lazy variant of :func:`bresenham_points` -- avoids building a list."""
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            return
        err2 = 2 * err
        if err2 > -dy:
            err -= dy
            x += sx
        if err2 < dx:
            err += dx
            y += sy


# ---------------------------------------------------------------------------
# Canvas-facing helpers
# ---------------------------------------------------------------------------

#: Algorithm selector accepted by :func:`draw_line`.
LINE_ALGORITHMS = ("bresenham", "dda")


def draw_line(
    canvas: Canvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: Color,
    algorithm: str = "bresenham",
) -> None:
    """Rasterise a single-pixel-wide line onto ``canvas``.

    ``algorithm`` selects between the two implementations.  The game uses
    Bresenham; the DDA path exists so the report and the in-game debug overlay
    can show that both produce the same figure.
    """
    if algorithm == "dda":
        points = dda_points(x0, y0, x1, y1)
    elif algorithm == "bresenham":
        points = bresenham_points(x0, y0, x1, y1)
    else:
        raise ValueError(f"unknown line algorithm {algorithm!r}; expected one of {LINE_ALGORITHMS}")
    for px, py in points:
        canvas.put_pixel(px, py, color)


def draw_polyline(
    canvas: Canvas,
    points: Sequence[tuple[float, float]],
    color: Color,
    closed: bool = False,
    algorithm: str = "bresenham",
) -> None:
    """Draw a chain of connected line segments."""
    if len(points) < 2:
        return
    for index in range(len(points) - 1):
        (x0, y0), (x1, y1) = points[index], points[index + 1]
        draw_line(canvas, x0, y0, x1, y1, color, algorithm)
    if closed:
        (x0, y0), (x1, y1) = points[-1], points[0]
        draw_line(canvas, x0, y0, x1, y1, color, algorithm)


def draw_thick_line(
    canvas: Canvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: Color,
    thickness: int = 2,
) -> None:
    """Draw a line ``thickness`` pixels wide.

    The line is widened by running Bresenham repeatedly along parallel offsets
    of the segment's unit normal.  This keeps the width perpendicular to the
    line for every slope, unlike the cheaper trick of simply stamping extra
    pixels above and below each plotted point.
    """
    thickness = max(1, int(thickness))
    if thickness == 1:
        draw_line(canvas, x0, y0, x1, y1, color)
        return

    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-9:
        canvas.put_pixel(x0, y0, color)
        return

    # Unit normal to the segment.
    nx = -dy / length
    ny = dx / length

    half = (thickness - 1) / 2.0
    offset = -half
    while offset <= half + 1e-9:
        ox = nx * offset
        oy = ny * offset
        draw_line(canvas, x0 + ox, y0 + oy, x1 + ox, y1 + oy, color)
        offset += 0.5   # half-pixel stepping avoids gaps on diagonal lines


def draw_dashed_line(
    canvas: Canvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: Color,
    dash: int = 8,
    gap: int = 6,
) -> None:
    """Draw a dashed line by masking the generated Bresenham pixel run."""
    period = max(1, dash + gap)
    for index, (px, py) in enumerate(bresenham_iter(x0, y0, x1, y1)):
        if index % period < dash:
            canvas.put_pixel(px, py, color)


def draw_rect_outline(
    canvas: Canvas,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    color: Color,
    thickness: int = 1,
) -> None:
    """Draw a rectangle border out of four Bresenham lines."""
    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]
    for index in range(4):
        (ax, ay) = corners[index]
        (bx, by) = corners[(index + 1) % 4]
        if thickness <= 1:
            draw_line(canvas, ax, ay, bx, by, color)
        else:
            draw_thick_line(canvas, ax, ay, bx, by, color, thickness)
