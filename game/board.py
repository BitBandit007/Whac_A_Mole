"""
Board rendering.

The board is split into two passes for a reason:

``render_background``
    Everything that never changes -- backdrop, play field, borders, the back
    half of every hole.  It is rasterised **once** into its own canvas at
    startup and copied over the live canvas each frame.  Scan-converting the
    whole board with per-pixel Python every frame would not hold 60 FPS; doing
    it once costs a few milliseconds at launch.

``draw_hole_fronts``
    The front lip of each hole, drawn *after* the moles.  That single ordering
    trick is what makes a mole look like it is standing inside a hole rather
    than in front of one.

Both DDA and Bresenham are exercised on screen: the diagonal backdrop stripes
are drawn with DDA, everything else with Bresenham.  They are indistinguishable
in the result, which is precisely the point worth demonstrating.
"""

from __future__ import annotations

import math

import config
from graphics import circle as gcircle
from graphics import polygon as gpoly
from graphics.line import draw_dashed_line, draw_line, draw_thick_line
from graphics.raster import Canvas, Rect

from .entities import Hole


# ---------------------------------------------------------------------------
# Static background
# ---------------------------------------------------------------------------


def render_background(canvas: Canvas, holes: list[Hole]) -> None:
    """Draw every static element of the board into ``canvas``."""
    _draw_backdrop(canvas)
    _draw_play_field(canvas)
    _draw_field_border(canvas)
    for hole in holes:
        _draw_hole_back(canvas, hole)


def _draw_backdrop(canvas: Canvas) -> None:
    """Night-sky gradient with faint diagonal stripes drawn using DDA."""
    canvas.vertical_gradient(
        Rect(0, 0, canvas.width - 1, canvas.height - 1),
        config.COL_BACKDROP_TOP,
        config.COL_BACKDROP_BOTTOM,
    )

    # Diagonal stripes.  Drawn with the DDA algorithm specifically so that both
    # line rasterisers appear in the finished frame.
    stripe = (255, 255, 255)
    for x in range(-canvas.height, canvas.width, 46):
        for offset in (0, 1):
            points = _dda_stripe(x + offset, canvas)
            for px, py in points:
                canvas.blend_pixel(px, py, stripe, 0.030)


def _dda_stripe(x_start: int, canvas: Canvas) -> list[tuple[int, int]]:
    from graphics.line import dda_points

    return dda_points(x_start, 0, x_start + canvas.height, canvas.height - 1)


def _draw_play_field(canvas: Canvas) -> None:
    """The green playing surface, with mown stripes and a dashed centre grid."""
    field = Rect(config.FIELD_LEFT, config.FIELD_TOP, config.FIELD_RIGHT, config.FIELD_BOTTOM)

    # Alternating mown stripes give the eye a reference for the perspective and
    # give the edge detectors something structured to find.
    band_height = 26
    y = field.y_min
    light = True
    while y <= field.y_max:
        color = config.COL_FIELD if light else config.COL_FIELD_DARK
        canvas.fill_rect(Rect(field.x_min, y, field.x_max, min(y + band_height - 1, field.y_max)), color)
        y += band_height
        light = not light

    # Cell separators, dashed so they read as guides rather than walls.
    cell_width = (config.FIELD_RIGHT - config.FIELD_LEFT) / config.GRID_COLS
    cell_height = (config.FIELD_BOTTOM - config.FIELD_TOP) / config.GRID_ROWS

    for col in range(1, config.GRID_COLS):
        x = config.FIELD_LEFT + cell_width * col
        draw_dashed_line(canvas, x, field.y_min + 6, x, field.y_max - 6, config.COL_FIELD_LINE, 10, 8)

    for row in range(1, config.GRID_ROWS):
        y = config.FIELD_TOP + cell_height * row
        draw_dashed_line(canvas, field.x_min + 6, y, field.x_max - 6, y, config.COL_FIELD_LINE, 10, 8)


def _draw_field_border(canvas: Canvas) -> None:
    """A double frame with decorative studs at the corners and mid-spans."""
    outer = Rect(
        config.FIELD_LEFT - 8,
        config.FIELD_TOP - 8,
        config.FIELD_RIGHT + 8,
        config.FIELD_BOTTOM + 8,
    )

    for index, (rect, color, thickness) in enumerate(
        (
            (outer, config.COL_BORDER_DARK, 6),
            (outer.inflated(-5), config.COL_BORDER, 3),
        )
    ):
        corners = rect.corners()
        for corner_index in range(4):
            ax, ay = corners[corner_index]
            bx, by = corners[(corner_index + 1) % 4]
            draw_thick_line(canvas, ax, ay, bx, by, color, thickness)

    # Studs: filled circles at the corners and at the midpoint of each edge.
    stud_positions = list(outer.corners())
    cx, cy = outer.center
    stud_positions += [
        (cx, outer.y_min),
        (cx, outer.y_max),
        (outer.x_min, cy),
        (outer.x_max, cy),
    ]
    for sx, sy in stud_positions:
        gcircle.fill_circle(canvas, sx, sy, 9, config.COL_BORDER_DARK)
        gcircle.fill_circle(canvas, sx, sy, 6, config.COL_BORDER)
        gcircle.fill_circle(canvas, sx - 2, sy - 2, 2, (255, 250, 220))


def _draw_hole_back(canvas: Canvas, hole: Hole) -> None:
    """The hole opening and its rear rim.

    Drawn as a stack of concentric ellipses: a soft shadow on the grass, a
    raised rim, then the dark opening with a vertical gradient inside it so the
    hole reads as having depth.
    """
    # Shadow cast on the grass around the mound.
    for spread in range(7, 0, -1):
        gcircle.fill_ellipse_blend(
            canvas,
            hole.cx,
            hole.cy + 5,
            hole.rx + spread * 2,
            hole.ry + spread,
            (0, 0, 0),
            0.035,
        )

    # Raised earth rim.
    gcircle.fill_ellipse(canvas, hole.cx, hole.cy, hole.rx + 9, hole.ry + 7, config.COL_HOLE_RIM)
    gcircle.fill_ellipse(canvas, hole.cx, hole.cy - 2, hole.rx + 5, hole.ry + 4, config.COL_HOLE_RIM_LIGHT)

    # The opening itself, shaded from near-black at the back to a lift at the front.
    gcircle.fill_ellipse(canvas, hole.cx, hole.cy, hole.rx, hole.ry, config.COL_HOLE)
    for step in range(1, 5):
        factor = step / 5.0
        gcircle.fill_ellipse_blend(
            canvas,
            hole.cx,
            hole.cy + hole.ry * 0.30 * factor,
            hole.rx * (1.0 - 0.16 * factor),
            hole.ry * (1.0 - 0.30 * factor),
            (48, 34, 26),
            0.28,
        )


# ---------------------------------------------------------------------------
# Foreground pass
# ---------------------------------------------------------------------------


def draw_hole_fronts(canvas: Canvas, holes: list[Hole]) -> None:
    """Draw the near lip of every hole, on top of whatever is standing in it."""
    for hole in holes:
        _draw_hole_front(canvas, hole)


def _draw_hole_front(canvas: Canvas, hole: Hole) -> None:
    """The front half of the rim: an arc plus the earth below it.

    Only the lower half of the ellipse is redrawn, so the rim occludes the
    bottom of a mole while leaving the top of the hole open.
    """
    # Earth in front of the hole: the crescent between the rim ellipse and the
    # opening ellipse, restricted to the lower half.
    canvas.push_clip(
        Rect(
            int(hole.cx - hole.rx - 14),
            int(hole.rim_y),
            int(hole.cx + hole.rx + 14),
            int(hole.cy + hole.ry + 14),
        )
    )
    try:
        gcircle.fill_ellipse(canvas, hole.cx, hole.cy, hole.rx + 9, hole.ry + 7, config.COL_HOLE_RIM)
        gcircle.fill_ellipse(
            canvas, hole.cx, hole.cy + 2, hole.rx + 3, hole.ry + 2, config.COL_HOLE_RIM_LIGHT
        )
        gcircle.fill_ellipse(canvas, hole.cx, hole.cy, hole.rx, hole.ry, config.COL_HOLE)
    finally:
        canvas.pop_clip()

    # Front rim highlight, drawn as an arc so it fades at the sides.
    gcircle.draw_arc(
        canvas,
        hole.cx,
        hole.cy,
        int(hole.rx + 6),
        20.0,
        160.0,
        config.COL_HOLE_RIM_LIGHT,
        thickness=2,
    )


def draw_type_ring(canvas: Canvas, hole: Hole, color: tuple[int, int, int], phase: float) -> None:
    """A rotating dashed ring identifying an effect mole's filter type."""
    segments = 12
    # Sit the ring clearly outside the raised rim (rx + 9), otherwise the hole
    # front pass drawn later covers most of it and it reads as scratches.
    radius_x = hole.rx + 20
    radius_y = hole.ry + 15
    for index in range(segments):
        if index % 2:
            continue
        start = phase + index * (360.0 / segments)
        _draw_elliptical_arc(canvas, hole.cx, hole.cy, radius_x, radius_y, start, start + 22.0, color)


def _draw_elliptical_arc(
    canvas: Canvas,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    start_deg: float,
    end_deg: float,
    color: tuple[int, int, int],
) -> None:
    """Approximate an elliptical arc by chaining short Bresenham segments."""
    steps = max(2, int(abs(end_deg - start_deg) / 4.0))
    previous: tuple[float, float] | None = None
    for step in range(steps + 1):
        angle = math.radians(start_deg + (end_deg - start_deg) * step / steps)
        point = (cx + rx * math.cos(angle), cy + ry * math.sin(angle))
        if previous is not None:
            draw_thick_line(canvas, previous[0], previous[1], point[0], point[1], color, 2)
        previous = point
