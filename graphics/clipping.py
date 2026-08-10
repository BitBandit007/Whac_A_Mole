"""
Line and polygon clipping.

Syllabus reference: CSE 452, Week 8 -- "Implementation of clipping algorithms".

Three algorithms, each solving a different problem:

``cohen_sutherland``
    Line clipping by *region codes*.  Cheap rejection first: if both endpoints
    share an outside region the segment cannot possibly cross the window, so it
    is discarded without any intersection arithmetic at all.

``liang_barsky``
    Line clipping by *parametric* form.  The segment is written as
    ``P(t) = P0 + t (P1 - P0)`` and the four window edges become four
    inequalities in ``t``.  Solving them yields the visible interval directly,
    typically with fewer iterations than Cohen-Sutherland.

``sutherland_hodgman``
    Polygon clipping.  The subject polygon is passed through the four window
    edges one after another, each pass producing a new polygon.

Clipping is not decorative in this project.  When a mole rises out of a hole,
the part of it that is still underground is removed by clipping its silhouette
against the hole's rim -- that is what makes it look like it is *emerging*
rather than sliding over the board.
"""

from __future__ import annotations

from typing import Sequence

Point = tuple[float, float]
Window = tuple[float, float, float, float]   # (x_min, y_min, x_max, y_max)


# ---------------------------------------------------------------------------
# Cohen-Sutherland
# ---------------------------------------------------------------------------

INSIDE = 0b0000
LEFT = 0b0001
RIGHT = 0b0010
BOTTOM = 0b0100
TOP = 0b1000


def region_code(x: float, y: float, window: Window) -> int:
    """Compute the 4-bit Cohen-Sutherland outcode for a point.

    Each bit records which side of the window the point lies beyond.  Two
    points whose codes share a set bit are both outside the same edge, so the
    segment between them is trivially invisible.

    Note that ``TOP`` and ``BOTTOM`` follow *screen* orientation here: y grows
    downwards, so the smaller y is the top of the window.
    """
    x_min, y_min, x_max, y_max = window
    code = INSIDE
    if x < x_min:
        code |= LEFT
    elif x > x_max:
        code |= RIGHT
    if y < y_min:
        code |= TOP
    elif y > y_max:
        code |= BOTTOM
    return code


def cohen_sutherland(
    p0: Point, p1: Point, window: Window
) -> tuple[Point, Point] | None:
    """Clip a segment against a rectangular window.

    Returns the visible portion as a pair of endpoints, or ``None`` if the
    segment lies entirely outside.

    The loop performs two tests before doing any work:

    * ``code0 | code1 == 0`` -- both endpoints inside, accept immediately.
    * ``code0 & code1 != 0`` -- both endpoints beyond the same edge, reject
      immediately.

    Only when neither shortcut applies does it compute a single intersection,
    replace the outside endpoint with it, and try again.  Each iteration
    removes at least one outcode bit, so the loop terminates in at most four
    passes.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    x_min, y_min, x_max, y_max = window

    code0 = region_code(x0, y0, window)
    code1 = region_code(x1, y1, window)

    while True:
        if not (code0 | code1):
            return ((x0, y0), (x1, y1))         # trivially accepted
        if code0 & code1:
            return None                          # trivially rejected

        # Pick whichever endpoint is currently outside.
        outside_code = code0 if code0 else code1

        # Intersect with the corresponding window edge.  The guards on dy / dx
        # matter: a horizontal segment never meets a horizontal edge, and the
        # outcode test guarantees we only reach a branch whose denominator is
        # non-zero.
        if outside_code & TOP:
            x = x0 + (x1 - x0) * (y_min - y0) / (y1 - y0)
            y = y_min
        elif outside_code & BOTTOM:
            x = x0 + (x1 - x0) * (y_max - y0) / (y1 - y0)
            y = y_max
        elif outside_code & RIGHT:
            y = y0 + (y1 - y0) * (x_max - x0) / (x1 - x0)
            x = x_max
        else:   # LEFT
            y = y0 + (y1 - y0) * (x_min - x0) / (x1 - x0)
            x = x_min

        if outside_code == code0:
            x0, y0 = x, y
            code0 = region_code(x0, y0, window)
        else:
            x1, y1 = x, y
            code1 = region_code(x1, y1, window)


# ---------------------------------------------------------------------------
# Liang-Barsky
# ---------------------------------------------------------------------------


def liang_barsky(p0: Point, p1: Point, window: Window) -> tuple[Point, Point] | None:
    """Clip a segment using the Liang-Barsky parametric algorithm.

    Write the segment as ``P(t) = P0 + t . D`` for ``t`` in ``[0, 1]``.  Being
    inside the window means satisfying four inequalities of the form
    ``p_k . t <= q_k``:

    ==== ============ ==================
    k    p_k          q_k
    ==== ============ ==================
    1    -dx          x0 - x_min   (left)
    2    +dx          x_max - x0   (right)
    3    -dy          y0 - y_min   (top)
    4    +dy          y_max - y0   (bottom)
    ==== ============ ==================

    ``p_k < 0`` means the line enters through that edge, so it tightens
    ``t_enter``; ``p_k > 0`` means it leaves, tightening ``t_exit``.  If
    ``p_k == 0`` the line is parallel to that edge -- and if ``q_k < 0`` as
    well, it is parallel *and* outside, so it is rejected outright.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    x_min, y_min, x_max, y_max = window

    dx = x1 - x0
    dy = y1 - y0

    p = (-dx, dx, -dy, dy)
    q = (x0 - x_min, x_max - x0, y0 - y_min, y_max - y0)

    t_enter = 0.0
    t_exit = 1.0

    for pk, qk in zip(p, q):
        if abs(pk) < 1e-12:
            if qk < 0:
                return None            # parallel to this edge and outside it
            continue
        t = qk / pk
        if pk < 0:
            if t > t_exit:
                return None
            t_enter = max(t_enter, t)
        else:
            if t < t_enter:
                return None
            t_exit = min(t_exit, t)

    if t_enter > t_exit:
        return None

    return (
        (x0 + t_enter * dx, y0 + t_enter * dy),
        (x0 + t_exit * dx, y0 + t_exit * dy),
    )


# ---------------------------------------------------------------------------
# Sutherland-Hodgman polygon clipping
# ---------------------------------------------------------------------------


def _inside(point: Point, edge: str, window: Window) -> bool:
    """Is ``point`` on the visible side of one window edge?"""
    x, y = point
    x_min, y_min, x_max, y_max = window
    if edge == "left":
        return x >= x_min
    if edge == "right":
        return x <= x_max
    if edge == "top":
        return y >= y_min
    return y <= y_max      # "bottom"


def _edge_intersection(a: Point, b: Point, edge: str, window: Window) -> Point:
    """Where segment ``a -> b`` crosses one window edge."""
    ax, ay = a
    bx, by = b
    x_min, y_min, x_max, y_max = window

    if edge in ("left", "right"):
        x_edge = x_min if edge == "left" else x_max
        # Guard against a vertical segment (bx == ax); such a segment can only
        # reach this branch if it lies exactly on the edge.
        t = 0.0 if abs(bx - ax) < 1e-12 else (x_edge - ax) / (bx - ax)
        return (x_edge, ay + t * (by - ay))

    y_edge = y_min if edge == "top" else y_max
    t = 0.0 if abs(by - ay) < 1e-12 else (y_edge - ay) / (by - ay)
    return (ax + t * (bx - ax), y_edge)


def clip_polygon_against_edge(
    polygon: Sequence[Point], edge: str, window: Window
) -> list[Point]:
    """One pass of Sutherland-Hodgman against a single window edge.

    Walking the polygon edge by edge, each vertex pair falls into one of four
    cases:

    ============= ============= ==========================
    previous      current       output
    ============= ============= ==========================
    inside        inside        current
    inside        outside       intersection
    outside       inside        intersection, then current
    outside       outside       nothing
    ============= ============= ==========================
    """
    if not polygon:
        return []

    output: list[Point] = []
    previous = polygon[-1]
    previous_inside = _inside(previous, edge, window)

    for current in polygon:
        current_inside = _inside(current, edge, window)
        if current_inside:
            if not previous_inside:
                output.append(_edge_intersection(previous, current, edge, window))
            output.append(current)
        elif previous_inside:
            output.append(_edge_intersection(previous, current, edge, window))
        previous = current
        previous_inside = current_inside

    return output


def sutherland_hodgman(polygon: Sequence[Point], window: Window) -> list[Point]:
    """Clip a polygon against a rectangular window.

    The subject polygon is fed through the four edges in turn; the output of
    one stage is the input of the next.  Returns an empty list if nothing
    survives.

    Caveat worth knowing for the viva: this algorithm assumes a *convex* clip
    window (a rectangle qualifies).  A concave subject polygon can come out
    with degenerate connecting edges along the boundary -- harmless when
    filling, visible when stroking the outline.
    """
    result = list(polygon)
    for edge in ("left", "right", "top", "bottom"):
        result = clip_polygon_against_edge(result, edge, window)
        if not result:
            return []
    return result


# ---------------------------------------------------------------------------
# Helpers used by the renderer
# ---------------------------------------------------------------------------


def clip_polyline(
    points: Sequence[Point], window: Window, closed: bool = False
) -> list[tuple[Point, Point]]:
    """Clip every segment of a polyline, returning the surviving segments."""
    segments: list[tuple[Point, Point]] = []
    count = len(points)
    if count < 2:
        return segments
    limit = count if closed else count - 1
    for index in range(limit):
        a = points[index]
        b = points[(index + 1) % count]
        clipped = cohen_sutherland(a, b, window)
        if clipped is not None:
            segments.append(clipped)
    return segments


def clip_point(point: Point, window: Window) -> bool:
    """Point-in-window test -- the degenerate case of clipping."""
    return region_code(point[0], point[1], window) == INSIDE
