"""
A vector stroke font drawn with the project's own line rasteriser.

The brief rules out built-in drawing functions, and text is drawing.  So the
score, timer and every menu label are rendered here rather than by a font
engine.

Each glyph is a list of *strokes*, and each stroke is a polyline in a local
design grid 4 units wide and 6 units tall with the origin at the glyph's
top-left.  Rendering a character is therefore:

1.  Build a transformation matrix that scales the design grid to the requested
    pixel size and translates it to the pen position
    (``graphics.transform2d``).
2.  Map the stroke vertices through it.
3.  Rasterise each segment with Bresenham (``graphics.line``).

That makes the font a direct application of two syllabus topics rather than a
separate asset -- and it means the text scales to any size without bitmaps.
"""

from __future__ import annotations

from typing import Sequence

from .line import draw_line, draw_thick_line
from .raster import Canvas, Color
from .transform2d import apply, compose, scaling, translation

Point = tuple[float, float]
Glyph = list[list[Point]]

#: Design-grid dimensions.
GLYPH_WIDTH = 4.0
GLYPH_HEIGHT = 6.0
GLYPH_ADVANCE = 6.0        # includes 2 units of inter-character spacing


# ---------------------------------------------------------------------------
# Glyph table
# ---------------------------------------------------------------------------

_GLYPHS: dict[str, Glyph] = {
    " ": [],
    "A": [[(0, 6), (2, 0), (4, 6)], [(0.8, 3.6), (3.2, 3.6)]],
    "B": [
        [(0, 0), (0, 6)],
        [(0, 0), (3, 0), (4, 1), (4, 2), (3, 3), (0, 3)],
        [(3, 3), (4, 4), (4, 5), (3, 6), (0, 6)],
    ],
    "C": [[(4, 1), (3, 0), (1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5)]],
    "D": [[(0, 0), (0, 6)], [(0, 0), (3, 0), (4, 1), (4, 5), (3, 6), (0, 6)]],
    "E": [[(4, 0), (0, 0), (0, 6), (4, 6)], [(0, 3), (3, 3)]],
    "F": [[(4, 0), (0, 0), (0, 6)], [(0, 3), (3, 3)]],
    "G": [[(4, 1), (3, 0), (1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 3.4), (2.2, 3.4)]],
    "H": [[(0, 0), (0, 6)], [(4, 0), (4, 6)], [(0, 3), (4, 3)]],
    "I": [[(1, 0), (3, 0)], [(2, 0), (2, 6)], [(1, 6), (3, 6)]],
    "J": [[(4, 0), (4, 5), (3, 6), (1, 6), (0, 5)]],
    "K": [[(0, 0), (0, 6)], [(4, 0), (0, 3.2)], [(1.2, 2.4), (4, 6)]],
    "L": [[(0, 0), (0, 6), (4, 6)]],
    "M": [[(0, 6), (0, 0), (2, 2.6), (4, 0), (4, 6)]],
    "N": [[(0, 6), (0, 0), (4, 6), (4, 0)]],
    "O": [[(1, 0), (3, 0), (4, 1), (4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0)]],
    "P": [[(0, 6), (0, 0), (3, 0), (4, 1), (4, 2.4), (3, 3.4), (0, 3.4)]],
    "Q": [
        [(1, 0), (3, 0), (4, 1), (4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0)],
        [(2.4, 4.4), (4.3, 6.4)],
    ],
    "R": [[(0, 6), (0, 0), (3, 0), (4, 1), (4, 2.4), (3, 3.4), (0, 3.4)], [(2, 3.4), (4, 6)]],
    "S": [
        [
            (4, 1), (3, 0), (1, 0), (0, 1), (0, 2), (1, 3),
            (3, 3), (4, 4), (4, 5), (3, 6), (1, 6), (0, 5),
        ]
    ],
    "T": [[(0, 0), (4, 0)], [(2, 0), (2, 6)]],
    "U": [[(0, 0), (0, 5), (1, 6), (3, 6), (4, 5), (4, 0)]],
    "V": [[(0, 0), (2, 6), (4, 0)]],
    "W": [[(0, 0), (1, 6), (2, 3), (3, 6), (4, 0)]],
    "X": [[(0, 0), (4, 6)], [(4, 0), (0, 6)]],
    "Y": [[(0, 0), (2, 3), (4, 0)], [(2, 3), (2, 6)]],
    "Z": [[(0, 0), (4, 0), (0, 6), (4, 6)]],
    "0": [
        [(1, 0), (3, 0), (4, 1), (4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0)],
        [(0.6, 4.8), (3.4, 1.2)],
    ],
    "1": [[(0.8, 1.2), (2, 0), (2, 6)], [(0.8, 6), (3.2, 6)]],
    "2": [[(0, 1), (1, 0), (3, 0), (4, 1), (4, 2.2), (0, 6), (4, 6)]],
    "3": [[(0, 0), (4, 0), (1.8, 2.6)], [(1.4, 2.6), (3, 2.6), (4, 3.6), (4, 5), (3, 6), (1, 6), (0, 5)]],
    "4": [[(3, 6), (3, 0), (0, 4), (4, 4)]],
    "5": [[(4, 0), (0, 0), (0, 2.6), (3, 2.6), (4, 3.6), (4, 5), (3, 6), (1, 6), (0, 5)]],
    "6": [[(4, 1), (3, 0), (1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (1, 3), (0, 4)]],
    "7": [[(0, 0), (4, 0), (1.4, 6)]],
    "8": [
        [(1, 3), (0, 2), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2), (3, 3), (1, 3)],
        [(3, 3), (4, 4), (4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3)],
    ],
    "9": [[(0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0), (0, 1), (0, 2), (1, 3), (3, 3), (4, 2)]],
    ".": [[(1.8, 5.4), (2.2, 5.4), (2.2, 6), (1.8, 6), (1.8, 5.4)]],
    ",": [[(2.2, 5.2), (2.2, 6), (1.4, 6.9)]],
    ":": [[(1.9, 1.8), (2.3, 1.8), (2.3, 2.4), (1.9, 2.4), (1.9, 1.8)],
          [(1.9, 4.2), (2.3, 4.2), (2.3, 4.8), (1.9, 4.8), (1.9, 4.2)]],
    "-": [[(0.6, 3), (3.4, 3)]],
    "_": [[(0, 6.4), (4, 6.4)]],
    "+": [[(2, 1.4), (2, 4.6)], [(0.4, 3), (3.6, 3)]],
    "=": [[(0.4, 2.2), (3.6, 2.2)], [(0.4, 3.8), (3.6, 3.8)]],
    "/": [[(0, 6), (4, 0)]],
    "\\": [[(0, 0), (4, 6)]],
    "!": [[(2, 0), (2, 4)], [(1.8, 5.4), (2.2, 5.4), (2.2, 6), (1.8, 6), (1.8, 5.4)]],
    "?": [[(0, 1), (1, 0), (3, 0), (4, 1), (4, 2), (2, 3.6), (2, 4.2)],
          [(1.8, 5.4), (2.2, 5.4), (2.2, 6), (1.8, 6), (1.8, 5.4)]],
    "'": [[(2, 0), (2, 1.6)]],
    "\"": [[(1.4, 0), (1.4, 1.6)], [(2.6, 0), (2.6, 1.6)]],
    "(": [[(3, 0), (1.4, 1.6), (1.4, 4.4), (3, 6)]],
    ")": [[(1, 0), (2.6, 1.6), (2.6, 4.4), (1, 6)]],
    "[": [[(3, 0), (1.2, 0), (1.2, 6), (3, 6)]],
    "]": [[(1, 0), (2.8, 0), (2.8, 6), (1, 6)]],
    "<": [[(3.2, 1), (1, 3), (3.2, 5)]],
    ">": [[(0.8, 1), (3, 3), (0.8, 5)]],
    "*": [[(2, 1), (2, 5)], [(0.5, 1.9), (3.5, 4.1)], [(3.5, 1.9), (0.5, 4.1)]],
    "%": [
        [(0, 0.4), (1.2, 0.4), (1.2, 1.8), (0, 1.8), (0, 0.4)],
        [(4, 0), (0, 6)],
        [(2.8, 4.2), (4, 4.2), (4, 5.6), (2.8, 5.6), (2.8, 4.2)],
    ],
    "#": [[(1.2, 0.6), (0.6, 5.4)], [(3, 0.6), (2.4, 5.4)], [(0.2, 2), (3.8, 2)], [(0.2, 4), (3.8, 4)]],
    "x": [[(0.6, 1.6), (3.4, 4.6)], [(3.4, 1.6), (0.6, 4.6)]],
}


def glyph_strokes(character: str) -> Glyph:
    """Look up a character, falling back to upper case then to a blank box."""
    if character in _GLYPHS:
        return _GLYPHS[character]
    upper = character.upper()
    if upper in _GLYPHS:
        return _GLYPHS[upper]
    # Unknown character -> draw a hollow box so the gap is obvious rather than silent.
    return [[(0.4, 0.6), (3.6, 0.6), (3.6, 5.4), (0.4, 5.4), (0.4, 0.6)]]


def supported_characters() -> str:
    """Every character the font can render -- handy for a test sheet."""
    return "".join(sorted(_GLYPHS))


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def text_width(text: str, size: float, tracking: float = 0.0) -> float:
    """Width in pixels of ``text`` rendered at cap height ``size``."""
    if not text:
        return 0.0
    scale = size / GLYPH_HEIGHT
    return len(text) * (GLYPH_ADVANCE * scale + tracking) - tracking


def text_height(size: float) -> float:
    return float(size)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def draw_text(
    canvas: Canvas,
    text: str,
    x: float,
    y: float,
    size: float = 18.0,
    color: Color = (255, 255, 255),
    thickness: int = 1,
    align: str = "left",
    tracking: float = 0.0,
) -> float:
    """Draw ``text`` with its top-left (or top-centre / top-right) at ``(x, y)``.

    ``align`` accepts ``"left"``, ``"center"`` and ``"right"``.
    Returns the x coordinate just past the last glyph, so callers can chain
    runs of differently coloured text.
    """
    if not text:
        return x

    scale = size / GLYPH_HEIGHT
    advance = GLYPH_ADVANCE * scale + tracking

    total_width = text_width(text, size, tracking)
    if align == "center":
        pen_x = x - total_width / 2.0
    elif align == "right":
        pen_x = x - total_width
    else:
        pen_x = x

    for character in text:
        strokes = glyph_strokes(character)
        if strokes:
            # Scale the design grid, then move it to the pen position.
            matrix = compose(translation(pen_x, y), scaling(scale, scale))
            for stroke in strokes:
                screen_points = apply(matrix, stroke)
                for index in range(len(screen_points) - 1):
                    ax, ay = screen_points[index]
                    bx, by = screen_points[index + 1]
                    if thickness <= 1:
                        draw_line(canvas, ax, ay, bx, by, color)
                    else:
                        draw_thick_line(canvas, ax, ay, bx, by, color, thickness)
        pen_x += advance

    return pen_x


def draw_text_shadowed(
    canvas: Canvas,
    text: str,
    x: float,
    y: float,
    size: float = 18.0,
    color: Color = (255, 255, 255),
    shadow: Color = (0, 0, 0),
    thickness: int = 1,
    align: str = "left",
    offset: int = 2,
    tracking: float = 0.0,
) -> float:
    """Draw text twice -- once offset as a drop shadow, once in the fill colour.

    Worth the extra pass: the score has to stay readable while an edge-detection
    flash is inverting the colours underneath it.
    """
    draw_text(canvas, text, x + offset, y + offset, size, shadow, thickness, align, tracking)
    return draw_text(canvas, text, x, y, size, color, thickness, align, tracking)


def draw_text_lines(
    canvas: Canvas,
    lines: Sequence[str],
    x: float,
    y: float,
    size: float = 16.0,
    color: Color = (255, 255, 255),
    line_spacing: float = 1.6,
    thickness: int = 1,
    align: str = "left",
) -> float:
    """Draw several lines of text, returning the y position below the block."""
    step = size * line_spacing
    cursor_y = y
    for line in lines:
        draw_text(canvas, line, x, cursor_y, size, color, thickness, align)
        cursor_y += step
    return cursor_y
