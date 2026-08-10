"""
Heads-up display and full-screen overlays.

Every glyph on screen comes from ``graphics.text``, the vector stroke font, so
the HUD is drawn by the same Bresenham rasteriser as the rest of the game.  The
panels, bars and hearts are polygons and circles from the same package.

The HUD is redrawn every frame (unlike the board) because almost all of it
changes -- but it is cheap: a few hundred short line segments.
"""

from __future__ import annotations

import math

import config
from graphics import circle as gcircle
from graphics import polygon as gpoly
from graphics import transform2d as T
from graphics.line import draw_line, draw_thick_line
from graphics.raster import Canvas, Rect
from graphics.text import draw_text, draw_text_shadowed, text_width


# ---------------------------------------------------------------------------
# Shared panel chrome
# ---------------------------------------------------------------------------


def draw_panel(
    canvas: Canvas,
    rect: Rect,
    fill: tuple[int, int, int] = config.COL_PANEL,
    edge: tuple[int, int, int] = config.COL_PANEL_EDGE,
    alpha: float = 0.92,
    corner: float = 12.0,
) -> None:
    """A rounded, semi-transparent panel with a bright edge."""
    outline = gpoly.rounded_rectangle(rect.width, rect.height, corner, segments=5)
    placed = T.apply(T.translation(*rect.center), outline)
    gpoly.fill_polygon_blend(canvas, placed, fill, alpha)
    gpoly.draw_polygon(canvas, placed, edge, thickness=2)


def draw_progress_bar(
    canvas: Canvas,
    rect: Rect,
    fraction: float,
    fill: tuple[int, int, int],
    background: tuple[int, int, int] = (18, 16, 34),
) -> None:
    """A horizontal bar, filled left to right by ``fraction`` in ``[0, 1]``."""
    fraction = max(0.0, min(1.0, fraction))
    canvas.fill_rect(rect, background)
    filled_width = int(rect.width * fraction)
    if filled_width > 0:
        canvas.fill_rect(Rect(rect.x_min, rect.y_min, rect.x_min + filled_width - 1, rect.y_max), fill)
        # Gloss line along the top of the filled section.
        draw_line(
            canvas,
            rect.x_min,
            rect.y_min + 1,
            rect.x_min + filled_width - 1,
            rect.y_min + 1,
            tuple(min(255, c + 60) for c in fill),
        )
    gpoly.draw_polygon(
        canvas,
        [
            (rect.x_min, rect.y_min),
            (rect.x_max, rect.y_min),
            (rect.x_max, rect.y_max),
            (rect.x_min, rect.y_max),
        ],
        config.COL_PANEL_EDGE,
        thickness=1,
    )


def draw_heart(canvas: Canvas, cx: float, cy: float, size: float, color: tuple[int, int, int]) -> None:
    """A heart-shaped polygon, used for the lives indicator."""
    local = gpoly.heart(size)
    placed = T.apply(T.translation(cx, cy), local)
    gpoly.fill_polygon(canvas, placed, color)
    gpoly.draw_polygon(canvas, placed, tuple(int(c * 0.55) for c in color), thickness=1)


# ---------------------------------------------------------------------------
# In-game HUD
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# HUD layout
#
# The stroke font draws a glyph of height `size` starting at the given y, so a
# label at y with size s occupies exactly [y, y + s].  Every row below is
# placed from that arithmetic rather than by eye -- the previous layout put the
# "TIME" label at y=20 with size 13 (ending at 33) and the clock at y=30, and
# the two drew straight through each other.
#
# Panel interior runs from y = 8 to y = HUD_HEIGHT - 6 = 90.
# ---------------------------------------------------------------------------

ROW_LABEL_Y = 15          # small caption row      -> ends at 27
ROW_LABEL_SIZE = 12
ROW_VALUE_Y = 31          # large readout row      -> ends at 59
ROW_VALUE_SIZE = 28
ROW_CLOCK_Y = 30          # clock digits           -> ends at 52
ROW_CLOCK_SIZE = 22
ROW_BAR_TOP = 62          # progress bars          -> end at 72
ROW_BAR_BOTTOM = 72
ROW_FOOTER_Y = 76         # hint / filter readout  -> ends at 88
ROW_FOOTER_SIZE = 12


def draw_hud(canvas: Canvas, state: dict) -> None:
    """Draw the top status bar.

    ``state`` is a plain dictionary rather than the engine object so this
    function stays testable in isolation and has no reason to reach back into
    game logic.
    """
    panel = Rect(10, 8, canvas.width - 11, config.HUD_HEIGHT - 6)
    draw_panel(canvas, panel, alpha=0.90)

    # ---- score ---------------------------------------------------------
    draw_text(canvas, "SCORE", 32, ROW_LABEL_Y, ROW_LABEL_SIZE, config.COL_TEXT_DIM, 1)
    draw_text_shadowed(
        canvas, f"{state['score']:06d}", 32, ROW_VALUE_Y, ROW_VALUE_SIZE, config.COL_ACCENT, (0, 0, 0), 2
    )

    # ---- combo ---------------------------------------------------------
    multiplier = state["multiplier"]
    combo_color = config.COL_GOOD if multiplier > 1 else config.COL_TEXT_DIM
    draw_text(canvas, "COMBO", 232, ROW_LABEL_Y, ROW_LABEL_SIZE, config.COL_TEXT_DIM, 1)
    draw_text_shadowed(
        canvas, f"x{multiplier}", 232, ROW_VALUE_Y, ROW_VALUE_SIZE, combo_color, (0, 0, 0), 2
    )

    # Combo charge bar: how close the player is to the next multiplier step.
    charge = (state["combo"] % config.COMBO_STEP) / float(config.COMBO_STEP)
    if multiplier >= config.COMBO_MAX:
        charge = 1.0
    draw_progress_bar(canvas, Rect(232, ROW_BAR_TOP, 316, ROW_BAR_BOTTOM), charge, combo_color)

    # ---- timer ---------------------------------------------------------
    remaining = max(0.0, state["time_left"])
    fraction = remaining / float(config.ROUND_SECONDS)
    if fraction > 0.5:
        timer_color = config.COL_GOOD
    elif fraction > 0.2:
        timer_color = config.COL_ACCENT
    else:
        timer_color = config.COL_BAD

    draw_text(
        canvas, "TIME", canvas.width // 2, ROW_LABEL_Y, ROW_LABEL_SIZE,
        config.COL_TEXT_DIM, 1, align="center",
    )
    draw_text_shadowed(
        canvas,
        f"{int(remaining) // 60}:{int(remaining) % 60:02d}",
        canvas.width // 2,
        ROW_CLOCK_Y,
        ROW_CLOCK_SIZE,
        timer_color,
        (0, 0, 0),
        2,
        align="center",
    )
    draw_progress_bar(
        canvas,
        Rect(canvas.width // 2 - 150, ROW_BAR_TOP, canvas.width // 2 + 150, ROW_BAR_BOTTOM),
        fraction,
        timer_color,
    )

    # ---- lives ---------------------------------------------------------
    draw_text(
        canvas, "LIVES", canvas.width - 32, ROW_LABEL_Y, ROW_LABEL_SIZE,
        config.COL_TEXT_DIM, 1, align="right",
    )
    for index in range(config.STARTING_LIVES):
        cx = canvas.width - 40 - index * 34
        filled = index < state["lives"]
        color = config.COL_BAD if filled else (62, 58, 84)
        # The most recently lost life pulses briefly so the loss is noticed.
        pulse = 1.0
        if filled and index == state["lives"] - 1 and state.get("life_pulse", 0.0) > 0.0:
            pulse = 1.0 + 0.25 * math.sin(state["life_pulse"] * 22.0)
        draw_heart(canvas, cx, 46, 14 * pulse, color)

    # ---- difficulty -----------------------------------------------------
    draw_text(
        canvas, state["difficulty"], canvas.width - 32, ROW_FOOTER_Y, ROW_FOOTER_SIZE,
        config.COL_TEXT_DIM, 1, align="right",
    )

    # ---- active effect readout ------------------------------------------
    # Centred below the timer bar.  This line only appears for half a second at
    # a time, so it shares its row with the pause hint rather than reserving
    # space that would crowd the permanently visible readouts.
    effect = state.get("effect_name")
    if effect:
        draw_text_shadowed(
            canvas,
            f"FILTER: {effect.upper()}",
            canvas.width // 2,
            ROW_FOOTER_Y,
            ROW_FOOTER_SIZE,
            state.get("effect_color", config.COL_TEXT),
            (0, 0, 0),
            1,
            align="center",
        )
    else:
        draw_text(
            canvas, "P  PAUSE", canvas.width // 2, ROW_FOOTER_Y, ROW_FOOTER_SIZE,
            config.COL_TEXT_DIM, 1, align="center",
        )


def draw_floating_texts(canvas: Canvas, floaters) -> None:
    """Draw the score popups that rise from each hit."""
    for floater in floaters:
        fade = floater.fade
        color = tuple(
            int(c * fade + 20 * (1.0 - fade)) for c in floater.color
        )
        draw_text_shadowed(
            canvas,
            floater.text,
            floater.x,
            floater.y,
            floater.size,
            color,
            (0, 0, 0),
            2,
            align="center",
        )


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------


def dim_screen(canvas: Canvas, amount: float = 0.62) -> None:
    """Darken the whole frame so an overlay reads clearly on top of it."""
    canvas.fill_rect_blend(Rect(0, 0, canvas.width - 1, canvas.height - 1), (6, 4, 18), amount)


def draw_title(canvas: Canvas, y: float, pulse: float) -> None:
    """The game title, with a subtle breathing scale."""
    size = 52 + 3 * math.sin(pulse * 2.0)
    draw_text_shadowed(
        canvas, "WHAC-A-MOLE", canvas.width // 2, y, size, config.COL_ACCENT, (60, 20, 10), 3, align="center"
    )


def draw_menu(canvas: Canvas, selected_difficulty: str, best_score: int, pulse: float) -> None:
    """The title screen.

    ``best_score`` is the best for the *currently selected* mode, so moving
    between EASY / NORMAL / HARD changes the target shown.
    """
    dim_screen(canvas, 0.55)
    draw_title(canvas, 58, pulse)

    panel = Rect(canvas.width // 2 - 330, 224, canvas.width // 2 + 330, 448)
    draw_panel(canvas, panel)

    draw_text(canvas, "DIFFICULTY", canvas.width // 2, 244, 15, config.COL_TEXT_DIM, 1, align="center")

    # Difficulty selector: three pills, the active one highlighted.
    for index, name in enumerate(config.DIFFICULTY_ORDER):
        active = name == selected_difficulty
        cx = canvas.width // 2 + (index - 1) * 150
        pill = Rect(cx - 66, 270, cx + 66, 308)
        draw_panel(
            canvas,
            pill,
            fill=config.COL_ACCENT if active else (40, 36, 70),
            edge=config.COL_ACCENT if active else config.COL_PANEL_EDGE,
            alpha=0.95,
            corner=10,
        )
        draw_text(
            canvas,
            name,
            cx,
            280,
            18,
            (30, 24, 12) if active else config.COL_TEXT_DIM,
            2,
            align="center",
        )

    draw_text(
        canvas, "LEFT / RIGHT ARROWS TO CHANGE", canvas.width // 2, 320, 12,
        config.COL_TEXT_DIM, 1, align="center",
    )

    # Best for the selected mode.  Always shown, so an unplayed mode reads as a
    # target of zero rather than silently vanishing.
    draw_text(
        canvas,
        f"BEST ON {selected_difficulty}",
        canvas.width // 2,
        352,
        13,
        config.COL_TEXT_DIM,
        1,
        align="center",
    )
    draw_text_shadowed(
        canvas,
        f"{best_score:06d}",
        canvas.width // 2,
        372,
        34,
        config.COL_GOOD if best_score > 0 else config.COL_TEXT_DIM,
        (0, 0, 0),
        2,
        align="center",
    )

    # Controls.  The hammer is mouse-only; the keyboard just starts and quits.
    lines = [
        "MOVE MOUSE  AIM          LEFT CLICK  SWING",
        "P  PAUSE                 ESC         QUIT",
    ]
    y = 470
    for text in lines:
        draw_text(canvas, text, canvas.width // 2, y, 13, config.COL_TEXT, 1, align="center")
        y += 24

    blink = 0.5 + 0.5 * math.sin(pulse * 4.0)
    color = tuple(int(config.COL_TEXT[i] * (0.45 + 0.55 * blink)) for i in range(3))
    draw_text(canvas, "PRESS ENTER TO START", canvas.width // 2, 534, 24, color, 2, align="center")


def draw_pause(canvas: Canvas, pulse: float) -> None:
    dim_screen(canvas, 0.66)
    panel = Rect(canvas.width // 2 - 220, canvas.height // 2 - 118, canvas.width // 2 + 220, canvas.height // 2 + 118)
    draw_panel(canvas, panel)
    draw_text_shadowed(
        canvas, "PAUSED", canvas.width // 2, canvas.height // 2 - 88, 44, config.COL_ACCENT, (0, 0, 0), 3, align="center"
    )
    # Left-aligned from a common origin, not centred: these are key/action
    # pairs, and centring each line separately makes the two columns wander.
    rows = [
        ("P / ESC", "RESUME"),
        ("R", "RESTART ROUND"),
        ("M", "MAIN MENU"),
    ]
    key_x = canvas.width // 2 - 150
    action_x = canvas.width // 2 - 20
    y = canvas.height // 2 - 24
    for key, action in rows:
        draw_text(canvas, key, key_x, y, 16, config.COL_ACCENT, 1)
        draw_text(canvas, action, action_x, y, 16, config.COL_TEXT, 1)
        y += 30


def draw_game_over(canvas: Canvas, summary: dict, pulse: float) -> None:
    """End-of-round screen with a breakdown of what the player did."""
    dim_screen(canvas, 0.70)

    panel = Rect(canvas.width // 2 - 330, 96, canvas.width // 2 + 330, 616)
    draw_panel(canvas, panel)

    title = "TIME UP" if summary["reason"] == "time" else "OUT OF LIVES"
    draw_text_shadowed(canvas, title, canvas.width // 2, 124, 40, config.COL_BAD, (0, 0, 0), 3, align="center")

    draw_text(canvas, "FINAL SCORE", canvas.width // 2, 184, 14, config.COL_TEXT_DIM, 1, align="center")
    draw_text_shadowed(
        canvas, f"{summary['score']:06d}", canvas.width // 2, 208, 52, config.COL_ACCENT, (0, 0, 0), 3, align="center"
    )

    if summary.get("is_new_best"):
        blink = 0.5 + 0.5 * math.sin(pulse * 6.0)
        draw_text(
            canvas,
            f"NEW {summary['difficulty']} HIGH SCORE",
            canvas.width // 2,
            274,
            20,
            tuple(int(config.COL_GOOD[i] * (0.4 + 0.6 * blink)) for i in range(3)),
            2,
            align="center",
        )
    else:
        draw_text(
            canvas,
            f"BEST ON {summary['difficulty']}   {summary.get('mode_best', 0):06d}",
            canvas.width // 2,
            276,
            16,
            config.COL_TEXT_DIM,
            1,
            align="center",
        )

    rows = [
        ("HITS", str(summary["hits"])),
        ("MISSES", str(summary["misses"])),
        ("ESCAPED", str(summary["escaped"])),
        ("ACCURACY", f"{summary['accuracy']:.0f}%"),
        ("BEST COMBO", f"x{summary['best_combo']}"),
        ("DIFFICULTY", summary["difficulty"]),
    ]
    y = 312
    for label, value in rows:
        draw_text(canvas, label, canvas.width // 2 - 250, y, 15, config.COL_TEXT_DIM, 1)
        draw_text(canvas, value, canvas.width // 2 + 250, y, 15, config.COL_TEXT, 1, align="right")
        draw_line(
            canvas,
            canvas.width // 2 - 250,
            y + 21,
            canvas.width // 2 + 250,
            y + 21,
            (54, 48, 88),
        )
        y += 32

    footer = [
        ("ENTER  PLAY AGAIN      M  MAIN MENU      ESC  QUIT", 13, config.COL_TEXT),
        (f"SCORE RECORDED IN {config.HIGHSCORE_FILE}", 11, config.COL_TEXT_DIM),
    ]
    y = 530
    for text, size, color in footer:
        draw_text(canvas, text, canvas.width // 2, y, size, color, 1, align="center")
        y += 26


def draw_countdown(canvas: Canvas, value: str, scale: float) -> None:
    """The 3-2-1-GO countdown before a round starts."""
    dim_screen(canvas, 0.35)
    size = 90 * scale
    draw_text_shadowed(
        canvas, value, canvas.width // 2, canvas.height // 2 - size / 2, size, config.COL_ACCENT, (0, 0, 0), 4, align="center"
    )


def draw_toast(canvas: Canvas, message: str, fade: float) -> None:
    """A transient notification strip at the bottom of the screen."""
    if fade <= 0.01:
        return
    width = int(text_width(message, 15) + 48)
    rect = Rect(canvas.width // 2 - width // 2, canvas.height - 56, canvas.width // 2 + width // 2, canvas.height - 22)
    draw_panel(canvas, rect, alpha=0.85 * fade, corner=8)
    color = tuple(int(c * fade) for c in config.COL_TEXT)
    draw_text(canvas, message, canvas.width // 2, canvas.height - 46, 15, color, 1, align="center")
