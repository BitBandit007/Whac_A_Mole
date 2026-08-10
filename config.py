"""
Central configuration for the Whac-A-Mole project.

Every tunable number in the game lives here so that gameplay can be balanced
without touching the algorithm modules.  Keeping constants out of the
``graphics`` and ``imaging`` packages also keeps those packages reusable as
standalone libraries -- they are pure implementations of the CSE 452 syllabus
algorithms and know nothing about this particular game.

Course reference : CSE 452 - Graphics & Image Processing Lab
Institution      : Bangladesh University of Business and Technology (BUBT)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Window / framebuffer
# ---------------------------------------------------------------------------

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
WINDOW_TITLE = "Whac-A-Mole  |  CSE 452 Graphics & Image Processing"
TARGET_FPS = 60

# ---------------------------------------------------------------------------
# Colour palette (R, G, B)
#
# The palette is deliberately high-contrast: the image-processing filters
# (edge detection in particular) produce far more legible output when the
# source frame has strong luminance separation between neighbouring regions.
# ---------------------------------------------------------------------------

COL_BACKDROP_TOP = (18, 16, 42)
COL_BACKDROP_BOTTOM = (44, 24, 68)
COL_FIELD = (34, 92, 58)
COL_FIELD_DARK = (24, 70, 44)
COL_FIELD_LINE = (58, 132, 84)
COL_BORDER = (240, 214, 122)
COL_BORDER_DARK = (172, 140, 60)

COL_HOLE = (26, 18, 14)
COL_HOLE_RIM = (96, 66, 44)
COL_HOLE_RIM_LIGHT = (140, 100, 68)

COL_PANEL = (26, 22, 52)
COL_PANEL_EDGE = (108, 96, 190)
COL_TEXT = (238, 238, 250)
COL_TEXT_DIM = (150, 148, 180)
COL_ACCENT = (255, 208, 84)
COL_GOOD = (108, 224, 148)
COL_BAD = (238, 92, 96)

COL_MOLE_BODY = (150, 102, 62)
COL_MOLE_BODY_DARK = (112, 74, 44)
COL_MOLE_BELLY = (214, 176, 130)
COL_MOLE_EYE = (250, 250, 250)
COL_MOLE_PUPIL = (24, 18, 16)
COL_MOLE_NOSE = (232, 120, 128)

COL_HAMMER_HEAD = (206, 66, 62)
COL_HAMMER_HEAD_DARK = (150, 40, 40)
COL_HAMMER_BAND = (238, 238, 244)
COL_HAMMER_HANDLE = (156, 106, 58)
COL_HAMMER_HANDLE_DARK = (110, 72, 38)

# ---------------------------------------------------------------------------
# Play field geometry
#
# ``FIELD_RECT`` is the clipping window used by every clipping routine in
# ``graphics.clipping``.  Nothing is allowed to draw outside it.
# ---------------------------------------------------------------------------

HUD_HEIGHT = 96

FIELD_LEFT = 40
FIELD_TOP = HUD_HEIGHT + 20
FIELD_RIGHT = WINDOW_WIDTH - 40
FIELD_BOTTOM = WINDOW_HEIGHT - 34
FIELD_RECT = (FIELD_LEFT, FIELD_TOP, FIELD_RIGHT, FIELD_BOTTOM)

GRID_COLS = 3
GRID_ROWS = 3
HOLE_RX = 74           # hole ellipse horizontal radius
HOLE_RY = 30           # hole ellipse vertical radius (perspective squash)

# ---------------------------------------------------------------------------
# Mole behaviour
# ---------------------------------------------------------------------------

MOLE_RADIUS = 44
# Rise and fall are the emergence animation, not the time a mole is hittable
# (that is ``up_time`` below).  Slow enough to read as a mole climbing out of
# the hole -- and slow enough to actually see the clipping against the rim.
MOLE_RISE_TIME = 0.34          # seconds spent rising and scaling out of the hole
MOLE_FALL_TIME = 0.26          # seconds spent dropping back down
MOLE_HIT_TIME = 0.22           # squash animation after a successful hit

# Rarity weights used when choosing which kind of mole to spawn.
MOLE_TYPE_WEIGHTS = {
    "normal": 46,
    "blur": 12,
    "sharpen": 12,
    "edge": 12,
    "emboss": 12,
    "golden": 6,
}

MOLE_SCORE = {
    "normal": 10,
    "blur": 15,
    "sharpen": 15,
    "edge": 15,
    "emboss": 15,
    "golden": 50,
}

# Ring / bandana colour that identifies each mole type on screen.
MOLE_TYPE_COLOR = {
    "normal": (150, 102, 62),
    "blur": (96, 196, 236),
    "sharpen": (250, 156, 60),
    "edge": (128, 232, 120),
    "emboss": (206, 132, 244),
    "golden": (255, 208, 84),
}

MOLE_TYPE_LABEL = {
    "normal": "MOLE",
    "blur": "BLUR",
    "sharpen": "SHARP",
    "edge": "EDGE",
    "emboss": "EMBOSS",
    "golden": "GOLD",
}

# ---------------------------------------------------------------------------
# Difficulty levels
#
# ``up_time``      : how long a mole stays visible before escaping
# ``spawn_delay``  : gap between spawns (min, max) in seconds
# ``max_active``   : how many moles may be up simultaneously
# ---------------------------------------------------------------------------

DIFFICULTIES = {
    "EASY": {"up_time": (2.10, 2.60), "spawn_delay": (0.95, 1.45), "max_active": 2},
    "NORMAL": {"up_time": (1.50, 1.95), "spawn_delay": (0.70, 1.10), "max_active": 2},
    "HARD": {"up_time": (1.05, 1.40), "spawn_delay": (0.48, 0.80), "max_active": 3},
}
DIFFICULTY_ORDER = ["EASY", "NORMAL", "HARD"]

ROUND_SECONDS = 60
STARTING_LIVES = 5
COMBO_STEP = 4          # every N consecutive hits raises the multiplier
COMBO_MAX = 5           # multiplier ceiling

# ---------------------------------------------------------------------------
# Hammer
# ---------------------------------------------------------------------------

HAMMER_SWING_TIME = 0.16        # seconds for the downward strike
HAMMER_RECOVER_TIME = 0.13      # seconds to swing back up
HAMMER_REST_ANGLE = -32.0       # degrees, resting tilt
HAMMER_STRIKE_ANGLE = 42.0      # degrees at full impact
HAMMER_HIT_RADIUS = 44          # impact radius tested against mole centres

#: Mouse sensitivity.  The pointer is an absolute device, so this is applied as
#: a gain about the centre of the play field: the hammer moves
#: ``MOUSE_SENSITIVITY`` pixels for every pixel the pointer moves, and is then
#: clamped to the field.  1.0 would track the pointer exactly; above 1.0 means
#: less physical movement is needed to cross the board.
MOUSE_SENSITIVITY = 1.75

# ---------------------------------------------------------------------------
# Screen effects (the "effect mole" payload)
# ---------------------------------------------------------------------------

EFFECT_DURATION = 0.55          # seconds a filter flash stays on screen
EFFECT_PEAK = 0.90              # maximum blend strength of the filtered frame

# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

#: Plain-text high-score table, written next to the project root.  Text rather
#: than JSON so it can be opened and read directly during the demo.
HIGHSCORE_FILE = "highscores.txt"

#: Where ``tools/generate_report.py`` writes its figures.
CAPTURE_DIR = "captures"
