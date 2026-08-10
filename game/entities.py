"""
Game objects: holes, moles, the hammer and hit particles.

Everything here follows one rule: an object stores its geometry once, in its
own **local coordinate system**, and is placed on screen by a transformation
matrix rebuilt each frame.  No object ever mutates its own vertex list to move.
That is the point the project brief makes about using matrix transformations
rather than "simply redrawing objects at new coordinates", and it is what makes
the animations composable -- a mole can rise, squash and tilt at once because
those are three matrices multiplied together.

Local coordinate conventions
----------------------------
* **Mole** -- origin at the mole's feet, body extending upward (negative y).
  A mole is placed by translating its origin to the hole and scaling it up.
* **Hammer** -- origin at the centre of the head, handle extending down-right.
  The swing is a rotation about a pivot just below the head.
* **Particle** -- origin at the particle centre.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

import config
from graphics import circle as gcircle
from graphics import polygon as gpoly
from graphics import transform2d as T
from graphics.line import draw_line, draw_thick_line
from graphics.raster import Canvas, Rect

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Holes
# ---------------------------------------------------------------------------


@dataclass
class Hole:
    """One hole in the game board.

    ``index`` is the 0-based grid position, which doubles as the keyboard
    shortcut (keys 1-9 map to holes 0-8).
    """

    index: int
    cx: float
    cy: float
    rx: float = config.HOLE_RX
    ry: float = config.HOLE_RY

    @property
    def rim_y(self) -> float:
        """The y coordinate below which a mole is hidden inside the hole.

        Slightly *above* the ellipse centre so the mole is cut off just behind
        the front lip of the hole, which is what sells the depth.
        """
        return self.cy + self.ry * 0.30

    def clip_window(self) -> tuple[float, float, float, float]:
        """The clipping window for a mole standing in this hole.

        Intersected with the play field, this is what
        ``graphics.clipping.sutherland_hodgman`` receives -- the reason a mole
        looks like it is emerging rather than sliding over the board.
        """
        return (
            max(config.FIELD_LEFT, self.cx - self.rx - 40),
            config.FIELD_TOP,
            min(config.FIELD_RIGHT, self.cx + self.rx + 40),
            self.rim_y,
        )


def build_holes() -> list[Hole]:
    """Lay out the grid of holes evenly across the play field."""
    holes: list[Hole] = []
    field_width = config.FIELD_RIGHT - config.FIELD_LEFT
    field_height = config.FIELD_BOTTOM - config.FIELD_TOP

    cell_width = field_width / config.GRID_COLS
    cell_height = field_height / config.GRID_ROWS

    for row in range(config.GRID_ROWS):
        for col in range(config.GRID_COLS):
            cx = config.FIELD_LEFT + cell_width * (col + 0.5)
            # Push the row centre slightly down inside its cell so the mole,
            # which grows upward, has room above it.
            cy = config.FIELD_TOP + cell_height * (row + 0.66)
            holes.append(Hole(index=row * config.GRID_COLS + col, cx=cx, cy=cy))
    return holes


# ---------------------------------------------------------------------------
# Moles
# ---------------------------------------------------------------------------

MOLE_HIDDEN = "hidden"
MOLE_RISING = "rising"
MOLE_UP = "up"
MOLE_FALLING = "falling"
MOLE_HIT = "hit"


@dataclass
class Mole:
    """A mole occupying one hole, with its own small state machine.

    States advance in one direction only::

        hidden -> rising -> up -> falling -> hidden
                              \\-> hit -> hidden

    ``pop`` is the single number the renderer needs: 0 means fully underground,
    1 means fully out.  Every state derives it differently, which is what gives
    each phase its own feel.
    """

    hole: Hole
    kind: str = "normal"
    state: str = MOLE_HIDDEN
    timer: float = 0.0
    up_duration: float = 1.0
    was_hit: bool = False
    escaped: bool = False

    # ---- lifecycle ------------------------------------------------------

    def spawn(self, kind: str, up_duration: float) -> None:
        self.kind = kind
        self.state = MOLE_RISING
        self.timer = 0.0
        self.up_duration = up_duration
        self.was_hit = False
        self.escaped = False

    def update(self, dt: float) -> None:
        """Advance the state machine.  Sets ``escaped`` on the frame a mole
        finishes falling without having been hit -- the engine reads that flag
        to deduct a life."""
        if self.state == MOLE_HIDDEN:
            return

        self.timer += dt

        if self.state == MOLE_RISING:
            if self.timer >= config.MOLE_RISE_TIME:
                self.state = MOLE_UP
                self.timer = 0.0

        elif self.state == MOLE_UP:
            if self.timer >= self.up_duration:
                self.state = MOLE_FALLING
                self.timer = 0.0

        elif self.state == MOLE_FALLING:
            if self.timer >= config.MOLE_FALL_TIME:
                self.state = MOLE_HIDDEN
                self.timer = 0.0
                self.escaped = True

        elif self.state == MOLE_HIT:
            if self.timer >= config.MOLE_HIT_TIME:
                self.state = MOLE_HIDDEN
                self.timer = 0.0

    def hit(self) -> None:
        self.state = MOLE_HIT
        self.timer = 0.0
        self.was_hit = True

    # ---- derived quantities --------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.state != MOLE_HIDDEN

    @property
    def is_hittable(self) -> bool:
        """Only a mole that is meaningfully out of its hole can be struck."""
        return self.state in (MOLE_RISING, MOLE_UP) and self.pop > 0.45

    @property
    def pop(self) -> float:
        """How far out of the hole the mole is, in ``[0, 1]``."""
        if self.state == MOLE_HIDDEN:
            return 0.0
        if self.state == MOLE_RISING:
            t = min(1.0, self.timer / config.MOLE_RISE_TIME)
            # Ease-out with a slight overshoot, so the mole pops rather than glides.
            return _ease_out_back(t)
        if self.state == MOLE_UP:
            return 1.0
        if self.state == MOLE_FALLING:
            t = min(1.0, self.timer / config.MOLE_FALL_TIME)
            return 1.0 - t * t
        # MOLE_HIT -- drops fast while squashing.
        t = min(1.0, self.timer / config.MOLE_HIT_TIME)
        return max(0.0, 1.0 - t)

    @property
    def squash(self) -> tuple[float, float]:
        """Non-uniform scale factors applied on top of ``pop``.

        A struck mole is flattened vertically and widened horizontally, which
        is the standard squash-and-stretch cue that the hit registered.
        """
        if self.state != MOLE_HIT:
            return (1.0, 1.0)
        t = min(1.0, self.timer / config.MOLE_HIT_TIME)
        return (1.0 + 0.45 * math.sin(math.pi * t), 1.0 - 0.55 * math.sin(math.pi * t))

    @property
    def center(self) -> Point:
        """Screen position of the mole's body centre, used for hit testing."""
        matrix = self.matrix()
        return T.apply_point(matrix, 0.0, -config.MOLE_RADIUS)

    @property
    def color(self) -> tuple[int, int, int]:
        return config.MOLE_TYPE_COLOR[self.kind]

    # ---- transformation -------------------------------------------------

    def matrix(self) -> np.ndarray:
        """Build this mole's local-to-screen matrix for the current frame.

        Three transformations are composed, right to left:

        1.  ``S(squash)``          -- the hit reaction.
        2.  ``S(0.86 + 0.14 pop)`` -- the mole grows slightly as it emerges.
        3.  ``T(hole, rim + sink)``-- placed at the hole, sunk by however much
            of the rise is still to come.

        The sink term is what the clipping window then removes, so a mole at
        ``pop = 0.3`` is genuinely a full-size mole standing 70 percent below
        the rim -- not a small mole.
        """
        pop = self.pop
        sink = (1.0 - pop) * (config.MOLE_RADIUS * 2.35)
        grow = 0.86 + 0.14 * pop
        sx, sy = self.squash

        return T.compose(
            T.translation(self.hole.cx, self.hole.rim_y + sink),
            T.scaling(grow * sx, grow * sy),
        )


def _ease_out_back(t: float) -> float:
    """Ease-out curve that overshoots slightly before settling.

    ``f(t) = 1 + c3 (t-1)^3 + c1 (t-1)^2`` with the usual constants.  The
    overshoot is why the mole looks springy rather than mechanical.
    """
    c1 = 1.70158
    c3 = c1 + 1.0
    u = t - 1.0
    return 1.0 + c3 * u * u * u + c1 * u * u


# ---------------------------------------------------------------------------
# Mole geometry -- local coordinates, origin at the feet
# ---------------------------------------------------------------------------

_R = float(config.MOLE_RADIUS)


def mole_body_outline() -> list[Point]:
    """The mole's silhouette: a dome-topped body, in local coordinates.

    This is the polygon handed to Sutherland-Hodgman.  It extends below the
    origin on purpose, so there is always something for the clipper to remove.

    The dome's centre sits at ``y = -0.975 R`` with radius ``R``, so the crown
    of the head is at ``-1.975 R``.  The ear positions below are chosen to sit
    just outside that circle, which is what makes them peek out from behind the
    head instead of vanishing inside it.
    """
    outline = gpoly.capsule(width=2.0 * _R, height=1.35 * _R, segments=16)
    # capsule() is centred on the origin; shift it up so the flat base sits
    # a little *below* y = 0, i.e. below the hole rim.
    return [(x, y - 0.30 * _R) for x, y in outline]


def mole_collar_outline() -> list[Point]:
    """The type-coloured collar, drawn across the mole's shoulders."""
    return [
        (-0.96 * _R, -0.54 * _R),
        (0.96 * _R, -0.54 * _R),
        (1.02 * _R, -0.24 * _R),
        (-1.02 * _R, -0.24 * _R),
    ]


def _mix(base: tuple[int, int, int], accent: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    """Blend two colours -- used to tint a mole towards its filter colour."""
    return tuple(int(base[i] + (accent[i] - base[i]) * amount) for i in range(3))


def draw_mole(canvas: Canvas, mole: Mole, time_now: float) -> None:
    """Render one mole.

    Drawing order is strictly back to front: ears, body, cap, chest, collar,
    snout, nose, eyes, whiskers.  Ears are drawn *before* the body so the head
    overlaps their inner edge, which is what gives the head its silhouette.

    Two independent clipping mechanisms are in play:

    * the body silhouette is clipped analytically with Sutherland-Hodgman,
      producing a genuinely shorter polygon;
    * every smaller feature is clipped by the canvas scissor rectangle, which
      discards pixels as they are written.

    Both are needed.  The analytic clip is what stops the *outline stroke* from
    drawing a lid across the mole's waist where the polygon was cut.
    """
    if not mole.is_active:
        return

    matrix = mole.matrix()
    window = mole.hole.clip_window()
    scale = _matrix_scale(matrix)

    # Effect moles keep a recognisably mole-coloured body, tinted towards their
    # filter colour.  Fully recolouring them made them read as eggs rather than
    # animals; the cap and collar carry the type identity instead.
    if mole.kind == "normal":
        body_color = config.COL_MOLE_BODY
    elif mole.kind == "golden":
        body_color = _mix(config.COL_MOLE_BODY, mole.color, 0.80)
    else:
        body_color = _mix(config.COL_MOLE_BODY, mole.color, 0.42)
    dark = tuple(int(c * 0.70) for c in body_color)

    def at(lx: float, ly: float) -> Point:
        return T.apply_point(matrix, lx * _R, ly * _R)

    canvas.push_clip(Rect(*(int(v) for v in window)))
    try:
        # -- ears (behind the head) --------------------------------------
        for side in (-1.0, 1.0):
            ex, ey = at(side * 0.86, -1.58)
            gcircle.fill_circle(canvas, ex, ey, 0.32 * _R * scale, dark)
            gcircle.fill_circle(canvas, ex, ey, 0.17 * _R * scale, config.COL_MOLE_NOSE)

        # -- body silhouette (analytically clipped) ----------------------
        body = T.apply(matrix, mole_body_outline())
        clipped_body = gpoly.fill_polygon_clipped(canvas, body, body_color, window)
        if len(clipped_body) >= 3:
            gpoly.draw_polygon(canvas, clipped_body, dark, thickness=2, window=window)

        # -- filter cap ---------------------------------------------------
        # A beanie in the mole's filter colour: the head circle redrawn in that
        # colour, scissored to the crown.  Cheaper and cleaner than building a
        # separate arc polygon, and it follows the head shape exactly.
        if mole.kind != "normal":
            cap_x, cap_y = at(0.0, -1.44)
            dome_x, dome_y = at(0.0, -0.975)
            canvas.push_clip(Rect(int(cap_x - 2 * _R * scale), int(cap_y - 2 * _R * scale),
                                  int(cap_x + 2 * _R * scale), int(cap_y)))
            try:
                gcircle.fill_circle(canvas, dome_x, dome_y, _R * scale - 1, mole.color)
            finally:
                canvas.pop_clip()

            # Brim across the base of the cap, and a pompom on top.
            brim_l = at(-0.80, -1.44)
            brim_r = at(0.80, -1.44)
            draw_thick_line(
                canvas, brim_l[0], brim_l[1], brim_r[0], brim_r[1],
                tuple(int(c * 0.72) for c in mole.color), max(2, int(3 * scale)),
            )
            px, py = at(0.0, -2.10)
            gcircle.fill_circle(canvas, px, py, 0.17 * _R * scale, mole.color)
            gcircle.fill_circle(canvas, px, py, 0.09 * _R * scale, (255, 255, 255))

        # -- chest patch --------------------------------------------------
        bx, by = at(0.0, -0.34)
        gcircle.fill_ellipse(canvas, bx, by, 0.66 * _R * scale, 0.42 * _R * scale, config.COL_MOLE_BELLY)

        # -- collar in the mole's filter colour ---------------------------
        collar = T.apply(matrix, mole_collar_outline())
        gpoly.fill_polygon_clipped(canvas, collar, mole.color, window)
        gpoly.draw_polygon(
            canvas, collar, tuple(int(c * 0.65) for c in mole.color), thickness=1, window=window
        )

        # -- snout and nose ------------------------------------------------
        sx, sy = at(0.0, -0.84)
        gcircle.fill_ellipse(canvas, sx, sy, 0.54 * _R * scale, 0.36 * _R * scale, config.COL_MOLE_BELLY)

        nx, ny = at(0.0, -0.99)
        gcircle.fill_ellipse(canvas, nx, ny, 0.17 * _R * scale, 0.13 * _R * scale, config.COL_MOLE_NOSE)

        # Mouth: two short strokes meeting under the nose.
        mouth_top = at(0.0, -0.90)
        for side in (-1.0, 1.0):
            mouth_end = at(side * 0.20, -0.74)
            draw_line(canvas, mouth_top[0], mouth_top[1], mouth_end[0], mouth_end[1], dark)

        # -- eyes ----------------------------------------------------------
        blink = _blink_factor(time_now, mole.hole.index)
        for side in (-1.0, 1.0):
            ex, ey = at(side * 0.36, -1.28)
            gcircle.fill_ellipse(
                canvas, ex, ey, 0.21 * _R * scale, 0.21 * _R * scale * blink, config.COL_MOLE_EYE
            )
            if blink > 0.35:
                gcircle.fill_ellipse(
                    canvas,
                    ex + side * 0.04 * _R * scale,
                    ey,
                    0.11 * _R * scale,
                    0.11 * _R * scale * blink,
                    config.COL_MOLE_PUPIL,
                )
                # Catchlight -- one bright pixel cluster makes eyes look alive.
                gcircle.fill_circle(
                    canvas,
                    ex + side * 0.01 * _R * scale,
                    ey - 0.05 * _R * scale,
                    max(1.0, 0.04 * _R * scale),
                    (255, 255, 255),
                )

        # -- whiskers -------------------------------------------------------
        for side in (-1.0, 1.0):
            for offset, tilt in ((-0.10, -0.18), (0.0, 0.0), (0.10, 0.18)):
                ax, ay = at(side * 0.26, -0.86 + offset)
                bx2, by2 = at(side * 1.00, -0.86 + offset + tilt)
                draw_line(canvas, ax, ay, bx2, by2, dark)

        # -- hit flash --------------------------------------------------------
        if mole.state == MOLE_HIT:
            t = min(1.0, mole.timer / config.MOLE_HIT_TIME)
            flash_x, flash_y = at(0.0, -1.0)
            gcircle.fill_circle_blend(
                canvas, flash_x, flash_y, _R * (0.7 + 1.6 * t), (255, 255, 255), 0.55 * (1.0 - t)
            )
    finally:
        canvas.pop_clip()


def _matrix_scale(matrix: np.ndarray) -> float:
    """Extract a representative uniform scale from a transformation matrix.

    Circles and ellipses are scan-converted directly from a centre and radius
    rather than from a vertex list, so they need a scalar radius.  Taking the
    geometric mean of the two column norms is the standard way to reduce an
    anisotropic 2x2 block to one number.
    """
    sx = math.hypot(matrix[0, 0], matrix[1, 0])
    sy = math.hypot(matrix[0, 1], matrix[1, 1])
    return math.sqrt(max(sx * sy, 1e-6))


def _blink_factor(time_now: float, seed: int) -> float:
    """A value in ``[0, 1]`` that briefly dips to 0 to simulate a blink."""
    phase = (time_now * 0.9 + seed * 0.7) % 4.0
    if phase > 3.86:
        return max(0.0, abs(phase - 3.93) / 0.07)
    return 1.0


# ---------------------------------------------------------------------------
# Hammer
# ---------------------------------------------------------------------------

HAMMER_PIVOT: Point = (0.0, 26.0)


@dataclass
class Hammer:
    """The player's hammer.

    Position comes from the mouse (or from a keyboard hole selection); the
    swing is a rotation about :data:`HAMMER_PIVOT`, a point just below the
    head.  Rotating about the head's own centre would look like a twirl;
    rotating about the grip would throw the head far off the cursor.  A pivot
    just below the head splits the difference -- the head arcs enough to read
    as a swing while staying near the pointer.
    """

    x: float = 0.0
    y: float = 0.0
    swing_timer: float = -1.0        # negative means "not swinging"
    strike_emitted: bool = False

    def move_to(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    def start_swing(self) -> bool:
        """Begin a swing.  Returns False if one is already in progress."""
        if self.swinging:
            return False
        self.swing_timer = 0.0
        self.strike_emitted = False
        return True

    @property
    def swinging(self) -> bool:
        return self.swing_timer >= 0.0

    def update(self, dt: float) -> bool:
        """Advance the swing.  Returns True on the single frame of impact."""
        if not self.swinging:
            return False

        self.swing_timer += dt
        total = config.HAMMER_SWING_TIME + config.HAMMER_RECOVER_TIME

        impact = False
        if not self.strike_emitted and self.swing_timer >= config.HAMMER_SWING_TIME:
            self.strike_emitted = True
            impact = True

        if self.swing_timer >= total:
            self.swing_timer = -1.0

        return impact

    @property
    def angle(self) -> float:
        """Current rotation in degrees, interpolated across the swing."""
        if not self.swinging:
            return config.HAMMER_REST_ANGLE

        if self.swing_timer < config.HAMMER_SWING_TIME:
            # Downstroke: accelerate into the impact.
            t = self.swing_timer / config.HAMMER_SWING_TIME
            eased = t * t
        else:
            # Recovery: ease back out to rest.
            t = (self.swing_timer - config.HAMMER_SWING_TIME) / config.HAMMER_RECOVER_TIME
            t = min(1.0, t)
            eased = 1.0 - (t * (2.0 - t))

        return config.HAMMER_REST_ANGLE + (
            config.HAMMER_STRIKE_ANGLE - config.HAMMER_REST_ANGLE
        ) * eased

    def matrix(self) -> np.ndarray:
        """Local-to-screen matrix: rotate about the pivot, then translate."""
        return T.compose(
            T.translation(self.x, self.y),
            T.rotation(self.angle, about=HAMMER_PIVOT),
        )

    @property
    def impact_point(self) -> Point:
        """Where the head actually is, derived from the matrix rather than assumed.

        Hit detection uses this, so the collision always agrees with what is
        drawn -- change the pivot or the swing angles and the hit box follows
        automatically.
        """
        return T.apply_point(self.matrix(), 0.0, 0.0)


def hammer_head_outline() -> list[Point]:
    return gpoly.rounded_rectangle(88.0, 54.0, 14.0, segments=5)


def hammer_handle_outline() -> list[Point]:
    """A tapered handle running down and to the right from the head."""
    return [(-11.0, 20.0), (11.0, 20.0), (27.0, 112.0), (7.0, 116.0)]


def hammer_grip_outline() -> list[Point]:
    return [(15.0, 78.0), (24.0, 78.0), (29.0, 110.0), (18.0, 112.0)]


def draw_hammer(canvas: Canvas, hammer: Hammer) -> None:
    """Render the hammer: handle first, then head, then highlight band."""
    matrix = hammer.matrix()
    window = (
        float(config.FIELD_LEFT - 60),
        float(config.FIELD_TOP - 60),
        float(config.FIELD_RIGHT + 60),
        float(config.FIELD_BOTTOM + 60),
    )

    handle = T.apply(matrix, hammer_handle_outline())
    gpoly.fill_polygon_clipped(canvas, handle, config.COL_HAMMER_HANDLE, window)
    gpoly.draw_polygon(canvas, handle, config.COL_HAMMER_HANDLE_DARK, thickness=2, window=window)

    grip = T.apply(matrix, hammer_grip_outline())
    gpoly.fill_polygon_clipped(canvas, grip, config.COL_HAMMER_HANDLE_DARK, window)

    head = T.apply(matrix, hammer_head_outline())
    gpoly.fill_polygon_clipped(canvas, head, config.COL_HAMMER_HEAD, window)
    gpoly.draw_polygon(canvas, head, config.COL_HAMMER_HEAD_DARK, thickness=3, window=window)

    # White band across the head -- a rectangle in local space, so it follows
    # the rotation for free.
    band = T.apply(matrix, gpoly.rectangle(88.0, 12.0))
    gpoly.fill_polygon_clipped(canvas, band, config.COL_HAMMER_BAND, window)

    # Specular highlight on the upper-left of the head.
    hx, hy = T.apply_point(matrix, -22.0, -14.0)
    gcircle.fill_circle_blend(canvas, hx, hy, 9, (255, 255, 255), 0.35)


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------


@dataclass
class Particle:
    """A short-lived decoration spawned on impact.

    Each particle carries its own translation, rotation and scale, which are
    fed into ``transform2d.trs`` -- a compact demonstration of the full
    transformation chain on dozens of objects at once.
    """

    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple[int, int, int]
    size: float
    spin: float
    angle: float = 0.0
    shape: str = "star"

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 900.0 * dt          # gravity
        self.vx *= 0.99                # mild drag
        self.angle += self.spin * dt
        self.life -= dt

    @property
    def alive(self) -> bool:
        return self.life > 0.0

    @property
    def fade(self) -> float:
        return max(0.0, min(1.0, self.life / self.max_life))


@dataclass
class ParticleSystem:
    """A fixed-capacity pool of particles."""

    particles: list[Particle] = field(default_factory=list)
    capacity: int = 220

    def burst(
        self,
        x: float,
        y: float,
        color: tuple[int, int, int],
        count: int = 16,
        speed: float = 320.0,
    ) -> None:
        """Emit particles radially from a point."""
        for _ in range(count):
            if len(self.particles) >= self.capacity:
                break
            angle = random.uniform(0.0, 2.0 * math.pi)
            magnitude = random.uniform(0.35, 1.0) * speed
            self.particles.append(
                Particle(
                    x=x,
                    y=y,
                    vx=math.cos(angle) * magnitude,
                    vy=math.sin(angle) * magnitude - 140.0,
                    life=random.uniform(0.35, 0.75),
                    max_life=0.75,
                    color=color,
                    size=random.uniform(4.0, 10.0),
                    spin=random.uniform(-560.0, 560.0),
                    shape=random.choice(("star", "square", "triangle")),
                )
            )

    def update(self, dt: float) -> None:
        for particle in self.particles:
            particle.update(dt)
        self.particles = [p for p in self.particles if p.alive]

    def clear(self) -> None:
        self.particles.clear()

    def draw(self, canvas: Canvas) -> None:
        """Draw every live particle through its own TRS matrix."""
        window = (
            float(config.FIELD_LEFT),
            float(config.FIELD_TOP),
            float(config.FIELD_RIGHT),
            float(config.FIELD_BOTTOM),
        )
        for particle in self.particles:
            scale = particle.fade
            if scale <= 0.02:
                continue

            matrix = T.trs(
                translate=(particle.x, particle.y),
                rotate_degrees=particle.angle,
                scale=(scale, scale),
            )

            if particle.shape == "star":
                local = gpoly.star_polygon(particle.size, particle.size * 0.45, 5)
            elif particle.shape == "square":
                local = gpoly.rectangle(particle.size * 1.5, particle.size * 1.5)
            else:
                local = gpoly.regular_polygon(particle.size, 3)

            screen = T.apply(matrix, local)
            gpoly.fill_polygon_clipped(canvas, screen, particle.color, window)


# ---------------------------------------------------------------------------
# Floating score labels
# ---------------------------------------------------------------------------


@dataclass
class FloatingText:
    """A score popup that drifts upward and fades out."""

    text: str
    x: float
    y: float
    life: float
    max_life: float
    color: tuple[int, int, int]
    size: float = 20.0

    def update(self, dt: float) -> None:
        self.y -= 62.0 * dt
        self.life -= dt

    @property
    def alive(self) -> bool:
        return self.life > 0.0

    @property
    def fade(self) -> float:
        return max(0.0, min(1.0, self.life / self.max_life))
