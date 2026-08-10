"""
The game engine: state machine, input handling and the main loop.

Structure
---------
The engine owns two canvases.  ``background`` holds the static board and is
scan-converted once at start-up.  ``canvas`` is the live frame: every frame
begins by copying the background over it, then draws the moving objects on top.
That split is what keeps a pure-Python rasteriser at 60 FPS -- the expensive
per-pixel work happens once, not sixty times a second.

Frame order matters and is deliberate:

1.  copy the pre-rendered background
2.  effect-mole rings, then moles (far row first, so near rows overlap them)
3.  the front lip of each hole -- drawn *after* the moles, which is what makes
    a mole look like it is standing inside a hole
4.  particles and floating score labels
5.  the image-processing flash, composited over the play field only
6.  the hammer, drawn after the flash so it stays readable
7.  the HUD, then any overlay

States
------
``MENU`` -> ``COUNTDOWN`` -> ``PLAYING`` -> ``GAME_OVER`` -> back to ``MENU``,
with ``PAUSED`` and ``LAB`` reachable from play and remembering where they came
from.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pygame

import config
from graphics.raster import Canvas, Rect
from graphics.text import draw_text

from . import board, highscore, hud
from .audio import AudioEngine
from .effects import EffectManager, screen_flash
from .entities import (
    FloatingText,
    Hammer,
    Mole,
    ParticleSystem,
    build_holes,
    draw_hammer,
    draw_mole,
)

STATE_MENU = "menu"
STATE_COUNTDOWN = "countdown"
STATE_PLAYING = "playing"
STATE_PAUSED = "paused"
STATE_GAME_OVER = "game_over"

COUNTDOWN_STEPS = ("3", "2", "1", "GO")
COUNTDOWN_STEP_TIME = 0.55


class Game:
    """Owns every runtime object and runs the main loop."""

    def __init__(self, difficulty: str = "NORMAL", muted: bool = False):
        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)

        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.present_surface = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # ---- rendering targets -----------------------------------------
        self.canvas = Canvas(config.WINDOW_WIDTH, config.WINDOW_HEIGHT, config.COL_BACKDROP_TOP)
        self.background = Canvas(config.WINDOW_WIDTH, config.WINDOW_HEIGHT, config.COL_BACKDROP_TOP)

        # ---- world -----------------------------------------------------
        self.holes = build_holes()
        self.moles = [Mole(hole=hole) for hole in self.holes]
        self.hammer = Hammer(x=config.WINDOW_WIDTH / 2, y=config.WINDOW_HEIGHT / 2)
        self.particles = ParticleSystem()
        self.floaters: list[FloatingText] = []

        print("[boot] scan-converting static board ...")
        board.render_background(self.background, self.holes)
        print("[boot] board ready")

        # ---- subsystems ------------------------------------------------
        self.audio = AudioEngine(pygame)
        self.audio.muted = muted
        self.effects = EffectManager(threaded=True)

        # ---- session ---------------------------------------------------
        self.difficulty = difficulty if difficulty in config.DIFFICULTIES else "NORMAL"
        self.state = STATE_MENU
        self.running = True
        self.quit_reason = "unknown"
        self.elapsed = 0.0
        self.show_debug = False

        self.countdown_timer = 0.0
        self.toast_message = ""
        self.toast_timer = 0.0
        self.miss_flash = 0.0
        self.life_pulse = 0.0

        # The hammer replaces the pointer, so the system cursor is hidden.  It
        # would otherwise sit somewhere else entirely once the sensitivity gain
        # and the play-field clamp are applied.
        pygame.mouse.set_visible(False)

        self.summary: dict = {}
        self._reset_round()

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def _reset_round(self) -> None:
        for mole in self.moles:
            mole.state = "hidden"
            mole.timer = 0.0
        self.particles.clear()
        self.floaters.clear()
        self.effects.clear()

        # Cancel any swing still in flight, so a round never begins with the
        # hammer half-way through a stroke it can no longer complete.
        self.hammer.swing_timer = -1.0
        self.hammer.strike_emitted = False

        self.score = 0
        self.lives = config.STARTING_LIVES
        self.time_left = float(config.ROUND_SECONDS)
        self.combo = 0
        self.best_combo = 0
        self.hits = 0
        self.misses = 0
        self.escaped = 0
        self.spawn_timer = 0.6
        self.miss_flash = 0.0
        self.life_pulse = 0.0

    def _start_countdown(self) -> None:
        self._reset_round()
        self.state = STATE_COUNTDOWN
        self.countdown_timer = 0.0
        self.audio.play("start", 0.7)

    def _end_round(self, reason: str) -> None:
        swings = self.hits + self.misses
        accuracy = (self.hits / swings * 100.0) if swings else 0.0

        # `record` only returns True when this round beat the stored best for
        # *this mode*, which is what the "NEW HIGH SCORE" banner means.
        previous_best = highscore.best_score(self.difficulty)
        is_new_best = highscore.record(self.score, self.difficulty, self.hits, accuracy)
        mode_best = max(previous_best, self.score)

        self.summary = {
            "reason": reason,
            "score": self.score,
            "hits": self.hits,
            "misses": self.misses,
            "escaped": self.escaped,
            "accuracy": accuracy,
            "best_combo": max(1, self._multiplier_for(self.best_combo)),
            "difficulty": self.difficulty,
            "is_new_best": is_new_best,
            "mode_best": mode_best,
        }
        self.state = STATE_GAME_OVER
        self.audio.play("game_over", 0.8)

    @property
    def multiplier(self) -> int:
        return self._multiplier_for(self.combo)

    @staticmethod
    def _multiplier_for(combo: int) -> int:
        return min(config.COMBO_MAX, 1 + combo // config.COMBO_STEP)

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def _difficulty_settings(self) -> dict:
        return config.DIFFICULTIES[self.difficulty]

    def _choose_kind(self) -> str:
        """Weighted random pick over the mole types."""
        kinds = list(config.MOLE_TYPE_WEIGHTS)
        weights = [config.MOLE_TYPE_WEIGHTS[kind] for kind in kinds]
        return random.choices(kinds, weights=weights, k=1)[0]

    def _try_spawn(self, dt: float) -> None:
        settings = self._difficulty_settings()
        self.spawn_timer -= dt
        if self.spawn_timer > 0.0:
            return

        active = sum(1 for mole in self.moles if mole.is_active)
        if active >= settings["max_active"]:
            # Board is full -- check back shortly rather than banking up spawns.
            self.spawn_timer = 0.12
            return

        free = [mole for mole in self.moles if not mole.is_active]
        if not free:
            self.spawn_timer = 0.12
            return

        mole = random.choice(free)
        mole.spawn(self._choose_kind(), random.uniform(*settings["up_time"]))
        self.spawn_timer = random.uniform(*settings["spawn_delay"])

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def quit(self, reason: str) -> None:
        """End the main loop, recording why.

        The reason is printed on exit.  A game that closes without saying
        anything is indistinguishable from a game that crashed silently, and
        both Esc and the window close button end up here.
        """
        self.quit_reason = reason
        self.running = False

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit("window closed")

            elif event.type == pygame.MOUSEMOTION:
                self._aim_at_pointer(event.pos)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.state == STATE_PLAYING:
                    self._swing()
                elif self.state == STATE_MENU:
                    self._start_countdown()

            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

    def _handle_keydown(self, event) -> None:
        key = event.key

        # ---- global shortcuts ------------------------------------------
        if key == pygame.K_F1:
            self.show_debug = not self.show_debug
            return
        if key == pygame.K_m and self.state in (STATE_PLAYING, STATE_MENU):
            if self.state == STATE_MENU:
                muted = self.audio.toggle_mute()
                self._toast("SOUND MUTED" if muted else "SOUND ON")
                return

        # ---- per-state --------------------------------------------------
        if self.state == STATE_MENU:
            self._handle_menu_key(key)
        elif self.state == STATE_PLAYING:
            self._handle_playing_key(key)
        elif self.state == STATE_PAUSED:
            self._handle_paused_key(key)
        elif self.state == STATE_GAME_OVER:
            self._handle_game_over_key(key)

    def _handle_menu_key(self, key: int) -> None:
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._start_countdown()
        elif key in (pygame.K_LEFT, pygame.K_a):
            index = config.DIFFICULTY_ORDER.index(self.difficulty)
            self.difficulty = config.DIFFICULTY_ORDER[(index - 1) % len(config.DIFFICULTY_ORDER)]
            self.audio.play("ui")
        elif key in (pygame.K_RIGHT, pygame.K_d):
            index = config.DIFFICULTY_ORDER.index(self.difficulty)
            self.difficulty = config.DIFFICULTY_ORDER[(index + 1) % len(config.DIFFICULTY_ORDER)]
            self.audio.play("ui")
        elif key == pygame.K_ESCAPE:
            self.quit("escape pressed on the menu")

    def _handle_playing_key(self, key: int) -> None:
        # The hammer is mouse-only: aiming and swinging have no keyboard
        # equivalent, so the only keys that do anything during play are the
        # ones that leave it.
        if key in (pygame.K_p, pygame.K_ESCAPE):
            self.state = STATE_PAUSED
            self.audio.play("ui")

    def _handle_paused_key(self, key: int) -> None:
        if key in (pygame.K_p, pygame.K_ESCAPE):
            self.state = STATE_PLAYING
            self.audio.play("ui")
        elif key == pygame.K_r:
            self._start_countdown()
        elif key == pygame.K_m:
            self.state = STATE_MENU
            self.audio.play("ui")

    def _handle_game_over_key(self, key: int) -> None:
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._start_countdown()
        elif key == pygame.K_m:
            self.state = STATE_MENU
            self.audio.play("ui")
        elif key == pygame.K_ESCAPE:
            self.quit("escape pressed on the game over screen")

    def _aim_at_pointer(self, position: tuple[int, int]) -> None:
        """Map the pointer to a hammer position, with sensitivity applied.

        A mouse is an *absolute* device -- the pointer is already wherever the
        operating system put it -- so sensitivity cannot be a simple multiplier
        on movement without the hammer drifting away from the pointer over
        time.  Instead the offset from the centre of the play field is scaled:

            hammer = centre + (pointer - centre) * MOUSE_SENSITIVITY

        The mapping stays one-to-one and drift-free (centre always maps to
        centre), but the hammer travels further than the hand does, so less
        physical movement covers the board.  The result is clamped to the play
        field, which is why the system cursor is hidden -- past the clamp the
        two would visibly separate.
        """
        centre_x = (config.FIELD_LEFT + config.FIELD_RIGHT) / 2.0
        centre_y = (config.FIELD_TOP + config.FIELD_BOTTOM) / 2.0
        gain = config.MOUSE_SENSITIVITY

        x = centre_x + (position[0] - centre_x) * gain
        y = centre_y + (position[1] - centre_y) * gain

        margin = 6
        x = min(max(x, config.FIELD_LEFT + margin), config.FIELD_RIGHT - margin)
        y = min(max(y, config.FIELD_TOP + margin), config.FIELD_BOTTOM - margin)

        self.hammer.move_to(x, y)

    # ------------------------------------------------------------------
    # Gameplay
    # ------------------------------------------------------------------

    def _swing(self) -> None:
        if not self.hammer.start_swing():
            return

    def _resolve_impact(self) -> None:
        """Called on the single frame the hammer reaches full extension."""
        impact_x, impact_y = self.hammer.impact_point

        best: Mole | None = None
        best_distance = float("inf")
        reach = config.HAMMER_HIT_RADIUS + config.MOLE_RADIUS * 0.80

        for mole in self.moles:
            if not mole.is_hittable:
                continue
            cx, cy = mole.center
            distance = math.hypot(cx - impact_x, cy - impact_y)
            if distance <= reach and distance < best_distance:
                best = mole
                best_distance = distance

        if best is None:
            self._register_miss(impact_x, impact_y)
        else:
            self._register_hit(best)

    def _register_hit(self, mole: Mole) -> None:
        mole.hit()

        previous_multiplier = self.multiplier
        self.combo += 1
        self.best_combo = max(self.best_combo, self.combo)
        multiplier = self.multiplier

        points = config.MOLE_SCORE[mole.kind] * multiplier
        self.score += points
        self.hits += 1

        cx, cy = mole.center
        self.particles.burst(cx, cy, mole.color, count=18, speed=340.0)
        self.particles.burst(cx, cy, (255, 255, 255), count=6, speed=200.0)

        label = f"+{points}"
        if multiplier > 1:
            label += f"  x{multiplier}"
        self.floaters.append(
            FloatingText(text=label, x=cx, y=cy - 40, life=0.85, max_life=0.85, color=mole.color, size=22)
        )

        if mole.kind == "golden":
            self.audio.play("golden", 0.9)
        else:
            self.audio.play("hit", 0.85)

        if multiplier > previous_multiplier:
            self.audio.play("combo", 0.7)
            self.floaters.append(
                FloatingText(
                    text=f"COMBO x{multiplier}",
                    x=config.WINDOW_WIDTH / 2,
                    y=config.FIELD_TOP + 26,
                    life=1.1,
                    max_life=1.1,
                    color=config.COL_GOOD,
                    size=26,
                )
            )

        # The filter flash is queued from the live frame, so what gets filtered
        # is exactly what the player was looking at when they connected.
        self.effects.trigger(mole.kind, self.canvas)

    def _register_miss(self, x: float, y: float) -> None:
        self.misses += 1
        self.combo = 0
        self.miss_flash = 0.22
        self.particles.burst(x, y, (150, 150, 170), count=8, speed=170.0)
        self.audio.play("miss", 0.6)

    def _register_escape(self, mole: Mole) -> None:
        self.escaped += 1
        self.combo = 0
        self.lives -= 1
        self.life_pulse = 0.6
        self.audio.play("escape", 0.6)

        if self.lives <= 0:
            self.audio.play("life_lost", 0.8)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _update(self, dt: float) -> None:
        self.elapsed += dt
        self.effects.update(dt)

        self.toast_timer = max(0.0, self.toast_timer - dt)
        self.miss_flash = max(0.0, self.miss_flash - dt)
        self.life_pulse = max(0.0, self.life_pulse - dt)

        if self.state == STATE_COUNTDOWN:
            self.countdown_timer += dt
            if self.countdown_timer >= COUNTDOWN_STEP_TIME * len(COUNTDOWN_STEPS):
                self.state = STATE_PLAYING
            return

        if self.state != STATE_PLAYING:
            # Particles keep settling on the game-over screen so it does not
            # freeze mid-explosion.
            self.particles.update(dt)
            self._update_floaters(dt)
            return

        # ---- live round -------------------------------------------------
        self.time_left -= dt

        if self.hammer.update(dt):
            self._resolve_impact()

        self._try_spawn(dt)

        for mole in self.moles:
            mole.update(dt)
            if mole.escaped:
                mole.escaped = False
                self._register_escape(mole)

        self.particles.update(dt)
        self._update_floaters(dt)

        if self.time_left <= 0.0:
            self.time_left = 0.0
            self._end_round("time")
        elif self.lives <= 0:
            self._end_round("lives")

    def _update_floaters(self, dt: float) -> None:
        for floater in self.floaters:
            floater.update(dt)
        self.floaters = [floater for floater in self.floaters if floater.alive]

    def _toast(self, message: str, duration: float = 2.2) -> None:
        self.toast_message = message
        self.toast_timer = duration

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self) -> None:
        self._render_world()

        if self.state == STATE_MENU:
            # The menu shows the best for the *selected* mode, so switching
            # difficulty switches the target you are chasing.
            hud.draw_menu(
                self.canvas,
                self.difficulty,
                highscore.best_score(self.difficulty),
                self.elapsed,
            )
        else:
            hud.draw_hud(self.canvas, self._hud_state())
            hud.draw_floating_texts(self.canvas, self.floaters)

            if self.state == STATE_COUNTDOWN:
                step = min(len(COUNTDOWN_STEPS) - 1, int(self.countdown_timer / COUNTDOWN_STEP_TIME))
                phase = (self.countdown_timer % COUNTDOWN_STEP_TIME) / COUNTDOWN_STEP_TIME
                hud.draw_countdown(self.canvas, COUNTDOWN_STEPS[step], 0.7 + 0.5 * (1.0 - phase))
            elif self.state == STATE_PAUSED:
                hud.draw_pause(self.canvas, self.elapsed)
            elif self.state == STATE_GAME_OVER:
                hud.draw_game_over(self.canvas, self.summary, self.elapsed)

        if self.miss_flash > 0.0:
            screen_flash(self.canvas, config.COL_BAD, 0.18 * (self.miss_flash / 0.22))

        if self.toast_timer > 0.0:
            hud.draw_toast(self.canvas, self.toast_message, min(1.0, self.toast_timer / 0.4))

        if self.show_debug:
            self._draw_debug()

        self._present()

    def _render_world(self) -> None:
        self.canvas.copy_from(self.background)

        # Rings marking effect moles, drawn under the moles themselves.
        for mole in self.moles:
            if mole.is_active and mole.kind != "normal":
                board.draw_type_ring(self.canvas, mole.hole, mole.color, self.elapsed * 60.0)

        # Far rows first so nearer moles overlap them correctly.
        for mole in sorted(self.moles, key=lambda m: m.hole.cy):
            draw_mole(self.canvas, mole, self.elapsed)

        board.draw_hole_fronts(self.canvas, self.holes)
        self.particles.draw(self.canvas)

        # The filter flash goes over the field but under the hammer and HUD.
        self.effects.composite(self.canvas)

        if self.state in (STATE_PLAYING, STATE_PAUSED, STATE_COUNTDOWN):
            draw_hammer(self.canvas, self.hammer)

    def _hud_state(self) -> dict:
        return {
            "score": self.score,
            "lives": self.lives,
            "time_left": self.time_left,
            "combo": self.combo,
            "multiplier": self.multiplier,
            "difficulty": self.difficulty,
            "life_pulse": self.life_pulse,
            "effect_name": self.effects.label,
            "effect_color": self.effects.color,
        }

    def _draw_debug(self) -> None:
        fps = self.clock.get_fps()
        active = sum(1 for mole in self.moles if mole.is_active)
        lines = [
            f"FPS {fps:5.1f}",
            f"MOLES {active}  PARTICLES {len(self.particles.particles)}",
            f"STATE {self.state.upper()}",
        ]
        y = config.WINDOW_HEIGHT - 74
        for line in lines:
            draw_text(self.canvas, line, 16, y, 13, (140, 255, 160), 1)
            y += 20

    def _present(self) -> None:
        """Hand the finished framebuffer to the display."""
        self.canvas.to_pygame_surface(pygame, self.present_surface)
        self.screen.blit(self.present_surface, (0, 0))
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("[boot] entering main loop")
        while self.running:
            # Clamp dt so that a long stall (a debugger breakpoint, or the
            # matplotlib report window opening) cannot teleport the simulation.
            dt = min(self.clock.tick(config.TARGET_FPS) / 1000.0, 0.05)

            self._handle_events()
            self._update(dt)
            self._render()

        print(f"[exit] {self.quit_reason}")
        self.shutdown()

    def shutdown(self) -> None:
        self.effects.shutdown()
        self.audio.shutdown()
        pygame.quit()


def main(difficulty: str = "NORMAL", muted: bool = False) -> int:
    game = Game(difficulty=difficulty, muted=muted)
    try:
        game.run()
    except KeyboardInterrupt:
        game.shutdown()
    return 0
