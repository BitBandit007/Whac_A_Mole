"""
Live image-processing effects -- the "effect mole" payload.

When an effect mole is struck, the play field is captured and run through one
of the spatial filters from the ``imaging`` package, then blended back over the
live frame as a flash that fades out.  Every hit is therefore a demonstration
of a real filter on a real frame, at full resolution and full precision.

Why the filter runs on a worker thread
--------------------------------------
A 3x3 correlation over the 920x564 play field in three channels is around
15 million multiply-accumulates.  NumPy does that in roughly 70 milliseconds --
fine as a batch operation, but four dropped frames if it happens inside the
game loop.

So the work is handed to a single background thread.  NumPy releases the
interpreter lock during array arithmetic, so the filter genuinely runs while
the game keeps rendering.  The result arrives a frame or two later and is
blended in as the flash fades up, which hides the latency completely.

The captured frame is a *frozen* snapshot, not the live one.  Over a flash
lasting half a second that reads as an intentional freeze-flash; recomputing
the filter every frame would cost the 70 ms per frame this design exists to
avoid.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

import config
from graphics.raster import Canvas
from imaging import edges, enhance, filters, kernels
from imaging.convolution import correlate2d
from imaging.quantise import to_uint8

#: The screen region effects are applied to.  The HUD is deliberately excluded
#: so the score and timer stay readable through the flash.
EFFECT_REGION = (
    config.FIELD_TOP,
    config.FIELD_BOTTOM + 1,
    config.FIELD_LEFT,
    config.FIELD_RIGHT + 1,
)

#: Human-readable name shown in the HUD for each mole type.
EFFECT_LABELS = {
    "blur": "mean 5x5",
    "sharpen": "laplacian sharpen",
    "edge": "sobel gradient",
    "emboss": "emboss",
    "golden": "histogram equalise",
}


# ---------------------------------------------------------------------------
# The filters themselves
# ---------------------------------------------------------------------------


def compute_effect(kind: str, region: np.ndarray) -> np.ndarray:
    """Apply the filter associated with a mole type.

    Runs on the worker thread.  Each branch uses the float32 fast path of
    ``correlate2d``: the extra precision of float64 is invisible after the
    result is quantised back to 8 bits, and halving the memory traffic nearly
    halves the runtime.
    """
    if kind == "blur":
        # Mean filter -- every neighbour weighted equally.
        response = correlate2d(region, kernels.MEAN_5, dtype=np.float32)
        return to_uint8(response)

    if kind == "sharpen":
        # Identity + Laplacian: the detected detail added back onto the image.
        response = correlate2d(region, kernels.SHARPEN_4, dtype=np.float32)
        return to_uint8(response)

    if kind == "edge":
        # Sobel gradient magnitude, colourised so the edges glow rather than
        # turning the play field into a grey wireframe.
        gray = enhance.to_gray(region).astype(np.float32)
        gx = correlate2d(gray, kernels.SOBEL_X, dtype=np.float32)
        gy = correlate2d(gray, kernels.SOBEL_Y, dtype=np.float32)
        magnitude = np.sqrt(gx * gx + gy * gy)
        peak = float(magnitude.max())
        if peak > 1e-6:
            magnitude *= 255.0 / peak
        return edges.edges_as_rgb(magnitude.astype(np.uint8), (150, 255, 170))

    if kind == "emboss":
        # Directional derivative biased to mid-grey, then collapsed to
        # luminance so it reads as stamped metal rather than tinted relief.
        response = correlate2d(region, kernels.EMBOSS, dtype=np.float32) + 128.0
        embossed = to_uint8(response)
        return enhance.to_rgb(enhance.to_gray(embossed))

    if kind == "golden":
        # Histogram equalisation plus a warm tint -- a point operation rather
        # than a spatial filter, for contrast with the other four.
        equalised = enhance.histogram_equalization(region)
        return enhance.tint(equalised, config.MOLE_TYPE_COLOR["golden"], 0.22)

    return np.asarray(region, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Effect manager
# ---------------------------------------------------------------------------


class EffectManager:
    """Schedules filter jobs and blends their results over the frame."""

    def __init__(self, threaded: bool = True):
        self.threaded = threaded
        self._executor = ThreadPoolExecutor(max_workers=1) if threaded else None
        self._pending: Future | None = None
        self._pending_kind: str | None = None

        self.image: np.ndarray | None = None
        self.kind: str | None = None
        self.timer: float = 0.0
        self.duration: float = config.EFFECT_DURATION

    # -- lifecycle -------------------------------------------------------

    def trigger(self, kind: str, canvas: Canvas) -> None:
        """Capture the play field and queue the filter for ``kind``.

        If a job is already in flight the request is dropped rather than
        queued.  Two flashes cannot be shown at once anyway, and letting the
        queue grow would mean effects arriving long after the hit that caused
        them.
        """
        if kind not in EFFECT_LABELS:
            return
        if self._pending is not None and not self._pending.done():
            return

        y0, y1, x0, x1 = EFFECT_REGION
        region = canvas.pixels[y0:y1, x0:x1].copy()

        if self._executor is None:
            self._adopt(kind, compute_effect(kind, region))
            return

        self._pending_kind = kind
        self._pending = self._executor.submit(compute_effect, kind, region)

    def _adopt(self, kind: str, image: np.ndarray) -> None:
        self.image = image
        self.kind = kind
        self.timer = 0.0

    def update(self, dt: float) -> None:
        """Collect any finished job and advance the fade."""
        if self._pending is not None and self._pending.done():
            try:
                result = self._pending.result()
            except Exception as error:
                # A failed filter must never take the game down with it -- but
                # it must not vanish silently either, or a broken effect looks
                # like an effect that simply never fires.
                print(
                    f"[effects] {self._pending_kind} filter failed: "
                    f"{error.__class__.__name__}: {error}"
                )
                result = None
            if result is not None and self._pending_kind is not None:
                self._adopt(self._pending_kind, result)
            self._pending = None
            self._pending_kind = None

        if self.image is not None:
            self.timer += dt
            if self.timer >= self.duration:
                self.image = None
                self.kind = None
                self.timer = 0.0

    def clear(self) -> None:
        self.image = None
        self.kind = None
        self.timer = 0.0

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    # -- rendering -------------------------------------------------------

    @property
    def strength(self) -> float:
        """Blend weight for this frame.

        The envelope ramps up over the first fifth of the effect and decays
        smoothly afterwards, so the flash arrives as a punch and leaves as a
        fade rather than snapping off.
        """
        if self.image is None:
            return 0.0
        t = self.timer / self.duration
        if t < 0.20:
            envelope = t / 0.20
        else:
            u = (t - 0.20) / 0.80
            envelope = (1.0 - u) ** 1.7
        return config.EFFECT_PEAK * max(0.0, min(1.0, envelope))

    @property
    def label(self) -> str | None:
        return EFFECT_LABELS.get(self.kind) if self.kind else None

    @property
    def color(self) -> tuple[int, int, int]:
        return config.MOLE_TYPE_COLOR.get(self.kind or "normal", config.COL_TEXT)

    def composite(self, canvas: Canvas) -> None:
        """Blend the filtered snapshot over the play field."""
        alpha = self.strength
        if self.image is None or alpha <= 0.01:
            return

        y0, y1, x0, x1 = EFFECT_REGION
        target = canvas.pixels[y0:y1, x0:x1]
        if target.shape != self.image.shape:
            return

        base = target.astype(np.float32)
        overlay = self.image.astype(np.float32)
        np.copyto(
            target,
            to_uint8(base + (overlay - base) * alpha),
        )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def screen_flash(canvas: Canvas, color: tuple[int, int, int], strength: float) -> None:
    """A plain colour flash, used for misses and life loss."""
    if strength <= 0.01:
        return
    from graphics.raster import Rect

    canvas.fill_rect_blend(Rect(0, 0, canvas.width - 1, canvas.height - 1), color, strength)


def vignette(canvas: Canvas, strength: float = 0.35) -> None:
    """Darken the frame towards its edges.

    Implemented as a separable radial ramp computed once and cached, so it
    costs one multiply per pixel per frame rather than a per-pixel distance.
    """
    mask = _vignette_mask(canvas.width, canvas.height)
    scale = 1.0 - strength * mask
    np.copyto(
        canvas.pixels,
        to_uint8(canvas.pixels.astype(np.float32) * scale[:, :, None]),
    )


_VIGNETTE_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _vignette_mask(width: int, height: int) -> np.ndarray:
    key = (width, height)
    cached = _VIGNETTE_CACHE.get(key)
    if cached is not None:
        return cached

    ys = (np.arange(height, dtype=np.float32) / max(height - 1, 1)) * 2.0 - 1.0
    xs = (np.arange(width, dtype=np.float32) / max(width - 1, 1)) * 2.0 - 1.0
    radius = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2) / np.sqrt(2.0)
    mask = np.clip((radius - 0.55) / 0.45, 0.0, 1.0) ** 1.6

    _VIGNETTE_CACHE[key] = mask
    return mask
