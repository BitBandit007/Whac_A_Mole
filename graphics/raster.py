"""
Framebuffer and pixel-level primitives.

The whole project renders into a plain NumPy array of shape ``(height, width, 3)``
holding 8-bit RGB samples.  That single decision buys us two things:

1.  The scan-conversion algorithms in this package can write pixels directly,
    with no graphics library in the way -- which is exactly what the project
    brief requires ("without relying on built-in graphics drawing functions").
2.  The finished frame *is already an image matrix*, so the digital image
    processing routines in the ``imaging`` package can operate on it with no
    conversion step at all.  The graphics half and the image-processing half of
    the syllabus meet in this array.

Pixel addressing follows the image-processing convention ``pixels[y, x]``
(row first), because that is how Gonzalez & Woods index an image and how every
convolution kernel in ``imaging`` is written.

A note on speed
---------------
``put_pixel`` is the honest, one-pixel-at-a-time primitive and is what the
line and circle algorithms use.  ``put_span`` fills a *horizontal run* of
pixels that an algorithm has already decided upon.  Span filling is what makes
area fills fast enough for 60 FPS: the algorithm still computes which pixels
belong to the shape, NumPy merely performs the memory write.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

Color = Sequence[int]


class Rect:
    """An axis-aligned rectangle with **inclusive** integer bounds.

    Inclusive bounds are used throughout because the clipping algorithms in
    ``graphics.clipping`` are defined over a closed window
    ``[x_min, x_max] x [y_min, y_max]``, and mixing inclusive and exclusive
    conventions is the classic source of off-by-one artefacts along a clip
    boundary.
    """

    __slots__ = ("x_min", "y_min", "x_max", "y_max")

    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        self.x_min = int(x_min)
        self.y_min = int(y_min)
        self.x_max = int(x_max)
        self.y_max = int(y_max)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_size(cls, x: float, y: float, width: float, height: float) -> "Rect":
        return cls(x, y, x + width - 1, y + height - 1)

    def copy(self) -> "Rect":
        return Rect(self.x_min, self.y_min, self.x_max, self.y_max)

    # -- queries -----------------------------------------------------------

    @property
    def width(self) -> int:
        return self.x_max - self.x_min + 1

    @property
    def height(self) -> int:
        return self.y_max - self.y_min + 1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def intersect(self, other: "Rect") -> "Rect":
        return Rect(
            max(self.x_min, other.x_min),
            max(self.y_min, other.y_min),
            min(self.x_max, other.x_max),
            min(self.y_max, other.y_max),
        )

    def is_empty(self) -> bool:
        return self.x_min > self.x_max or self.y_min > self.y_max

    def inflated(self, dx: int, dy: int | None = None) -> "Rect":
        if dy is None:
            dy = dx
        return Rect(self.x_min - dx, self.y_min - dy, self.x_max + dx, self.y_max + dy)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def corners(self) -> list[tuple[int, int]]:
        """The four corners in clockwise order (screen coordinates)."""
        return [
            (self.x_min, self.y_min),
            (self.x_max, self.y_min),
            (self.x_max, self.y_max),
            (self.x_min, self.y_max),
        ]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Rect({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})"


class Canvas:
    """An RGB framebuffer with a clip-rectangle stack.

    Parameters
    ----------
    width, height:
        Framebuffer dimensions in pixels.
    background:
        Colour used by :meth:`clear`.

    Attributes
    ----------
    pixels:
        ``uint8`` array of shape ``(height, width, 3)``.  This is the live
        buffer -- image-processing routines may read it, filter it and write
        the result straight back.
    """

    def __init__(self, width: int, height: int, background: Color = (0, 0, 0)):
        self.width = int(width)
        self.height = int(height)
        self.background = tuple(background)
        self.pixels = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.pixels[:, :] = self.background

        self._bounds = Rect(0, 0, self.width - 1, self.height - 1)
        self._clip_stack: list[Rect] = [self._bounds.copy()]

    # ------------------------------------------------------------------
    # Clip rectangle management
    # ------------------------------------------------------------------

    @property
    def clip(self) -> Rect:
        """The rectangle that currently limits all drawing."""
        return self._clip_stack[-1]

    def push_clip(self, rect: Rect) -> Rect:
        """Intersect ``rect`` with the active clip and make it current."""
        new_clip = self.clip.intersect(rect)
        self._clip_stack.append(new_clip)
        return new_clip

    def pop_clip(self) -> None:
        if len(self._clip_stack) > 1:
            self._clip_stack.pop()

    def reset_clip(self) -> None:
        self._clip_stack = [self._bounds.copy()]

    # ------------------------------------------------------------------
    # Whole-buffer operations
    # ------------------------------------------------------------------

    def clear(self, color: Color | None = None) -> None:
        self.pixels[:, :] = self.background if color is None else color

    def copy_from(self, other: "Canvas") -> None:
        """Copy another canvas of identical size into this one.

        Used every frame to restore the pre-rendered static background before
        the moving objects are drawn on top of it.
        """
        if other.pixels.shape != self.pixels.shape:
            raise ValueError(
                f"canvas size mismatch: {other.pixels.shape} vs {self.pixels.shape}"
            )
        np.copyto(self.pixels, other.pixels)

    def clone(self) -> "Canvas":
        twin = Canvas(self.width, self.height, self.background)
        np.copyto(twin.pixels, self.pixels)
        return twin

    def snapshot(self) -> np.ndarray:
        """Return an independent copy of the framebuffer.

        This is the bridge into the ``imaging`` package: the returned array is
        an ordinary ``(H, W, 3) uint8`` image.
        """
        return self.pixels.copy()

    # ------------------------------------------------------------------
    # Pixel primitives
    # ------------------------------------------------------------------

    def put_pixel(self, x: int, y: int, color: Color) -> None:
        """Write one pixel, discarding it if it falls outside the clip window.

        This is the single point through which every scan-conversion algorithm
        writes to the screen.
        """
        xi = int(x)
        yi = int(y)
        clip = self._clip_stack[-1]
        if clip.x_min <= xi <= clip.x_max and clip.y_min <= yi <= clip.y_max:
            self.pixels[yi, xi] = color

    def get_pixel(self, x: int, y: int) -> tuple[int, int, int]:
        xi, yi = int(x), int(y)
        if 0 <= xi < self.width and 0 <= yi < self.height:
            return tuple(int(c) for c in self.pixels[yi, xi])
        return (0, 0, 0)

    def blend_pixel(self, x: int, y: int, color: Color, alpha: float) -> None:
        """Alpha-blend one pixel: ``dst = (1 - a) * dst + a * src``.

        Used for anti-aliased highlights and for fading effects in and out.
        """
        if alpha <= 0.0:
            return
        if alpha >= 1.0:
            self.put_pixel(x, y, color)
            return
        xi, yi = int(x), int(y)
        clip = self._clip_stack[-1]
        if not (clip.x_min <= xi <= clip.x_max and clip.y_min <= yi <= clip.y_max):
            return
        dst = self.pixels[yi, xi].astype(np.float32)
        src = np.asarray(color, dtype=np.float32)
        self.pixels[yi, xi] = np.clip(dst + (src - dst) * alpha, 0, 255).astype(np.uint8)

    def put_span(self, y: int, x_start: int, x_end: int, color: Color) -> None:
        """Fill the inclusive horizontal run ``[x_start, x_end]`` on row ``y``.

        The *caller* has already decided that this run belongs to the shape --
        typically a scan-line of a circle or polygon.  Only the clipping and
        the memory write happen here.
        """
        yi = int(y)
        clip = self._clip_stack[-1]
        if yi < clip.y_min or yi > clip.y_max:
            return
        x0 = max(int(x_start), clip.x_min)
        x1 = min(int(x_end), clip.x_max)
        if x0 > x1:
            return
        self.pixels[yi, x0 : x1 + 1] = color

    def blend_span(self, y: int, x_start: int, x_end: int, color: Color, alpha: float) -> None:
        """Alpha-blended version of :meth:`put_span`."""
        if alpha <= 0.0:
            return
        if alpha >= 1.0:
            self.put_span(y, x_start, x_end, color)
            return
        yi = int(y)
        clip = self._clip_stack[-1]
        if yi < clip.y_min or yi > clip.y_max:
            return
        x0 = max(int(x_start), clip.x_min)
        x1 = min(int(x_end), clip.x_max)
        if x0 > x1:
            return
        strip = self.pixels[yi, x0 : x1 + 1].astype(np.float32)
        src = np.asarray(color, dtype=np.float32)
        self.pixels[yi, x0 : x1 + 1] = np.clip(
            strip + (src - strip) * alpha, 0, 255
        ).astype(np.uint8)

    def fill_rect(self, rect: Rect, color: Color) -> None:
        """Fill a rectangle by emitting one span per scan-line."""
        clipped = self.clip.intersect(rect)
        if clipped.is_empty():
            return
        self.pixels[
            clipped.y_min : clipped.y_max + 1, clipped.x_min : clipped.x_max + 1
        ] = color

    def fill_rect_blend(self, rect: Rect, color: Color, alpha: float) -> None:
        if alpha <= 0.0:
            return
        if alpha >= 1.0:
            self.fill_rect(rect, color)
            return
        clipped = self.clip.intersect(rect)
        if clipped.is_empty():
            return
        region = self.pixels[
            clipped.y_min : clipped.y_max + 1, clipped.x_min : clipped.x_max + 1
        ].astype(np.float32)
        src = np.asarray(color, dtype=np.float32)
        self.pixels[
            clipped.y_min : clipped.y_max + 1, clipped.x_min : clipped.x_max + 1
        ] = np.clip(region + (src - region) * alpha, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Gradients (used for the backdrop)
    # ------------------------------------------------------------------

    def vertical_gradient(self, rect: Rect, top_color: Color, bottom_color: Color) -> None:
        """Paint a top-to-bottom linear colour ramp, one span per scan-line."""
        clipped = self.clip.intersect(rect)
        if clipped.is_empty():
            return
        top = np.asarray(top_color, dtype=np.float32)
        bottom = np.asarray(bottom_color, dtype=np.float32)
        height = rect.height
        for y in range(clipped.y_min, clipped.y_max + 1):
            t = (y - rect.y_min) / max(height - 1, 1)
            color = top + (bottom - top) * t
            self.put_span(y, clipped.x_min, clipped.x_max, color.astype(np.uint8))

    # ------------------------------------------------------------------
    # Bulk pixel plotting
    # ------------------------------------------------------------------

    def put_points(self, points: Iterable[tuple[int, int]], color: Color) -> None:
        """Plot an iterable of points.  Convenience for algorithm generators."""
        for x, y in points:
            self.put_pixel(x, y, color)

    # ------------------------------------------------------------------
    # Interop
    # ------------------------------------------------------------------

    def to_pygame_surface(self, pygame_module, surface=None):
        """Present the framebuffer through a pygame surface.

        pygame stores surfaces column-major ``(width, height, 3)`` while we
        store the image row-major, so a transposed *view* is handed over.  No
        drawing happens here -- this is purely how the finished array of pixels
        reaches the monitor.
        """
        view = np.transpose(self.pixels, (1, 0, 2))
        if surface is None:
            return pygame_module.surfarray.make_surface(view)
        pygame_module.surfarray.blit_array(surface, view)
        return surface

    def load_image(self, image: np.ndarray) -> None:
        """Replace the framebuffer contents with ``image`` (same shape)."""
        if image.shape != self.pixels.shape:
            raise ValueError(f"image shape {image.shape} != canvas {self.pixels.shape}")
        np.copyto(self.pixels, image.astype(np.uint8, copy=False))
