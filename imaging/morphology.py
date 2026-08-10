"""
Morphological image processing.

Syllabus reference: CSE 452, Week 11 -- "Morphological operations: Basics of
set theory, Dilation and erosion, Structuring element, opening and closing,
hit or miss transformation".

Morphology treats a binary image as a *set* of foreground pixel coordinates and
probes it with a second, much smaller set called the **structuring element**
(SE).  Every operation below is defined in those set-theoretic terms:

``dilation``    ``A (+) B = { z : (B_hat)_z  intersects A }``
                -- the SE is reflected, translated to each position, and the
                position is kept if it overlaps the object at all.  Grows the
                object, fills small gaps, joins nearby components.

``erosion``     ``A (-) B = { z : B_z  is contained in A }``
                -- kept only if the SE fits *entirely* inside the object.
                Shrinks the object and deletes anything smaller than the SE.

``opening``     ``A o B = (A (-) B) (+) B``
                -- erode then dilate.  Removes small protrusions and isolated
                specks while restoring the size of what survives.

``closing``     ``A . B = (A (+) B) (-) B``
                -- dilate then erode.  Fills small holes and thin gaps while
                restoring the outer size.

Opening and closing are *idempotent*: applying either one twice changes
nothing the second time.  That is what makes them filters rather than merely
size changes.
"""

from __future__ import annotations

import numpy as np

from .enhance import to_gray
from .segmentation import otsu_threshold


# ---------------------------------------------------------------------------
# Structuring elements
# ---------------------------------------------------------------------------


def square_se(size: int = 3) -> np.ndarray:
    """Solid square SE -- 8-connected, the most common default."""
    size = max(1, int(size) | 1)
    return np.ones((size, size), dtype=bool)


def cross_se(size: int = 3) -> np.ndarray:
    """Plus-shaped SE -- 4-connected, preserves rectilinear structure."""
    size = max(1, int(size) | 1)
    element = np.zeros((size, size), dtype=bool)
    centre = size // 2
    element[centre, :] = True
    element[:, centre] = True
    return element


def disk_se(radius: int = 2) -> np.ndarray:
    """Approximately circular SE -- isotropic, so it does not favour any axis."""
    radius = max(1, int(radius))
    size = 2 * radius + 1
    ax = np.arange(size) - radius
    xx, yy = np.meshgrid(ax, ax)
    return (xx * xx + yy * yy) <= radius * radius


def line_se(length: int = 5, horizontal: bool = True) -> np.ndarray:
    """A 1-pixel-wide line SE -- extracts structure of one orientation only."""
    length = max(1, int(length) | 1)
    element = np.zeros((length, length), dtype=bool)
    centre = length // 2
    if horizontal:
        element[centre, :] = True
    else:
        element[:, centre] = True
    return element


NAMED_STRUCTURING_ELEMENTS = {
    "square 3x3": square_se(3),
    "square 5x5": square_se(5),
    "cross 3x3": cross_se(3),
    "cross 5x5": cross_se(5),
    "disk r=2": disk_se(2),
    "disk r=3": disk_se(3),
}


# ---------------------------------------------------------------------------
# Binary conversion
# ---------------------------------------------------------------------------


def binarize(image: np.ndarray, level: int | None = None) -> np.ndarray:
    """Convert any image to a boolean foreground mask.

    ``level`` defaults to Otsu's threshold, so morphology can be demonstrated
    on an arbitrary game frame without hand-tuning a level first.
    """
    array = np.asarray(image)
    if array.dtype == bool:
        return array
    gray = to_gray(array)
    if level is None:
        level = otsu_threshold(gray)
    # ``>`` matches the convention used by ``segmentation.apply_threshold``:
    # class 0 is [0, T], so the foreground is strictly above T.
    return gray > level


def to_display(binary: np.ndarray) -> np.ndarray:
    """Render a boolean mask as an 8-bit black-and-white image."""
    return (np.asarray(binary).astype(np.uint8)) * 255


# ---------------------------------------------------------------------------
# Core operations -- reference implementations
# ---------------------------------------------------------------------------


def dilate_naive(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Direct set-theoretic dilation, written as explicit loops.

    For every pixel, the reflected SE is centred on it and the pixel is set if
    *any* SE element lands on foreground.  Slow, but it is the definition
    transcribed literally.
    """
    binary = np.asarray(binary, dtype=bool)
    se = np.asarray(se, dtype=bool)
    height, width = binary.shape
    sh, sw = se.shape
    oy, ox = sh // 2, sw // 2

    output = np.zeros_like(binary)
    for y in range(height):
        for x in range(width):
            hit = False
            for j in range(sh):
                for i in range(sw):
                    if not se[j, i]:
                        continue
                    yy = y + j - oy
                    xx = x + i - ox
                    if 0 <= yy < height and 0 <= xx < width and binary[yy, xx]:
                        hit = True
                        break
                if hit:
                    break
            output[y, x] = hit
    return output


# ---------------------------------------------------------------------------
# Core operations -- vectorised
# ---------------------------------------------------------------------------


def _shifted_views(binary: np.ndarray, se: np.ndarray, pad_value: bool):
    """Yield the image shifted to every active position of the SE.

    Dilation ORs these together, erosion ANDs them.  Choosing the padding value
    is what makes the border behave correctly: pad with ``False`` for dilation
    (outside the image there is no object to grow from) and with ``True`` for
    erosion (so the border is not eaten away by an SE hanging off the edge).
    """
    binary = np.asarray(binary, dtype=bool)
    se = np.asarray(se, dtype=bool)
    height, width = binary.shape
    sh, sw = se.shape
    oy, ox = sh // 2, sw // 2

    padded = np.full((height + 2 * oy, width + 2 * ox), pad_value, dtype=bool)
    padded[oy : oy + height, ox : ox + width] = binary

    for j in range(sh):
        for i in range(sw):
            if se[j, i]:
                yield padded[j : j + height, i : i + width]


def dilate(binary: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Binary dilation.  Grows the foreground by the shape of the SE."""
    if se is None:
        se = square_se(3)
    result = np.zeros(np.asarray(binary).shape, dtype=bool)
    for view in _shifted_views(binary, se, pad_value=False):
        result |= view
    return result


def erode(binary: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Binary erosion.  Keeps only positions where the SE fits entirely inside."""
    if se is None:
        se = square_se(3)
    result = np.ones(np.asarray(binary).shape, dtype=bool)
    for view in _shifted_views(binary, se, pad_value=True):
        result &= view
    return result


def opening(binary: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Erosion followed by dilation -- removes small bright specks."""
    return dilate(erode(binary, se), se)


def closing(binary: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Dilation followed by erosion -- fills small dark holes."""
    return erode(dilate(binary, se), se)


# ---------------------------------------------------------------------------
# Derived operations
# ---------------------------------------------------------------------------


def morphological_gradient(binary: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """``dilate(A) - erode(A)`` -- a one-SE-wide band around every boundary."""
    return dilate(binary, se) & ~erode(binary, se)


def boundary_extraction(binary: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """``A - erode(A)`` -- the inner boundary of every object.

    Sits just inside the object, unlike the morphological gradient which
    straddles it.
    """
    binary = np.asarray(binary, dtype=bool)
    return binary & ~erode(binary, se)


def top_hat(binary: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """``A - opening(A)`` -- whatever the opening removed.

    Isolates bright detail smaller than the SE, which is the standard way to
    correct uneven background illumination.
    """
    binary = np.asarray(binary, dtype=bool)
    return binary & ~opening(binary, se)


def black_hat(binary: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """``closing(A) - A`` -- whatever the closing filled in."""
    binary = np.asarray(binary, dtype=bool)
    return closing(binary, se) & ~binary


def hit_or_miss(
    binary: np.ndarray, se_foreground: np.ndarray, se_background: np.ndarray
) -> np.ndarray:
    """Hit-or-miss transform: shape detection by simultaneous fit and anti-fit.

    ``A (*) B = (A (-) B_fg)  intersect  (A^c (-) B_bg)``

    A position is kept only if the foreground SE fits inside the object *and*
    the background SE fits inside the complement.  Requiring both is what makes
    this a shape *detector* rather than a size filter: it finds exactly the
    configurations matching the template, such as isolated points, corners or
    line ends.

    The two structuring elements must not overlap -- a pixel cannot be required
    to be foreground and background at once.
    """
    binary = np.asarray(binary, dtype=bool)
    fg = np.asarray(se_foreground, dtype=bool)
    bg = np.asarray(se_background, dtype=bool)

    if fg.shape != bg.shape:
        raise ValueError("hit-or-miss structuring elements must have the same shape")
    if np.any(fg & bg):
        raise ValueError("hit-or-miss structuring elements must not overlap")

    return erode(binary, fg) & erode(~binary, bg)


def corner_detector_se() -> tuple[np.ndarray, np.ndarray]:
    """A hit-or-miss template pair matching a top-left convex corner."""
    foreground = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ],
        dtype=bool,
    )
    background = np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    )
    return foreground, background


def isolated_point_se() -> tuple[np.ndarray, np.ndarray]:
    """A hit-or-miss template pair matching a single isolated foreground pixel."""
    foreground = np.zeros((3, 3), dtype=bool)
    foreground[1, 1] = True
    background = np.ones((3, 3), dtype=bool)
    background[1, 1] = False
    return foreground, background


def thin_once(binary: np.ndarray) -> np.ndarray:
    """One pass of morphological thinning, ``A - (A (*) B)``.

    Repeated application peels an object down towards its skeleton.  A single
    pass is exposed here because it is the clearest way to *see* what the
    hit-or-miss transform is being used for.
    """
    foreground, background = corner_detector_se()
    return np.asarray(binary, dtype=bool) & ~hit_or_miss(binary, foreground, background)


# ---------------------------------------------------------------------------
# Grayscale morphology
# ---------------------------------------------------------------------------


def grayscale_dilate(image: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Grayscale dilation -- local maximum over the SE.

    The binary definitions generalise by replacing set union with maximum and
    set intersection with minimum, which is why the max and min filters in
    ``imaging.filters`` are the same operations by another name.
    """
    from .filters import max_filter

    size = 3 if se is None else max(se.shape)
    return max_filter(image, size)


def grayscale_erode(image: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Grayscale erosion -- local minimum over the SE."""
    from .filters import min_filter

    size = 3 if se is None else max(se.shape)
    return min_filter(image, size)


# ---------------------------------------------------------------------------
# Menu dispatch
# ---------------------------------------------------------------------------

#: Names exposed to the in-game lab viewer.
MORPHOLOGY_MENU = (
    "dilation",
    "erosion",
    "opening",
    "closing",
    "gradient",
    "boundary",
    "top hat",
    "black hat",
    "hit-or-miss (corner)",
    "thinning",
)


def apply_named_morphology(
    image: np.ndarray, name: str, se: np.ndarray | None = None
) -> np.ndarray:
    """Dispatch a morphological operation by the names in :data:`MORPHOLOGY_MENU`.

    The image is binarised with Otsu first, and the boolean result is converted
    back to a displayable 8-bit image.
    """
    binary = binarize(image)
    if se is None:
        se = square_se(3)

    table = {
        "dilation": lambda b: dilate(b, se),
        "erosion": lambda b: erode(b, se),
        "opening": lambda b: opening(b, se),
        "closing": lambda b: closing(b, se),
        "gradient": lambda b: morphological_gradient(b, se),
        "boundary": lambda b: boundary_extraction(b, se),
        "top hat": lambda b: top_hat(b, se),
        "black hat": lambda b: black_hat(b, se),
        "hit-or-miss (corner)": lambda b: hit_or_miss(b, *corner_detector_se()),
        "thinning": thin_once,
    }
    if name not in table:
        raise KeyError(f"unknown morphological operation {name!r}")
    return to_display(table[name](binary))
