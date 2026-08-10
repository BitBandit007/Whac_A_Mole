"""
Edge detection: first-order operators, second-order operators, and Canny.

Syllabus reference: CSE 452, Week 9 -- "Edge Detection: First order, second
order edge operators, Canny's edge detection".

An edge is a place where intensity changes quickly, so every detector here is
a derivative in disguise.

**First order** operators estimate the gradient ``grad f = (df/dx, df/dy)`` and
report its magnitude.  They produce thick responses (a ramp edge is steep over
several pixels) and their output depends on how the mask trades noise
rejection against localisation -- Roberts is sharp but noisy, Sobel is smoother
but blunter.

**Second order** operators use the Laplacian ``d2f/dx2 + d2f/dy2``, which is
zero in flat regions, zero at the *centre* of a ramp, and changes sign across
an edge.  Detecting edges then means finding zero crossings, which localises
them to a single pixel -- at the price of much greater noise sensitivity,
since differentiating twice amplifies noise twice.

**Canny** is not another mask but a whole procedure, designed against three
explicit criteria: good detection, good localisation, and a single response
per edge.  It is the only detector here that outputs thin, connected contours.
"""

from __future__ import annotations

import numpy as np

from . import kernels
from .convolution import correlate2d, normalise_response
from .enhance import to_gray, to_rgb
from .quantise import to_uint8


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------


def gradients(
    image: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the raw ``(gx, gy)`` responses of a pair of derivative masks."""
    gray = to_gray(image).astype(np.float64)
    gx = correlate2d(gray, kernel_x)
    gy = correlate2d(gray, kernel_y)
    return gx, gy


def gradient_magnitude(gx: np.ndarray, gy: np.ndarray, exact: bool = True) -> np.ndarray:
    """Combine two gradient components into an edge strength.

    ``exact=True`` uses the Euclidean norm ``sqrt(gx^2 + gy^2)``.
    ``exact=False`` uses the ``|gx| + |gy|`` approximation, which is what
    hardware implementations use because it avoids a square root; it
    overestimates diagonal edges by up to 41 percent.
    """
    if exact:
        return np.sqrt(gx * gx + gy * gy)
    return np.abs(gx) + np.abs(gy)


def gradient_direction(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Gradient angle in degrees, in ``[0, 180)``.

    The gradient points across the edge, not along it.  Only the orientation
    matters for edge work, so opposite directions are folded together modulo
    180 degrees.
    """
    return (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0


# ---------------------------------------------------------------------------
# First-order operators
# ---------------------------------------------------------------------------


def roberts(image: np.ndarray, normalise: bool = True) -> np.ndarray:
    """Roberts cross-gradient operator (2x2 diagonal differences).

    The smallest possible derivative mask.  Fast and well localised, but with
    no smoothing at all it responds strongly to single-pixel noise, and its
    even-sized mask has no true centre pixel -- the response sits half a pixel
    off the true edge.
    """
    gx, gy = gradients(image, kernels.ROBERTS_X, kernels.ROBERTS_Y)
    magnitude = gradient_magnitude(gx, gy)
    return normalise_response(magnitude) if normalise else magnitude


def prewitt(image: np.ndarray, normalise: bool = True) -> np.ndarray:
    """Prewitt operator: central difference across, uniform average along.

    The averaging column gives it noise rejection Roberts lacks, at the cost of
    a slightly thicker response.
    """
    gx, gy = gradients(image, kernels.PREWITT_X, kernels.PREWITT_Y)
    magnitude = gradient_magnitude(gx, gy)
    return normalise_response(magnitude) if normalise else magnitude


def sobel(image: np.ndarray, normalise: bool = True) -> np.ndarray:
    """Sobel operator -- Prewitt with binomial weighting along the edge.

    Weighting the centre row twice is a small 1-2-1 smoothing pass, which makes
    Sobel noticeably steadier than Prewitt on noisy input.  It is the default
    first-order detector in this project and the one the "edge mole" triggers.
    """
    gx, gy = gradients(image, kernels.SOBEL_X, kernels.SOBEL_Y)
    magnitude = gradient_magnitude(gx, gy)
    return normalise_response(magnitude) if normalise else magnitude


def scharr(image: np.ndarray, normalise: bool = True) -> np.ndarray:
    """Scharr operator -- Sobel-like weights optimised for rotational symmetry."""
    gx, gy = gradients(image, kernels.SCHARR_X, kernels.SCHARR_Y)
    magnitude = gradient_magnitude(gx, gy)
    return normalise_response(magnitude) if normalise else magnitude


def sobel_components(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The horizontal and vertical Sobel responses, separately and displayable.

    Useful in the report: ``gx`` alone shows only vertical edges and ``gy``
    alone only horizontal ones, which makes the directional nature of the
    operator obvious in a way the combined magnitude hides.
    """
    gx, gy = gradients(image, kernels.SOBEL_X, kernels.SOBEL_Y)
    return normalise_response(np.abs(gx)), normalise_response(np.abs(gy))


# ---------------------------------------------------------------------------
# Second-order operators
# ---------------------------------------------------------------------------


def laplacian(image: np.ndarray, connectivity: int = 8, normalise: bool = True) -> np.ndarray:
    """Laplacian second-derivative operator.

    Isotropic -- it has no preferred direction, unlike the gradient masks.  Its
    raw output is signed; normalising maps the zero level to mid-grey so both
    sides of every edge stay visible.
    """
    kernel = kernels.LAPLACIAN_8 if connectivity == 8 else kernels.LAPLACIAN_4
    gray = to_gray(image).astype(np.float64)
    response = correlate2d(gray, kernel)
    return normalise_response(response) if normalise else response


def laplacian_of_gaussian(
    image: np.ndarray, size: int = 9, sigma: float = 1.4, normalise: bool = True
) -> np.ndarray:
    """Laplacian of Gaussian (Marr-Hildreth operator).

    Smoothing and the second derivative are folded into one mask.  Doing them
    separately would work too, but combining them is both faster and better
    behaved, and ``sigma`` becomes an explicit scale control: large sigma finds
    only coarse structure, small sigma finds fine detail.
    """
    kernel = kernels.laplacian_of_gaussian_kernel(size, sigma)
    gray = to_gray(image).astype(np.float64)
    response = correlate2d(gray, kernel)
    return normalise_response(response) if normalise else response


def zero_crossings(response: np.ndarray, threshold: float = 4.0) -> np.ndarray:
    """Locate sign changes in a second-derivative response.

    A pixel is marked when it and one of its right/lower neighbours have
    opposite signs *and* the jump between them exceeds ``threshold``.  The
    magnitude test is essential: without it, every noise ripple through zero
    would be reported as an edge.
    """
    response = np.asarray(response, dtype=np.float64)
    height, width = response.shape
    edges = np.zeros((height, width), dtype=bool)

    right = response[:, 1:]
    left = response[:, :-1]
    horizontal = (np.sign(left) != np.sign(right)) & (np.abs(left - right) > threshold)
    edges[:, :-1] |= horizontal

    down = response[1:, :]
    up = response[:-1, :]
    vertical = (np.sign(up) != np.sign(down)) & (np.abs(up - down) > threshold)
    edges[:-1, :] |= vertical

    return (edges * 255).astype(np.uint8)


def marr_hildreth(
    image: np.ndarray, size: int = 9, sigma: float = 1.4, threshold: float = 4.0
) -> np.ndarray:
    """Full Marr-Hildreth detector: LoG followed by zero-crossing detection."""
    gray = to_gray(image).astype(np.float64)
    kernel = kernels.laplacian_of_gaussian_kernel(size, sigma)
    response = correlate2d(gray, kernel)
    return zero_crossings(response, threshold)


# ---------------------------------------------------------------------------
# Canny
# ---------------------------------------------------------------------------


def _shift(padded: np.ndarray, dy: int, dx: int, height: int, width: int) -> np.ndarray:
    """View of a 1-pixel-padded array shifted by ``(dy, dx)``."""
    return padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin an edge map to single-pixel width.

    A gradient magnitude ridge is several pixels wide.  Only the pixel at the
    crest of that ridge is a true edge, so each pixel is compared with its two
    neighbours *along the gradient direction* and zeroed unless it is at least
    as large as both.

    The continuous gradient angle is quantised to the four directions an 8-
    connected pixel grid can actually represent: 0, 45, 90 and 135 degrees.
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    height, width = magnitude.shape
    padded = np.zeros((height + 2, width + 2), dtype=np.float64)
    padded[1:-1, 1:-1] = magnitude

    angle = np.asarray(direction, dtype=np.float64) % 180.0

    # Neighbour pairs for each quantised direction, in screen coordinates
    # (y increases downwards).
    sectors = [
        ((angle < 22.5) | (angle >= 157.5), (0, -1), (0, 1)),        # 0   deg
        ((angle >= 22.5) & (angle < 67.5), (-1, 1), (1, -1)),        # 45  deg
        ((angle >= 67.5) & (angle < 112.5), (-1, 0), (1, 0)),        # 90  deg
        ((angle >= 112.5) & (angle < 157.5), (-1, -1), (1, 1)),      # 135 deg
    ]

    suppressed = np.zeros_like(magnitude)
    for mask, (dy_a, dx_a), (dy_b, dx_b) in sectors:
        neighbour_a = _shift(padded, dy_a, dx_a, height, width)
        neighbour_b = _shift(padded, dy_b, dx_b, height, width)
        is_peak = (magnitude >= neighbour_a) & (magnitude >= neighbour_b)
        suppressed = np.where(mask & is_peak, magnitude, suppressed)

    return suppressed


def hysteresis_threshold(
    suppressed: np.ndarray, low: float, high: float, max_iterations: int = 256
) -> np.ndarray:
    """Double thresholding with connectivity-driven edge tracking.

    A single threshold always fails one way or the other: set it high and real
    edges break into dashes, set it low and noise gets through.  Canny's answer
    is two thresholds --

    * above ``high``  -> *strong*, certainly an edge;
    * between the two -> *weak*, an edge only if it connects to a strong one;
    * below ``low``   -> discarded.

    Weak pixels are then repeatedly absorbed while they touch an accepted
    pixel, so a faint stretch of a genuine contour is kept while an isolated
    faint blob is not.  The loop stops as soon as nothing new is absorbed.
    """
    suppressed = np.asarray(suppressed, dtype=np.float64)
    strong = suppressed >= high
    weak = (suppressed >= low) & ~strong

    accepted = strong.copy()
    height, width = suppressed.shape

    for _ in range(max_iterations):
        padded = np.zeros((height + 2, width + 2), dtype=bool)
        padded[1:-1, 1:-1] = accepted

        # 8-connected neighbourhood: OR together all eight shifts.
        neighbourhood = np.zeros_like(accepted)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbourhood |= _shift(padded, dy, dx, height, width)

        newly_accepted = weak & neighbourhood & ~accepted
        if not newly_accepted.any():
            break
        accepted |= newly_accepted

    return (accepted * 255).astype(np.uint8)


def canny(
    image: np.ndarray,
    sigma: float = 1.2,
    kernel_size: int = 5,
    low_ratio: float = 0.10,
    high_ratio: float = 0.26,
) -> np.ndarray:
    """Canny edge detector -- the full five-stage pipeline.

    1.  **Smooth** with a Gaussian, because differentiating raw pixels
        amplifies noise.
    2.  **Differentiate** with Sobel to get gradient magnitude and direction.
    3.  **Non-maximum suppression** to thin ridges to one pixel.
    4.  **Double threshold** to classify strong and weak candidates.
    5.  **Hysteresis** to keep the weak ones that connect to strong ones.

    The thresholds are given as *ratios* of the peak gradient magnitude rather
    than absolute levels, so the detector adapts to how contrasty the frame
    happens to be -- important here because the game screen ranges from a dim
    menu to a bright flash.
    """
    gray = to_gray(image).astype(np.float64)

    # 1. Smooth
    smoothed = correlate2d(gray, kernels.gaussian_kernel(kernel_size, sigma))

    # 2. Gradients
    gx = correlate2d(smoothed, kernels.SOBEL_X)
    gy = correlate2d(smoothed, kernels.SOBEL_Y)
    magnitude = gradient_magnitude(gx, gy)
    direction = gradient_direction(gx, gy)

    # 3. Thin
    suppressed = non_maximum_suppression(magnitude, direction)

    # 4 & 5. Threshold and track
    #
    # The thresholds are relative to the peak gradient, so a frame with no real
    # structure would have its own floating-point noise scaled up into a full
    # set of "edges".  An absolute floor of one intensity level per pixel of
    # gradient rules that out: below it there is nothing to detect.
    peak = float(suppressed.max())
    if peak <= 1.0:
        return np.zeros(gray.shape, dtype=np.uint8)
    return hysteresis_threshold(suppressed, peak * low_ratio, peak * high_ratio)


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------


def edge_overlay(
    image: np.ndarray, edge_map: np.ndarray, color: tuple[int, int, int] = (255, 64, 64)
) -> np.ndarray:
    """Paint an edge map over the original image in a highlight colour.

    Far more informative than showing the edge map alone -- it makes it obvious
    which structures the detector actually locked onto.
    """
    base = to_rgb(image).copy()
    mask = np.asarray(edge_map) > 0
    base[mask] = color
    return base


def edges_as_rgb(edge_map: np.ndarray, color: tuple[int, int, int] = (240, 255, 240)) -> np.ndarray:
    """Colourise a single-channel edge map for display on the game screen."""
    mask = np.asarray(edge_map, dtype=np.float64) / 255.0
    tinted = mask[:, :, None] * np.asarray(color, dtype=np.float64)
    return to_uint8(tinted)


#: Names exposed to the in-game lab viewer.
EDGE_MENU = (
    "roberts",
    "prewitt",
    "sobel",
    "scharr",
    "sobel gx",
    "sobel gy",
    "laplacian 4",
    "laplacian 8",
    "laplacian of gaussian",
    "marr-hildreth",
    "canny",
)


def apply_named_edge(image: np.ndarray, name: str) -> np.ndarray:
    """Dispatch an edge detector by the display names in :data:`EDGE_MENU`."""
    table = {
        "roberts": roberts,
        "prewitt": prewitt,
        "sobel": sobel,
        "scharr": scharr,
        "sobel gx": lambda img: sobel_components(img)[0],
        "sobel gy": lambda img: sobel_components(img)[1],
        "laplacian 4": lambda img: laplacian(img, 4),
        "laplacian 8": lambda img: laplacian(img, 8),
        "laplacian of gaussian": lambda img: laplacian_of_gaussian(img),
        "marr-hildreth": lambda img: marr_hildreth(img),
        "canny": lambda img: canny(img),
    }
    if name not in table:
        raise KeyError(f"unknown edge operator {name!r}")
    return table[name](image)
