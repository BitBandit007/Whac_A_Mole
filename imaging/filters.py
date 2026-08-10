"""
Spatial filters: smoothing, sharpening and directional effects.

Syllabus reference: CSE 452, Week 6 -- "Different types of filter in digital
image, image enhancement".

Two families appear here, and the distinction is the one worth remembering:

**Linear filters** are weighted sums of the neighbourhood, so they are exactly
a convolution: mean, Gaussian, sharpen, emboss, motion blur.  They obey
superposition, they are separable when their mask is rank 1, and they can be
analysed in the frequency domain.

**Rank (order-statistic) filters** sort the neighbourhood and pick a value from
it: median, min, max.  They are *not* convolutions and have no frequency
response -- which is precisely why the median filter can delete salt-and-pepper
noise without smearing edges, something no linear filter can do.

These are the functions the game calls when an effect mole is hit.
"""

from __future__ import annotations

import numpy as np

from . import kernels
from .convolution import correlate2d, pad_image
from .quantise import to_uint8


# ---------------------------------------------------------------------------
# Linear smoothing
# ---------------------------------------------------------------------------


def mean_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Box / averaging filter.

    Every pixel becomes the unweighted average of its ``size x size``
    neighbourhood.  Cheap and effective against Gaussian noise, but it blurs
    edges as readily as it blurs noise, and a large box mask leaves visible
    rectangular ringing.
    """
    kernel = kernels.box_kernel(size)
    return to_uint8(correlate2d(image, kernel))


def gaussian_filter(image: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing.

    Weights fall off with distance, so nearby pixels count for more than
    distant ones.  The result is a far more natural blur than a box filter,
    with no rectangular artefacts.
    """
    kernel = kernels.gaussian_kernel(size, sigma)
    return to_uint8(correlate2d(image, kernel))


def motion_blur(image: np.ndarray, length: int = 9, horizontal: bool = True) -> np.ndarray:
    """Directional blur -- averages along a single row or column.

    Simulates the smear a camera records when the subject moves during the
    exposure.  Used for the golden mole's exit flourish.
    """
    length = max(3, int(length) | 1)
    kernel = np.zeros((length, length), dtype=np.float64)
    if horizontal:
        kernel[length // 2, :] = 1.0 / length
    else:
        kernel[:, length // 2] = 1.0 / length
    return to_uint8(correlate2d(image, kernel))


# ---------------------------------------------------------------------------
# Rank / order-statistic filters
# ---------------------------------------------------------------------------


def _neighbourhood_stack(image: np.ndarray, size: int, mode: str = "replicate") -> np.ndarray:
    """Build an array holding every neighbour of every pixel.

    Returns shape ``(size*size, H, W[, C])``: element ``k`` is the image
    shifted so that neighbour ``k`` lands on the centre pixel.  Any rank
    statistic -- median, min, max, percentile -- is then one reduction along
    axis 0.

    This is the same reorganisation trick used by ``correlate2d``: loop over
    the small window, not over the large image.
    """
    size = max(1, int(size) | 1)
    radius = size // 2
    padded = pad_image(image, radius, radius, mode)
    height, width = image.shape[:2]

    planes = []
    for dy in range(size):
        for dx in range(size):
            planes.append(padded[dy : dy + height, dx : dx + width])
    return np.stack(planes, axis=0)


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Median filter -- a non-linear, edge-preserving smoother.

    Replaces each pixel with the median of its neighbourhood.  Because the
    median is an *order* statistic rather than an average, an isolated extreme
    value (a salt or pepper pixel) is discarded outright rather than being
    averaged into its neighbours.  A step edge, by contrast, survives intact:
    on either side of it the majority of the window still holds the correct
    level.
    """
    stack = _neighbourhood_stack(image, size)
    return to_uint8(np.median(stack, axis=0))


def min_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Minimum filter -- grayscale erosion.  Shrinks bright regions."""
    stack = _neighbourhood_stack(image, size)
    return to_uint8(stack.min(axis=0))


def max_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Maximum filter -- grayscale dilation.  Grows bright regions."""
    stack = _neighbourhood_stack(image, size)
    return to_uint8(stack.max(axis=0))


def midpoint_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Average of the local minimum and maximum.

    Works well against uniform and Gaussian noise; sits between the mean and
    the median in behaviour.
    """
    stack = _neighbourhood_stack(image, size)
    return to_uint8((stack.min(axis=0) + stack.max(axis=0)) / 2.0)


def alpha_trimmed_mean_filter(image: np.ndarray, size: int = 3, trim: int = 2) -> np.ndarray:
    """Mean of the neighbourhood after discarding the ``trim`` extremes at each end.

    A tunable compromise: ``trim = 0`` is the mean filter, ``trim`` at its
    maximum is the median filter, and values in between handle images carrying
    both impulse *and* Gaussian noise.
    """
    stack = _neighbourhood_stack(image, size)
    count = stack.shape[0]
    trim = int(np.clip(trim, 0, (count - 1) // 2))
    ordered = np.sort(stack, axis=0)
    kept = ordered[trim : count - trim]
    return to_uint8(kept.mean(axis=0))


# ---------------------------------------------------------------------------
# Sharpening
# ---------------------------------------------------------------------------


def sharpen(image: np.ndarray, strength: float = 1.0, connectivity: int = 4) -> np.ndarray:
    """Laplacian sharpening: ``g = f + strength . laplacian(f)``.

    The Laplacian responds only where intensity is changing, so adding it back
    onto the original exaggerates exactly the transitions -- edges gain
    contrast while flat regions are untouched.
    """
    kernel = kernels.sharpen_kernel(strength, connectivity)
    return to_uint8(correlate2d(image, kernel))


def unsharp_mask(
    image: np.ndarray, amount: float = 1.0, size: int = 5, sigma: float = 1.0
) -> np.ndarray:
    """Unsharp masking: ``g = f + amount . (f - blur(f))``.

    The difference ``f - blur(f)`` is the *mask*: everything the blur removed,
    which is exactly the fine detail.  Scaling it back on is a gentler, more
    controllable sharpen than a raw Laplacian because ``sigma`` decides which
    spatial frequencies get boosted.

    ``amount > 1`` is high-boost filtering.
    """
    original = np.asarray(image, dtype=np.float64)
    blurred = correlate2d(image, kernels.gaussian_kernel(size, sigma))
    detail = original - blurred
    return to_uint8(original + float(amount) * detail)


def high_boost(image: np.ndarray, boost: float = 1.5, size: int = 5) -> np.ndarray:
    """High-boost filtering: ``g = (A - 1) f + highpass(f)``.

    Equivalent to unsharp masking with ``amount = A - 1``; written separately
    because the textbook presents it in this form.
    """
    return unsharp_mask(image, amount=max(0.0, float(boost) - 1.0), size=size)


# ---------------------------------------------------------------------------
# Directional / artistic
# ---------------------------------------------------------------------------


def emboss(image: np.ndarray, strength: float = 1.0, bias: float = 128.0) -> np.ndarray:
    """Emboss -- a directional derivative rendered as relief.

    The mask is antisymmetric about the main diagonal, so it measures the rate
    of change along that diagonal.  The response is signed; adding ``bias``
    (mid-grey) maps zero change to grey, positive slopes to light and negative
    slopes to dark.  The eye reads that as a surface lit from the top-left.
    """
    kernel = kernels.EMBOSS * float(strength)
    response = correlate2d(image, kernel) + float(bias)
    return to_uint8(response)


def emboss_gray(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Emboss rendered in grey, which reads as metal rather than tinted relief.

    The colour version embosses each channel independently, so strongly
    coloured regions come out tinted.  Collapsing to luminance first gives the
    familiar monochrome stamped-metal look.
    """
    from .enhance import to_gray, to_rgb

    gray = to_gray(image)
    embossed = emboss(gray, strength)
    return to_rgb(embossed) if np.asarray(image).ndim == 3 else embossed


def posterize(image: np.ndarray, levels: int = 6) -> np.ndarray:
    """Quantise each channel to ``levels`` values -- intensity level slicing.

    A useful companion to the segmentation module: it shows what "reducing the
    number of grey levels" does before any thresholding decision is made.
    """
    levels = max(2, int(levels))
    array = np.asarray(image, dtype=np.float64)
    step = 255.0 / (levels - 1)
    return to_uint8(np.round(array / step) * step)


#: Names exposed to the in-game lab viewer.
FILTER_MENU = (
    "mean 3x3",
    "mean 5x5",
    "gaussian 5x5",
    "median 3x3",
    "min 3x3",
    "max 3x3",
    "sharpen",
    "unsharp mask",
    "emboss",
    "motion blur",
    "posterize",
)


def apply_named_filter(image: np.ndarray, name: str) -> np.ndarray:
    """Dispatch a filter by the display names used in :data:`FILTER_MENU`."""
    table = {
        "mean 3x3": lambda img: mean_filter(img, 3),
        "mean 5x5": lambda img: mean_filter(img, 5),
        "gaussian 5x5": lambda img: gaussian_filter(img, 5, 1.2),
        "median 3x3": lambda img: median_filter(img, 3),
        "min 3x3": lambda img: min_filter(img, 3),
        "max 3x3": lambda img: max_filter(img, 3),
        "sharpen": lambda img: sharpen(img, 1.0),
        "unsharp mask": lambda img: unsharp_mask(img, 1.4),
        "emboss": lambda img: emboss(img),
        "motion blur": lambda img: motion_blur(img, 9),
        "posterize": lambda img: posterize(img, 5),
    }
    if name not in table:
        raise KeyError(f"unknown filter {name!r}")
    return table[name](image)
