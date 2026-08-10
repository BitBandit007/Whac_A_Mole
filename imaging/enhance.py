"""
Point operations and histogram processing.

Syllabus reference: CSE 452, Week 6 -- "image enhancement".

Everything in this module is a *point* operation: the output at a pixel depends
only on the input at that same pixel (and, for histogram equalisation, on the
global intensity distribution).  No neighbourhood is involved, which is what
separates these from the spatial filters in ``imaging.filters``.

A point operation is fully described by its intensity transformation function
``s = T(r)``.  Reading the shape of ``T`` tells you what the operation does:
a straight line with negative slope is a negative, a concave curve lifts dark
detail, a convex curve suppresses it.
"""

from __future__ import annotations

import numpy as np

from .quantise import to_uint8

#: ITU-R BT.601 luminance weights.  Green dominates because human vision is
#: most sensitive to it; a naive (R+G+B)/3 average gets the perceived
#: brightness of coloured regions noticeably wrong.
LUMA_WEIGHTS = (0.299, 0.587, 0.114)


# ---------------------------------------------------------------------------
# Colour space helpers
# ---------------------------------------------------------------------------


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to 8-bit grayscale using the luminance weights.

    Grayscale input is returned unchanged, so callers can pass either.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    weights = np.array(LUMA_WEIGHTS, dtype=np.float64)
    luminance = array[:, :, :3].astype(np.float64) @ weights
    return to_uint8(luminance)


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Promote a grayscale image to three identical channels."""
    array = np.asarray(image)
    if array.ndim == 3:
        return array.astype(np.uint8, copy=False)
    return np.repeat(array[:, :, None], 3, axis=2).astype(np.uint8)


# ---------------------------------------------------------------------------
# Intensity transformations
# ---------------------------------------------------------------------------


def negative(image: np.ndarray) -> np.ndarray:
    """Photographic negative: ``s = 255 - r``.

    The simplest possible transformation, and genuinely useful: detail buried
    in a large dark region is much easier to read once that region is bright.
    """
    return (255 - np.asarray(image, dtype=np.int16)).clip(0, 255).astype(np.uint8)


def log_transform(image: np.ndarray, c: float | None = None) -> np.ndarray:
    """Log transformation: ``s = c . log(1 + r)``.

    Expands the dark end of the range and compresses the bright end, so it
    reveals detail in shadows.  ``c`` defaults to ``255 / log(256)``, the value
    that maps the input range exactly onto the output range.
    """
    array = np.asarray(image, dtype=np.float64)
    if c is None:
        c = 255.0 / np.log(1.0 + 255.0)
    return to_uint8(c * np.log(1.0 + array))


def inverse_log_transform(image: np.ndarray) -> np.ndarray:
    """Inverse of :func:`log_transform` -- expands the bright end instead."""
    array = np.asarray(image, dtype=np.float64) / 255.0
    return to_uint8((np.exp(array * np.log(256.0)) - 1.0))


def gamma_correction(image: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """Power-law (gamma) transformation: ``s = c . r^gamma``.

    * ``gamma < 1`` brightens, expanding dark tones (like the log transform,
      but with a tunable amount).
    * ``gamma > 1`` darkens, expanding bright tones.
    * ``gamma = 1`` is the identity.

    The image is normalised to ``[0, 1]`` before the power so that ``gamma``
    behaves the same regardless of bit depth.
    """
    array = np.asarray(image, dtype=np.float64) / 255.0
    return to_uint8(float(c) * np.power(array, float(gamma)) * 255.0)


def contrast_stretch(
    image: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0
) -> np.ndarray:
    """Linear contrast stretching between two intensity percentiles.

    Maps ``[r_low, r_high]`` onto the full ``[0, 255]`` range.  Percentiles
    rather than the raw min and max are used deliberately: a single stray black
    or white pixel would otherwise pin the endpoints and the stretch would do
    almost nothing.
    """
    array = np.asarray(image, dtype=np.float64)
    low = float(np.percentile(array, low_percentile))
    high = float(np.percentile(array, high_percentile))
    if high - low < 1e-9:
        return np.asarray(image).astype(np.uint8, copy=True)
    stretched = (array - low) * (255.0 / (high - low))
    return to_uint8(stretched)


def brightness_contrast(
    image: np.ndarray, brightness: float = 0.0, contrast: float = 1.0
) -> np.ndarray:
    """Affine intensity map: ``s = contrast . (r - 128) + 128 + brightness``.

    Pivoting the contrast about mid-grey rather than about zero keeps the
    overall exposure steady while the slope changes.
    """
    array = np.asarray(image, dtype=np.float64)
    result = float(contrast) * (array - 128.0) + 128.0 + float(brightness)
    return to_uint8(result)


def threshold_binary(image: np.ndarray, level: int = 128) -> np.ndarray:
    """Hard threshold to pure black and white -- the crudest point operation."""
    gray = to_gray(image)
    return np.where(gray >= level, 255, 0).astype(np.uint8)


def bit_plane(image: np.ndarray, plane: int) -> np.ndarray:
    """Extract one bit plane (0 = least significant, 7 = most significant).

    The high planes carry the recognisable structure of the image; the low
    planes are close to noise.  That asymmetry is the entire basis of bit-plane
    image compression, and it is striking to see directly.
    """
    plane = int(np.clip(plane, 0, 7))
    gray = to_gray(image)
    return (((gray >> plane) & 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Histogram processing
# ---------------------------------------------------------------------------


def histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Intensity histogram of a grayscale image (or of its luminance).

    Counted with ``np.bincount`` rather than a Python loop, but the quantity is
    the plain definition: ``h(k)`` is the number of pixels with intensity
    ``k``.
    """
    gray = to_gray(image).astype(np.int64).ravel()
    counts = np.bincount(gray, minlength=256)
    if bins == 256:
        return counts
    return counts.reshape(bins, -1).sum(axis=1)


def normalised_histogram(image: np.ndarray) -> np.ndarray:
    """Histogram scaled to a probability distribution, ``p(k) = h(k) / MN``."""
    counts = histogram(image).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def cumulative_histogram(image: np.ndarray) -> np.ndarray:
    """The cumulative distribution function of the intensities."""
    return np.cumsum(normalised_histogram(image))


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Global histogram equalisation.

    The transformation is the CDF of the image's own intensity distribution,
    scaled to the display range::

        s_k = (L - 1) . sum_{j=0..k} p(j)

    Intuition: intensities that are common get spread far apart, intensities
    that are rare get squeezed together.  The result approximates a uniform
    histogram, which maximises the use of the available dynamic range.

    For colour input only the luminance is equalised and the original hue is
    restored afterwards.  Equalising R, G and B independently would shift the
    colours, because each channel would get a different transformation.
    """
    array = np.asarray(image)

    if array.ndim == 2:
        cdf = cumulative_histogram(array)
        lookup = to_uint8(np.round(cdf * 255.0))
        return lookup[array]

    gray = to_gray(array)
    equalised = histogram_equalization(gray)

    # Rescale each channel by the ratio the luminance changed by, so colour
    # relationships are preserved.  The +1 guards against dividing by zero on
    # pure black pixels.
    original = gray.astype(np.float64) + 1.0
    ratio = (equalised.astype(np.float64) + 1.0) / original
    result = array[:, :, :3].astype(np.float64) * ratio[:, :, None]
    return to_uint8(result)


def histogram_equalization_rgb(image: np.ndarray) -> np.ndarray:
    """Per-channel equalisation.

    Included for contrast with :func:`histogram_equalization`: this version
    *does* shift the colours, sometimes dramatically.  Comparing the two side
    by side makes the reason obvious.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        return histogram_equalization(array)
    channels = [histogram_equalization(array[:, :, c]) for c in range(3)]
    return np.stack(channels, axis=2)


def histogram_stretch(image: np.ndarray) -> np.ndarray:
    """Full-range linear stretch using the true minimum and maximum."""
    array = np.asarray(image, dtype=np.float64)
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-9:
        return np.asarray(image).astype(np.uint8, copy=True)
    return to_uint8((array - low) * 255.0 / (high - low))


# ---------------------------------------------------------------------------
# Image arithmetic
# ---------------------------------------------------------------------------


def blend(image_a: np.ndarray, image_b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear cross-fade: ``(1 - alpha) . A + alpha . B``.

    This is how a filter flash is mixed back over the live frame during play,
    so the effect fades in and out instead of snapping.
    """
    a = np.asarray(image_a, dtype=np.float64)
    b = np.asarray(image_b, dtype=np.float64)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return to_uint8(a + (b - a) * alpha)


def image_difference(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Absolute difference of two images -- the standard change detector."""
    a = np.asarray(image_a, dtype=np.float64)
    b = np.asarray(image_b, dtype=np.float64)
    return to_uint8(np.abs(a - b))


def add_images(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Saturating addition (no wraparound)."""
    a = np.asarray(image_a, dtype=np.float64)
    b = np.asarray(image_b, dtype=np.float64)
    return to_uint8(a + b)


def tint(image: np.ndarray, color: tuple[int, int, int], amount: float = 0.3) -> np.ndarray:
    """Pull an image towards a colour -- used to colour-code each mole effect."""
    array = to_rgb(image).astype(np.float64)
    target = np.asarray(color, dtype=np.float64)
    amount = float(np.clip(amount, 0.0, 1.0))
    return to_uint8(array + (target - array) * amount)
