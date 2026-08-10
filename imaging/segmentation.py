"""
Image segmentation by thresholding.

Syllabus reference: CSE 452, Week 10 -- "Image Segmentation: Segmentation of
grey level images, pixel based approach - Multi level thresholding, local
thresholding, threshold detection method".

Segmentation splits an image into meaningful regions.  The pixel-based approach
does it with intensity alone: choose a level ``T`` and call everything above it
foreground.  All the difficulty is in choosing ``T``.

Three answers appear here, in increasing sophistication:

``iterative_threshold``
    Guess a level, average the two groups it produces, move the level to the
    midpoint of those averages, repeat until it settles.  A fixed-point method,
    also known as the isodata algorithm.

``otsu_threshold``
    Choose the level that maximises the variance *between* the two classes.
    Exhaustive over all 256 levels, and computed from the histogram alone, so
    it is exact and fast.

``adaptive_mean_threshold`` / ``adaptive_gaussian_threshold``
    Abandon the single global level entirely and compute one per pixel from its
    own neighbourhood.  This is what handles uneven illumination, where no
    global level can possibly work.
"""

from __future__ import annotations

import numpy as np

from . import kernels
from .convolution import correlate2d, separable_correlate
from .enhance import histogram, to_gray
from .quantise import to_uint8


def _local_mean(gray: np.ndarray, block_size: int) -> np.ndarray:
    """Neighbourhood mean, computed as two 1D passes.

    Local thresholding wants a large window -- 31x31 is typical -- and a naive
    2D pass over a 961-element mask is far slower than it needs to be.  The box
    mask is separable, so one horizontal pass followed by one vertical pass
    gives an identical result for 62 array operations instead of 961.
    """
    factor = kernels.box_kernel_1d(block_size)
    return separable_correlate(gray, factor, factor)


def _local_gaussian_mean(gray: np.ndarray, block_size: int, sigma: float) -> np.ndarray:
    """Gaussian-weighted neighbourhood mean, also as two 1D passes."""
    factor = kernels.gaussian_kernel_1d(block_size, sigma)
    return separable_correlate(gray, factor, factor)


# ---------------------------------------------------------------------------
# Global threshold selection
# ---------------------------------------------------------------------------


def iterative_threshold(image: np.ndarray, tolerance: float = 0.5, max_iterations: int = 64) -> int:
    """Threshold detection by iterative intensity averaging (isodata).

    Algorithm:

    1.  Start with ``T`` at the mean intensity.
    2.  Split the pixels into those below ``T`` and those at or above it.
    3.  Compute the mean of each group, ``mu1`` and ``mu2``.
    4.  Set ``T = (mu1 + mu2) / 2``.
    5.  Repeat until ``T`` stops moving by more than ``tolerance``.

    Converges in a handful of iterations for a bimodal histogram.  It is
    computed from the histogram rather than the pixels so the cost does not
    depend on image size.
    """
    counts = histogram(image).astype(np.float64)
    levels = np.arange(256, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 128

    threshold = float((counts * levels).sum() / total)

    for _ in range(max_iterations):
        below = levels < threshold
        above = ~below

        weight_below = counts[below].sum()
        weight_above = counts[above].sum()
        if weight_below == 0 or weight_above == 0:
            break

        mean_below = (counts[below] * levels[below]).sum() / weight_below
        mean_above = (counts[above] * levels[above]).sum() / weight_above
        new_threshold = (mean_below + mean_above) / 2.0

        if abs(new_threshold - threshold) < tolerance:
            threshold = new_threshold
            break
        threshold = new_threshold

    return int(round(threshold))


def otsu_threshold(image: np.ndarray) -> int:
    """Otsu's method -- maximise between-class variance.

    Splitting the histogram at level ``t`` produces two classes with weights
    ``w0, w1`` and means ``mu0, mu1``.  The between-class variance is::

        sigma_b^2(t) = w0(t) . w1(t) . (mu0(t) - mu1(t))^2

    Otsu's insight is that total variance is fixed, so maximising the variance
    *between* the classes is the same as minimising the variance *within* them
    -- but the between-class form can be evaluated for every ``t`` in one pass
    over the histogram using cumulative sums, instead of re-scanning the image
    256 times.

    Returns the level that maximises it, using the convention that class 0 is
    ``[0, T]`` and class 1 is ``[T+1, 255]`` -- so a pixel is foreground when
    ``f > T``, matching :func:`apply_threshold`.

    **Ties.**  On an image with a gap in its histogram -- two clean intensity
    clusters and nothing between them -- the criterion is *exactly equal* for
    every level in the gap.  Taking the first maximum, as a plain ``argmax``
    does, puts the threshold hard against the lower cluster, where a single
    noisy pixel flips its class.  The midpoint of the maximising plateau is
    taken instead, which is the most robust point in the gap and also what the
    iterative method converges to.
    """
    counts = histogram(image).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 128

    probability = counts / total
    levels = np.arange(256, dtype=np.float64)

    # Cumulative class weights and cumulative first moments.
    weight0 = np.cumsum(probability)
    weight1 = 1.0 - weight0
    moment = np.cumsum(probability * levels)
    global_mean = moment[-1]

    # Avoid dividing by zero where a class is empty; those levels score zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        mean0 = np.where(weight0 > 0, moment / weight0, 0.0)
        mean1 = np.where(weight1 > 0, (global_mean - moment) / weight1, 0.0)

    between_class_variance = np.nan_to_num(weight0 * weight1 * (mean0 - mean1) ** 2)

    peak = float(between_class_variance.max())
    if peak <= 0.0:
        return 128                      # single-intensity image: nothing to split

    # All levels within rounding distance of the peak form the plateau.
    plateau = np.flatnonzero(between_class_variance >= peak * (1.0 - 1e-12))
    return int((plateau[0] + plateau[-1]) // 2)


def multilevel_threshold(image: np.ndarray, classes: int = 3) -> list[int]:
    """Multi-level Otsu: choose ``classes - 1`` thresholds.

    The same between-class variance criterion generalised to more than two
    groups.  For three classes the search is over all ordered pairs
    ``(t1, t2)``, which is 32 640 combinations -- small enough to evaluate
    exhaustively on the histogram.

    Only 2 and 3 classes are supported; beyond that an exhaustive search stops
    being practical and the standard approach switches to a recursive or
    tree-based method.
    """
    classes = int(classes)
    if classes <= 2:
        return [otsu_threshold(image)]
    if classes > 3:
        raise ValueError("multilevel_threshold supports 2 or 3 classes")

    counts = histogram(image).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return [85, 170]

    probability = counts / total
    levels = np.arange(256, dtype=np.float64)

    cumulative_weight = np.cumsum(probability)
    cumulative_moment = np.cumsum(probability * levels)

    def class_stats(low: int, high: int) -> tuple[float, float]:
        """Weight and mean of the intensity range ``[low, high]``."""
        weight = cumulative_weight[high] - (cumulative_weight[low - 1] if low > 0 else 0.0)
        moment = cumulative_moment[high] - (cumulative_moment[low - 1] if low > 0 else 0.0)
        if weight <= 0.0:
            return 0.0, 0.0
        return weight, moment / weight

    global_mean = cumulative_moment[-1]
    best_variance = -1.0
    best = (85, 170)

    for t1 in range(1, 254):
        w0, m0 = class_stats(0, t1)
        if w0 <= 0.0:
            continue
        for t2 in range(t1 + 1, 255):
            w1, m1 = class_stats(t1 + 1, t2)
            w2, m2 = class_stats(t2 + 1, 255)
            if w1 <= 0.0 or w2 <= 0.0:
                continue
            variance = (
                w0 * (m0 - global_mean) ** 2
                + w1 * (m1 - global_mean) ** 2
                + w2 * (m2 - global_mean) ** 2
            )
            if variance > best_variance:
                best_variance = variance
                best = (t1, t2)

    return list(best)


# ---------------------------------------------------------------------------
# Applying thresholds
# ---------------------------------------------------------------------------


def apply_threshold(image: np.ndarray, level: int, invert: bool = False) -> np.ndarray:
    """Binarise at a single level.

    Uses the textbook convention ``g = 1 if f > T``, so a level returned by
    :func:`otsu_threshold` or :func:`iterative_threshold` -- both of which
    define class 0 as ``[0, T]`` -- segments the image correctly.  Using
    ``>=`` here instead would shift every threshold by one level and, on an
    image whose lower cluster sits exactly at ``T``, would put that entire
    cluster in the wrong class.
    """
    gray = to_gray(image)
    binary = gray > level
    if invert:
        binary = ~binary
    return (binary * 255).astype(np.uint8)


def apply_multilevel(image: np.ndarray, thresholds: list[int]) -> np.ndarray:
    """Quantise into ``len(thresholds) + 1`` evenly spaced grey levels.

    Each class is rendered at its own flat intensity, so the segmentation is
    visible as distinct bands rather than as a binary mask.
    """
    gray = to_gray(image).astype(np.int32)
    ordered = sorted(int(t) for t in thresholds)
    class_count = len(ordered) + 1

    labels = np.zeros(gray.shape, dtype=np.int32)
    for threshold in ordered:
        labels += (gray > threshold).astype(np.int32)

    step = 255.0 / max(class_count - 1, 1)
    return to_uint8(labels * step)


def otsu(image: np.ndarray) -> np.ndarray:
    """Convenience: compute Otsu's level and binarise with it."""
    return apply_threshold(image, otsu_threshold(image))


# ---------------------------------------------------------------------------
# Local / adaptive thresholding
# ---------------------------------------------------------------------------


def adaptive_mean_threshold(
    image: np.ndarray, block_size: int = 31, offset: float = 8.0, invert: bool = False
) -> np.ndarray:
    """Local thresholding against the neighbourhood mean.

    Each pixel is compared with the average of a ``block_size`` window centred
    on it, minus a constant ``offset``.  Because the reference level follows
    the illumination, a page that is brightly lit on one side and shadowed on
    the other still binarises correctly -- something no global threshold can
    manage.

    ``offset`` biases the decision away from the local mean.  Without it, a
    perfectly flat region would come out as pure noise, since half its pixels
    sit either side of their own average by definition.
    """
    gray = to_gray(image).astype(np.float64)
    local_mean = _local_mean(gray, block_size)
    binary = gray >= (local_mean - float(offset))
    if invert:
        binary = ~binary
    return (binary * 255).astype(np.uint8)


def adaptive_gaussian_threshold(
    image: np.ndarray,
    block_size: int = 31,
    sigma: float = 6.0,
    offset: float = 8.0,
    invert: bool = False,
) -> np.ndarray:
    """Local thresholding against a Gaussian-weighted neighbourhood mean.

    Same idea as :func:`adaptive_mean_threshold`, but nearby pixels count for
    more than distant ones, so the reference level responds more smoothly to
    illumination changes and produces fewer blocky artefacts.
    """
    gray = to_gray(image).astype(np.float64)
    local_mean = _local_gaussian_mean(gray, block_size, sigma)
    binary = gray >= (local_mean - float(offset))
    if invert:
        binary = ~binary
    return (binary * 255).astype(np.uint8)


def niblack_threshold(
    image: np.ndarray, block_size: int = 25, k: float = -0.2
) -> np.ndarray:
    """Niblack's local method: ``T(x, y) = mean + k . stddev``.

    Adds the local *standard deviation* to the local mean, so the threshold
    tightens where the neighbourhood is busy and relaxes where it is flat.
    The local variance comes from ``E[x^2] - E[x]^2``, both terms being box
    filters, which keeps it to two convolutions.
    """
    gray = to_gray(image).astype(np.float64)
    local_mean = _local_mean(gray, block_size)
    local_mean_square = _local_mean(gray * gray, block_size)
    local_variance = np.maximum(local_mean_square - local_mean ** 2, 0.0)
    local_std = np.sqrt(local_variance)
    threshold = local_mean + float(k) * local_std
    return ((gray >= threshold) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def threshold_report(image: np.ndarray) -> dict[str, object]:
    """Every threshold this module can compute, for side-by-side comparison."""
    return {
        "otsu": otsu_threshold(image),
        "iterative": iterative_threshold(image),
        "mean": int(round(float(to_gray(image).mean()))),
        "multilevel": multilevel_threshold(image, 3),
    }


#: Names exposed to the in-game lab viewer.
SEGMENTATION_MENU = (
    "otsu",
    "iterative",
    "multilevel (3 class)",
    "adaptive mean",
    "adaptive gaussian",
    "niblack",
)


def apply_named_segmentation(image: np.ndarray, name: str) -> np.ndarray:
    """Dispatch a segmentation method by the names in :data:`SEGMENTATION_MENU`."""
    table = {
        "otsu": otsu,
        "iterative": lambda img: apply_threshold(img, iterative_threshold(img)),
        "multilevel (3 class)": lambda img: apply_multilevel(img, multilevel_threshold(img, 3)),
        "adaptive mean": lambda img: adaptive_mean_threshold(img, 31, 8.0),
        "adaptive gaussian": lambda img: adaptive_gaussian_threshold(img, 31, 6.0, 8.0),
        "niblack": lambda img: niblack_threshold(img, 25, -0.2),
    }
    if name not in table:
        raise KeyError(f"unknown segmentation method {name!r}")
    return table[name](image)
