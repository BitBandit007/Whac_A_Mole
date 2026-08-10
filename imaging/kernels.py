"""
Named convolution masks.

Syllabus reference: CSE 452, Week 6 -- "Different types of filter in digital
image, image enhancement".

Collecting the masks in one place makes the *shape* of each filter visible.
Reading down this file you can see directly why a mean mask blurs (all weights
positive, summing to one), why a Laplacian mask sharpens (centre positive,
surround negative, weights summing to zero) and why Sobel detects edges (an
antisymmetric difference in one axis, smoothing in the other).

Sign convention
---------------
The gradient masks are written in *correlation* form, i.e. as they are laid
directly over the image.  ``imaging.convolution.convolve2d`` rotates a mask by
180 degrees before applying it; ``correlate2d`` does not.  For symmetric masks
such as mean or Gaussian the distinction has no effect, which is exactly why
so many textbooks use the two words interchangeably -- but it matters for the
Sobel and emboss masks, where a flipped mask reverses the sign of the response.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Smoothing (low-pass) masks
# ---------------------------------------------------------------------------

#: 3x3 box / mean mask.  Every neighbour contributes equally.
MEAN_3 = np.ones((3, 3), dtype=np.float64) / 9.0

#: 5x5 box mask -- stronger blur, wider support.
MEAN_5 = np.ones((5, 5), dtype=np.float64) / 25.0

#: 3x3 weighted average approximating a Gaussian (sigma about 0.85).
#: Centre weight 4, edge 2, corner 1 -- the classic binomial mask.
GAUSSIAN_3 = np.array(
    [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ],
    dtype=np.float64,
) / 16.0

#: 5x5 binomial Gaussian approximation (sigma about 1.0).
GAUSSIAN_5 = np.array(
    [
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ],
    dtype=np.float64,
) / 256.0

#: Horizontal motion blur -- averages along one row only.
MOTION_BLUR_5 = np.zeros((5, 5), dtype=np.float64)
MOTION_BLUR_5[2, :] = 1.0 / 5.0


# ---------------------------------------------------------------------------
# Sharpening (high-pass) masks
# ---------------------------------------------------------------------------

#: Laplacian with 4-connectivity.  Weights sum to zero, so flat regions map to
#: zero response and only intensity changes survive.
LAPLACIAN_4 = np.array(
    [
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ],
    dtype=np.float64,
)

#: Laplacian with 8-connectivity -- includes the diagonal neighbours.
LAPLACIAN_8 = np.array(
    [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ],
    dtype=np.float64,
)

#: Sharpening mask = identity + Laplacian.  Adds the detected detail back onto
#: the original image in a single pass.
SHARPEN_4 = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
    dtype=np.float64,
)

SHARPEN_8 = np.array(
    [
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# First-order gradient masks
# ---------------------------------------------------------------------------

#: Roberts cross -- 2x2 diagonal differences.  Cheapest edge operator; very
#: sensitive to noise because it averages nothing.
ROBERTS_X = np.array([[1, 0], [0, -1]], dtype=np.float64)
ROBERTS_Y = np.array([[0, 1], [-1, 0]], dtype=np.float64)

#: Prewitt -- central difference in one axis, uniform average in the other.
PREWITT_X = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ],
    dtype=np.float64,
)
PREWITT_Y = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ],
    dtype=np.float64,
)

#: Sobel -- Prewitt with the centre row/column weighted twice, which is a
#: binomial smoothing along the edge direction.  Better noise rejection.
SOBEL_X = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.float64,
)
SOBEL_Y = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],
    dtype=np.float64,
)

#: Scharr -- optimised weights with better rotational symmetry than Sobel.
SCHARR_X = np.array(
    [
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3],
    ],
    dtype=np.float64,
)
SCHARR_Y = np.array(
    [
        [-3, -10, -3],
        [0, 0, 0],
        [3, 10, 3],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Directional / artistic masks
# ---------------------------------------------------------------------------

#: Emboss -- an antisymmetric mask along the main diagonal.  It is a directional
#: derivative; adding a mid-grey bias afterwards is what turns the signed
#: response into the familiar "stamped metal" look.
EMBOSS = np.array(
    [
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2],
    ],
    dtype=np.float64,
)

EMBOSS_STRONG = np.array(
    [
        [-4, -2, 0],
        [-2, 1, 2],
        [0, 2, 4],
    ],
    dtype=np.float64,
)

#: Outline / edge-enhance mask -- Laplacian-8 by another name.
OUTLINE = LAPLACIAN_8.copy()


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------


def box_kernel(size: int) -> np.ndarray:
    """An ``size x size`` normalised averaging mask."""
    size = max(1, int(size) | 1)      # force odd
    return np.ones((size, size), dtype=np.float64) / float(size * size)


def box_kernel_1d(size: int) -> np.ndarray:
    """The 1D factor of a box mask.

    A box mask is separable: ``B_2D = b b^T``.  Applying ``b`` once horizontally
    and once vertically costs ``2n`` operations per pixel instead of ``n^2``,
    which is the difference between a 31x31 local-mean threshold taking 961
    array passes and taking 62.
    """
    size = max(1, int(size) | 1)
    return np.ones(size, dtype=np.float64) / float(size)


def gaussian_kernel_1d(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """The 1D factor of a Gaussian mask.

    The 2D Gaussian factorises exactly, ``G(x, y) = g(x) g(y)``, because the
    exponent ``-(x^2 + y^2) / 2 sigma^2`` splits into a sum.  Normalising each
    1D factor to unit sum reproduces the normalised 2D mask.
    """
    size = max(1, int(size) | 1)
    radius = size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(ax ** 2) / (2.0 * float(sigma) ** 2))
    return kernel / kernel.sum()


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Sample a 2D Gaussian onto an odd-sized grid and normalise it.

    ``G(x, y) = exp(-(x^2 + y^2) / (2 sigma^2))``

    The leading ``1 / (2 pi sigma^2)`` is omitted because the mask is
    normalised to unit sum afterwards, which is what actually guarantees the
    filter preserves average brightness.
    """
    size = max(1, int(size) | 1)
    radius = size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * float(sigma) ** 2))
    return kernel / kernel.sum()


def laplacian_of_gaussian_kernel(size: int = 9, sigma: float = 1.4) -> np.ndarray:
    """The LoG ("Mexican hat") mask, a second-order edge operator.

    ``LoG(x, y) = ((x^2 + y^2 - 2 sigma^2) / sigma^4) exp(-(x^2+y^2)/(2 sigma^2))``

    Smoothing and second differentiation are combined into one mask, which is
    both faster and less noise-sensitive than running a Laplacian on a raw
    image.  The mask is shifted to sum exactly to zero so that a constant
    region produces exactly zero response.
    """
    size = max(3, int(size) | 1)
    radius = size // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx ** 2 + yy ** 2
    s2 = float(sigma) ** 2
    kernel = ((r2 - 2.0 * s2) / (s2 * s2)) * np.exp(-r2 / (2.0 * s2))
    return kernel - kernel.mean()


def sharpen_kernel(strength: float = 1.0, connectivity: int = 4) -> np.ndarray:
    """Identity plus a scaled Laplacian.

    ``strength = 0`` returns the identity (no change); larger values push more
    high-frequency detail back into the image.
    """
    laplacian = LAPLACIAN_4 if connectivity == 4 else LAPLACIAN_8
    identity = np.zeros_like(laplacian)
    identity[1, 1] = 1.0
    return identity + float(strength) * laplacian


#: Lookup used by the in-game lab viewer and the post-game report.
NAMED_KERNELS: dict[str, np.ndarray] = {
    "mean 3x3": MEAN_3,
    "mean 5x5": MEAN_5,
    "gaussian 3x3": GAUSSIAN_3,
    "gaussian 5x5": GAUSSIAN_5,
    "motion blur": MOTION_BLUR_5,
    "laplacian 4": LAPLACIAN_4,
    "laplacian 8": LAPLACIAN_8,
    "sharpen 4": SHARPEN_4,
    "sharpen 8": SHARPEN_8,
    "prewitt x": PREWITT_X,
    "prewitt y": PREWITT_Y,
    "sobel x": SOBEL_X,
    "sobel y": SOBEL_Y,
    "scharr x": SCHARR_X,
    "scharr y": SCHARR_Y,
    "emboss": EMBOSS,
}
