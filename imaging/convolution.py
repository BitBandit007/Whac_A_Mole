"""
Correlation and convolution -- the engine every spatial filter runs on.

Syllabus reference: CSE 452, Week 6 -- spatial filtering; the correlation /
convolution pair is the foundation of the whole image-processing half of the
course.

Definitions
-----------
For an image ``f`` and a mask ``w`` of size ``(2a+1) x (2b+1)``:

**Correlation** slides the mask over the image as written::

    g(x, y) = sum_{s=-a..a} sum_{t=-b..b}  w(s, t) . f(x + s, y + t)

**Convolution** rotates the mask by 180 degrees first::

    g(x, y) = sum_{s=-a..a} sum_{t=-b..b}  w(s, t) . f(x - s, y - t)

The two agree whenever the mask is symmetric, which covers mean and Gaussian
smoothing -- and is why the terms get used loosely.  They differ in sign for
antisymmetric masks such as Sobel or emboss, so both are implemented here and
the game's filters are explicit about which one they want.

Two implementations of each
---------------------------
``correlate2d_naive`` is the direct four-nested-loop transcription of the
formula above.  It is the version to read, and the version the tests treat as
ground truth.

``correlate2d`` computes exactly the same result but loops over the *mask*
(nine iterations for a 3x3) instead of over the pixels, accumulating shifted
copies of the padded image.  Same arithmetic, same result -- but roughly three
orders of magnitude faster, which is what makes a full-screen filter flash
possible at 60 FPS.  ``tests/test_imaging.py`` asserts the two agree.
"""

from __future__ import annotations

import numpy as np

from .quantise import to_uint8, to_uint8_scaled

PadMode = str

#: Border handling strategies.
PAD_MODES = ("zero", "replicate", "reflect", "wrap")


# ---------------------------------------------------------------------------
# Border padding
# ---------------------------------------------------------------------------


def pad_image(image: np.ndarray, pad_y: int, pad_x: int, mode: PadMode = "replicate") -> np.ndarray:
    """Extend an image border so a mask can be centred on edge pixels.

    ``zero``
        Pad with 0.  Simple, but darkens the border of a smoothed image.
    ``replicate``
        Repeat the outermost row/column.  The default: it introduces no false
        edges, which matters because a false border edge would light up every
        edge detector in this package.
    ``reflect``
        Mirror the image about its border.
    ``wrap``
        Treat the image as periodic.
    """
    if pad_y == 0 and pad_x == 0:
        return image.astype(np.float64, copy=True)

    array = image.astype(np.float64, copy=False)
    height, width = array.shape[:2]

    if mode == "zero":
        shape = (height + 2 * pad_y, width + 2 * pad_x) + array.shape[2:]
        out = np.zeros(shape, dtype=np.float64)
        out[pad_y : pad_y + height, pad_x : pad_x + width] = array
        return out

    if mode == "replicate":
        row_index = np.clip(np.arange(-pad_y, height + pad_y), 0, height - 1)
        col_index = np.clip(np.arange(-pad_x, width + pad_x), 0, width - 1)
    elif mode == "reflect":
        row_index = _reflect_index(np.arange(-pad_y, height + pad_y), height)
        col_index = _reflect_index(np.arange(-pad_x, width + pad_x), width)
    elif mode == "wrap":
        row_index = np.arange(-pad_y, height + pad_y) % height
        col_index = np.arange(-pad_x, width + pad_x) % width
    else:
        raise ValueError(f"unknown pad mode {mode!r}; expected one of {PAD_MODES}")

    # Fancy-index the rows, then the columns.  Works unchanged for grayscale
    # (H, W) and colour (H, W, 3) images.
    return array[row_index][:, col_index]


def _reflect_index(indices: np.ndarray, size: int) -> np.ndarray:
    """Fold out-of-range indices back into ``[0, size)`` by mirroring."""
    period = 2 * size - 2 if size > 1 else 1
    folded = np.abs(indices) % period
    return np.where(folded >= size, period - folded, folded)


# ---------------------------------------------------------------------------
# Reference implementations -- the readable ones
# ---------------------------------------------------------------------------


def correlate2d_naive(
    image: np.ndarray, kernel: np.ndarray, mode: PadMode = "replicate"
) -> np.ndarray:
    """Direct transcription of the correlation sum.  Grayscale only.

    Kept deliberately slow and literal: four nested loops, one multiply-
    accumulate per mask element per pixel.  This is the version to point at
    when explaining what correlation *is*.
    """
    if image.ndim != 2:
        raise ValueError("correlate2d_naive expects a 2D grayscale image")

    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2

    padded = pad_image(image, pad_y, pad_x, mode)
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            total = 0.0
            for s in range(kh):
                for t in range(kw):
                    total += kernel[s, t] * padded[y + s, x + t]
            output[y, x] = total
    return output


def convolve2d_naive(
    image: np.ndarray, kernel: np.ndarray, mode: PadMode = "replicate"
) -> np.ndarray:
    """Convolution = correlation with the mask rotated 180 degrees."""
    return correlate2d_naive(image, np.flip(np.asarray(kernel, dtype=np.float64)), mode)


# ---------------------------------------------------------------------------
# Vectorised implementations -- the fast ones
# ---------------------------------------------------------------------------


def correlate2d(
    image: np.ndarray,
    kernel: np.ndarray,
    mode: PadMode = "replicate",
    dtype: type = np.float64,
) -> np.ndarray:
    """Correlate a 2D or 3-channel image with ``kernel``.

    Rather than visiting every pixel in Python, the loop runs over the *mask*.
    For each mask element ``w(s, t)`` the whole padded image is shifted by
    ``(s, t)`` and accumulated with weight ``w(s, t)``.  Summing those shifted,
    weighted copies is precisely the correlation sum, just reorganised so the
    per-pixel work happens inside NumPy.

    A 3x3 mask therefore costs nine array operations regardless of image size.
    Zero-valued mask entries are skipped entirely, which is a real saving for
    sparse masks such as the Laplacian or a motion blur.

    ``dtype`` selects the working precision.  ``float64`` is the default and is
    what the tests check against the naive implementation.  The live game passes
    ``float32``, which halves the memory traffic -- and memory traffic, not
    arithmetic, is what limits this loop on a full-screen frame.  Both are far
    more precise than the 8-bit result needs.
    """
    array = np.asarray(image)
    kernel = np.asarray(kernel, dtype=dtype)
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2

    # Colour images are handled in a single pass rather than one call per
    # channel: the padded array carries its channel axis along untouched, and
    # the accumulation broadcasts over it.  One pad instead of three matters
    # when this runs on a full 1000x700 frame during gameplay.
    padded = pad_image(array, pad_y, pad_x, mode).astype(dtype, copy=False)
    height, width = array.shape[:2]
    output = np.zeros(array.shape, dtype=dtype)
    scratch = np.empty(array.shape, dtype=dtype)

    for s in range(kh):
        for t in range(kw):
            weight = kernel[s, t]
            if weight == 0.0:
                continue
            # Multiply into a reused scratch buffer and accumulate in place, so
            # the loop allocates nothing per mask element.
            np.multiply(padded[s : s + height, t : t + width], weight, out=scratch)
            np.add(output, scratch, out=output)
    return output


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    mode: PadMode = "replicate",
    dtype: type = np.float64,
) -> np.ndarray:
    """Convolve an image with ``kernel`` (mask rotated 180 degrees first)."""
    return correlate2d(image, np.flip(np.asarray(kernel, dtype=np.float64)), mode, dtype)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def apply_kernel(
    image: np.ndarray,
    kernel: np.ndarray,
    mode: PadMode = "replicate",
    offset: float = 0.0,
    use_convolution: bool = False,
) -> np.ndarray:
    """Filter an image and return a display-ready ``uint8`` result.

    ``offset`` is added before clipping.  Masks whose weights sum to zero (any
    derivative mask) produce signed output centred on zero; adding an offset of
    128 maps that to mid-grey so both polarities of edge stay visible instead
    of half of them being clipped to black.
    """
    operator = convolve2d if use_convolution else correlate2d
    result = operator(image, kernel, mode) + float(offset)
    return to_uint8(result)


def separable_correlate(
    image: np.ndarray,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    mode: PadMode = "replicate",
) -> np.ndarray:
    """Apply a separable mask as two 1D passes.

    A rank-1 mask ``K = k_y k_x^T`` can be applied as a horizontal pass
    followed by a vertical one.  For an ``n x n`` mask that cuts the cost from
    ``n^2`` operations per pixel to ``2n`` -- the reason Gaussian blur is cheap
    at large radii.  Both Gaussian and box masks are separable.
    """
    row = np.asarray(kernel_x, dtype=np.float64).reshape(1, -1)
    column = np.asarray(kernel_y, dtype=np.float64).reshape(-1, 1)
    return correlate2d(correlate2d(image, row, mode), column, mode)


def is_separable(kernel: np.ndarray, tolerance: float = 1e-8) -> bool:
    """Test whether a mask is rank 1, and therefore separable."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        return False
    return bool(np.linalg.matrix_rank(kernel, tol=tolerance) <= 1)


def normalise_response(response: np.ndarray) -> np.ndarray:
    """Rescale an arbitrary filter response to the full 0-255 display range.

    Derivative responses can be negative and can exceed 255; scaling by the
    actual min and max keeps every value visible instead of clipping the tails.
    """
    response = np.asarray(response, dtype=np.float64)
    return to_uint8_scaled(response, float(response.min()), float(response.max()))
