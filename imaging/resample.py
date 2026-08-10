"""
Image resampling: nearest-neighbour and bilinear interpolation.

Resizing an image means asking for intensities at coordinates that do not
exist, so a rule is needed for inventing them.  Two rules are implemented here,
and the difference between them is one of the clearest illustrations of what
interpolation actually buys you.

``resize_nearest``
    Round the source coordinate and take that pixel.  Exact, fast, and
    preserves hard edges perfectly -- which is why it is the right choice for
    displaying a binary mask or an edge map, where a blended value would be
    meaningless.  Enlarging with it produces visible blocks.

``resize_bilinear``
    Take the four surrounding pixels and blend them by distance.  Smooth
    results when enlarging, and much better behaved when shrinking, but it
    softens genuine edges.

Both are used by the in-game lab viewer to fit processed frames into their
display panels.
"""

from __future__ import annotations

import numpy as np

from .quantise import to_uint8


def resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize by nearest-neighbour sampling.

    The source coordinate for output pixel ``i`` is taken at the *centre* of
    that pixel, ``(i + 0.5) . scale - 0.5``, rather than at ``i . scale``.
    Sampling at the corner instead biases the whole image half a pixel up and
    to the left -- a small error that becomes obvious when an image is
    resampled repeatedly.
    """
    array = np.asarray(image)
    src_h, src_w = array.shape[:2]
    width, height = max(1, int(width)), max(1, int(height))

    row_scale = src_h / float(height)
    col_scale = src_w / float(width)

    rows = np.clip(((np.arange(height) + 0.5) * row_scale - 0.5).round(), 0, src_h - 1).astype(np.int64)
    cols = np.clip(((np.arange(width) + 0.5) * col_scale - 0.5).round(), 0, src_w - 1).astype(np.int64)

    return array[rows][:, cols]


def resize_bilinear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize by bilinear interpolation.

    For each output pixel the four surrounding source pixels are weighted by
    how close they are:

        f(x, y) = (1-a)(1-b) f00 + a(1-b) f10 + (1-a)b f01 + a b f11

    where ``a`` and ``b`` are the fractional parts of the source coordinate.
    Equivalent to interpolating along x twice and then once along y, which is
    where the name comes from.
    """
    array = np.asarray(image, dtype=np.float64)
    src_h, src_w = array.shape[:2]
    width, height = max(1, int(width)), max(1, int(height))

    row_scale = src_h / float(height)
    col_scale = src_w / float(width)

    src_y = np.clip((np.arange(height) + 0.5) * row_scale - 0.5, 0, src_h - 1)
    src_x = np.clip((np.arange(width) + 0.5) * col_scale - 0.5, 0, src_w - 1)

    y0 = np.floor(src_y).astype(np.int64)
    x0 = np.floor(src_x).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)

    b = (src_y - y0)[:, None]
    a = (src_x - x0)[None, :]

    if array.ndim == 3:
        b = b[:, :, None]
        a = a[:, :, None]

    f00 = array[np.ix_(y0, x0)] if array.ndim == 2 else array[y0][:, x0]
    f10 = array[np.ix_(y0, x1)] if array.ndim == 2 else array[y0][:, x1]
    f01 = array[np.ix_(y1, x0)] if array.ndim == 2 else array[y1][:, x0]
    f11 = array[np.ix_(y1, x1)] if array.ndim == 2 else array[y1][:, x1]

    top = f00 * (1.0 - a) + f10 * a
    bottom = f01 * (1.0 - a) + f11 * a
    return to_uint8(top * (1.0 - b) + bottom * b)


def scale_to_fit(image: np.ndarray, max_width: int, max_height: int, smooth: bool = True) -> np.ndarray:
    """Resize an image to fit inside a box while preserving its aspect ratio."""
    array = np.asarray(image)
    src_h, src_w = array.shape[:2]
    if src_h == 0 or src_w == 0:
        return array

    factor = min(max_width / float(src_w), max_height / float(src_h))
    width = max(1, int(round(src_w * factor)))
    height = max(1, int(round(src_h * factor)))

    resizer = resize_bilinear if smooth else resize_nearest
    return resizer(array, width, height)


def downsample(image: np.ndarray, factor: int = 2) -> np.ndarray:
    """Shrink by an integer factor after box-averaging.

    Decimating without averaging first is *aliasing*: high-frequency detail
    that the smaller grid cannot represent folds back as false low-frequency
    patterns.  Averaging each block before dropping samples removes that detail
    before it can alias -- the spatial-domain form of a pre-filter.
    """
    array = np.asarray(image, dtype=np.float64)
    factor = max(1, int(factor))
    if factor == 1:
        return np.asarray(image).astype(np.uint8, copy=True)

    height, width = array.shape[:2]
    trimmed_h = (height // factor) * factor
    trimmed_w = (width // factor) * factor
    array = array[:trimmed_h, :trimmed_w]

    if array.ndim == 3:
        blocks = array.reshape(trimmed_h // factor, factor, trimmed_w // factor, factor, array.shape[2])
        return to_uint8(blocks.mean(axis=(1, 3)))

    blocks = array.reshape(trimmed_h // factor, factor, trimmed_w // factor, factor)
    return to_uint8(blocks.mean(axis=(1, 3)))
