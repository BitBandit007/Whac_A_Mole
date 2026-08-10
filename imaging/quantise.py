"""
Converting filter results back to 8-bit pixels.

Every operation in this package computes in floating point and has to land
back on the 0-255 integer grid.  How that final step is done matters more than
it looks, so it lives in one place rather than being repeated at forty call
sites.

The trap
--------
``numpy``'s ``astype(np.uint8)`` **truncates** towards zero.  A mean filter
applied to a perfectly flat image of intensity 140 computes
``140 * (1/25) * 25``, which in binary floating point is 139.99999999999997 --
and truncation turns that into **139**.  The filter darkens an image it should
have left untouched, and repeated filtering walks the whole image towards
black.

Rounding to nearest fixes it: 139.99999999999997 rounds to 140, and a filter
whose weights sum to one now provably preserves a constant region.

``np.rint`` rounds halves to even, which is the unbiased choice.  Rounding
halves upward would add a systematic +0.5 bias of about 0.2 percent brightness
per pass -- invisible once, obvious after a chain of filters.
"""

from __future__ import annotations

import numpy as np


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Clip to ``[0, 255]``, round to nearest, and cast to ``uint8``.

    This is the single exit point from floating-point image arithmetic back to
    displayable pixels.
    """
    return np.rint(np.clip(array, 0, 255)).astype(np.uint8)


def to_uint8_scaled(array: np.ndarray, low: float, high: float) -> np.ndarray:
    """Rescale ``[low, high]`` onto ``[0, 255]`` and quantise.

    Used for signed filter responses, where the interesting range is not
    already 0-255.  A degenerate range (``high == low``) maps to black rather
    than dividing by zero.
    """
    span = float(high) - float(low)
    if abs(span) < 1e-12:
        return np.zeros(np.shape(array), dtype=np.uint8)
    return to_uint8((np.asarray(array, dtype=np.float64) - low) * (255.0 / span))


def to_binary_image(mask: np.ndarray) -> np.ndarray:
    """Render a boolean mask as an 8-bit black-and-white image.

    No rounding is involved -- the values are exactly 0 and 255 -- but routing
    it through here keeps every "produce a displayable image" path in one
    module.
    """
    return (np.asarray(mask).astype(np.uint8)) * 255
