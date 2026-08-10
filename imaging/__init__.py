"""
``imaging`` -- from-scratch digital image processing package.

Every operation in this package is implemented from its mathematical
definition.  No filtering, thresholding or morphological routine is imported
from OpenCV, SciPy or ``skimage``; NumPy is used only for array storage and
elementwise arithmetic.

Image convention
----------------
Images are NumPy arrays.  Grayscale images have shape ``(H, W)``, colour images
``(H, W, 3)``.  The public entry points accept ``uint8`` input and return
``uint8`` unless documented otherwise; the arithmetic in between is done in
``float64`` so that intermediate results are not clipped or wrapped.

Module map
----------
``kernels``       Named convolution masks (mean, Gaussian, Sobel, emboss, ...).
``convolution``   Correlation and convolution, naive and vectorised.
``filters``       Smoothing and sharpening spatial filters.
``edges``         Roberts, Prewitt, Sobel, Laplacian, LoG and Canny.
``enhance``       Point operations: negative, log, gamma, histogram equalisation.
``segmentation``  Otsu, iterative, multi-level and local thresholding.
``morphology``    Dilation, erosion, opening, closing, hit-or-miss.
"""

from .convolution import convolve2d, correlate2d
from .enhance import to_gray

__all__ = ["convolve2d", "correlate2d", "to_gray"]
