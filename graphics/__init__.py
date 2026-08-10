"""
``graphics`` -- from-scratch 2D raster graphics package.

This package contains hand-written implementations of the scan-conversion,
transformation and clipping algorithms taught in CSE 452.  Nothing in here
calls a built-in drawing primitive: every visible pixel is decided by one of
these algorithms and written into a NumPy framebuffer.

Module map
----------
``raster``       Framebuffer, pixel/span writing, clip-rectangle management.
``line``         DDA and Bresenham line generation (all eight octants).
``circle``       Bresenham circle, midpoint circle, midpoint ellipse, arcs.
``polygon``      Polygon outlines, scan-line fill, shape helpers.
``transform2d``  Homogeneous 3x3 matrices: translate / rotate / scale / shear.
``clipping``     Cohen-Sutherland, Liang-Barsky, Sutherland-Hodgman.
``text``         Vector stroke font rendered through the line rasteriser.
"""

from .raster import Canvas, Rect

__all__ = ["Canvas", "Rect"]
