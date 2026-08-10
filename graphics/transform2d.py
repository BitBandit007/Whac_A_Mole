"""
2D geometric transformations in homogeneous coordinates.

Syllabus reference: CSE 452, Weeks 4-5 -- "Geometric transformation of basic
shapes such as rotation, translation, scaling" and "Modeling 2D transformation
with keyboard and mouse to control movement of objects".

Every transformation is a 3x3 matrix acting on column vectors ``[x, y, 1]^T``.
The third homogeneous row is what makes translation -- which is *not* a linear
map on its own -- expressible as a matrix, and therefore composable with
rotation and scaling by plain matrix multiplication.

    | x' |   | a  b  tx | | x |
    | y' | = | c  d  ty | | y |
    | 1  |   | 0  0  1  | | 1 |

Nothing in the game moves an object by adding an offset to its coordinates.
The mole's pop-up is a scaling matrix, the hammer's swing is a rotation about
a pivot, and particles are driven by a composed translate-rotate-scale chain.
Object geometry is always stored in its own *local* coordinate system and
mapped to screen coordinates by a matrix built fresh each frame.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Primitive matrices
# ---------------------------------------------------------------------------


def identity() -> np.ndarray:
    """The 3x3 identity -- leaves every point where it is."""
    return np.eye(3, dtype=np.float64)


def translation(tx: float, ty: float) -> np.ndarray:
    """Translation by ``(tx, ty)``.

        | 1  0  tx |
        | 0  1  ty |
        | 0  0  1  |
    """
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 2] = float(tx)
    matrix[1, 2] = float(ty)
    return matrix


def scaling(sx: float, sy: float | None = None, about: Point | None = None) -> np.ndarray:
    """Scaling by ``(sx, sy)``, optionally about a fixed point.

        | sx  0   0 |
        | 0   sy  0 |
        | 0   0   1 |

    A bare scaling matrix always scales about the origin.  Scaling about an
    arbitrary fixed point ``p`` is the classic three-step composition
    ``T(p) . S . T(-p)``: move the fixed point to the origin, scale, move back.
    The game uses this so a mole grows *out of its hole* rather than out of the
    top-left corner of the screen.
    """
    if sy is None:
        sy = sx
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = float(sx)
    matrix[1, 1] = float(sy)
    if about is None:
        return matrix
    px, py = about
    return compose(translation(px, py), matrix, translation(-px, -py))


def rotation(angle_degrees: float, about: Point | None = None) -> np.ndarray:
    """Rotation by ``angle_degrees``, optionally about a pivot point.

        | cos(t)  -sin(t)  0 |
        | sin(t)   cos(t)  0 |
        | 0        0       1 |

    Screen coordinates put +y downwards, so a positive angle appears to rotate
    *clockwise* on the monitor.  The hammer relies on that: its strike angle is
    a positive number.

    As with scaling, rotating about a pivot is ``T(p) . R . T(-p)`` -- the
    hammer rotates about the base of its handle, not about its centroid.
    """
    theta = math.radians(float(angle_degrees))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = cos_t
    matrix[0, 1] = -sin_t
    matrix[1, 0] = sin_t
    matrix[1, 1] = cos_t
    if about is None:
        return matrix
    px, py = about
    return compose(translation(px, py), matrix, translation(-px, -py))


def shear(shx: float = 0.0, shy: float = 0.0) -> np.ndarray:
    """Shear transformation.

        | 1    shx  0 |
        | shy  1    0 |
        | 0    0    1 |

    Used for the squash-and-stretch on a mole that has just been hit.
    """
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 1] = float(shx)
    matrix[1, 0] = float(shy)
    return matrix


def reflection_x() -> np.ndarray:
    """Reflect about the x axis (negate y)."""
    return scaling(1.0, -1.0)


def reflection_y() -> np.ndarray:
    """Reflect about the y axis (negate x)."""
    return scaling(-1.0, 1.0)


def reflection_origin() -> np.ndarray:
    """Reflect through the origin (negate both axes)."""
    return scaling(-1.0, -1.0)


# ---------------------------------------------------------------------------
# Composition and application
# ---------------------------------------------------------------------------


def compose(*matrices: np.ndarray) -> np.ndarray:
    """Multiply matrices left to right: ``compose(A, B, C)`` returns ``A @ B @ C``.

    Because points are column vectors, the *rightmost* matrix is applied to the
    point first.  Reading a composition right-to-left therefore reads it in
    execution order -- a detail worth stating explicitly, since reversing it is
    the single most common transformation bug.
    """
    result = identity()
    for matrix in matrices:
        result = result @ matrix
    return result


def apply(matrix: np.ndarray, points: Sequence[Point]) -> list[Point]:
    """Transform a sequence of points by ``matrix``.

    The points are promoted to homogeneous coordinates, multiplied in one
    batch, then projected back by dividing through by ``w``.  (For the affine
    matrices used here ``w`` is always 1, but the division keeps the routine
    correct if a projective matrix is ever passed in.)
    """
    if not len(points):
        return []
    array = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([array, np.ones(len(array))])   # N x 3
    transformed = homogeneous @ matrix.T                          # N x 3
    w = transformed[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1.0, w)
    cartesian = transformed[:, :2] / w
    return [(float(px), float(py)) for px, py in cartesian]


def apply_point(matrix: np.ndarray, x: float, y: float) -> Point:
    """Transform a single point."""
    vector = np.array([float(x), float(y), 1.0])
    result = matrix @ vector
    w = result[2] if abs(result[2]) > 1e-12 else 1.0
    return (float(result[0] / w), float(result[1] / w))


def inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert a transformation matrix.

    Needed for hit testing: rather than transforming every mole into screen
    space to see whether the hammer struck it, the hammer's impact point can be
    pulled back into the mole's local space with the inverse matrix.
    """
    return np.linalg.inv(matrix)


# ---------------------------------------------------------------------------
# Convenience builders used by the game
# ---------------------------------------------------------------------------


def trs(
    translate: Point = (0.0, 0.0),
    rotate_degrees: float = 0.0,
    scale: Point = (1.0, 1.0),
) -> np.ndarray:
    """Build the standard translate-rotate-scale chain.

    Returns ``T . R . S``, so a point is scaled first, then rotated, then
    translated -- the order that keeps rotation and scaling anchored at the
    object's own origin.
    """
    return compose(
        translation(translate[0], translate[1]),
        rotation(rotate_degrees),
        scaling(scale[0], scale[1]),
    )


def centroid(points: Sequence[Point]) -> Point:
    """Arithmetic mean of a point set -- a convenient default pivot."""
    if not len(points):
        return (0.0, 0.0)
    array = np.asarray(points, dtype=np.float64)
    return (float(array[:, 0].mean()), float(array[:, 1].mean()))


def bounding_box(points: Sequence[Point]) -> tuple[float, float, float, float]:
    """Axis-aligned bounds ``(x_min, y_min, x_max, y_max)`` of a point set."""
    array = np.asarray(points, dtype=np.float64)
    return (
        float(array[:, 0].min()),
        float(array[:, 1].min()),
        float(array[:, 0].max()),
        float(array[:, 1].max()),
    )
