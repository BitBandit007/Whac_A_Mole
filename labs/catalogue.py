"""
The catalogue of image-processing operations, and a dispatcher for them.

Pure data plus one function -- no user interface.  The game itself demonstrates
five operations, one per effect mole; this catalogue is what
``tools/generate_report.py`` walks to produce a labelled before/after figure
for every operation in the ``imaging`` package, which is what goes into the
written lab report next to each algorithm's explanation.

Each entry is ``(operation name, one-line explanation)``.  The explanations are
the captions for the report figures.
"""

from __future__ import annotations

import numpy as np

from imaging import edges, enhance, filters, morphology, segmentation

#: ``(category, [(operation name, one-line explanation), ...])``
CATALOGUE: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "SPATIAL FILTERS",
        [
            ("mean 3x3", "Unweighted average of the 3x3 neighbourhood. Blurs noise and edges alike."),
            ("mean 5x5", "Wider box mask. Stronger blur, and the square mask starts to show."),
            ("gaussian 5x5", "Distance-weighted average. Natural blur with no boxy artefacts."),
            ("median 3x3", "Order statistic, not a sum. Removes impulse noise but keeps edges sharp."),
            ("min 3x3", "Local minimum. Grayscale erosion: bright regions shrink."),
            ("max 3x3", "Local maximum. Grayscale dilation: bright regions grow."),
            ("sharpen", "Identity plus Laplacian. Adds the detected detail back onto the image."),
            ("unsharp mask", "Original plus the difference from its own blur. Gentler than a raw Laplacian."),
            ("emboss", "Antisymmetric diagonal mask plus a mid-grey bias. Reads as surface relief."),
            ("motion blur", "Averaging along one row only. Simulates movement during exposure."),
            ("posterize", "Quantises to a few grey levels. Shows what reducing bit depth costs."),
        ],
    ),
    (
        "EDGE DETECTION",
        [
            ("roberts", "2x2 diagonal differences. Smallest and fastest operator, and the noisiest."),
            ("prewitt", "3x3 central difference with uniform averaging along the edge."),
            ("sobel", "Prewitt with 1-2-1 weighting. Better noise rejection; the project default."),
            ("scharr", "Sobel-like weights tuned for rotational symmetry."),
            ("sobel gx", "Horizontal derivative alone. Responds only to vertical edges."),
            ("sobel gy", "Vertical derivative alone. Responds only to horizontal edges."),
            ("laplacian 4", "Second derivative, 4-connected. Isotropic, no preferred direction."),
            ("laplacian 8", "Second derivative including diagonals. Stronger, noisier."),
            ("laplacian of gaussian", "Smoothing and second derivative in one mask. The Mexican hat."),
            ("marr-hildreth", "LoG followed by zero-crossing detection. Single-pixel contours."),
            ("canny", "Smooth, differentiate, thin, double threshold, track. Thin connected edges."),
        ],
    ),
    (
        "SEGMENTATION",
        [
            ("otsu", "Threshold maximising between-class variance. Computed from the histogram."),
            ("iterative", "Repeatedly average the two classes and split at the midpoint."),
            ("multilevel (3 class)", "Otsu generalised to two thresholds and three regions."),
            ("adaptive mean", "One threshold per pixel from its own neighbourhood mean."),
            ("adaptive gaussian", "Local threshold with distance weighting. Smoother than the box version."),
            ("niblack", "Local mean plus k times the local standard deviation."),
        ],
    ),
    (
        "MORPHOLOGY",
        [
            ("dilation", "Keep a position if the structuring element touches the object at all."),
            ("erosion", "Keep it only if the structuring element fits entirely inside."),
            ("opening", "Erode then dilate. Deletes specks smaller than the element."),
            ("closing", "Dilate then erode. Fills holes smaller than the element."),
            ("gradient", "Dilation minus erosion. A band straddling every boundary."),
            ("boundary", "Object minus its erosion. The inner boundary."),
            ("top hat", "Object minus its opening. Isolates small bright detail."),
            ("black hat", "Closing minus the object. Isolates small dark detail."),
            ("hit-or-miss (corner)", "Fit and anti-fit at once. Detects one exact configuration."),
            ("thinning", "Object minus its hit-or-miss response. One step towards a skeleton."),
        ],
    ),
    (
        "ENHANCEMENT",
        [
            ("grayscale", "Luminance-weighted collapse to one channel. Green dominates."),
            ("negative", "s = 255 - r. Reveals detail buried inside large dark regions."),
            ("log transform", "s = c log(1 + r). Expands shadows, compresses highlights."),
            ("gamma 0.45", "Power law below 1. Brightens and lifts dark tones."),
            ("gamma 2.20", "Power law above 1. Darkens and stretches bright tones."),
            ("contrast stretch", "Linear rescale between two intensity percentiles."),
            ("histogram equalise", "Transform by the image's own CDF. Approximates a flat histogram."),
            ("histogram equalise rgb", "Per-channel equalisation. Note how the colours shift."),
            ("bit plane 7", "The most significant bit. Carries the recognisable structure."),
            ("bit plane 3", "A middle bit. Structure is already largely gone."),
            ("bit plane 0", "The least significant bit. Essentially noise."),
        ],
    ),
]


def apply_operation(category: str, name: str, image: np.ndarray) -> np.ndarray:
    """Run one catalogue entry against ``image``.

    Raises ``KeyError`` for an unknown category or operation, so a typo in the
    catalogue fails loudly rather than silently returning the input.
    """
    if category == "SPATIAL FILTERS":
        return filters.apply_named_filter(image, name)

    if category == "EDGE DETECTION":
        return edges.apply_named_edge(image, name)

    if category == "SEGMENTATION":
        return segmentation.apply_named_segmentation(image, name)

    if category == "MORPHOLOGY":
        return morphology.apply_named_morphology(image, name)

    if category == "ENHANCEMENT":
        table = {
            "grayscale": enhance.to_gray,
            "negative": enhance.negative,
            "log transform": enhance.log_transform,
            "gamma 0.45": lambda img: enhance.gamma_correction(img, 0.45),
            "gamma 2.20": lambda img: enhance.gamma_correction(img, 2.20),
            "contrast stretch": enhance.contrast_stretch,
            "histogram equalise": enhance.histogram_equalization,
            "histogram equalise rgb": enhance.histogram_equalization_rgb,
            "bit plane 7": lambda img: enhance.bit_plane(img, 7),
            "bit plane 3": lambda img: enhance.bit_plane(img, 3),
            "bit plane 0": lambda img: enhance.bit_plane(img, 0),
        }
        if name not in table:
            raise KeyError(f"unknown enhancement operation {name!r}")
        return table[name](image)

    raise KeyError(f"unknown category {category!r}")


def operation_count() -> int:
    """Total number of operations in the catalogue."""
    return sum(len(operations) for _, operations in CATALOGUE)
