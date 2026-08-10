"""
The post-game Matplotlib report.

This is the deliverable named in the project brief: after a round ends, the
final frame is processed with the course's image-processing algorithms and the
original is displayed alongside the filtered versions.

Two figures are produced:

``show_filter_report``
    The brief's requirement -- original, blurred, sharpened, edge-detected and
    embossed, plus the intensity histogram.

``show_pipeline_report``
    A walk through the five stages of Canny, which is the one algorithm in the
    syllabus whose intermediate steps are more instructive than its output.

Both run on a downsampled copy of the frame, and both save a PNG next to the
project as well as opening a window, so the figures can go straight into the
lab report.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np

import config
from imaging import edges, enhance, filters, kernels, morphology, resample, segmentation
from imaging.convolution import correlate2d


def _capture_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(root, config.CAPTURE_DIR)
    os.makedirs(folder, exist_ok=True)
    return folder


def _save(figure, stem: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(_capture_dir(), f"{stem}-{stamp}.png")
    figure.savefig(path, dpi=120, bbox_inches="tight", facecolor=figure.get_facecolor())
    return path


def _style_axis(axis, title: str) -> None:
    axis.set_title(title, fontsize=10, color="#e8e8f4", pad=6)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#4a4478")


# ---------------------------------------------------------------------------
# The brief's report
# ---------------------------------------------------------------------------


def build_filter_report(frame: np.ndarray, downsample_factor: int = 2):
    """Build the original-versus-filtered figure.  Returns ``(figure, path)``."""
    import matplotlib

    import matplotlib.pyplot as plt

    source = resample.downsample(np.asarray(frame, dtype=np.uint8), downsample_factor)

    panels = [
        ("Original frame", source, None),
        ("Mean filter 5x5", filters.mean_filter(source, 5), "blur mole"),
        ("Laplacian sharpen", filters.sharpen(source, 1.0), "sharpen mole"),
        ("Sobel gradient magnitude", edges.sobel(source), "edge mole"),
        ("Emboss", filters.emboss(source), "emboss mole"),
        ("Histogram equalisation", enhance.histogram_equalization(source), "golden mole"),
        ("Median filter 3x3", filters.median_filter(source, 3), None),
        ("Canny edges", edges.canny(source), None),
    ]

    figure, axes = plt.subplots(3, 3, figsize=(13.5, 8.6))
    figure.patch.set_facecolor("#14122a")

    for axis, (title, image, tag) in zip(axes.ravel(), panels):
        display = image if image.ndim == 3 else image
        axis.imshow(display, cmap=None if display.ndim == 3 else "gray", vmin=None if display.ndim == 3 else 0, vmax=None if display.ndim == 3 else 255)
        _style_axis(axis, title if tag is None else f"{title}\n({tag})")
        axis.set_facecolor("#14122a")

    # Ninth panel: the intensity histogram of the original frame.
    histogram_axis = axes.ravel()[8]
    counts = enhance.histogram(source)
    histogram_axis.bar(np.arange(256), counts, width=1.0, color="#7ad2ff")
    otsu_level = segmentation.otsu_threshold(source)
    histogram_axis.axvline(otsu_level, color="#ffd054", linewidth=1.6, label=f"Otsu = {otsu_level}")
    histogram_axis.set_title("Intensity histogram", fontsize=10, color="#e8e8f4", pad=6)
    histogram_axis.set_facecolor("#1c1838")
    histogram_axis.tick_params(colors="#9a97bb", labelsize=7)
    histogram_axis.legend(fontsize=8, facecolor="#1c1838", edgecolor="#4a4478", labelcolor="#e8e8f4")
    for spine in histogram_axis.spines.values():
        spine.set_color("#4a4478")

    figure.suptitle(
        "Whac-A-Mole  |  CSE 452  |  Final frame processed with the course's spatial filters",
        color="#ffd054",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    return figure, _save(figure, "filter-report")


# ---------------------------------------------------------------------------
# Canny pipeline walkthrough
# ---------------------------------------------------------------------------


def build_pipeline_report(frame: np.ndarray, downsample_factor: int = 2):
    """Show every intermediate stage of the Canny detector."""
    import matplotlib.pyplot as plt

    source = resample.downsample(np.asarray(frame, dtype=np.uint8), downsample_factor)
    gray = enhance.to_gray(source).astype(np.float64)

    smoothed = correlate2d(gray, kernels.gaussian_kernel(5, 1.2))
    gx = correlate2d(smoothed, kernels.SOBEL_X)
    gy = correlate2d(smoothed, kernels.SOBEL_Y)
    magnitude = edges.gradient_magnitude(gx, gy)
    direction = edges.gradient_direction(gx, gy)
    suppressed = edges.non_maximum_suppression(magnitude, direction)

    peak = float(suppressed.max()) or 1.0
    final = edges.hysteresis_threshold(suppressed, peak * 0.10, peak * 0.26)

    panels = [
        ("1. Grayscale", gray, "gray"),
        ("2. Gaussian smoothed", smoothed, "gray"),
        ("3. Gradient magnitude", magnitude, "gray"),
        ("4. Gradient direction", direction, "twilight"),
        ("5. Non-maximum suppression", suppressed, "gray"),
        ("6. Hysteresis threshold", final, "gray"),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(13.5, 6.4))
    figure.patch.set_facecolor("#14122a")

    for axis, (title, image, cmap) in zip(axes.ravel(), panels):
        axis.imshow(image, cmap=cmap)
        _style_axis(axis, title)

    figure.suptitle(
        "Canny edge detection, stage by stage  |  CSE 452 Week 9",
        color="#ffd054",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    return figure, _save(figure, "canny-pipeline")


# ---------------------------------------------------------------------------
# Morphology and segmentation walkthrough
# ---------------------------------------------------------------------------


def build_morphology_report(frame: np.ndarray, downsample_factor: int = 2):
    """Otsu segmentation followed by the four basic morphological operations."""
    import matplotlib.pyplot as plt

    source = resample.downsample(np.asarray(frame, dtype=np.uint8), downsample_factor)
    level = segmentation.otsu_threshold(source)
    binary = morphology.binarize(source, level)
    element = morphology.square_se(3)

    panels = [
        ("Original", enhance.to_gray(source)),
        (f"Otsu threshold (T = {level})", morphology.to_display(binary)),
        ("Dilation", morphology.to_display(morphology.dilate(binary, element))),
        ("Erosion", morphology.to_display(morphology.erode(binary, element))),
        ("Opening", morphology.to_display(morphology.opening(binary, element))),
        ("Closing", morphology.to_display(morphology.closing(binary, element))),
        ("Morphological gradient", morphology.to_display(morphology.morphological_gradient(binary, element))),
        ("Boundary extraction", morphology.to_display(morphology.boundary_extraction(binary, element))),
    ]

    figure, axes = plt.subplots(2, 4, figsize=(15.0, 6.2))
    figure.patch.set_facecolor("#14122a")

    for axis, (title, image) in zip(axes.ravel(), panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        _style_axis(axis, title)

    figure.suptitle(
        "Segmentation and morphology  |  CSE 452 Weeks 10-11",
        color="#ffd054",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    return figure, _save(figure, "morphology-report")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def show_filter_report(frame: np.ndarray, block: bool = False) -> str | None:
    """Build and display the filter report.  Returns the saved path."""
    return _show(build_filter_report, frame, block)


def show_pipeline_report(frame: np.ndarray, block: bool = False) -> str | None:
    return _show(build_pipeline_report, frame, block)


def show_morphology_report(frame: np.ndarray, block: bool = False) -> str | None:
    return _show(build_morphology_report, frame, block)


def _show(builder, frame: np.ndarray, block: bool) -> str | None:
    """Shared plumbing: build, save, and display without stalling the game.

    ``plt.show(block=False)`` keeps the game responsive while the figure window
    is open.  If no interactive backend is available -- a headless machine, or
    matplotlib falling back to Agg -- the figure is still written to disk and
    the path returned, so nothing is lost.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[report] matplotlib is not installed; skipping figure")
        return None

    try:
        figure, path = builder(frame)
    except Exception as error:
        print(f"[report] could not build figure: {error.__class__.__name__}: {error}")
        return None

    try:
        plt.show(block=block)
    except Exception:
        pass

    print(f"[report] saved {path}")
    return path


def save_all_reports(frame: np.ndarray) -> list[str]:
    """Generate every figure and return the list of saved paths.

    Used by ``tools/generate_report.py`` to produce the images for the written
    lab report in one command.
    """
    paths: list[str] = []
    for builder in (build_filter_report, build_pipeline_report, build_morphology_report):
        try:
            _, path = builder(frame)
            paths.append(path)
        except Exception as error:
            print(f"[report] {builder.__name__} failed: {error}")
    return paths
