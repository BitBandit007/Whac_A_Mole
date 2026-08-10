#!/usr/bin/env python3
"""
Generate every report figure without playing a round.

Useful when writing the lab report: it builds a representative game frame,
runs it through the whole ``imaging`` package, and writes the figures into
``captures/``.

Usage::

    python tools/generate_report.py                 # all figures from a synthetic frame
    python tools/generate_report.py --image shot.png  # use your own screenshot
    python tools/generate_report.py --sheet         # one PNG per lab operation

The ``--sheet`` mode is the one to run before writing up: it produces a
separate labelled image for every single operation in the catalogue, which is
what you paste into the report next to each algorithm's explanation.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running this file directly from the tools/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from graphics.raster import Canvas
from imaging import enhance


def build_demo_frame() -> np.ndarray:
    """Render a board with one mole of every type, plus the hammer and HUD.

    A frame with all six mole types visible gives every filter something to
    work on -- flat regions, hard edges, fine whisker detail and saturated
    colour all in one image.
    """
    from game.board import draw_hole_fronts, render_background
    from game.entities import Hammer, Mole, build_holes, draw_hammer, draw_mole
    from game.hud import draw_hud

    holes = build_holes()
    canvas = Canvas(config.WINDOW_WIDTH, config.WINDOW_HEIGHT, config.COL_BACKDROP_TOP)
    render_background(canvas, holes)

    kinds = ["normal", "blur", "sharpen", "edge", "emboss", "golden"]
    for index, kind in enumerate(kinds):
        mole = Mole(hole=holes[index])
        mole.spawn(kind, 1.0)
        mole.state = "up"
        draw_mole(canvas, mole, 0.0)

    draw_hole_fronts(canvas, holes)
    draw_hammer(canvas, Hammer(x=520.0, y=430.0))

    draw_hud(
        canvas,
        {
            "score": 4820,
            "lives": 2,
            "time_left": 41.0,
            "combo": 6,
            "multiplier": 2,
            "difficulty": "NORMAL",
            "life_pulse": 0.0,
            "effect_name": None,
            "effect_color": config.COL_TEXT,
        },
    )
    return canvas.snapshot()


def load_frame(path: str) -> np.ndarray:
    image = plt.imread(path)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]          # drop alpha
    return image


def generate_sheet(frame: np.ndarray, folder: str) -> list[str]:
    """Write one labelled PNG per operation in the lab catalogue."""
    from labs.catalogue import CATALOGUE, apply_operation
    from imaging import resample

    source = resample.downsample(frame, 2)
    written: list[str] = []

    for category, operations in CATALOGUE:
        slug_category = category.lower().replace(" ", "-")
        for name, description in operations:
            slug = name.replace(" ", "-").replace("(", "").replace(")", "")
            try:
                result = enhance.to_rgb(apply_operation(category, name, source))
            except Exception as error:
                print(f"  !! {category} / {name}: {error.__class__.__name__}: {error}")
                continue

            figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            figure.patch.set_facecolor("#14122a")
            for axis, (image, title) in zip(axes, ((source, "Original"), (result, name))):
                axis.imshow(image)
                axis.set_title(title, color="#e8e8f4", fontsize=11)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_color("#4a4478")
            figure.suptitle(f"{category}  |  {description}", color="#ffd054", fontsize=10)
            figure.tight_layout(rect=(0, 0, 1, 0.93))

            path = os.path.join(folder, f"op-{slug_category}-{slug}.png")
            figure.savefig(path, dpi=100, bbox_inches="tight", facecolor=figure.get_facecolor())
            plt.close(figure)
            written.append(path)
            print(f"  {os.path.basename(path)}")

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate CSE 452 report figures.")
    parser.add_argument("--image", help="use this PNG instead of a synthetic game frame")
    parser.add_argument(
        "--sheet",
        action="store_true",
        help="also write one labelled figure per lab operation",
    )
    arguments = parser.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(root, config.CAPTURE_DIR)
    os.makedirs(folder, exist_ok=True)

    if arguments.image:
        print(f"loading {arguments.image} ...")
        frame = load_frame(arguments.image)
    else:
        print("rendering a demonstration frame ...")
        frame = build_demo_frame()
        plt.imsave(os.path.join(folder, "demo-frame.png"), frame)

    print("building the main report figures ...")
    from labs.post_game_report import save_all_reports

    for path in save_all_reports(frame):
        print(f"  {os.path.basename(path)}")

    if arguments.sheet:
        print("building the per-operation sheet ...")
        generate_sheet(frame, folder)

    print(f"\ndone -- figures are in {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
