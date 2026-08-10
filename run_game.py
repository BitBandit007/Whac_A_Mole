from __future__ import annotations

import argparse
import sys

import config


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_game.py",
        description="Whac-A-Mole built from 2D graphics algorithms and DIP techniques.",
    )
    parser.add_argument(
        "--difficulty",
        choices=config.DIFFICULTY_ORDER,
        default="NORMAL",
        help="starting difficulty (default: NORMAL)",
    )
    parser.add_argument("--mute", action="store_true", help="start with sound disabled")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="render one frame headlessly and exit; verifies the install without a window",
    )
    return parser.parse_args(argv)


def check_dependencies() -> bool:
    """Report clearly on anything missing rather than dying on an ImportError."""
    missing: list[str] = []

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    try:
        import pygame  # noqa: F401
    except ImportError:
        missing.append("pygame-ce  (or pygame on Python <= 3.13)")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[warn] matplotlib is not installed -- the post-game report will be skipped.")

    if missing:
        print("Missing required packages: " + ", ".join(missing))
        print("Install them with:\n    pip install -r requirements.txt")
        return False
    return True


def selftest() -> int:
    """Render a single frame without opening a window.

    Useful on a lab machine where the display or the audio device may be
    unavailable: it exercises the whole pipeline -- board scan-conversion, mole
    transformation, clipping, the HUD font and an image-processing pass -- and
    writes the frame to ``captures/selftest.png``.
    """
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import numpy as np

    from game.board import render_background
    from game.entities import build_holes, Hammer, Mole, draw_hammer, draw_mole
    from game.hud import draw_hud
    from graphics.raster import Canvas
    from imaging import edges, filters

    print("selftest: building canvas ...")
    canvas = Canvas(config.WINDOW_WIDTH, config.WINDOW_HEIGHT, config.COL_BACKDROP_TOP)
    holes = build_holes()
    render_background(canvas, holes)

    print("selftest: drawing moles and hammer ...")
    for index, kind in enumerate(("normal", "blur", "sharpen", "edge", "emboss", "golden")):
        mole = Mole(hole=holes[index])
        mole.spawn(kind, 1.0)
        mole.state = "up"
        draw_mole(canvas, mole, 0.0)

    hammer = Hammer(x=520.0, y=430.0)
    draw_hammer(canvas, hammer)

    draw_hud(
        canvas,
        {
            "score": 1234,
            "lives": 2,
            "time_left": 41.0,
            "combo": 6,
            "multiplier": 2,
            "difficulty": "NORMAL",
            "life_pulse": 0.0,
            "effect_name": "sobel gradient",
            "effect_color": config.MOLE_TYPE_COLOR["edge"],
        },
    )

    print("selftest: running image-processing pass ...")
    frame = canvas.snapshot()
    assert filters.mean_filter(frame, 3).shape == frame.shape
    assert edges.sobel(frame).shape == frame.shape[:2]

    os.makedirs(config.CAPTURE_DIR, exist_ok=True)
    path = os.path.join(config.CAPTURE_DIR, "selftest.png")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.imsave(path, frame)
        print(f"selftest: wrote {path}")
    except ImportError:
        print("selftest: matplotlib missing, skipped writing the PNG")

    print("selftest: OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)

    if not check_dependencies():
        return 1

    if arguments.selftest:
        return selftest()

    from game.engine import main as run_game

    return run_game(difficulty=arguments.difficulty, muted=arguments.mute)


if __name__ == "__main__":
    sys.exit(main())
