"""
``game`` -- the Whac-A-Mole application layer.

This package contains no algorithms.  It is the part that decides *what* to
draw and *when*; every actual pixel is produced by ``graphics``, and every
image effect by ``imaging``.  Keeping that boundary strict is what lets the two
algorithm packages be read, tested and marked on their own.

Module map
----------
``entities``   Hole, Mole, Hammer and Particle -- state and geometry.
``board``      Pre-renders the static background; draws the hole front lips.
``hud``        Score panel, timer bar, lives, combo meter, overlays.
``effects``    Applies image-processing filters to the live frame.
``audio``      Procedurally synthesised sound effects (no asset files).
``highscore``  JSON-backed high-score table.
``engine``     State machine and main loop.
"""
