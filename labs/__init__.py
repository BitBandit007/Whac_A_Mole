"""
``labs`` -- report generation for the written submission.

The game demonstrates five image-processing operations through its effect
moles.  The syllabus covers considerably more than five, so this package makes
the rest reachable for the report:

``catalogue``
    Every operation in the ``imaging`` package, with a one-line explanation of
    each, plus a dispatcher.  Pure data -- no user interface.

``post_game_report``
    The Matplotlib figures described in the project brief: the final frame of a
    round shown alongside its blurred, sharpened, edge-detected and embossed
    versions, plus a stage-by-stage walk through Canny and a segmentation /
    morphology sheet.

Both are driven by ``tools/generate_report.py``, which renders its own
demonstration frame -- a board with one mole of every type, the hammer and the
HUD, so every filter has flat regions, hard edges, fine whisker detail and
saturated colour to work on::

    python tools/generate_report.py --sheet

Pass ``--image`` to use a screenshot of your own instead.  The game does not
write screenshots by itself.
"""
