# Syllabus → code map

Every topic in the CSE 452 course outline, and exactly where it is implemented
and used. Week numbers follow the course plan in the course outline document.

---

## Graphics half

### Week 1 — DDA and Bresenham line drawing

| | |
|---|---|
| **Implemented in** | `graphics/line.py` |
| **Functions** | `dda_points()`, `bresenham_points()`, `bresenham_iter()`, `draw_line()`, `draw_thick_line()`, `draw_dashed_line()` |
| **Used by** | every outline in the game; the stroke font (`graphics/text.py`); the dashed grid and diagonal backdrop stripes (`game/board.py`) |
| **Visible where** | the backdrop stripes are drawn with **DDA**, everything else with **Bresenham** — deliberately, so both appear in the finished frame |
| **Tested by** | `tests/test_graphics.py` — endpoints, 8-connectivity, ≤ ½-pixel deviation from the true line, reversibility, DDA/Bresenham agreement across all 8 octants |

Bresenham is the default because it is integer-only: the decision variable
`err = dx - dy` is updated by comparing `2·err` against `-dy` and `dx`, so
there is no division, no floating point and no rounding.

### Week 2 — Bresenham and midpoint circle drawing

| | |
|---|---|
| **Implemented in** | `graphics/circle.py` |
| **Functions** | `bresenham_circle_points()` (`d = 3 − 2r`), `midpoint_circle_points()` (`p = 1 − r`), `midpoint_ellipse_points()` (two-region), `draw_arc()`, `fill_circle()`, `fill_ellipse()`, `fill_ring()` |
| **Used by** | mole ears, eyes, pupils, nose, snout, chest, cap; hole openings and rims (ellipses, because the board is drawn in perspective); border studs; particles; the hammer highlight |
| **Tested by** | `tests/test_graphics.py` — every pixel within ¾ px of the true radius, full 8-fold symmetry, midpoint ≈ Bresenham, ellipse equation satisfied, filled area matches π r² to 2 % |

Both circle algorithms are kept and both are used, so the report can show that
two different decision variables produce the same figure.

### Week 3 — Basic shapes

| | |
|---|---|
| **Implemented in** | `graphics/polygon.py` |
| **Functions** | `draw_polygon()`, `fill_polygon()`, `polygon_spans()`, `regular_polygon()`, `star_polygon()`, `rectangle()`, `rounded_rectangle()`, `capsule()`, `heart()`, `arrow()`, `signed_area()`, `point_in_polygon()` |
| **Used by** | mole body silhouette (`capsule`); hammer head (`rounded_rectangle`) and handle; HUD panels; lives indicator (`heart`); particles (`star_polygon`, `regular_polygon`) |
| **Tested by** | `tests/test_graphics.py` — rectangle and triangle fill areas match the shoelace formula, concave (arrow) fill leaves the notch empty, `point_in_polygon` agrees with the fill |

The fill is the standard **scan-line / edge-list** algorithm with the even-odd
rule. Edges are treated as half-open in y (`y_min ≤ y < y_max`) so that a
scan-line passing exactly through a vertex contributes the right number of
crossings — the classic vertex bug, handled explicitly.

### Weeks 4–5 — Geometric transformations, keyboard and mouse control

| | |
|---|---|
| **Implemented in** | `graphics/transform2d.py` |
| **Functions** | `translation()`, `rotation()`, `scaling()`, `shear()`, `reflection_*()`, `compose()`, `apply()`, `apply_point()`, `inverse()`, `trs()` |
| **Used by** | **mole rise** — `game/entities.py::Mole.matrix()` composes translate ∘ scale; **hammer swing** — `Hammer.matrix()` rotates about a pivot; **hit squash** — non-uniform scaling; **particles** — full TRS chain; **the stroke font** — every glyph is scaled and translated by a matrix |
| **Mouse control** | `game/engine.py::_aim_at_pointer()` — the pointer position is mapped to the hammer by scaling its offset from the field centre (a 2D scaling about a fixed point, the same construction as `scaling(about=...)`), then clamped to the field |
| **Tested by** | `tests/test_graphics.py` — rotation preserves distance and is clockwise on screen, rotation/scaling about a pivot fixes that pivot, `compose(A,B)` applies B first, inverse round-trips, shear has determinant 1 |

All matrices are 3×3 in homogeneous coordinates. No object ever moves by
adding an offset to its stored vertices — geometry is stored once in local
space and mapped to the screen by a matrix rebuilt each frame.

The hammer's collision point is read back **out of** its matrix
(`Hammer.impact_point` calls `apply_point(matrix, 0, 0)`), so the hit box can
never disagree with what is drawn.

### Week 8 — Clipping algorithms

| | |
|---|---|
| **Implemented in** | `graphics/clipping.py` |
| **Functions** | `region_code()`, `cohen_sutherland()`, `liang_barsky()`, `sutherland_hodgman()`, `clip_polygon_against_edge()`, `clip_polyline()` |
| **Used by** | **mole emergence** — `game/entities.py::draw_mole()` clips the body silhouette against `Hole.clip_window()`, whose bottom edge is the hole rim; **hammer and particles** — clipped to the play field; **scissor clipping** — `Canvas.push_clip()` / `pop_clip()` discards pixels as they are written |
| **Tested by** | `tests/test_clipping.py` — outcodes for every region, trivial accept/reject, **Cohen–Sutherland and Liang–Barsky give identical results on 4 000 random segments**, clipped endpoints always inside the window, clipping never increases polygon area |

This is the topic most likely to be implemented and then not actually used.
Here it is load-bearing: `tests/test_game.py::test_mole_never_draws_below_its_hole_rim`
fails if the clipper is removed.

---

## Image processing half

### Week 6 — Spatial filters and image enhancement

| | |
|---|---|
| **Implemented in** | `imaging/convolution.py`, `imaging/kernels.py`, `imaging/filters.py`, `imaging/enhance.py` |
| **Correlation / convolution** | `correlate2d_naive()` and `convolve2d_naive()` (four nested loops — the literal formula); `correlate2d()` and `convolve2d()` (vectorised over the mask); `separable_correlate()`; `pad_image()` with zero / replicate / reflect / wrap |
| **Smoothing** | `mean_filter()`, `gaussian_filter()`, `motion_blur()` |
| **Rank filters** | `median_filter()`, `min_filter()`, `max_filter()`, `midpoint_filter()`, `alpha_trimmed_mean_filter()` |
| **Sharpening** | `sharpen()`, `unsharp_mask()`, `high_boost()`, `emboss()` |
| **Enhancement** | `negative()`, `log_transform()`, `gamma_correction()`, `contrast_stretch()`, `histogram()`, `histogram_equalization()`, `bit_plane()` |
| **Used by** | the effect moles (`game/effects.py`); every report figure |
| **Tested by** | `tests/test_imaging.py` — **vectorised == naive for every mask**, correlation ≠ convolution for asymmetric masks, mean preserves a constant, derivative masks annihilate a constant, median removes salt-and-pepper *and* preserves a step edge, gamma < 1 brightens, equalisation widens a narrow histogram |

The correlation/convolution distinction is made explicit rather than glossed
over: the two agree for symmetric masks (mean, Gaussian) and differ in sign for
antisymmetric ones (Sobel, emboss), and there is a test for both facts.

### Week 9 — Edge detection: first order, second order, Canny

| | |
|---|---|
| **Implemented in** | `imaging/edges.py` |
| **First order** | `roberts()`, `prewitt()`, `sobel()`, `scharr()`, `sobel_components()` |
| **Second order** | `laplacian()` (4- and 8-connected), `laplacian_of_gaussian()`, `zero_crossings()`, `marr_hildreth()` |
| **Canny** | `canny()` — with `non_maximum_suppression()` and `hysteresis_threshold()` exposed separately so each stage can be shown on its own |
| **Used by** | the **edge mole**; `labs/post_game_report.py::build_pipeline_report()` renders all six stages side by side |
| **Tested by** | `tests/test_imaging.py` — each operator localises a step edge to within ±2 columns, G<sub>x</sub> and G<sub>y</sub> respond to perpendicular edges, NMS thins a 3-px ridge to 1 px, hysteresis keeps connected weak pixels and drops isolated ones, Canny output is strictly binary and silent on a flat image |

Canny is the full five-stage pipeline: Gaussian smoothing → Sobel gradients →
non-maximum suppression → double threshold → connectivity-driven edge tracking.
Thresholds are expressed as ratios of the peak gradient so the detector adapts
to frame contrast, with an absolute floor so a flat frame cannot have its own
floating-point noise amplified into false edges.

### Week 10 — Image segmentation: multi-level, local, threshold detection

| | |
|---|---|
| **Implemented in** | `imaging/segmentation.py` |
| **Threshold detection** | `iterative_threshold()` (isodata), `otsu_threshold()` (between-class variance) |
| **Multi-level** | `multilevel_threshold()` — Otsu generalised to two thresholds, exhaustive over all 32 640 ordered pairs; `apply_multilevel()` |
| **Local thresholding** | `adaptive_mean_threshold()`, `adaptive_gaussian_threshold()`, `niblack_threshold()` |
| **Used by** | `build_morphology_report()`; the Otsu level is marked on the histogram in the filter report; the `--sheet` figures |
| **Tested by** | `tests/test_imaging.py` — Otsu finds the valley of a bimodal histogram and agrees with isodata, multi-level returns ordered thresholds and three distinct levels, adaptive thresholding succeeds on an illumination gradient where a global threshold cannot |

All the local methods use **separable** box and Gaussian passes: a 31×31 window
is applied as two 1D passes (62 operations per pixel) instead of one 2D pass
(961), which is what makes them fast enough to run over a batch of figures.

Otsu's tie-break is documented: on a histogram with a clean gap the criterion
is exactly flat across the gap, so the **midpoint of the maximising plateau** is
taken rather than the first maximum, which would sit hard against the lower
cluster.

### Week 11 — Morphological operations

| | |
|---|---|
| **Implemented in** | `imaging/morphology.py` |
| **Structuring elements** | `square_se()`, `cross_se()`, `disk_se()`, `line_se()` |
| **Basic** | `dilate()`, `erode()`, `opening()`, `closing()` — plus `dilate_naive()` as the readable reference |
| **Derived** | `morphological_gradient()`, `boundary_extraction()`, `top_hat()`, `black_hat()` |
| **Hit-or-miss** | `hit_or_miss()`, `corner_detector_se()`, `isolated_point_se()`, `thin_once()` |
| **Grayscale** | `grayscale_dilate()`, `grayscale_erode()` (identical to the max and min filters — the connection is noted in the code) |
| **Used by** | `build_morphology_report()`; the `--sheet` report figures |
| **Tested by** | `tests/test_imaging.py` — vectorised == naive, **erosion and dilation are duals**, opening removes specks, closing fills holes, both are idempotent, opening is anti-extensive and closing extensive, hit-or-miss finds an isolated point and rejects overlapping structuring elements |

Each operation is documented in its set-theoretic definition, e.g.
`A ⊖ B = { z : B_z ⊆ A }`, and the code follows that definition directly.

---

## Beyond the outline

Included because they make the project work, and each is a recognised topic:

| Topic | Where | Why it is there |
|---|---|---|
| Scan-line polygon fill | `graphics/polygon.py` | filling anything at all |
| Vector stroke font | `graphics/text.py` | text without a font engine — an application of Bresenham + transformations |
| Nearest-neighbour & bilinear interpolation | `imaging/resample.py` | scaling frames into the report figure panels |
| Anti-aliasing by blending | `graphics/raster.py` | `blend_pixel()`, `blend_span()` for glows and fades |
| Separable convolution | `imaging/convolution.py`, `imaging/kernels.py` | 2n instead of n² per pixel |
| Quantisation and rounding | `imaging/quantise.py` | one documented float → uint8 exit point |
| Procedural audio synthesis | `game/audio.py` | no binary assets in the repository |

---

## Where to look first

If a marker has five minutes:

1. `imaging/convolution.py` — the naive and vectorised implementations side by side, with the test that proves they agree.
2. `graphics/clipping.py` + `game/entities.py::draw_mole()` — clipping doing real work.
3. `graphics/transform2d.py` + `Mole.matrix()` / `Hammer.matrix()` — transformations driving the animation.
4. The figures from `python tools/generate_report.py --sheet` — one per algorithm, each captioned.
