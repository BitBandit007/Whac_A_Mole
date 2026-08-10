# Viva preparation

The questions a CSE 452 examiner is most likely to ask about this project, with
answers and the exact file to open.

Read this before the demo. The point of the project is that you can explain
every algorithm in it — a working game that you cannot explain scores worse
than a simpler one you can.

---

## About the project as a whole

**Q. Show me one place where you use a built-in drawing function.**

There isn't one. The framebuffer is a NumPy array of shape `(700, 1000, 3)`.
Every pixel is written by `Canvas.put_pixel()` or `Canvas.put_span()` in
`graphics/raster.py`, and the only thing that decides which pixels those are is
a scan-conversion algorithm we wrote. pygame is used for the window, the event
loop, timing and audio — `Canvas.to_pygame_surface()` is the single point of
contact, and it hands over the finished array whole.

**Q. Then why is NumPy allowed?**

It is storage and arithmetic, not graphics. `pixels[y, x] = colour` is an array
assignment. NumPy has no line, circle, filter or threshold function in use
here. The one place it does real work is span filling: our algorithm computes
*which* run of pixels belongs to the shape, and NumPy performs the memory
write.

**Q. How does the graphics half connect to the image processing half?**

Through the framebuffer. Because the finished frame is already an `(H, W, 3)`
`uint8` array, it *is* an image in exactly the form the image-processing
routines expect. There is no conversion step. Hit a blur mole and
`imaging/filters.py` runs on the same array the rasteriser just wrote to.

---

## Lines

**Q. Difference between DDA and Bresenham?**

DDA computes the slope, steps along the major axis in unit increments and
rounds: floating-point arithmetic and one round per pixel. Bresenham keeps an
integer decision variable that tracks twice the difference between the true
line and the midpoint of the two candidate pixels; the sign of that variable
picks the next pixel. Same figure, integer-only arithmetic.

Both are in `graphics/line.py`, and `tests/test_graphics.py` asserts they agree
across all eight octants.

**Q. Why is Bresenham better?**

No division, no floating point, no rounding — so it is faster and exactly
reproducible. That is why it goes into hardware.

**Q. How do you handle all eight octants without eight separate cases?**

`graphics/line.py::bresenham_points()`. Take absolute deltas `dx`, `dy`, and
separate step directions `sx`, `sy` (±1). Initialise `err = dx − dy`. Each
iteration compares `2·err` against `−dy` and `dx`; each crossing commits a step
along that axis. The direction signs handle the octant, the two comparisons
handle the slope.

**Q. Where does the +6 / +10 in the circle algorithm come from?**

That is Bresenham's *circle*, not the line. See below.

---

## Circles

**Q. Explain the midpoint circle decision parameter.**

`graphics/circle.py::midpoint_circle_points()`. The decision parameter is the
circle function evaluated at the midpoint between the two candidate pixels:

```
p = f(x+1, y−½) = (x+1)² + (y−½)² − r²
```

Multiply out and drop the constant ¼ (it never changes the sign) and you get
the integer starting value `p = 1 − r`. If `p < 0` the midpoint is inside the
circle, so keep `y` and update `p += 2x + 1`; otherwise decrement `y` and update
`p += 2(x − y) + 1`.

**Q. And Bresenham's circle?**

Same idea, different algebra: `d = 3 − 2r`, updated by `4x + 6` or
`4(x − y) + 10`. Both are implemented (`bresenham_circle_points()`), and a test
asserts the two produce almost identical pixel sets — they differ by at most a
few pixels at the octant boundary.

**Q. Why compute only one octant?**

Eight-fold symmetry. One computed point `(x, y)` gives eight pixels by sign and
coordinate swaps — `_eight_way_symmetry()`. That is the whole point of the
algorithm: one eighth of the arithmetic.

**Q. Why do you need a separate ellipse algorithm?**

An ellipse has only four-fold symmetry, and its slope passes through −1 partway
round the quadrant. So the first quadrant splits into two regions: in region 1
`x` drives the loop, and when `2ry²x ≥ 2rx²y` the algorithm switches to region 2
where `y` drives it. `midpoint_ellipse_points()`. The holes are ellipses because
the board is drawn in perspective.

---

## Transformations

**Q. Why homogeneous coordinates?**

Translation is not a linear map, so it cannot be written as a 2×2 matrix.
Adding a third coordinate makes it one, which means translation, rotation and
scaling all compose by ordinary matrix multiplication.

**Q. Show me a transformation actually driving an animation.**

`game/entities.py::Mole.matrix()`:

```python
T.compose(
    T.translation(hole.cx, hole.rim_y + sink),
    T.scaling(grow * sx, grow * sy),
)
```

`sink` is how far below the rim the mole still is, `grow` is its rise scale,
and `sx, sy` are the squash factors applied when it is struck. Nothing adds an
offset to a stored vertex — the geometry lives in local space and this matrix
maps it to the screen every frame.

`Hammer.matrix()` is `translation ∘ rotation(angle, about=pivot)`.

**Q. Why does `compose(A, B)` apply B first?**

Points are column vectors, so `A·B·p` applies `B` to `p` first. Reading a
composition right to left reads it in execution order. There is a test for
this, because reversing it is the most common transformation bug.

**Q. How do you rotate about a point that isn't the origin?**

`T(p) · R · T(−p)`: move the pivot to the origin, rotate, move back.
`graphics/transform2d.py::rotation(angle, about=...)` does exactly that, and a
test asserts the pivot is fixed. The hammer rotates about a point just below
its head; the mole scales about its hole.

**Q. Positive angle — clockwise or anticlockwise?**

Clockwise *on screen*, because screen y points downward. The maths is the
standard rotation matrix; the visual sense flips because of the coordinate
system. There is a test asserting `rotation(90)` maps `(1,0)` to `(0,1)`.

---

## Clipping

**Q. Explain Cohen–Sutherland.**

`graphics/clipping.py`. Each endpoint gets a 4-bit outcode recording which
sides of the window it lies beyond. Then:

- `code0 | code1 == 0` — both inside, **trivially accept**.
- `code0 & code1 != 0` — both beyond the same edge, **trivially reject**.
- Otherwise compute one intersection, replace the outside endpoint, repeat.

Each iteration clears at least one outcode bit, so it terminates in at most
four passes. The value is the two trivial tests: most segments are decided with
no intersection arithmetic at all.

**Q. Then why also implement Liang–Barsky?**

It is parametric rather than geometric. Write the segment as
`P(t) = P₀ + t·D`; the four window edges become four inequalities `pₖ·t ≤ qₖ`.
`pₖ < 0` means the line enters through that edge (tightening `t_enter`),
`pₖ > 0` means it leaves (tightening `t_exit`). If `pₖ = 0` the line is parallel
to that edge, and if `qₖ < 0` as well it is parallel *and* outside — reject.
It usually needs fewer iterations because it solves for the interval directly.

`tests/test_clipping.py` runs both over 4 000 random segments and requires
identical answers.

**Q. Explain Sutherland–Hodgman.**

Polygon clipping. The subject polygon is passed through the four window edges
one at a time; the output of one stage is the input of the next. For each
vertex pair: inside→inside emits the current vertex; inside→outside emits the
intersection; outside→inside emits the intersection then the vertex;
outside→outside emits nothing.

**Q. What is its limitation?**

It assumes a **convex clip window** (a rectangle qualifies). A *concave subject*
polygon can come out with degenerate connecting edges along the boundary —
harmless when filling, visible when stroking the outline. Documented in the
function's docstring.

**Q. Where is clipping actually used? (Expect this one.)**

`game/entities.py::draw_mole()`. A mole at 30 % risen is a **full-size** mole
standing 70 % below the rim, not a small mole. `Hole.clip_window()` returns a
window whose bottom edge is the hole rim, and the body silhouette is clipped
against it with Sutherland–Hodgman. Delete the clipper and moles slide over the
board instead of emerging from it.

Two mechanisms are used together: the analytic polygon clip (which produces a
genuinely shorter polygon, so the *outline stroke* doesn't draw a lid across
the mole's waist) and the canvas scissor rectangle (which discards pixels as
they are written, for the circular features).

`tests/test_game.py::test_mole_never_draws_below_its_hole_rim` fails if either
is removed.

---

## Correlation and convolution

**Q. What is the difference?**

Correlation slides the mask over the image as written:
`g(x,y) = ΣΣ w(s,t)·f(x+s, y+t)`.
Convolution rotates the mask 180° first:
`g(x,y) = ΣΣ w(s,t)·f(x−s, y−t)`.

They are identical for symmetric masks (mean, Gaussian) — which is why the
terms get used loosely — and differ in **sign** for antisymmetric ones (Sobel,
emboss). Both are implemented; `tests/test_imaging.py` asserts both facts.

**Q. Your fast implementation doesn't look like the formula. Is it still your
algorithm?**

Yes, and both versions are in the file. `correlate2d_naive()` is the direct
four-nested-loop transcription. `correlate2d()` computes the same sum
reorganised: instead of looping over pixels it loops over the **mask**, and for
each mask element accumulates the whole padded image shifted by `(s, t)` and
scaled by `w(s, t)`. Summing nine shifted weighted copies *is* the correlation
sum — just with the per-pixel work inside NumPy instead of inside Python.

A 3×3 mask costs nine array operations regardless of image size, which is what
makes a full-screen filter possible at 60 FPS. The test
`test_vectorised_matches_naive` asserts they agree exactly for every mask.

**Q. What are the border options and which did you choose?**

Zero, replicate, reflect, wrap — `pad_image()`. The default is **replicate**,
because zero padding introduces a false step at the border, and a false step is
exactly what every edge detector in the package would light up on. There is a
test showing zero padding darkens the border of a bright image while replicate
does not.

**Q. What is a separable mask?**

One that factorises as `K = k_y · k_xᵀ`. Then it can be applied as a horizontal
pass followed by a vertical one — `2n` operations per pixel instead of `n²`. Box
and Gaussian masks are separable; LoG is not. `is_separable()` tests it by
matrix rank. The adaptive thresholds use it: a 31×31 window costs 62 operations
instead of 961.

---

## Filters

**Q. Mean versus median — why keep both?**

Mean is a **linear** filter: a weighted sum, so it is a convolution, it obeys
superposition and it has a frequency response. Median is a **rank** filter: it
sorts the neighbourhood and picks the middle value, so it is not a convolution
at all.

That difference is why the median can delete salt-and-pepper noise without
smearing edges: an isolated extreme value is discarded outright rather than
averaged in, and at a step edge the majority of the window still holds the
correct level. No linear filter can do this. There are tests for both halves:
the median removes impulse noise *and* preserves a step edge that the mean
blurs.

**Q. How does sharpening work?**

`g = f + Laplacian(f)`. The Laplacian responds only where intensity is
changing, so adding it back exaggerates exactly the transitions while leaving
flat regions untouched. Fold that into one mask and you get the familiar
centre-5, neighbours-−1 kernel: `kernels.SHARPEN_4`.

Unsharp masking is the gentler version: `g = f + amount·(f − blur(f))`. The
difference `f − blur(f)` is the fine detail; `sigma` decides which frequencies
get boosted.

**Q. Why does emboss need a +128?**

It is a directional derivative, so its response is **signed**. Without a bias,
every negative slope clips to black and half the relief disappears. Adding
mid-grey maps zero change to grey, positive slopes to light and negative to
dark — which the eye reads as a surface lit from the top-left.

---

## Edge detection

**Q. Roberts, Prewitt, Sobel — why three?**

They trade noise rejection against localisation. Roberts is 2×2 diagonal
differences: sharpest, but it averages nothing so it responds to single-pixel
noise, and its even-sized mask has no true centre. Prewitt adds a uniform
average along the edge. Sobel weights the centre row twice — a 1-2-1 binomial
smoothing along the edge — which makes it noticeably steadier. Sobel is the
project default.

**Q. First order versus second order?**

First order estimates the gradient and reports its magnitude; an edge is a
**maximum**. The response is thick, because a ramp edge is steep over several
pixels.

Second order uses the Laplacian, which is zero in flat regions, zero at the
*centre* of a ramp, and changes sign across an edge; an edge is a **zero
crossing**, which localises it to a single pixel. The cost is noise:
differentiating twice amplifies noise twice.

**Q. Why LoG instead of Laplacian?**

Because a raw Laplacian on a noisy image is unusable. LoG folds Gaussian
smoothing and the second derivative into one mask — faster than doing them
separately, and `sigma` becomes an explicit scale control.

**Q. Walk me through Canny.** *(Very likely.)*

Five stages, `imaging/edges.py::canny()`:

1. **Gaussian smooth** — differentiating raw pixels amplifies noise.
2. **Sobel gradients** — magnitude and direction.
3. **Non-maximum suppression** — a gradient ridge is several pixels wide; compare each pixel with its two neighbours *along the gradient direction* and zero it unless it is the crest. The angle is quantised to 0°/45°/90°/135°, the four directions an 8-connected grid can represent.
4. **Double threshold** — above `high` is strong, between is weak, below `low` is discarded.
5. **Hysteresis** — repeatedly absorb weak pixels that touch an accepted pixel.

**Q. Why two thresholds?**

A single threshold always fails one way or the other: set it high and real
edges break into dashes; set it low and noise gets through. Two thresholds plus
connectivity keeps a faint stretch of a genuine contour (it connects to a strong
part) while rejecting an isolated faint blob.

The Canny pipeline figure from `python tools/generate_report.py` shows all six
stages side by side.

---

## Segmentation

**Q. Derive Otsu's criterion.**

Split the histogram at level `t` into two classes with weights `w₀, w₁` and
means `μ₀, μ₁`. The between-class variance is
`σ²_b(t) = w₀(t)·w₁(t)·(μ₀(t) − μ₁(t))²`.

Total variance is fixed, so **maximising the variance between the classes is
the same as minimising the variance within them** — but the between-class form
can be evaluated for all 256 levels in one pass over the histogram using
cumulative sums, instead of rescanning the image 256 times.

**Q. Your Otsu has a tie-break. Why?**

On an image with a clean gap in its histogram the criterion is *exactly equal*
for every level in the gap. A plain `argmax` returns the first, putting the
threshold hard against the lower cluster where one noisy pixel flips its class.
We take the **midpoint of the maximising plateau**, which is the most robust
point in the gap and also what the iterative method converges to.

**Q. Why is the threshold `f > T` and not `f ≥ T`?**

Because both threshold-selection methods define class 0 as `[0, T]`. Using `≥`
would shift every threshold by one level, and on an image whose lower cluster
sits exactly at `T` it would put that whole cluster in the wrong class. It is
also Gonzalez & Woods' convention.

**Q. When does a global threshold fail?**

Under uneven illumination — no single level can work when the background on one
side is brighter than the foreground on the other. That is what
`adaptive_mean_threshold()` is for: it compares each pixel with the mean of its
own neighbourhood, so the reference level follows the illumination.

**Q. Why the offset `C` in adaptive thresholding?**

Without it, a perfectly flat region comes out as pure noise — by definition half
its pixels sit either side of their own local average. The offset biases the
decision away from the local mean so flat regions resolve consistently.

---

## Morphology

**Q. Define dilation and erosion.**

`A ⊕ B = { z : (B̂)_z ∩ A ≠ ∅ }` — keep a position if the reflected structuring
element overlaps the object at all. Grows the object.

`A ⊖ B = { z : B_z ⊆ A }` — keep it only if the element fits **entirely** inside.
Shrinks the object and deletes anything smaller than the element.

**Q. Opening and closing?**

Opening is `(A ⊖ B) ⊕ B`: erode then dilate, removing small protrusions and
specks while restoring the size of what survives. Closing is `(A ⊕ B) ⊖ B`:
fills small holes and gaps.

**Q. Why are they called filters rather than just size changes?**

Because they are **idempotent** — applying either twice changes nothing the
second time. There is a test for this.

**Q. What does hit-or-miss do that erosion cannot?**

`A ⊛ B = (A ⊖ B_fg) ∩ (Aᶜ ⊖ B_bg)`. It requires the foreground element to fit
inside the object **and** the background element to fit inside the complement.
Requiring both makes it a shape *detector* rather than a size filter — it finds
exactly the configuration in the template: isolated points, corners, line ends.
The two elements must not overlap, and the code raises `ValueError` if they do.

**Q. What is the structuring element for, and does its shape matter?**

It is the probe. Its shape decides what survives: a `cross_se` preserves
rectilinear structure, a `disk_se` is isotropic so it does not favour any axis,
a `line_se` extracts structure of one orientation only.

**Q. How does this relate to the min and max filters?**

They are the same operations. Grayscale morphology replaces set union with
maximum and intersection with minimum, so grayscale dilation *is* the max filter
and grayscale erosion *is* the min filter. `grayscale_dilate()` says so and
delegates to `filters.max_filter()`.

---

## Engineering questions

**Q. How do you get 60 FPS out of pure Python?**

Three things:

1. **The static board is scan-converted once** at start-up and copied over each frame; only moving objects are re-rasterised.
2. **Area fills emit spans, not pixels.** The algorithm produces one `(y, x_start, x_end)` run per scan-line; NumPy does the memory write.
3. **Convolution loops over the mask, not the image** — nine array operations for a 3×3, regardless of resolution.

**Q. A full-screen filter takes 70 ms. How is that not four dropped frames?**

It runs on a worker thread (`game/effects.py`). NumPy releases the interpreter
lock during array arithmetic, so the filter genuinely runs in parallel while the
game keeps rendering. The result arrives a frame or two later and is blended in
as the flash fades up, which hides the latency.

The trade-off, stated honestly: the filtered frame is a **frozen snapshot**, not
the live one. Over a half-second flash that reads as an intentional freeze; the
alternative would be recomputing 70 ms of work every frame.

**Q. You mentioned a rounding bug. What was it?**

`numpy`'s `astype(np.uint8)` **truncates**. A mean filter over a flat image of
intensity 140 computes `140 · (1/25) · 25`, which in binary floating point is
139.99999999999997 — truncated to **139**. The filter darkened an image it
should have left untouched, and repeated filtering would walk the whole image
towards black.

Every conversion now goes through `imaging/quantise.py::to_uint8()`, which
rounds to nearest (`np.rint`, halves-to-even, so there is no upward bias).
`test_mean_filter_preserves_a_constant_image` is the regression test.

**Q. How do you know your fast code is correct?**

291 tests, and the important ones compare an optimised implementation against
the literal definition: vectorised correlation against the four-nested-loop
version, vectorised dilation against the naive one, Cohen–Sutherland against
Liang–Barsky over 4 000 random segments. The rest assert defining properties —
a mean filter preserves a constant, a derivative mask annihilates one, opening
is idempotent, erosion and dilation are duals.

```bash
python -m pytest tests -q
```

---

## If something goes wrong in the demo

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: pygame` | `pip install -r requirements.txt` — on Python 3.13+ it must be `pygame-ce` |
| No window / running over remote desktop | `python run_game.py --selftest` renders a frame to `captures/selftest.png` with no display |
| No sound | Expected on some lab machines; `game/audio.py` disables itself and prints one line. The game plays normally |
| Asked for the Matplotlib report | `python tools/generate_report.py --sheet` — figures land in `captures/` |
| Asked where scores are kept | `highscores.txt` next to the project — open it in Notepad |
| Matplotlib window doesn't open | The figures are still written to `captures/` — open them from there |
| Slow first frame | The static board is scan-converted at start-up; it happens once |
