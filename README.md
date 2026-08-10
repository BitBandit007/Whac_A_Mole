# Whac-A-Mole

**An interactive game built with 2D computer graphics algorithms and digital image processing techniques**

Course: **CSE 452 — Graphics & Image Processing Lab**
Department of Computer Science and Engineering
Bangladesh University of Business and Technology (BUBT)

| Name | ID | Intake |
|---|---|---|
| Md. Mohaimin Mahin | 20235203067 | 44-02 |
| Tahmid Hasan Siam | 20235203054 | 44-02 |

---

## What this is

A complete, playable arcade game in which **every visible pixel is produced by an algorithm from the CSE 452 syllabus.**

There is no `pygame.draw.circle`, no `pygame.draw.line`, no font engine, no OpenCV, no SciPy, no `skimage`. The game renders into a plain NumPy array by running hand-written scan-conversion algorithms, and pygame is used only to open a window, read input, keep time, and put the finished array on screen.

That single design decision is what ties the two halves of the course together: because the framebuffer *is* an image matrix, the digital image processing routines can filter the live game screen with no conversion step at all. Hit a **blur mole** and a mean filter runs over the actual frame you are looking at.

---

## Quick start

```bash
pip install -r requirements.txt
python run_game.py
```

Other ways to run it:

```bash
python run_game.py --difficulty HARD    # start on hard
python run_game.py --mute               # no sound
python run_game.py --selftest           # render one frame headlessly and exit
```

`--selftest` needs no display or sound card. Use it to check the install on a lab machine, or over remote desktop; it writes `captures/selftest.png` and exits.

**Requires Python 3.9+.** On Python 3.13 and newer install `pygame-ce` (already in `requirements.txt`); it imports as `pygame`, so nothing in the code changes.

---

## How to play

The hammer is **mouse-only** — there is no keyboard aiming or swinging.

| Input | Action |
|---|---|
| Move mouse | Aim the hammer |
| Left click | Swing |
| `←` `→` (menu) | Change difficulty |
| `Enter` (menu) | Start |
| `P` | Pause |
| `Esc` | Quit (from the menu), resume (from pause) |
| `F1` | FPS / debug overlay |

The system cursor is hidden — the hammer *is* the pointer. Mouse sensitivity is
`MOUSE_SENSITIVITY` in `config.py` (default 1.75): the hammer travels 1.75 px
for every pixel the pointer moves, measured from the centre of the play field,
so less physical movement covers the board. The mapping stays one-to-one, so
returning the pointer to a spot always returns the hammer to the same spot.

Hit moles before they escape. Every mole that gets away costs a life. Four consecutive hits raise the score multiplier, up to ×5; a miss or an escape resets it.

### The moles

Each mole type is bound to one image-processing operation, applied to the live play field the moment you hit it.

| Mole | Colour | Operation | Points |
|---|---|---|---|
| Plain | brown | — | 10 |
| Blur | cyan | Mean filter 5×5 | 15 |
| Sharpen | orange | Identity + Laplacian | 15 |
| Edge | green | Sobel gradient magnitude | 15 |
| Emboss | violet | Emboss mask + mid-grey bias | 15 |
| Golden | gold | Histogram equalisation | 50 |

---

## High scores

The best round for **each mode** is kept in **`highscores.txt`** next to the
project — plain text, so it can be opened and read during the demo:

```
# MODE   |  SCORE | HITS | ACCURACY | DATE
# -------+--------+------+----------+-----------------
  EASY   |   6650 |   71 |    90.2% | 2026-08-09 23:52
  NORMAL |   4820 |   52 |    82.5% | 2026-08-09 23:52
  HARD   |   3110 |   41 |    76.0% | 2026-08-09 23:52
```

Exactly one row per mode — a worse round leaves the file untouched. The menu
shows the best for whichever difficulty is currently selected, so switching
between EASY / NORMAL / HARD switches the target you are chasing, and the
game-over screen only says *NEW HIGH SCORE* when you beat that mode's own best.

A mode that has never been played simply has no row. A missing, hand-edited or
garbled file loads as an empty table rather than crashing the game, and only
the unparseable rows are dropped.

---

## Reports for the write-up

The game is deliberately just the game — no browsing UI and no screenshots
written behind your back. The figures for the written report come from a
separate tool:

```bash
python tools/generate_report.py            # renders its own demonstration frame
python tools/generate_report.py --sheet    # plus one labelled figure per operation
python tools/generate_report.py --image myscreenshot.png
```

Three figures are produced, all saved into `captures/`:

1. **Filter report** — the final frame beside its blurred, sharpened, edge-detected, embossed, equalised, median-filtered and Canny versions, plus the intensity histogram with Otsu's threshold marked. *(This is the figure named in §3 of the project proposal.)*
2. **Canny pipeline** — all six stages of the detector: grayscale → Gaussian → gradient magnitude → gradient direction → non-maximum suppression → hysteresis.
3. **Segmentation and morphology** — Otsu binarisation followed by dilation, erosion, opening, closing, gradient and boundary extraction.

`--sheet` additionally writes a separate labelled before/after PNG for **every
one of the 49 operations** in `labs/catalogue.py`, across five categories:

- **Spatial filters** — mean, Gaussian, median, min, max, sharpen, unsharp mask, emboss, motion blur, posterise
- **Edge detection** — Roberts, Prewitt, Sobel, Scharr, separate G<sub>x</sub>/G<sub>y</sub>, Laplacian (4- and 8-connected), Laplacian of Gaussian, Marr–Hildreth, Canny
- **Segmentation** — Otsu, iterative (isodata), multi-level Otsu, adaptive mean, adaptive Gaussian, Niblack
- **Morphology** — dilation, erosion, opening, closing, gradient, boundary, top-hat, black-hat, hit-or-miss, thinning
- **Enhancement** — grayscale, negative, log, gamma, contrast stretch, histogram equalisation (luminance and per-channel), bit-plane slicing

That is the mode to run before writing up: one figure per algorithm, each with
its caption already written.

---

## Project layout

```
├── run_game.py              entry point (--selftest, --difficulty, --mute)
├── config.py                every tunable constant
│
├── graphics/                2D raster graphics, from scratch
│   ├── raster.py            framebuffer, pixel/span writes, clip stack
│   ├── line.py              DDA and Bresenham
│   ├── circle.py            Bresenham circle, midpoint circle, midpoint ellipse
│   ├── polygon.py           outlines, scan-line fill, shape constructors
│   ├── transform2d.py       homogeneous 3×3 translate / rotate / scale / shear
│   ├── clipping.py          Cohen–Sutherland, Liang–Barsky, Sutherland–Hodgman
│   └── text.py              vector stroke font drawn with Bresenham
│
├── imaging/                 digital image processing, from scratch
│   ├── kernels.py           named masks + Gaussian / LoG / separable builders
│   ├── convolution.py       correlation and convolution (naive and vectorised)
│   ├── filters.py           smoothing, rank and sharpening filters
│   ├── edges.py             Roberts → Prewitt → Sobel → Laplacian → LoG → Canny
│   ├── enhance.py           point operations and histogram processing
│   ├── segmentation.py      Otsu, isodata, multi-level, adaptive thresholding
│   ├── morphology.py        dilation, erosion, opening, closing, hit-or-miss
│   ├── resample.py          nearest-neighbour and bilinear interpolation
│   └── quantise.py          the single float → uint8 exit point
│
├── game/                    application layer (no algorithms live here)
│   ├── entities.py          Hole, Mole, Hammer, Particle + their geometry
│   ├── board.py             static background render, hole front lips
│   ├── hud.py               score panel, timer, lives, menus, overlays
│   ├── effects.py           threaded filter flashes
│   ├── audio.py             procedurally synthesised sound (no asset files)
│   ├── highscore.py         plain-text high-score table
│   └── engine.py            state machine and main loop
│
├── labs/
│   ├── catalogue.py         all 49 operations + captions + dispatcher
│   └── post_game_report.py  the Matplotlib figures
│
├── tests/                   291 tests
├── tools/generate_report.py batch figure generation
└── docs/
    ├── COURSE_MAPPING.md    syllabus topic → file → function
    └── VIVA_NOTES.md        the questions you should expect, answered
```

The boundary between `graphics`/`imaging` and `game` is strict: the two algorithm packages are standalone libraries that know nothing about Whac-A-Mole, and the game layer contains no algorithms. Each can be read and marked on its own.

---

## Tests

```bash
python -m pytest tests -q
```

291 tests, about 12 seconds. They run headlessly (SDL's dummy video and audio drivers), so no display is needed.

They check *algorithmic properties*, not saved screenshots:

- Bresenham is 8-connected, hits both endpoints, stays within half a pixel of the true line, and gives the same figure drawn in either direction
- DDA and Bresenham agree on all eight octants
- Circle pixels lie within ¾ of a pixel of the true radius and have full 8-fold symmetry
- A filled disc's pixel count matches π r² to within 2 %
- Cohen–Sutherland and Liang–Barsky return identical results across 4 000 random segments
- Clipping never increases polygon area
- **The vectorised convolution matches the naive four-nested-loop version exactly** — this is the important one; the fast path used at runtime is verified against the literal definition of the formula
- A mean filter preserves a constant image; every derivative mask annihilates one
- A median filter deletes salt-and-pepper noise *and* preserves a step edge, where a mean filter cannot
- Erosion and dilation are duals; opening and closing are idempotent and correctly ordered
- Hysteresis keeps weak edge pixels connected to strong ones and drops isolated ones
- Rotation preserves distance; `compose(A, B)` applies B first; inverse round-trips
- The engine runs end to end: every state renders, the clip stack never leaks, the effect thread delivers

---

## Notes on the implementation

A few decisions worth knowing about, because they are the ones a marker is most likely to ask about.

**The background is scan-converted once.** A pure-Python rasteriser cannot redraw a full board sixty times a second. The static board is rendered into its own canvas at start-up and copied over the live frame each tick; only the moving objects are re-rasterised. Start-up costs a few hundred milliseconds, and the game holds 60 FPS.

**Area fills use spans, not per-pixel loops.** The scan-conversion algorithm still decides *which* pixels belong to the shape — it produces one `(y, x_start, x_end)` run per scan-line — and NumPy performs the memory write. The algorithm is ours; only the byte-copy is delegated.

**Convolution loops over the mask, not the image.** `correlate2d_naive` is the direct four-nested-loop transcription of the formula and is what the tests treat as ground truth. The version used at runtime computes the same sum reorganised as nine shifted, weighted array additions — identical arithmetic, roughly a thousand times faster. Both are kept, and the tests assert they agree.

**Filter flashes run on a worker thread.** A full-screen 3×3 correlation takes about 70 ms. Doing that inside the game loop would drop four frames on every hit, so it is handed to a background thread; NumPy releases the interpreter lock during array arithmetic, so it genuinely runs in parallel while the game keeps rendering.

**Clipping is load-bearing, not decorative.** A mole rising out of its hole is a full-size mole standing mostly below the rim, with its silhouette clipped by Sutherland–Hodgman against a window whose bottom edge *is* the hole rim. Remove the clipper and moles slide over the board instead of emerging from it. There is a test for exactly this: nothing may be drawn below `hole.rim_y`.

**All text is vector strokes.** Each glyph is a set of polylines in a 4×6 design grid, scaled by a transformation matrix and rasterised with Bresenham. The font is an application of two syllabus topics rather than an imported asset.

**Sound is synthesised, not sampled.** Every effect is a NumPy waveform built at start-up, so the repository contains no binary audio files. If no audio device is available the module disables itself and the game runs silently.

**One float → uint8 exit point.** `numpy`'s `astype(np.uint8)` truncates: a mean filter over a flat image of intensity 140 computes 139.99999999999997 and stores **139**, darkening an image it should have left untouched. Every conversion goes through `imaging/quantise.py`, which rounds to nearest instead. There is a test asserting a mean filter preserves a constant image.

---

## Attribution

`graphics/` and `imaging/` are original implementations written from the algorithm definitions in the course textbooks:

- *Computer Graphics* (Schaum's Outline) — Plastock & Kalley
- *Digital Image Processing* — Gonzalez & Woods

NumPy, pygame and Matplotlib are used only for array storage, windowing/input/audio, and plotting respectively.
