"""
High-score persistence: the best round for each difficulty, in a text file.

The table holds **exactly one row per mode** -- the best round ever played on
EASY, on NORMAL and on HARD.  A rolling history of every round made the file
grow and made "is this a high score?" ambiguous; one row per mode is what the
menu actually needs to display, and it keeps the file short enough to read at a
glance during the demo:

    # Whac-A-Mole -- best score per mode
    # CSE 452 Graphics & Image Processing Lab
    #
    # MODE   |  SCORE | HITS | ACCURACY | DATE
    # -------+--------+------+----------+-----------------
      EASY   |   4820 |   52 |    82.5% | 2026-08-09 22:54
      HARD   |   3110 |   41 |    76.0% | 2026-08-09 23:07

Modes that have never been played simply have no row.

Everything is written defensively.  A missing, unreadable, hand-edited or
half-written file yields an empty table rather than an exception, and a single
malformed row is skipped without taking the rest of the table with it.

Parsing rule: blank lines and lines beginning with ``#`` are comments; any
other line must split into five ``|``-separated fields whose first field is a
known difficulty name.
"""

from __future__ import annotations

import os
from datetime import datetime

import config

FIELD_COUNT = 5


def _path() -> str:
    """Resolve the score file next to the project root, not the working directory."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, config.HIGHSCORE_FILE)


def _header() -> str:
    return (
        "# Whac-A-Mole -- best score per mode\n"
        "# CSE 452 Graphics & Image Processing Lab\n"
        "#\n"
        "# Updated automatically whenever a mode's best is beaten. Safe to delete.\n"
        "#\n"
        "# MODE   |  SCORE | HITS | ACCURACY | DATE\n"
        "# -------+--------+------+----------+-----------------\n"
    )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _parse_line(line: str) -> tuple[str, dict] | None:
    """Turn one table row into ``(mode, entry)``, or ``None`` if it is not one."""
    fields = [part.strip() for part in line.split("|")]
    if len(fields) != FIELD_COUNT:
        return None

    mode, score, hits, accuracy, date = fields
    if mode not in config.DIFFICULTIES:
        return None

    try:
        return mode, {
            "score": int(score),
            "difficulty": mode,
            "hits": int(hits),
            "accuracy": float(accuracy.rstrip("%")),
            "date": date,
        }
    except ValueError:
        # A hand-edited or truncated row -- skip it rather than failing the load.
        return None


def load() -> dict[str, dict]:
    """Read the table as ``{difficulty: entry}``.  Empty dict on any problem."""
    path = _path()
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except (OSError, UnicodeDecodeError):
        return {}

    table: dict[str, dict] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parsed = _parse_line(stripped)
        if parsed is None:
            continue
        mode, entry = parsed
        # A duplicated mode (hand-edited file) keeps the better row.
        if mode not in table or entry["score"] > table[mode]["score"]:
            table[mode] = entry
    return table


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _format_line(entry: dict) -> str:
    """Render one entry as an aligned table row."""
    return (
        f"  {entry['difficulty']:<6s} | {entry['score']:6d} | {entry['hits']:4d} | "
        f"{entry['accuracy']:7.1f}% | {entry['date']}\n"
    )


def save(table: dict[str, dict]) -> bool:
    """Write the table.  Returns False if the file could not be written."""
    try:
        with open(_path(), "w", encoding="utf-8") as handle:
            handle.write(_header())
            # Always in menu order, so the file reads the same way the game does.
            for mode in config.DIFFICULTY_ORDER:
                entry = table.get(mode)
                if entry is not None:
                    handle.write(_format_line(entry))
        return True
    except OSError:
        return False


def record(score: int, difficulty: str, hits: int, accuracy: float) -> bool:
    """Record a finished round.  Returns True if it beat that mode's best.

    Only the best round per mode is kept, so a worse round leaves the file
    untouched.
    """
    table = load()
    previous = table.get(difficulty)
    previous_best = previous["score"] if previous else -1

    if int(score) <= previous_best:
        return False

    table[difficulty] = {
        "score": int(score),
        "difficulty": difficulty,
        "hits": int(hits),
        "accuracy": round(float(accuracy), 1),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    save(table)
    return True


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def best_score(difficulty: str | None = None) -> int:
    """Best score for one mode, or across all modes when ``difficulty`` is None."""
    table = load()
    if difficulty is not None:
        entry = table.get(difficulty)
        return entry["score"] if entry else 0
    return max((entry["score"] for entry in table.values()), default=0)


def best_entry(difficulty: str) -> dict | None:
    """The full record for one mode's best round, or None if never played."""
    return load().get(difficulty)


def all_bests() -> dict[str, dict]:
    """Every mode's best round, keyed by difficulty."""
    return load()
