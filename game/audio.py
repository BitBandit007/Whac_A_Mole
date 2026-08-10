"""
Procedurally synthesised sound effects.

There are no audio files in this repository.  Every sound is generated as a
NumPy waveform at start-up and handed to pygame as a raw sample buffer, which
keeps the project a single self-contained checkout and means the sounds can be
described in code rather than credited to a download.

It is also a neat companion to the image-processing half of the project: an
audio envelope is a 1D signal, shaped by exactly the same kind of arithmetic
the 2D filters use on images.

Every entry point is written to fail soft.  If SDL cannot open an audio device
-- common on a lab machine, a remote desktop or a CI runner -- the module
disables itself and the game runs silently rather than crashing.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44100
BIT_DEPTH = -16
CHANNELS = 2


# ---------------------------------------------------------------------------
# Waveform building blocks
# ---------------------------------------------------------------------------


def _time_axis(duration: float) -> np.ndarray:
    return np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)


def _sine(frequency: float, duration: float) -> np.ndarray:
    return np.sin(2.0 * np.pi * frequency * _time_axis(duration))

def _sweep(start_hz: float, end_hz: float, duration: float) -> np.ndarray:
    """A linear frequency sweep (chirp).

    The instantaneous frequency is integrated to get phase; sweeping the
    frequency directly inside ``sin(2 pi f t)`` would produce the wrong pitch
    contour and an audible discontinuity.
    """
    t = _time_axis(duration)
    if len(t) == 0:
        return t
    frequency = np.linspace(start_hz, end_hz, len(t))
    phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    return np.sin(phase)


def _square(frequency: float, duration: float) -> np.ndarray:
    return np.sign(_sine(frequency, duration))


def _noise(duration: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, int(SAMPLE_RATE * duration))


def _envelope(length: int, attack: float = 0.01, decay: float = 0.9) -> np.ndarray:
    """A simple attack/decay amplitude envelope.

    The attack ramp matters more than it looks: starting a waveform at full
    amplitude produces a step discontinuity, which is heard as a click.
    """
    if length <= 0:
        return np.zeros(0)
    attack_samples = max(1, int(length * attack))
    decay_samples = max(1, length - attack_samples)
    return np.concatenate(
        [
            np.linspace(0.0, 1.0, attack_samples),
            np.linspace(1.0, 0.0, decay_samples) ** (1.0 / max(decay, 0.05)),
        ]
    )[:length]


def _mix(*waves: np.ndarray) -> np.ndarray:
    """Sum waveforms of differing lengths, zero-padding the shorter ones."""
    if not waves:
        return np.zeros(0)
    length = max(len(w) for w in waves)
    total = np.zeros(length)
    for wave in waves:
        total[: len(wave)] += wave
    return total


def _normalise(wave: np.ndarray, level: float = 0.75) -> np.ndarray:
    peak = float(np.max(np.abs(wave))) if len(wave) else 0.0
    if peak < 1e-9:
        return wave
    return wave * (level / peak)


def _to_stereo_int16(wave: np.ndarray) -> np.ndarray:
    """Convert a mono float waveform in ``[-1, 1]`` to interleaved 16-bit stereo."""
    clipped = np.clip(wave, -1.0, 1.0)
    samples = (clipped * 32767.0).astype(np.int16)
    return np.ascontiguousarray(np.column_stack([samples, samples]))


# ---------------------------------------------------------------------------
# The sounds
# ---------------------------------------------------------------------------


def _make_hit() -> np.ndarray:
    """A satisfying thwack: a noise transient over a fast downward pitch drop."""
    thump = _sweep(420.0, 90.0, 0.16) * _envelope(int(SAMPLE_RATE * 0.16), 0.005, 0.55)
    click = _noise(0.05, seed=1) * _envelope(int(SAMPLE_RATE * 0.05), 0.002, 0.30)
    body = _sine(180.0, 0.12) * _envelope(int(SAMPLE_RATE * 0.12), 0.01, 0.6)
    return _normalise(_mix(thump, click * 0.55, body * 0.4), 0.80)


def _make_miss() -> np.ndarray:
    """A dull whiff -- filtered noise with no tonal centre."""
    air = _noise(0.14, seed=2) * _envelope(int(SAMPLE_RATE * 0.14), 0.02, 0.35)
    # A short moving-average pass rolls off the high end, turning white noise
    # into something closer to a rush of air.  Same idea as a mean filter,
    # one dimension down.
    kernel = np.ones(28) / 28.0
    air = np.convolve(air, kernel, mode="same")
    low = _sine(120.0, 0.12) * _envelope(int(SAMPLE_RATE * 0.12), 0.01, 0.5) * 0.35
    return _normalise(_mix(air, low), 0.45)


def _make_escape() -> np.ndarray:
    """A descending two-tone blip for a mole that got away."""
    first = _sine(520.0, 0.07) * _envelope(int(SAMPLE_RATE * 0.07), 0.01, 0.6)
    second = np.concatenate(
        [np.zeros(int(SAMPLE_RATE * 0.07)), _sine(390.0, 0.10) * _envelope(int(SAMPLE_RATE * 0.10), 0.01, 0.6)]
    )
    return _normalise(_mix(first, second), 0.55)


def _make_combo() -> np.ndarray:
    """A rising three-note arpeggio when the multiplier steps up."""
    notes = [660.0, 880.0, 1180.0]
    segments = []
    for index, frequency in enumerate(notes):
        gap = np.zeros(int(SAMPLE_RATE * 0.055 * index))
        tone = _sine(frequency, 0.09) * _envelope(int(SAMPLE_RATE * 0.09), 0.01, 0.5)
        segments.append(np.concatenate([gap, tone]))
    return _normalise(_mix(*segments), 0.55)


def _make_golden() -> np.ndarray:
    """A bright shimmer for the bonus mole."""
    sparkle = _mix(
        _sweep(880.0, 1760.0, 0.22) * _envelope(int(SAMPLE_RATE * 0.22), 0.01, 0.45),
        _sweep(1320.0, 2640.0, 0.18) * _envelope(int(SAMPLE_RATE * 0.18), 0.02, 0.4) * 0.5,
    )
    return _normalise(sparkle, 0.6)


def _make_life_lost() -> np.ndarray:
    """A downward buzz on losing a life."""
    buzz = _square(220.0, 0.20) * _envelope(int(SAMPLE_RATE * 0.20), 0.01, 0.4)
    drop = _sweep(300.0, 110.0, 0.22) * _envelope(int(SAMPLE_RATE * 0.22), 0.01, 0.5)
    return _normalise(_mix(buzz * 0.35, drop), 0.6)


def _make_game_over() -> np.ndarray:
    """A descending minor arpeggio."""
    notes = [523.0, 440.0, 349.0, 262.0]
    segments = []
    for index, frequency in enumerate(notes):
        gap = np.zeros(int(SAMPLE_RATE * 0.14 * index))
        tone = _sine(frequency, 0.30) * _envelope(int(SAMPLE_RATE * 0.30), 0.01, 0.4)
        segments.append(np.concatenate([gap, tone]))
    return _normalise(_mix(*segments), 0.6)


def _make_start() -> np.ndarray:
    """A rising fanfare when a round begins."""
    notes = [392.0, 523.0, 659.0, 784.0]
    segments = []
    for index, frequency in enumerate(notes):
        gap = np.zeros(int(SAMPLE_RATE * 0.09 * index))
        tone = _sine(frequency, 0.18) * _envelope(int(SAMPLE_RATE * 0.18), 0.01, 0.5)
        segments.append(np.concatenate([gap, tone]))
    return _normalise(_mix(*segments), 0.6)


def _make_ui() -> np.ndarray:
    """A short click for menu navigation."""
    tone = _sine(1200.0, 0.035) * _envelope(int(SAMPLE_RATE * 0.035), 0.01, 0.4)
    return _normalise(tone, 0.35)


_GENERATORS = {
    "hit": _make_hit,
    "miss": _make_miss,
    "escape": _make_escape,
    "combo": _make_combo,
    "golden": _make_golden,
    "life_lost": _make_life_lost,
    "game_over": _make_game_over,
    "start": _make_start,
    "ui": _make_ui,
}


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


class AudioEngine:
    """Owns the synthesised sounds and plays them on request.

    ``enabled`` is False whenever the mixer could not be opened, in which case
    every method becomes a no-op.
    """

    def __init__(self, pygame_module):
        self.pygame = pygame_module
        self.enabled = False
        self.muted = False
        self.sounds: dict[str, object] = {}

        try:
            if not pygame_module.mixer.get_init():
                pygame_module.mixer.init(
                    frequency=SAMPLE_RATE, size=BIT_DEPTH, channels=CHANNELS, buffer=512
                )
            for name, generator in _GENERATORS.items():
                buffer = _to_stereo_int16(generator())
                self.sounds[name] = pygame_module.sndarray.make_sound(buffer)
            self.enabled = True
        except Exception as error:      # pragma: no cover - hardware dependent
            print(f"[audio] disabled ({error.__class__.__name__}: {error})")
            self.enabled = False

    def play(self, name: str, volume: float = 1.0) -> None:
        if not self.enabled or self.muted:
            return
        sound = self.sounds.get(name)
        if sound is None:
            return
        try:
            sound.set_volume(max(0.0, min(1.0, volume)))
            sound.play()
        except Exception:               # pragma: no cover - hardware dependent
            pass

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        return self.muted

    def shutdown(self) -> None:
        if not self.enabled:
            return
        try:
            self.pygame.mixer.stop()
        except Exception:               # pragma: no cover
            pass
