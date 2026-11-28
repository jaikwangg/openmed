"""Synthetic phonocardiogram-like WAV bytes with S1/S2 truth; never a recording.

Each heart sound is a damped low-frequency sine burst: S1 is longer, lower and
louder than S2. Onsets are fixed first, so the truth is exact by construction,
and the audio is rendered from them. Optional Gaussian noise comes from a seeded
``random.Random`` and optional clipping caps the waveform at a fraction of full
scale. Output is mono 16-bit PCM written with the standard ``wave`` module into
memory; nothing is written to disk and nothing touches the network.

Conventions:

* One cardiac cycle lasts ``60 / heart_rate_bpm`` seconds. S1 opens cycle ``k``
  at ``k * cycle`` and S2 follows after ``systole_fraction * cycle`` seconds.
* Onset sample indices are ``round(time * sample_rate_hz)``, so spacings can
  differ from the exact interval by one sample.
* Only onsets whose whole burst fits inside the clip are rendered and reported.
"""

from __future__ import annotations

import io
import math
import random
import struct
import wave
from dataclasses import dataclass

_FULL_SCALE = 32767
# (frequency Hz, duration s, decay time constant s, peak amplitude) per sound.
S1_BURST = (45.0, 0.10, 0.025, 0.8)
S2_BURST = (65.0, 0.08, 0.018, 0.6)


class HeartSoundFixtureError(ValueError):
    """Raised for parameters the generator cannot render."""


@dataclass(frozen=True)
class HeartSoundFixture:
    """In-memory WAV bytes plus the S1 and S2 onset truth in samples."""

    wav_bytes: bytes
    sample_rate_hz: int
    frame_count: int
    heart_rate_bpm: float
    s1_onsets: tuple[int, ...]
    s2_onsets: tuple[int, ...]


def _positive(value, code):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HeartSoundFixtureError(code)
    if not math.isfinite(value) or value <= 0:
        raise HeartSoundFixtureError(code)


def synthetic_heart_sounds(
    *,
    heart_rate_bpm: float = 72.0,
    sample_rate_hz: int = 4000,
    duration_s: float = 5.0,
    systole_fraction: float = 0.35,
    noise_sd: float = 0.0,
    clip_level: float = 1.0,
    seed: int = 0,
) -> HeartSoundFixture:
    """Render S1/S2 bursts into 16-bit mono WAV bytes and return their onsets.

    ``noise_sd`` and ``clip_level`` are fractions of full scale. The seed only
    drives the noise, so ``noise_sd=0`` gives the same bytes for every seed.
    """

    _positive(heart_rate_bpm, "heart_sound_invalid_heart_rate")
    if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, int):
        raise HeartSoundFixtureError("heart_sound_invalid_sample_rate")
    if sample_rate_hz <= 0:
        raise HeartSoundFixtureError("heart_sound_invalid_sample_rate")
    _positive(duration_s, "heart_sound_invalid_duration")
    _positive(clip_level, "heart_sound_invalid_clip_level")
    if clip_level > 1:
        raise HeartSoundFixtureError("heart_sound_invalid_clip_level")
    if isinstance(noise_sd, bool) or not isinstance(noise_sd, (int, float)):
        raise HeartSoundFixtureError("heart_sound_invalid_noise")
    if not math.isfinite(noise_sd) or noise_sd < 0:
        raise HeartSoundFixtureError("heart_sound_invalid_noise")
    if not 0 < systole_fraction < 1:
        raise HeartSoundFixtureError("heart_sound_invalid_systole")

    cycle_s = 60.0 / heart_rate_bpm
    systole_s = systole_fraction * cycle_s
    # Each burst must end before the next sound starts, or the truth would
    # point into the tail of the previous one.
    if S1_BURST[1] >= systole_s or S2_BURST[1] >= cycle_s - systole_s:
        raise HeartSoundFixtureError("heart_sound_overlapping_bursts")

    frame_count = round(duration_s * sample_rate_hz)
    samples = [0.0] * frame_count
    onsets: dict[str, list[int]] = {"s1": [], "s2": []}
    cycle = 0
    while round(cycle * cycle_s * sample_rate_hz) < frame_count:
        start_s = cycle * cycle_s
        for name, offset_s, burst in (
            ("s1", 0.0, S1_BURST),
            ("s2", systole_s, S2_BURST),
        ):
            onset = round((start_s + offset_s) * sample_rate_hz)
            frequency, length_s, decay_s, amplitude = burst
            length = round(length_s * sample_rate_hz)
            if onset + length > frame_count:
                continue
            onsets[name].append(onset)
            for i in range(length):
                t = i / sample_rate_hz
                samples[onset + i] += (
                    amplitude
                    * math.exp(-t / decay_s)
                    * math.sin(2 * math.pi * frequency * t)
                )
        cycle += 1

    rng = random.Random(seed)
    frames = bytearray()
    for value in samples:
        if noise_sd:
            value += rng.gauss(0.0, noise_sd)
        value = max(-clip_level, min(clip_level, value))
        frames += struct.pack("<h", round(value * _FULL_SCALE))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(bytes(frames))
    return HeartSoundFixture(
        wav_bytes=buffer.getvalue(),
        sample_rate_hz=sample_rate_hz,
        frame_count=frame_count,
        heart_rate_bpm=float(heart_rate_bpm),
        s1_onsets=tuple(onsets["s1"]),
        s2_onsets=tuple(onsets["s2"]),
    )
