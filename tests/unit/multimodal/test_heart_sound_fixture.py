"""Determinism, WAV metadata and S1/S2 truth checks for the heart-sound fixture.

The truth is checked against the requested heart rate and against an
independent burst detector that reads the rendered samples back, so a change
that keeps the onset arithmetic but moves the audible sounds fails.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import io
import math
import os
import socket
import sys
import wave
from pathlib import Path

import pytest

from openmed.multimodal.wav_metadata import WAVE_FORMAT_PCM, read_wav_metadata
from tests.fixtures.multimodal import heart_sounds
from tests.fixtures.multimodal.heart_sounds import (
    S1_BURST,
    S2_BURST,
    HeartSoundFixtureError,
    synthetic_heart_sounds,
)

NOISY = {"noise_sd": 0.05, "clip_level": 0.5, "seed": 7}
PINNED_SHA256 = {
    "default": "5ee6a5487f9f875387b3c8ccd76c01b8f22c187ad22e793075bde347ad50f92b",
    "noisy_clipped": "a8ba7a2ac7c22382f2c2bfe6b6f2fe97c67d49edcac6072c02d66502bea87b81",
}


def _samples(fixture):
    with wave.open(io.BytesIO(fixture.wav_bytes), "rb") as reader:
        raw = reader.readframes(reader.getnframes())
    return memoryview(raw).cast("h").tolist()


def _bursts(samples, sample_rate_hz):
    """Return ``(onset, length)`` for runs of sound separated by >= 10 ms silence.

    A burst starts with ``sin(0) == 0``, so the onset is one sample before the
    first non-zero sample.
    """

    gap = round(0.01 * sample_rate_hz)
    bursts = []
    start = last = None
    for index, value in enumerate(samples):
        if value == 0:
            continue
        if start is None or index - last > gap:
            if start is not None:
                bursts.append((start - 1, last - start + 2))
            start = index
        last = index
    if start is not None:
        bursts.append((start - 1, last - start + 2))
    return bursts


@pytest.mark.parametrize("kwargs", [{}, NOISY, {"heart_rate_bpm": 140.0}])
def test_same_arguments_give_identical_bytes(kwargs):
    first = synthetic_heart_sounds(**kwargs)
    assert synthetic_heart_sounds(**kwargs) == first


def test_seed_drives_noise_only():
    assert synthetic_heart_sounds(seed=1) == synthetic_heart_sounds(seed=2)
    noisy_a = synthetic_heart_sounds(noise_sd=0.05, seed=1)
    noisy_b = synthetic_heart_sounds(noise_sd=0.05, seed=2)
    assert noisy_a.wav_bytes != noisy_b.wav_bytes
    assert noisy_a.s1_onsets == noisy_b.s1_onsets
    assert noisy_a.s2_onsets == noisy_b.s2_onsets


@pytest.mark.parametrize(
    "sample_rate_hz,duration_s", [(2000, 3.0), (4000, 5.0), (8000, 2.5)]
)
def test_read_wav_metadata_reports_requested_rate_and_duration(
    sample_rate_hz, duration_s
):
    fixture = synthetic_heart_sounds(
        sample_rate_hz=sample_rate_hz, duration_s=duration_s, **NOISY
    )
    metadata = read_wav_metadata(fixture.wav_bytes)
    assert metadata.format_code == WAVE_FORMAT_PCM
    assert metadata.channels == 1
    assert metadata.bit_depth == 16
    assert metadata.sample_rate_hz == sample_rate_hz
    assert metadata.frame_count == fixture.frame_count
    assert metadata.data_byte_count == 2 * fixture.frame_count
    assert metadata.duration_seconds == pytest.approx(duration_s)


@pytest.mark.parametrize("heart_rate_bpm", [40.0, 60.0, 72.0, 110.5, 150.0])
@pytest.mark.parametrize("sample_rate_hz", [2000, 4000])
def test_truth_spacing_matches_heart_rate(heart_rate_bpm, sample_rate_hz):
    fixture = synthetic_heart_sounds(
        heart_rate_bpm=heart_rate_bpm, sample_rate_hz=sample_rate_hz, duration_s=6.0
    )
    cycle = 60.0 / heart_rate_bpm * sample_rate_hz
    systole = 0.35 * cycle
    s1, s2 = fixture.s1_onsets, fixture.s2_onsets
    assert len(s1) >= 3 and len(s2) >= 3
    assert s1[0] == 0
    events = sorted([(i, "s1") for i in s1] + [(i, "s2") for i in s2])
    assert [name for _, name in events] == ["s1", "s2"] * len(s2) + ["s1"] * (
        len(s1) - len(s2)
    )
    for first, second in zip(s1, s1[1:]):
        assert abs((second - first) - cycle) <= 1
    for onset_s1, onset_s2 in zip(s1, s2):
        assert abs((onset_s2 - onset_s1) - systole) <= 1
    for onset_s2, next_s1 in zip(s2, s1[1:]):
        assert abs((next_s1 - onset_s2) - (cycle - systole)) <= 1
    expected_cycles = math.ceil(6.0 * heart_rate_bpm / 60.0)
    assert len(s1) in (expected_cycles - 1, expected_cycles)


@pytest.mark.parametrize("heart_rate_bpm", [45.0, 72.0, 130.0])
@pytest.mark.parametrize("sample_rate_hz", [2000, 4000, 8000])
def test_independent_detector_recovers_truth(heart_rate_bpm, sample_rate_hz):
    fixture = synthetic_heart_sounds(
        heart_rate_bpm=heart_rate_bpm, sample_rate_hz=sample_rate_hz
    )
    bursts = _bursts(_samples(fixture), sample_rate_hz)
    s1_length = round(S1_BURST[1] * sample_rate_hz)
    s2_length = round(S2_BURST[1] * sample_rate_hz)
    detected_s1 = tuple(onset for onset, length in bursts if length == s1_length)
    detected_s2 = tuple(onset for onset, length in bursts if length == s2_length)
    assert len(detected_s1) + len(detected_s2) == len(bursts)
    assert detected_s1 == fixture.s1_onsets
    assert detected_s2 == fixture.s2_onsets


def test_s1_is_louder_than_s2_and_silence_separates_them():
    fixture = synthetic_heart_sounds()
    samples = _samples(fixture)
    rate = fixture.sample_rate_hz
    s1_peak = max(abs(v) for v in samples[: round(S1_BURST[1] * rate)])
    onset = fixture.s2_onsets[0]
    s2_peak = max(abs(v) for v in samples[onset : onset + round(S2_BURST[1] * rate)])
    assert s1_peak > s2_peak > 0
    gap = samples[round(S1_BURST[1] * rate) : onset]
    assert gap and not any(gap)


def test_noise_fills_silence_and_clipping_caps_the_peak():
    clean = _samples(synthetic_heart_sounds())
    clipped = _samples(synthetic_heart_sounds(clip_level=0.3))
    noisy = _samples(synthetic_heart_sounds(noise_sd=0.01, seed=3))
    cap = round(0.3 * 32767)
    assert max(abs(v) for v in clean) > cap
    assert max(abs(v) for v in clipped) == cap
    assert sum(abs(v) == cap for v in clipped) > 1
    assert sum(v == 0 for v in noisy) < sum(v == 0 for v in clean) // 10


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"heart_rate_bpm": 0}, "heart_sound_invalid_heart_rate"),
        ({"heart_rate_bpm": math.nan}, "heart_sound_invalid_heart_rate"),
        ({"heart_rate_bpm": True}, "heart_sound_invalid_heart_rate"),
        ({"sample_rate_hz": 4000.0}, "heart_sound_invalid_sample_rate"),
        ({"sample_rate_hz": 0}, "heart_sound_invalid_sample_rate"),
        ({"duration_s": -1.0}, "heart_sound_invalid_duration"),
        ({"duration_s": math.inf}, "heart_sound_invalid_duration"),
        ({"noise_sd": -0.1}, "heart_sound_invalid_noise"),
        ({"noise_sd": math.nan}, "heart_sound_invalid_noise"),
        ({"clip_level": 0}, "heart_sound_invalid_clip_level"),
        ({"clip_level": 1.5}, "heart_sound_invalid_clip_level"),
        ({"systole_fraction": 0}, "heart_sound_invalid_systole"),
        ({"systole_fraction": 1}, "heart_sound_invalid_systole"),
        ({"heart_rate_bpm": 400.0}, "heart_sound_overlapping_bursts"),
        ({"systole_fraction": 0.05}, "heart_sound_overlapping_bursts"),
    ],
)
def test_unrenderable_parameters_fail_with_stable_codes(kwargs, code):
    with pytest.raises(HeartSoundFixtureError, match=f"^{code}$"):
        synthetic_heart_sounds(**kwargs)


def test_standard_library_only_and_no_files_or_network(monkeypatch):
    source = Path(heart_sounds.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            assert root == "__future__" or root in sys.stdlib_module_names

    def refuse(*_args, **_kwargs):
        raise AssertionError("fixture attempted I/O")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(os, "open", refuse)
    monkeypatch.setattr(socket, "socket", refuse)
    synthetic_heart_sounds(**NOISY)


@pytest.mark.parametrize("name,kwargs", [("default", {}), ("noisy_clipped", NOISY)])
def test_output_matches_pinned_digest(name, kwargs):
    digest = hashlib.sha256(synthetic_heart_sounds(**kwargs).wav_bytes).hexdigest()
    assert digest == PINNED_SHA256[name]
