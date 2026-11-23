"""Check ``iter_grapheme_clusters`` against Unicode's GraphemeBreakTest data."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.core.decoding import iter_grapheme_clusters

FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "unicode"
FIXTURE_PATH = FIXTURE_DIR / "grapheme_break_subset.txt"
LICENSE_PATH = FIXTURE_DIR / "grapheme_break_subset.LICENSE.md"

# Intended deviations from UAX #29, as (text, UAX #29 boundaries) pairs. None of
# the conformance rows hits one, so each is pinned here as a strict xfail:
# - GB999 inside an ideographic description sequence: UAX #29 breaks after the
#   operator, OpenMed keeps the whole sequence (operator plus components) as one
#   cluster so a span never splits a described ideograph.
_KNOWN_DEVIATIONS = (
    pytest.param(
        "⿰氵青",
        (0, 1, 2, 3),
        id="GB999-ideographic-description-sequence",
        marks=pytest.mark.xfail(
            reason="GB999: an ideographic description sequence stays one cluster",
            strict=True,
        ),
    ),
)


def _parse_line(data: str) -> tuple[str, tuple[int, ...]]:
    text = ""
    boundaries: list[int] = []
    for token in data.split():
        if token == "÷":
            boundaries.append(len(text))
        elif token != "×":
            text += chr(int(token, 16))
    return text, tuple(boundaries)


def _conformance_rows() -> list[pytest.param]:
    rows = []
    lines = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        data = line.partition("#")[0].strip()
        if data:
            rows.append(pytest.param(*_parse_line(data), id=f"line-{number}"))
    return rows


def _boundaries(text: str) -> tuple[int, ...]:
    return (0, *(end for _, end in iter_grapheme_clusters(text)))


def test_fixture_states_unicode_version_and_carries_license() -> None:
    header = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()[0]

    assert header == "# GraphemeBreakTest-15.1.0.txt"
    assert "UNICODE LICENSE V3" in LICENSE_PATH.read_text(encoding="utf-8")
    assert len(_conformance_rows()) > 600


def test_parse_line_reads_break_and_no_break_marks() -> None:
    assert _parse_line("÷ 0061 × 0308 ÷ 0062 ÷") == (
        "äb",
        (0, 2, 3),
    )


@pytest.mark.parametrize(("text", "expected"), _conformance_rows())
def test_iter_grapheme_clusters_matches_grapheme_break_test(
    text: str,
    expected: tuple[int, ...],
) -> None:
    assert _boundaries(text) == expected


@pytest.mark.parametrize(("text", "expected"), _KNOWN_DEVIATIONS)
def test_known_deviation_from_uax29(text: str, expected: tuple[int, ...]) -> None:
    assert _boundaries(text) == expected
