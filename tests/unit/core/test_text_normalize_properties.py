"""Property tests for the source map built by ``normalize_unicode_with_offsets``.

These pin the invariants a native port of the offset map must keep: one map
entry per normalized code point, source spans inside the input, source
starts that never move backwards, and a whole-output projection that covers
the whole input.
"""

from __future__ import annotations

import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from openmed.core.text_normalize import OffsetMap, normalize_unicode_with_offsets

_ALPHABET = st.one_of(
    st.characters(min_codepoint=0x41, max_codepoint=0x5A),  # Latin capitals
    st.characters(min_codepoint=0x61, max_codepoint=0x7A),  # Latin small letters
    st.sampled_from(" .-"),
    # Combining marks from several combining classes (230, 220, 202, 216, 240),
    # so canonical reordering and blocked composition both get exercised.
    st.sampled_from("̧̛̣̀́̈ͅ"),
    st.characters(min_codepoint=0xFF10, max_codepoint=0xFF5A),  # fullwidth forms
    st.characters(min_codepoint=0xFB00, max_codepoint=0xFB06),  # Latin ligatures
    st.characters(min_codepoint=0x1100, max_codepoint=0x1112),  # Hangul L jamo
    st.characters(min_codepoint=0x1161, max_codepoint=0x1175),  # Hangul V jamo
    st.characters(min_codepoint=0x11A8, max_codepoint=0x11C2),  # Hangul T jamo
    st.characters(min_codepoint=0x0660, max_codepoint=0x0669),  # Arabic-Indic digits
    st.characters(min_codepoint=0x0966, max_codepoint=0x096F),  # Devanagari digits
)


def _code_points(text: str) -> str:
    return " ".join(f"U+{ord(character):04X}" for character in text)


def _check_offset_map(text: str, normalized: str, offset_map: OffsetMap) -> None:
    where = f"input [{_code_points(text)}] -> output [{_code_points(normalized)}]"

    assert len(offset_map) == len(normalized), where
    assert len(offset_map.source_spans) == len(normalized), where
    assert offset_map.original_length == len(text), where

    for index, (start, end) in enumerate(offset_map.source_spans):
        assert 0 <= start < end <= len(text), f"{where}: span {index} is {start, end}"
        assert offset_map[index] == start, f"{where}: index {index}"

    starts = list(offset_map)
    assert starts == sorted(starts), f"{where}: source starts {starts} move backwards"

    if text:
        assert offset_map.to_original_span(0, len(normalized)) == (0, len(text)), (
            f"{where}: spans {offset_map.source_spans} do not cover the input"
        )


@pytest.mark.fuzz
@settings(max_examples=500, deadline=None)
@given(st.text(alphabet=_ALPHABET, max_size=24))
def test_generated_offset_maps_keep_their_invariants(text):
    note(f"input code points: {_code_points(text)}")
    normalized, offset_map = normalize_unicode_with_offsets(text)

    _check_offset_map(text, normalized, offset_map)


@pytest.mark.fuzz
@settings(max_examples=200, deadline=None)
@given(st.text(alphabet=_ALPHABET, min_size=1, max_size=24))
def test_projected_spans_cover_every_selected_source_character(text):
    note(f"input code points: {_code_points(text)}")
    normalized, offset_map = normalize_unicode_with_offsets(text)

    for start in range(len(normalized)):
        for end in range(start + 1, len(normalized) + 1):
            source_start, source_end = offset_map.to_original_span(start, end)
            spans = offset_map.source_spans[start:end]
            assert source_start == min(span[0] for span in spans)
            assert source_end == max(span[1] for span in spans)


@pytest.mark.fuzz
@pytest.mark.parametrize(
    "text",
    [
        "ạ́",  # reordered marks: U+0323 composes into the base
        "̣́",  # reordered marks with no base
        "é́",  # second identical mark is blocked from composing
        "가ᅡ",  # second vowel jamo is left after the syllable
    ],
)
def test_reordered_and_blocked_marks_stay_covered(text):
    normalized, offset_map = normalize_unicode_with_offsets(text)

    _check_offset_map(text, normalized, offset_map)
    # The first output code point absorbed source code points from the whole
    # combining sequence, so redacting it alone must cover all of them.
    assert offset_map.to_original_span(0, 1) == (0, len(text))


@pytest.mark.fuzz
def test_an_off_by_one_source_index_breaks_the_properties():
    text = "xﬁle Ａ́ 각"
    normalized, offset_map = normalize_unicode_with_offsets(text)
    _check_offset_map(text, normalized, offset_map)

    shifted = list(offset_map.source_spans)
    start, end = shifted[-1]
    shifted[-1] = (start + 1, end + 1)
    broken = OffsetMap(
        indices=tuple(span[0] for span in shifted),
        source_spans=tuple(shifted),
        original_length=offset_map.original_length,
    )

    with pytest.raises(AssertionError):
        _check_offset_map(text, normalized, broken)
