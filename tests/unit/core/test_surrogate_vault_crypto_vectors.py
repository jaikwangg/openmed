"""Golden vectors that pin the encrypted surrogate mapping envelope bytes.

``encrypt_mapping`` is deterministic (AES-SIV), so a change to canonical JSON,
key derivation or associated data shows up here as a byte mismatch instead of
silently making previously stored mappings unreadable.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cryptography")

from openmed.core import surrogate_vault_crypto  # noqa: E402
from openmed.core.surrogate_vault_crypto import (  # noqa: E402
    ENCRYPTION_SCHEME,
    SCHEMA_VERSION,
    decrypt_mapping,
    encrypt_mapping,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "parity"
    / "surrogate_vault_crypto_v1.json"
)
VECTORS: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
CASES = {case["id"]: case for case in VECTORS["cases"]}
NEGATIVE_CASES = VECTORS["negative_cases"]


def _key(case: dict[str, Any]) -> bytes:
    return bytes.fromhex(case["key_hex"])


def _ciphertext(envelope: str) -> bytes:
    return base64.b64decode(json.loads(envelope)["ciphertext"])


def test_fixture_is_synthetic_and_matches_module_constants() -> None:
    assert VECTORS["synthetic"] is True
    assert VECTORS["encryption_scheme"] == ENCRYPTION_SCHEME
    assert VECTORS["envelope_schema_version"] == SCHEMA_VERSION
    assert len(CASES) == len(VECTORS["cases"])
    for case in VECTORS["cases"]:
        for key, value in case["mapping"].items():
            assert key.startswith("<SURROGATE_") and key.endswith(">")
            assert value.startswith("<SYNTHETIC_SOURCE_") and value.endswith(">")


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda case: case["id"])
def test_encrypt_matches_golden_envelope_and_round_trips(case: dict[str, Any]) -> None:
    expected = case["envelope"].encode("ascii")

    assert encrypt_mapping(case["mapping"], _key(case)) == expected
    assert decrypt_mapping(expected, _key(case)) == case["mapping"]


@pytest.mark.parametrize("case", NEGATIVE_CASES, ids=lambda case: case["id"])
def test_negative_vectors_raise_documented_errors(case: dict[str, Any]) -> None:
    error = getattr(surrogate_vault_crypto, case["error"])
    base = CASES[case["base"]]

    with pytest.raises(error, match=case["message"]) as raised:
        decrypt_mapping(case["envelope"], _key(case))

    for key, value in base["mapping"].items():
        assert key not in str(raised.value)
        assert value not in str(raised.value)


def test_negative_vectors_differ_from_their_base_as_described() -> None:
    by_id = {case["id"]: case for case in NEGATIVE_CASES}
    base = CASES["single-entry"]
    base_ciphertext = _ciphertext(base["envelope"])

    for case_id, index in (("flipped-tag-byte", 0), ("flipped-ciphertext-byte", -1)):
        flipped = bytearray(base_ciphertext)
        flipped[index] ^= 0x01
        assert by_id[case_id]["key_hex"] == base["key_hex"]
        assert _ciphertext(by_id[case_id]["envelope"]) == bytes(flipped)

    wrong_key = by_id["wrong-key"]
    assert wrong_key["envelope"] == base["envelope"]
    assert len(_key(wrong_key)) == len(_key(base))
    assert wrong_key["key_hex"] != base["key_hex"]

    extra = json.loads(by_id["extra-envelope-field"]["envelope"])
    assert set(extra) - set(json.loads(base["envelope"])) == {"key_id"}
    assert extra["ciphertext"] == json.loads(base["envelope"])["ciphertext"]

    assert len(_key(by_id["short-key"])) == len(_key(base)) - 1
