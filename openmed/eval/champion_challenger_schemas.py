"""Draft 2020-12 JSON Schemas for champion-challenger artifacts.

:func:`build_comparison_record_schema` describes the signed comparison record
written by :meth:`~openmed.eval.champion_challenger.PromotionVerdict.to_dict`,
and :func:`build_champion_pointer_schema` describes ``gates/champion.json`` as
written by :meth:`~openmed.eval.champion_challenger.ChampionPointer.to_dict`.
Reviewers and other tools can validate either file without importing OpenMed.

The schemas check shape only. Whether ``record_hash`` matches the record's
content is still checked by :meth:`PromotionVerdict.verify`.
"""

from __future__ import annotations

from typing import Any, Final

from openmed.eval.champion_challenger import (
    CATEGORY_GATE,
    CATEGORY_LATENCY,
    CATEGORY_LEAKAGE,
    CATEGORY_QUANT,
    CATEGORY_RECALL,
    CHAMPION_POINTER_SCHEMA_VERSION,
    COMPARISON_RECORD_ARTIFACT,
    COMPARISON_RECORD_SCHEMA_VERSION,
    HIGHER_IS_BETTER,
    HOLD,
    LOWER_IS_BETTER,
    PROMOTE,
    REGRESSION,
    REJECT,
    TIE,
    WIN,
)

CHAMPION_CHALLENGER_SCHEMA_DIALECT: Final = (
    "https://json-schema.org/draft/2020-12/schema"
)

_RECORD_HASH_PATTERN: Final = r"^sha256:[0-9a-f]{64}(?![\s\S])"


def _string() -> dict[str, Any]:
    return {"type": "string"}


def _string_list() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def _number_map() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": {"type": "number"}}


def _closed_object(properties: dict[str, Any], *optional: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(name for name in properties if name not in optional),
        "properties": properties,
    }


def build_comparison_record_schema() -> dict[str, Any]:
    """Build the JSON Schema for a champion-challenger comparison record.

    Returns:
        A new JSON-compatible Draft 2020-12 schema mapping. Mutating the
        returned mapping cannot affect a later export.
    """

    model_ref = _closed_object({"repo_id": _string(), "repro_hash": _string()})
    ledger_entry = _closed_object(
        {
            "axis": {"type": "string", "minLength": 1},
            "blocking": {"type": "boolean"},
            "category": {
                "type": "string",
                "enum": [
                    CATEGORY_RECALL,
                    CATEGORY_LEAKAGE,
                    CATEGORY_QUANT,
                    CATEGORY_LATENCY,
                    CATEGORY_GATE,
                ],
            },
            "challenger": {"type": "number"},
            "champion": {"type": "number"},
            "delta": {"type": "number"},
            "direction": {
                "type": "string",
                "enum": [HIGHER_IS_BETTER, LOWER_IS_BETTER],
            },
            "outcome": {"type": "string", "enum": [WIN, TIE, REGRESSION]},
            # Present only on entries for an axis the challenger did not report.
            "reason": {"type": "string", "minLength": 1},
            "tolerance": {"type": "number", "minimum": 0},
        },
        "reason",
    )
    ignored_f1 = _closed_object(
        {
            "challenger": _number_map(),
            # Absent when there was no incumbent champion to compare against.
            "champion": _number_map(),
            "note": _string(),
        },
        "champion",
    )

    schema = _closed_object(
        {
            "artifact": {"type": "string", "const": COMPARISON_RECORD_ARTIFACT},
            "challenger": {"$ref": "#/$defs/model_ref"},
            "champion": {"$ref": "#/$defs/model_ref"},
            "family": _string(),
            "format": _string(),
            "ignored_f1": ignored_f1,
            "key": _string(),
            "ledger": {"type": "array", "items": {"$ref": "#/$defs/ledger_entry"}},
            "reasons": _string_list(),
            "record_hash": {"type": "string", "pattern": _RECORD_HASH_PATTERN},
            "regressions": _string_list(),
            "schema_version": {
                "type": "integer",
                "const": COMPARISON_RECORD_SCHEMA_VERSION,
            },
            "tier": _string(),
            "ties": _string_list(),
            "verdict": {"type": "string", "enum": [PROMOTE, HOLD, REJECT]},
            "wins": _string_list(),
        }
    )
    return {
        "$schema": CHAMPION_CHALLENGER_SCHEMA_DIALECT,
        "$id": (
            "urn:openmed:eval:champion_challenger_comparison:"
            f"{COMPARISON_RECORD_SCHEMA_VERSION}"
        ),
        **schema,
        "$defs": {"ledger_entry": ledger_entry, "model_ref": model_ref},
    }


def build_champion_pointer_schema() -> dict[str, Any]:
    """Build the JSON Schema for the committed champion pointer store.

    Returns:
        A new JSON-compatible Draft 2020-12 schema mapping. Mutating the
        returned mapping cannot affect a later export.
    """

    entry = _closed_object(
        {
            "champion": _string(),
            "family": _string(),
            "format": _string(),
            "key": _string(),
            "previous_champion": {"type": ["string", "null"]},
            # Empty until a promotion writes the comparison record's hash.
            "record_hash": {
                "anyOf": [
                    {"type": "string", "const": ""},
                    {"type": "string", "pattern": _RECORD_HASH_PATTERN},
                ]
            },
            "tier": _string(),
        }
    )
    schema = _closed_object(
        {
            "entries": {
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/pointer_entry"},
            },
            "schema_version": {
                "type": "integer",
                "const": CHAMPION_POINTER_SCHEMA_VERSION,
            },
        }
    )
    return {
        "$schema": CHAMPION_CHALLENGER_SCHEMA_DIALECT,
        "$id": f"urn:openmed:eval:champion_pointer:{CHAMPION_POINTER_SCHEMA_VERSION}",
        **schema,
        "$defs": {"pointer_entry": entry},
    }


__all__ = [
    "CHAMPION_CHALLENGER_SCHEMA_DIALECT",
    "build_champion_pointer_schema",
    "build_comparison_record_schema",
]
