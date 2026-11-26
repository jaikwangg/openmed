"""Tests for the champion-challenger promotion framework (#1274).

All fixtures are synthetic and generated algorithmically: no real PHI, model
outputs, or network access is involved. Every verdict is derived purely from two
committed gate reports.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.eval.champion_challenger import (
    COMPARISON_RECORD_ARTIFACT,
    HOLD,
    PROMOTE,
    REJECT,
    ChampionPointer,
    ChampionPointerEntry,
    PromotionVerdict,
    apply_promotion,
    build_promotion_issue_draft,
    compare_champion_challenger,
    write_comparison_record,
)
from openmed.eval.champion_challenger_schemas import (
    build_champion_pointer_schema,
    build_comparison_record_schema,
)
from openmed.eval.release_gates import QUARANTINED, RELEASABLE, GateCheck, GateReport

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]


def _report(
    *,
    decision: str = RELEASABLE,
    per_label_recall: dict[str, float] | None = None,
    residual_leakage_rate: float = 0.001,
    critical_leakage_count: int = 0,
    quant_recall_delta: float | None = None,
    p50_ms: float | None = 10.0,
    p95_ms: float | None = 20.0,
    ram_mb: float | None = 128.0,
    gate_results: tuple[GateCheck, ...] = (),
    repo_id: str = "OpenMed/pii-tiny-mlx-fp-v3",
    family: str = "pii",
    tier: str = "tiny",
    fmt: str = "mlx-fp",
    eval_set_hash: str = "a" * 8,
    leakage_fixture_hash: str = "b" * 8,
) -> GateReport:
    """Build a synthetic gate report for a champion or challenger."""

    recall = per_label_recall or {"SSN": 0.9999, "API_KEY": 0.9995}
    return GateReport(
        repo_id=repo_id,
        family=family,
        tier=tier,
        param_count=1_000_000,
        format=fmt,
        per_label_recall=recall,
        per_label_precision={label: 0.999 for label in recall},
        critical_leakage_count=critical_leakage_count,
        residual_leakage_rate=residual_leakage_rate,
        quant_recall_delta=quant_recall_delta,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        ram_mb=ram_mb,
        eval_set_hash=eval_set_hash,
        leakage_fixture_hash=leakage_fixture_hash,
        decision=decision,
        gate_results=gate_results,
    )


def _payload_with_f1(report: GateReport, micro_f1: float) -> dict:
    """Return the report payload with an aggregate F1 metric attached."""

    payload = report.to_dict()
    payload["metrics"] = {"micro_f1": micro_f1}
    return payload


# -- acceptance criterion 1: gate criterion (recall over F1) ------------------


def test_reject_on_higher_f1_but_lower_critical_recall() -> None:
    """Higher F1 cannot rescue a critical-label recall regression: REJECT."""

    champion = _payload_with_f1(
        _report(per_label_recall={"SSN": 0.9999, "API_KEY": 0.9995}),
        micro_f1=0.900,
    )
    # Challenger has strictly higher aggregate F1 but drops SSN recall by 0.0049,
    # well past the G7 tolerance of 0.002.
    challenger = _payload_with_f1(
        _report(per_label_recall={"SSN": 0.9950, "API_KEY": 0.9999}),
        micro_f1=0.980,
    )

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == REJECT
    assert "per_label_recall.SSN" in verdict.regressions
    # Aggregate F1 was captured but never used as a promotion reason.
    assert verdict.ignored_f1["challenger"]["micro_f1"] == 0.980
    assert all("f1" not in reason.lower() for reason in verdict.reasons)


def test_aggregate_f1_never_changes_verdict() -> None:
    """Two challengers identical but for F1 produce identical verdicts + ledgers."""

    champion = _report()
    low_f1 = _payload_with_f1(_report(residual_leakage_rate=0.0005), micro_f1=0.10)
    high_f1 = _payload_with_f1(_report(residual_leakage_rate=0.0005), micro_f1=0.99)

    low = compare_champion_challenger(champion, low_f1)
    high = compare_champion_challenger(champion, high_f1)

    assert low.verdict == high.verdict == PROMOTE
    assert [e.to_dict() for e in low.ledger] == [e.to_dict() for e in high.ledger]


def test_hold_when_only_f1_improves() -> None:
    """A challenger that only improves aggregate F1 is held, not promoted."""

    champion = _report()
    challenger = _payload_with_f1(_report(), micro_f1=0.99)

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == HOLD
    assert verdict.wins == ()


def test_promote_on_recall_and_leakage_win() -> None:
    """A releasable challenger that wins on recall/leakage is promoted."""

    champion = _report(
        per_label_recall={"SSN": 0.9999, "API_KEY": 0.9950},
        residual_leakage_rate=0.0020,
    )
    challenger = _report(
        per_label_recall={"SSN": 0.9999, "API_KEY": 0.9999},
        residual_leakage_rate=0.0010,
    )

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == PROMOTE
    assert "per_label_recall.API_KEY" in verdict.wins
    assert "residual_leakage_rate" in verdict.wins
    assert verdict.regressions == ()


def test_leakage_regression_downgrades_win_to_hold() -> None:
    """A recall win offset by a leakage regression is held, not promoted."""

    champion = _report(
        per_label_recall={"SSN": 0.9999, "API_KEY": 0.9950},
        residual_leakage_rate=0.0010,
    )
    challenger = _report(
        per_label_recall={"SSN": 0.9999, "API_KEY": 0.9999},
        residual_leakage_rate=0.0020,
    )

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == HOLD
    assert "per_label_recall.API_KEY" in verdict.wins
    assert "residual_leakage_rate" in verdict.regressions


def test_within_tolerance_recall_drop_is_a_tie() -> None:
    """A critical-recall drop within the G7 tolerance is a tie, not a regression."""

    # SSN drops 0.0014 (within the 0.002 tolerance) -> tie; API_KEY rises well
    # past tolerance -> a clear win.
    champion = _report(per_label_recall={"SSN": 0.9999, "API_KEY": 0.9950})
    challenger = _report(per_label_recall={"SSN": 0.9985, "API_KEY": 0.9999})

    verdict = compare_champion_challenger(champion, challenger)

    assert "per_label_recall.SSN" in verdict.ties
    assert verdict.verdict == PROMOTE  # API_KEY win, no blocking regression


def test_not_releasable_challenger_is_rejected() -> None:
    """A challenger failing its own floors (QUARANTINED) is rejected."""

    champion = _report()
    challenger = _report(decision=QUARANTINED)

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == REJECT
    assert any("RELEASABLE" in reason for reason in verdict.reasons)


def test_gate_fix_counts_as_promotion_win() -> None:
    """A gate the champion failed that the challenger now passes is a win."""

    champion = _report(gate_results=(GateCheck("G11", False, reason="miss"),))
    challenger = _report(gate_results=(GateCheck("G11", True),))

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == PROMOTE
    assert "gate.G11" in verdict.wins


def test_mismatched_coordinates_are_rejected() -> None:
    """A champion and challenger from different (family, tier, format) cannot compare."""

    champion = _report(family="clinical", tier="large", fmt="onnx")
    challenger = _report(family="pii", tier="tiny", fmt="mlx-fp")

    with pytest.raises(ValueError, match="different rollout coordinates"):
        compare_champion_challenger(champion, challenger)


def test_mismatched_evidence_corpora_are_rejected(subtests) -> None:
    """Head-to-head metrics must come from the same committed evidence corpora."""
    for field, value in (
        ("eval_set_hash", "different-eval"),
        ("leakage_fixture_hash", "different-leakage"),
    ):
        with subtests.test(field=field, value=value):
            champion = _report()
            challenger = _report(**{field: value})

            with pytest.raises(ValueError, match=field):
                compare_champion_challenger(champion, challenger)


def test_bootstrap_without_champion_promotes_releasable() -> None:
    """With no incumbent, a releasable challenger bootstraps the pointer."""

    assert compare_champion_challenger(None, _report()).verdict == PROMOTE
    assert (
        compare_champion_challenger(None, _report(decision=QUARANTINED)).verdict
        == REJECT
    )


# -- acceptance criterion 2: byte-identical reproducibility -------------------


def test_ledger_reproduces_byte_identically(tmp_path: Path) -> None:
    """The comparison record reproduces byte-for-byte from the two reports."""

    champion = _report(residual_leakage_rate=0.0020)
    challenger = _report(residual_leakage_rate=0.0010)

    first = compare_champion_challenger(champion, challenger)
    second = compare_champion_challenger(champion, challenger)
    assert first.to_json() == second.to_json()

    path_a = write_comparison_record(first, tmp_path / "a.json")
    path_b = write_comparison_record(second, tmp_path / "b.json")
    assert path_a.read_bytes() == path_b.read_bytes()

    # The embedded content hash is a self-consistent signature.
    assert first.verify()
    # Round-tripping through the persisted record preserves the exact payload.
    reloaded = PromotionVerdict.from_dict(json.loads(path_a.read_text()))
    assert reloaded.to_json() == first.to_json()
    assert reloaded.record_hash == first.record_hash


@pytest.mark.slow
def test_record_reproduces_across_process(tmp_path: Path) -> None:
    """A record written by the CLI matches the in-process record byte-for-byte."""

    champion_path = tmp_path / "champion.json"
    challenger_path = tmp_path / "challenger.json"
    champion = _report(residual_leakage_rate=0.0020)
    challenger = _report(residual_leakage_rate=0.0010)
    champion_path.write_text(json.dumps(champion.to_dict()), encoding="utf-8")
    challenger_path.write_text(json.dumps(challenger.to_dict()), encoding="utf-8")

    in_process = write_comparison_record(
        compare_champion_challenger(champion, challenger),
        tmp_path / "in_process.json",
    )

    cli_record = tmp_path / "cli.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_WORKTREE_ROOT), env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openmed.eval.champion_challenger",
            "--champion",
            str(champion_path),
            "--challenger",
            str(challenger_path),
            "--record",
            str(cli_record),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert cli_record.read_bytes() == in_process.read_bytes()


# -- acceptance criterion 3: rollback safety ----------------------------------


def test_promote_writes_pointer_and_record(tmp_path: Path) -> None:
    """PROMOTE advances the pointer and writes the comparison record."""

    pointer_path = tmp_path / "champion.json"
    ChampionPointer().save(pointer_path)
    record_path = tmp_path / "record.json"

    champion = _report(residual_leakage_rate=0.0020)
    challenger = _report(
        residual_leakage_rate=0.0010, repo_id="OpenMed/pii-tiny-mlx-fp-v4"
    )
    verdict = compare_champion_challenger(champion, challenger)
    assert verdict.verdict == PROMOTE

    result = apply_promotion(
        verdict,
        pointer_path=pointer_path,
        record_path=record_path,
        challenger_version="v4",
    )

    assert result.promoted is True
    assert record_path.is_file()
    pointer = ChampionPointer.load(pointer_path)
    entry = pointer.get("pii", "tiny", "mlx-fp")
    assert entry is not None
    assert entry.champion == "v4"
    assert entry.record_hash == verdict.record_hash


def test_hold_or_reject_never_advances_pointer(subtests, tmp_path) -> None:
    """A HOLD/REJECT verdict leaves the committed champion pointer untouched."""
    case_root = tmp_path
    for challenger_kwargs in [
        {"per_label_recall": {"SSN": 0.9950, "API_KEY": 0.9999}},  # REJECT
        {},  # HOLD (all ties)
    ]:
        with subtests.test(challenger_kwargs=challenger_kwargs):
            tmp_path = case_root / str(len(list(case_root.iterdir())))
            tmp_path.mkdir()

            pointer_path = tmp_path / "champion.json"
            seeded = ChampionPointer(
                entries={
                    "pii::tiny::mlx-fp": ChampionPointerEntry(
                        family="pii",
                        tier="tiny",
                        format="mlx-fp",
                        champion="v3",
                        record_hash="sha256:" + "0" * 64,
                    )
                }
            )
            seeded.save(pointer_path)
            before = pointer_path.read_bytes()

            champion = _report()
            challenger = _report(**challenger_kwargs)
            verdict = compare_champion_challenger(champion, challenger)
            assert verdict.verdict in {HOLD, REJECT}

            record_path = tmp_path / "record.json"
            result = apply_promotion(
                verdict,
                pointer_path=pointer_path,
                record_path=record_path,
            )

            assert result.promoted is False
            # The pointer file is byte-for-byte unchanged and the incumbent survives.
            assert pointer_path.read_bytes() == before
            assert (
                ChampionPointer.load(pointer_path).get("pii", "tiny", "mlx-fp").champion
                == "v3"
            )
            # No pointer/record is advanced; an offline issue draft is emitted instead.
            assert not record_path.exists()
            assert result.issue_draft is not None
            assert result.issue_draft["verdict"] == verdict.verdict


def test_issue_draft_is_offline_and_deterministic() -> None:
    """The issue draft is a deterministic, network-free artifact."""

    champion = _report()
    challenger = _report(decision=QUARANTINED)
    verdict = compare_champion_challenger(champion, challenger)

    first = build_promotion_issue_draft(verdict)
    second = build_promotion_issue_draft(verdict)
    assert first == second
    assert first["labels"][-1] == "reject"


# -- acceptance criterion 4: derivable with no model call ---------------------


def test_verdict_derivable_from_committed_reports(tmp_path: Path) -> None:
    """The verdict is derived from two committed report files, no model call."""

    champion_path = tmp_path / "champion.json"
    challenger_path = tmp_path / "challenger.json"
    champion_path.write_text(
        json.dumps(_report(residual_leakage_rate=0.0020).to_dict()),
        encoding="utf-8",
    )
    challenger_path.write_text(
        json.dumps(_report(residual_leakage_rate=0.0010).to_dict()),
        encoding="utf-8",
    )

    verdict = compare_champion_challenger(champion_path, challenger_path)

    assert verdict.verdict == PROMOTE
    assert verdict.challenger_repro_hash
    assert verdict.champion_repro_hash


def test_committed_champion_pointer_is_loadable() -> None:
    """The committed gates/champion.json seed loads cleanly."""

    pointer = ChampionPointer.load()
    assert pointer.to_dict()["schema_version"] == 1


def test_champion_pointer_save_is_atomic_on_write_failure(
    tmp_path, monkeypatch
) -> None:
    """A failed mid-write leaves the committed pointer intact, no temp files."""

    pointer_path = tmp_path / "champion.json"
    ChampionPointer().save(pointer_path)
    assert [p.name for p in tmp_path.iterdir()] == ["champion.json"]
    # Use a different, valid serialization so an in-place overwrite is visible.
    pointer_path.write_bytes(pointer_path.read_bytes().rstrip(b"\n"))
    committed = pointer_path.read_bytes()

    def explode(src: object, dst: object) -> None:
        raise OSError("simulated full disk")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError, match="simulated full disk"):
        ChampionPointer().save(pointer_path)

    assert pointer_path.read_bytes() == committed
    assert [p.name for p in tmp_path.iterdir()] == ["champion.json"]


# -- missing axes: a challenger that does not measure what the champion did ----


def test_challenger_missing_a_critical_label_recall_blocks() -> None:
    """A critical label the champion reports and the challenger omits is REJECT."""

    champion = _report(per_label_recall={"SSN": 0.9999, "API_KEY": 0.9995})
    # The challenger never measured API_KEY, so it cannot be shown to hold the
    # champion's recall there. That is a blocking regression, not a skipped axis.
    challenger = _report(per_label_recall={"SSN": 0.9999})

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == REJECT
    entry = next(e for e in verdict.ledger if e.axis == "per_label_recall.API_KEY")
    assert entry.reason == "missing_in_challenger"
    assert entry.blocking is True
    assert any("per_label_recall.API_KEY" in reason for reason in verdict.reasons)


def test_challenger_missing_a_gate_the_champion_passed_blocks() -> None:
    """A gate the champion passed that the challenger omits is REJECT."""

    champion = _report(
        gate_results=(GateCheck("G3", True), GateCheck("G4", True)),
    )
    challenger = _report(gate_results=(GateCheck("G4", True),))

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == REJECT
    entry = next(e for e in verdict.ledger if e.axis == "gate.G3")
    assert entry.reason == "missing_in_challenger"
    assert entry.blocking is True
    assert any("gate.G3" in reason for reason in verdict.reasons)


def test_champion_missing_the_axis_stays_unaligned() -> None:
    """An axis the champion never reported keeps today's behaviour: no entry."""

    champion = _report(per_label_recall={"SSN": 0.9999})
    challenger = _report(per_label_recall={"SSN": 0.9999, "API_KEY": 0.9995})

    verdict = compare_champion_challenger(champion, challenger)

    assert not any(e.axis == "per_label_recall.API_KEY" for e in verdict.ledger)
    assert all("missing_in_challenger" != e.reason for e in verdict.ledger)


def test_challenger_missing_a_gate_the_champion_failed_does_not_block() -> None:
    """Dropping a gate the champion already failed is not a regression."""

    champion = _report(gate_results=(GateCheck("G3", False, reason="miss"),))
    challenger = _report(gate_results=())

    verdict = compare_champion_challenger(champion, challenger)

    assert not any(e.axis == "gate.G3" for e in verdict.ledger)


@pytest.mark.parametrize("missing_axis", ["recall", "gate"])
def test_missing_axis_reason_survives_verified_record_roundtrip(missing_axis):
    champion = _report(gate_results=(GateCheck("G3", True),))
    challenger = _report(
        per_label_recall={"SSN": 0.9999} if missing_axis == "recall" else None,
        gate_results=(GateCheck("G3", True),) if missing_axis == "recall" else (),
    )
    verdict = compare_champion_challenger(champion, challenger)
    restored = PromotionVerdict.from_dict(json.loads(verdict.to_json()))

    assert restored.verify()
    assert restored.to_json() == verdict.to_json()
    assert any(entry.reason == "missing_in_challenger" for entry in restored.ledger)

    tampered = verdict.to_dict()
    entry = next(item for item in tampered["ledger"] if item.get("reason"))
    entry["reason"] = "altered_reason"
    assert not PromotionVerdict.from_dict(tampered).verify()


# -- JSON Schema export (#4552) -----------------------------------------------


def _schema_validator(schema: dict[str, Any]) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _synthetic_records() -> dict[str, dict[str, Any]]:
    """Return one comparison record per verdict path, as written to disk."""

    champion = _report(per_label_recall={"SSN": 0.995, "API_KEY": 0.995})
    better = _report(
        per_label_recall={"SSN": 0.999, "API_KEY": 0.998},
        residual_leakage_rate=0.0005,
        repo_id="OpenMed/pii-tiny-mlx-fp-v4",
    )
    worse = _report(
        per_label_recall={"SSN": 0.97, "API_KEY": 0.995},
        repo_id="OpenMed/pii-tiny-mlx-fp-v4",
    )
    verdicts = {
        "promote": compare_champion_challenger(champion, better),
        "hold_f1_only": compare_champion_challenger(
            _payload_with_f1(champion, 0.80), _payload_with_f1(champion, 0.95)
        ),
        "reject": compare_champion_challenger(champion, worse),
        "missing_axis": compare_champion_challenger(
            _report(gate_results=(GateCheck("G3", True),)),
            _report(per_label_recall={"SSN": 0.9999}),
        ),
        "bootstrap": compare_champion_challenger(None, better),
        "bootstrap_rejected": compare_champion_challenger(
            None, _report(decision=QUARANTINED)
        ),
    }
    return {name: json.loads(verdict.to_json()) for name, verdict in verdicts.items()}


def test_comparison_records_from_synthetic_reports_validate(subtests) -> None:
    validator = _schema_validator(build_comparison_record_schema())
    records = _synthetic_records()

    assert {record["verdict"] for record in records.values()} == {
        PROMOTE,
        HOLD,
        REJECT,
    }
    assert any(entry.get("reason") for entry in records["missing_axis"]["ledger"])
    assert "champion" in records["hold_f1_only"]["ignored_f1"]
    for name, record in records.items():
        with subtests.test(record=name):
            validator.validate(record)


def test_comparison_record_schema_rejects_malformed_records(subtests) -> None:
    validator = _schema_validator(build_comparison_record_schema())
    record = _synthetic_records()["promote"]

    def unknown_verdict(doc):
        doc["verdict"] = "MAYBE"

    def missing_record_hash(doc):
        del doc["record_hash"]

    def malformed_record_hash(doc):
        doc["record_hash"] = "deadbeef"

    def wrong_artifact(doc):
        doc["artifact"] = "openmed.eval.something_else"

    def wrong_schema_version(doc):
        doc["schema_version"] = 2

    def unknown_outcome(doc):
        doc["ledger"][0]["outcome"] = "draw"

    def unknown_category(doc):
        doc["ledger"][0]["category"] = "f1"

    def missing_ledger_field(doc):
        del doc["ledger"][0]["blocking"]

    def string_ledger_value(doc):
        doc["ledger"][0]["delta"] = "0.004"

    def extra_field(doc):
        doc["raw_text"] = "synthetic"

    for mutate in (
        unknown_verdict,
        missing_record_hash,
        malformed_record_hash,
        wrong_artifact,
        wrong_schema_version,
        unknown_outcome,
        unknown_category,
        missing_ledger_field,
        string_ledger_value,
        extra_field,
    ):
        with subtests.test(case=mutate.__name__):
            doc = copy.deepcopy(record)
            mutate(doc)
            assert not validator.is_valid(doc)


def test_champion_pointer_schema_validates_empty_and_populated_stores() -> None:
    validator = _schema_validator(build_champion_pointer_schema())
    record_hash = compare_champion_challenger(None, _report()).record_hash
    populated = ChampionPointer()
    populated.set(
        ChampionPointerEntry(
            family="pii",
            tier="tiny",
            format="mlx-fp",
            champion="v3",
            record_hash=record_hash,
        )
    )
    populated.set(
        ChampionPointerEntry(
            family="pii",
            tier="small",
            format="onnx",
            champion="v5",
            record_hash=record_hash,
            previous_champion="v4",
        )
    )

    for store in (ChampionPointer(), populated):
        validator.validate(json.loads(store.to_json()))
    validator.validate(ChampionPointer.load().to_dict())


def test_champion_pointer_schema_rejects_malformed_stores(subtests) -> None:
    validator = _schema_validator(build_champion_pointer_schema())
    store = ChampionPointer()
    store.set(
        ChampionPointerEntry(family="pii", tier="tiny", format="mlx-fp", champion="v3")
    )
    document = store.to_dict()

    def missing_champion(doc):
        del doc["entries"]["pii::tiny::mlx-fp"]["champion"]

    def numeric_previous_champion(doc):
        doc["entries"]["pii::tiny::mlx-fp"]["previous_champion"] = 3

    def wrong_schema_version(doc):
        doc["schema_version"] = 2

    def entries_as_list(doc):
        doc["entries"] = []

    for mutate in (
        missing_champion,
        numeric_previous_champion,
        wrong_schema_version,
        entries_as_list,
    ):
        with subtests.test(case=mutate.__name__):
            doc = copy.deepcopy(document)
            mutate(doc)
            assert not validator.is_valid(doc)


def _clear_nested(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _clear_nested(child)
        value.clear()
    elif isinstance(value, list):
        for child in value:
            _clear_nested(child)
        value.clear()


@pytest.mark.parametrize(
    "build", [build_comparison_record_schema, build_champion_pointer_schema]
)
def test_schema_builders_return_fresh_byte_stable_mappings(build) -> None:
    first = build()
    rendered = json.dumps(first, sort_keys=True)
    _clear_nested(first)

    assert json.dumps(build(), sort_keys=True) == rendered
    assert build()["$id"].startswith("urn:openmed:eval:")


def test_comparison_record_schema_tracks_module_constants() -> None:
    schema = build_comparison_record_schema()
    entry = schema["$defs"]["ledger_entry"]["properties"]

    assert schema["properties"]["artifact"]["const"] == COMPARISON_RECORD_ARTIFACT
    assert schema["properties"]["verdict"]["enum"] == [PROMOTE, HOLD, REJECT]
    assert set(entry["outcome"]["enum"]) == {"win", "tie", "regression"}
