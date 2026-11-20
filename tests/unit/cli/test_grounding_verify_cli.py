"""Tests for ``openmed grounding verify`` over cached snapshot manifests."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from openmed.cli import main_module

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("grounding verify must not open a network connection")

    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket.socket, "connect", _refuse)


def _import_snapshot(cache_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main_module.main(
        [
            "grounding",
            "import",
            "--system",
            "icd10cm",
            "--input",
            str(FIXTURE),
            "--version",
            "synthetic-fixture-1",
            "--cache-dir",
            str(cache_dir),
        ]
    )
    capsys.readouterr()
    assert exit_code == 0


def _verify(
    cache_dir: Path, capsys: pytest.CaptureFixture[str], *extra: str
) -> tuple[int, dict]:
    exit_code = main_module.main(
        ["grounding", "verify", "--cache-dir", str(cache_dir), "--json", *extra]
    )
    return exit_code, json.loads(capsys.readouterr().out)


def test_freshly_imported_snapshot_verifies_ok(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_dir = tmp_path / "grounding"
    _import_snapshot(cache_dir, capsys)

    exit_code, envelope = _verify(cache_dir, capsys)

    assert exit_code == 0
    assert envelope["ok"] is True
    assert envelope["command"] == "grounding verify"
    assert envelope["outcome"] == "completed"
    assert envelope["data"]["snapshots"] == [
        {
            "system": "icd10cm",
            "version": "synthetic-fixture-1",
            "artifact": "concepts.jsonl",
            "status": "ok",
        }
    ]


def test_one_changed_byte_reports_mismatch_and_exits_non_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_dir = tmp_path / "grounding"
    _import_snapshot(cache_dir, capsys)
    artifact = cache_dir / "icd10cm" / "concepts.jsonl"
    data = bytearray(artifact.read_bytes())
    data[0] ^= 0x01
    artifact.write_bytes(bytes(data))

    exit_code, envelope = _verify(cache_dir, capsys, "--system", "icd10cm")

    assert exit_code == 1
    assert envelope["outcome"] == "check_failed"
    assert [item["status"] for item in envelope["data"]["snapshots"]] == ["mismatch"]
    assert str(tmp_path) not in json.dumps(envelope)


def test_deleted_artifact_reports_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_dir = tmp_path / "grounding"
    _import_snapshot(cache_dir, capsys)
    (cache_dir / "icd10cm" / "concepts.jsonl").unlink()

    exit_code, envelope = _verify(cache_dir, capsys)

    assert exit_code == 1
    assert [item["status"] for item in envelope["data"]["snapshots"]] == ["missing"]


def test_empty_cache_verifies_with_no_snapshots(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code, envelope = _verify(tmp_path / "empty", capsys)

    assert exit_code == 0
    assert envelope["data"]["snapshots"] == []
