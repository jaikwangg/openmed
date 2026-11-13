"""The doctor reports the Unicode data version behind offset computations."""

from __future__ import annotations

import argparse
import json
import unicodedata

from openmed.cli import main_module
from openmed.cli.main import _handle_doctor
from openmed.core import doctor as doctor_module
from openmed.core.doctor import run_diagnostics

SYNTHETIC_UNIDATA_VERSION = "99.0.0"


def _unicode_check(checks):
    return next(item for item in checks if item["name"] == "unicode_data_version")


def test_unicode_data_version_reports_patched_value(monkeypatch):
    monkeypatch.setattr(
        doctor_module.unicodedata, "unidata_version", SYNTHETIC_UNIDATA_VERSION
    )

    checks = []
    doctor_module._check_unicode_data(checks)

    assert checks == [
        {
            "name": "unicode_data_version",
            "status": "PASS",
            "details": SYNTHETIC_UNIDATA_VERSION,
            "version": SYNTHETIC_UNIDATA_VERSION,
        }
    ]


def test_run_diagnostics_includes_unicode_data_version():
    check = _unicode_check(run_diagnostics())

    assert check["status"] == "PASS"
    assert check["version"] == unicodedata.unidata_version
    assert check["details"] == unicodedata.unidata_version
    assert "hint" not in check


def test_unicode_data_version_never_fails(monkeypatch):
    class _Broken:
        @property
        def unidata_version(self):
            raise RuntimeError("SYNTHETIC_PRIVATE_FAILURE")

    monkeypatch.setattr(doctor_module, "unicodedata", _Broken())

    results = run_diagnostics()
    check = _unicode_check(results)

    assert check["status"] == "WARN"
    assert check["details"] == "version unavailable"
    assert "version" not in check
    assert "SYNTHETIC_PRIVATE_FAILURE" not in json.dumps(results)
    assert _handle_doctor(argparse.Namespace(json=True)) == 0


def test_doctor_text_output_shows_unicode_data_version(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor_module.unicodedata, "unidata_version", SYNTHETIC_UNIDATA_VERSION
    )

    exit_code = _handle_doctor(argparse.Namespace(json=False))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"PASS unicode_data_version: {SYNTHETIC_UNIDATA_VERSION}" in captured.out


def test_doctor_json_output_has_stable_unicode_key(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor_module.unicodedata, "unidata_version", SYNTHETIC_UNIDATA_VERSION
    )

    exit_code = main_module.main(["doctor", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    check = _unicode_check(payload["data"]["checks"])
    assert check["status"] == "PASS"
    assert check["version"] == SYNTHETIC_UNIDATA_VERSION
