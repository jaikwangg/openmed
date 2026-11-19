"""Offline tests for Maple ONNX Runtime bundle integrity."""

from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from openmed.onnx.maple_bundle import (
    MAPLE_BUNDLE_FILENAME,
    MapleBundleError,
    build_maple_onnx_bundle,
    main,
    validate_maple_onnx_bundle,
)


def _write_fake_export(root):
    files = {
        "decoder_model.ort": b"prefill-graph",
        "decoder_with_past_model.ort": b"decode-graph",
        "tokenizer.json": b'{"model":{"type":"BPE"}}',
        "tokenizer_config.json": b'{"model_max_length":131072}',
        "config.json": b'{"architectures":["MapleForCausalLM"]}',
    }
    for name, payload in files.items():
        (root / name).write_bytes(payload)


def test_builds_android_bundle_with_manifest_first_and_exact_integrity(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)

    result = build_maple_onnx_bundle(source, tmp_path / "maple.ommaple.zip")

    assert result.manifest["architecture"] == "MapleForCausalLM"
    assert result.manifest["runtime"] == "onnxruntime-mobile"
    assert result.manifest["quantization"] == "qmoe-4bit-blockwise-128"
    assert result.total_size_bytes > 0
    with zipfile.ZipFile(result.bundle_path) as archive:
        assert archive.namelist()[0] == MAPLE_BUNDLE_FILENAME
        assert all(
            entry.compress_type == zipfile.ZIP_STORED for entry in archive.infolist()
        )
    assert validate_maple_onnx_bundle(result.bundle_path).manifest == result.manifest


def test_builds_web_bundle_with_single_decoder_graph(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)

    result = build_maple_onnx_bundle(
        source,
        tmp_path / "maple-web.zip",
        prefill_path="decoder_model.ort",
        decode_path=None,
        runtime="onnxruntime-web",
    )

    assert result.manifest["runtime"] == "onnxruntime-web"
    assert result.manifest["graphs"]["decode_path"] is None
    assert result.manifest["cache"] is None


def test_builds_stateless_android_bundle_accepted_by_mobile_contract(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)

    result = build_maple_onnx_bundle(
        source,
        tmp_path / "maple-mobile-stateless.zip",
        prefill_path="decoder_model.ort",
        decode_path=None,
        runtime="onnxruntime-mobile",
    )

    assert result.manifest["runtime"] == "onnxruntime-mobile"
    assert result.manifest["graphs"]["decode_path"] is None
    assert result.manifest["cache"] is None


def test_builds_unified_cached_decoder_without_duplicate_payload(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)

    result = build_maple_onnx_bundle(
        source,
        tmp_path / "maple-mobile-unified.zip",
        prefill_path="decoder_model.ort",
        decode_path="decoder_model.ort",
        runtime="onnxruntime-mobile",
    )

    assert result.manifest["graphs"]["prefill_path"] == "decoder_model.ort"
    assert result.manifest["graphs"]["decode_path"] == "decoder_model.ort"
    assert result.manifest["cache"] is not None
    declared_paths = [item["path"] for item in result.manifest["files"]]
    assert declared_paths.count("decoder_model.ort") == 1
    with zipfile.ZipFile(result.bundle_path) as archive:
        assert archive.namelist().count("decoder_model.ort") == 1


def test_refuses_overwrite_and_path_traversal(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    output = tmp_path / "maple.zip"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="overwrite"):
        build_maple_onnx_bundle(source, output)
    with pytest.raises(MapleBundleError, match="unsafe bundle path"):
        build_maple_onnx_bundle(source, tmp_path / "new.zip", prefill_path="../x.ort")


def test_rejects_tampered_payload(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    original = build_maple_onnx_bundle(source, tmp_path / "maple.zip")
    tampered = tmp_path / "tampered.zip"

    with zipfile.ZipFile(original.bundle_path) as source_archive:
        with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED) as output:
            for entry in source_archive.infolist():
                payload = source_archive.read(entry)
                if entry.filename == "decoder_model.ort":
                    payload = b"tampered-data"
                output.writestr(entry.filename, payload)

    with pytest.raises(MapleBundleError, match="checksum mismatch"):
        validate_maple_onnx_bundle(tampered)


def test_rejects_undeclared_archive_entry(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    original = build_maple_onnx_bundle(source, tmp_path / "maple.zip")
    invalid = tmp_path / "invalid.zip"

    with zipfile.ZipFile(original.bundle_path) as source_archive:
        manifest = json.loads(source_archive.read(MAPLE_BUNDLE_FILENAME))
        with zipfile.ZipFile(invalid, "w", compression=zipfile.ZIP_STORED) as output:
            output.writestr(MAPLE_BUNDLE_FILENAME, json.dumps(manifest))
            for entry in source_archive.infolist()[1:]:
                output.writestr(entry.filename, source_archive.read(entry))
            output.writestr("undeclared.bin", b"not allowed")

    with pytest.raises(MapleBundleError, match="exactly match"):
        validate_maple_onnx_bundle(invalid)


def test_cli_builds_bundle_with_positional_arguments(tmp_path, capsys):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    bundle = tmp_path / "maple.zip"

    assert main([str(source), str(bundle)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["bundle"] == str(bundle.resolve())
    assert validate_maple_onnx_bundle(bundle).manifest["runtime"] == report["runtime"]


def test_cli_validate_reports_valid_bundle(tmp_path, capsys):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    built = build_maple_onnx_bundle(source, tmp_path / "maple.zip")
    with zipfile.ZipFile(built.bundle_path) as archive:
        manifest_sha256 = hashlib.sha256(
            archive.read(MAPLE_BUNDLE_FILENAME)
        ).hexdigest()

    assert main(["--validate", str(built.bundle_path)]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "status": "valid",
        "runtime": "onnxruntime-mobile",
        "file_count": 5,
        "total_size_bytes": built.total_size_bytes,
        "manifest_sha256": manifest_sha256,
    }


def test_cli_validate_reports_tampered_bundle_without_paths(tmp_path, capsys):
    source = tmp_path / "export"
    source.mkdir()
    _write_fake_export(source)
    original = build_maple_onnx_bundle(source, tmp_path / "maple.zip")
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(original.bundle_path) as source_archive:
        with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED) as output:
            for entry in source_archive.infolist():
                payload = source_archive.read(entry)
                if entry.filename == "decoder_model.ort":
                    payload = b"tampered-data"
                output.writestr(entry.filename, payload)

    assert main(["--validate", str(tampered)]) == 1

    output = capsys.readouterr().out
    assert json.loads(output) == {"status": "invalid", "code": "bundle_invalid"}
    assert str(tmp_path) not in output


def test_cli_validate_reports_missing_bundle_without_paths(tmp_path, capsys):
    missing = tmp_path / "missing.zip"

    assert main(["--validate", str(missing)]) == 1

    output = capsys.readouterr().out
    assert json.loads(output) == {"status": "invalid", "code": "bundle_not_found"}
    assert str(tmp_path) not in output


def test_cli_rejects_validate_mixed_with_build_arguments(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["--validate", str(tmp_path / "maple.zip"), str(tmp_path)])
    assert excinfo.value.code == 2

    with pytest.raises(SystemExit) as excinfo:
        main([str(tmp_path)])
    assert excinfo.value.code == 2
