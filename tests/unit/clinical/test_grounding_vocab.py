"""Tests for free vocabulary loading and alias indexing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from openmed.clinical.grounding import (
    FREE_VOCAB_SYSTEMS,
    RestrictedVocabularyError,
    VocabLoader,
    VocabSource,
    VocabularyChecksumError,
    load_grounding_snapshot_manifest_schema,
)
from openmed.core.offline import OfflineModeError


def _write_fixture(path: Path, system: str) -> Path:
    target = path / f"{system}.tsv"
    target.write_text(
        "code\tpreferred_term\tsynonyms\n"
        f"{system.upper()}-001\tAspirin\tacetylsalicylic acid|ASA\n"
        f"{system.upper()}-002\tMetformin\tGlucophage\n",
        encoding="utf-8",
    )
    return target


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loader_indexes_fixture_for_all_free_systems(tmp_path):
    registry = {}
    for system in FREE_VOCAB_SYSTEMS:
        fixture = _write_fixture(tmp_path, system)
        registry[system] = VocabSource(
            system=system,
            path=fixture,
            sha256=_sha256(fixture),
        )

    loader = VocabLoader(cache_dir=tmp_path / "cache", registry=registry)

    for system in FREE_VOCAB_SYSTEMS:
        index = loader.get_index(system)
        assert index.concept_count == 2
        assert index["aspirin"] == f"{system.upper()}-001"
        assert index.get("acetylsalicylic acid") == f"{system.upper()}-001"
        assert index.lookup("ASA").preferred_term == "Aspirin"


def test_system_aliases_resolve_to_fixture_index(tmp_path):
    fixture = _write_fixture(tmp_path, "icd10cm")
    loader = VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "icd10cm": VocabSource(
                system="icd10cm",
                path=fixture,
                sha256=_sha256(fixture),
            )
        },
    )

    assert loader.get_index("icd-10-cm")["ASA"] == "ICD10CM-001"


def test_offline_mode_blocks_empty_cache_download(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENMED_OFFLINE", "1")

    def fail_download(url: str, target: Path, timeout: float) -> None:
        raise AssertionError("download should not be attempted")

    loader = VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "rxnorm": VocabSource(
                system="rxnorm",
                url="https://example.invalid/rxnorm.tsv",
                sha256="0" * 64,
            )
        },
        downloader=fail_download,
    )

    with pytest.raises(OfflineModeError, match="vocabulary download for rxnorm"):
        loader.get_index("rxnorm")


def test_offline_mode_serves_cached_vocab_without_download(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    cached_dir = tmp_path / "cache" / "rxnorm"
    cached_dir.mkdir(parents=True)
    cached_fixture = _write_fixture(cached_dir, "concepts")

    def fail_download(url: str, target: Path, timeout: float) -> None:
        raise AssertionError("download should not be attempted")

    loader = VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "rxnorm": VocabSource(
                system="rxnorm",
                url="https://example.invalid/rxnorm.tsv",
                sha256=_sha256(cached_fixture),
            )
        },
        downloader=fail_download,
    )

    assert loader.get_index("rxnorm").get("ASA") == "CONCEPTS-001"


def test_restricted_vocabularies_point_to_user_key_path(subtests):
    for system in ["umls", "snomed", "snomed-ct"]:
        with subtests.test(system=system):
            loader = VocabLoader()

            with pytest.raises(RestrictedVocabularyError, match="user-key-gated"):
                loader.get_index(system)


def test_checksum_mismatch_rejects_local_source(tmp_path):
    fixture = _write_fixture(tmp_path, "mesh")
    loader = VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "mesh": VocabSource(
                system="mesh",
                path=fixture,
                sha256="0" * 64,
            )
        },
    )

    with pytest.raises(VocabularyChecksumError, match="Checksum mismatch"):
        loader.get_index("mesh")


def test_downloaded_artifact_requires_checksum(tmp_path):
    loader = VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "hpo": VocabSource(
                system="hpo",
                url="https://example.invalid/hp.obo",
            )
        },
    )

    with pytest.raises(VocabularyChecksumError, match="without a SHA-256 checksum"):
        loader.get_index("hpo")


def _manifest_validator() -> Draft202012Validator:
    schema = load_grounding_snapshot_manifest_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _imported_manifest(tmp_path: Path, system: str) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fixture = _write_fixture(tmp_path, system)
    loader = VocabLoader(cache_dir=tmp_path / "cache", registry={})
    loader.import_snapshot(
        system,
        fixture,
        version="2026-01",
        license_note="Synthetic fixture",
    )
    manifest_path = tmp_path / "cache" / system / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_imported_snapshot_manifests_validate_against_schema(tmp_path):
    validator = _manifest_validator()

    for system in FREE_VOCAB_SYSTEMS:
        manifest = _imported_manifest(tmp_path / system, system)
        assert list(validator.iter_errors(manifest)) == []


def test_snapshot_manifest_schema_matches_manifest_fields(tmp_path):
    schema = load_grounding_snapshot_manifest_schema()
    manifest = _imported_manifest(tmp_path, "loinc")

    assert set(schema["properties"]) == set(manifest)
    assert set(schema["required"]) == set(manifest)
    assert schema["properties"]["system"]["enum"] == list(FREE_VOCAB_SYSTEMS)


@pytest.mark.parametrize(
    "digest",
    [
        "a" * 64,
        "sha256:" + "a" * 63,
        "sha256:" + "A" * 64,
        "sha256:" + "g" * 64,
        "md5:" + "a" * 64,
    ],
)
def test_snapshot_manifest_schema_rejects_malformed_digest(tmp_path, digest):
    manifest = _imported_manifest(tmp_path, "rxnorm")
    manifest["sha256"] = digest

    assert not _manifest_validator().is_valid(manifest)


@pytest.mark.parametrize("system", ["snomed", "umls", "SNOMED", "rx-norm"])
def test_snapshot_manifest_schema_rejects_non_free_system(tmp_path, system):
    manifest = _imported_manifest(tmp_path, "rxnorm")
    manifest["system"] = system

    assert not _manifest_validator().is_valid(manifest)


def test_snapshot_manifest_schema_rejects_missing_blank_and_extra_fields(tmp_path):
    validator = _manifest_validator()
    manifest = _imported_manifest(tmp_path, "hpo")

    missing = dict(manifest)
    del missing["sha256"]
    blank = {**manifest, "version": "   "}
    extra = {**manifest, "source_text": "synthetic"}

    for payload in (missing, blank, extra):
        assert not validator.is_valid(payload)
