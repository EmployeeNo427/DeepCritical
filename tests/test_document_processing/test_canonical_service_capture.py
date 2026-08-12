from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pytest
from defusedxml import ElementTree
from pypdf import PdfReader

from DeepResearch.src.document_processing.alignment import DoclingGrobidAligner
from DeepResearch.src.document_processing.canonical import (
    CanonicalAnchorRole,
    CanonicalDiagnosticSeverity,
    CanonicalDocumentView,
    CanonicalizationConfig,
    build_canonical_document_view,
    canonical_document_bytes,
    load_canonical_document,
)
from DeepResearch.src.document_processing.clients import GrobidClient
from DeepResearch.src.document_processing.models import (
    ArtifactLocationRole,
    ContentSpanSet,
    DataProductRef,
    DocumentArtifact,
    RuntimeAttestationSource,
    sha256_bytes,
)
from DeepResearch.src.document_processing.native_contracts import (
    DOCLING_COMPONENT_VERSION,
    DOCLING_CONTAINER_IMAGE,
    DOCLING_SERVE_VERSION,
    GROBID_COMPONENT_VERSION,
    GROBID_CONTAINER_IMAGE,
    OCR_COMPONENT_VERSION,
    OCR_CONTAINER_IMAGE,
    OCR_DIGEST_RUNNER_VERSION,
    REMOTE_ATTESTATION_CONTRACT_VERSION,
    RUNTIME_ATTESTATION_SCHEMA_VERSION,
)
from DeepResearch.src.document_processing.storage import ContentAddressedStore
from DeepResearch.src.document_processing.validation import (
    DoclingQualityValidator,
    build_pdf_content_spans,
    docling_document_sha256,
    validate_content_integrity,
)
from tests.test_document_processing.test_live_stack_contract import (
    rich_raster_pdf_bytes,
)

FIXTURE_ROOT = (
    Path(__file__).parents[1]
    / "fixtures"
    / "document_processing"
    / "canonical"
    / "service_capture"
    / "v1"
)

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_OCI_REFERENCE = re.compile(r"\S+@sha256:[0-9a-f]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EPHEMERAL_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
_TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"

_FILE_CONTRACTS = {
    "README.md": ("fixture-documentation", "text/markdown"),
    "docling_document.json": (
        "selected-native-docling-document",
        "application/json",
    ),
    "grobid.tei.xml": ("selected-native-grobid-tei", "application/tei+xml"),
    "source-rich-raster.pdf": ("synthetic-raster-source", "application/pdf"),
}

_EXPECTED_CAPTURE = {
    "evidence_artifact_digest": (
        "sha256:7dbdfb0c9f30f6d881739fa3b12c2e7afa872265c7434d032d1668c636311feb"
    ),
    "evidence_artifact_id": 9138405936,
    "evidence_artifact_name": "document-processing-all-real-31587722084",
    "native_inventory_sha256": (
        "eda08ebbdfb858cb4e56b9e5cb316e84fe07945d49127260dfdfa0a286361447"
    ),
    "raw_manifest_schema": "deepcritical-live-all-real-evidence-v1",
    "raw_manifest_sha256": (
        "4b1c5cdd20eca223a341c0e83f48adbef9a6c912431bafa3a9b0721b54ae85c8"
    ),
    "repository": "EmployeeNo427/DeepCritical",
    "tested_revision": "a2c91ddcd7f4c38cd011a11487e02296c3c356a8",
    "workflow_job": "all-real",
    "workflow_path": ".github/workflows/document-processing-live.yml",
    "workflow_run_attempt": 1,
    "workflow_run_id": 31587722084,
}

_EXPECTED_SERVICE_IDENTITIES = {
    "docling": {
        "container_image_id": (
            "sha256:d3ad0dd8e3baac86423e207de7f7c4aaa9dd1f6ddbc888110db29bb1dccaa1bb"
        ),
        "container_reference": (
            "quay.io/docling-project/docling-serve-cpu:v1.21.0@"
            "sha256:c7d56cf78c45ab61406bc2dfebbac562"
            "c16e38538393f838991a949577cd3d0a"
        ),
    },
    "grobid": {
        "base_image_reference": (
            "grobid/grobid:0.9.0-full@"
            "sha256:52774a8375b30a9bd541936964b48600"
            "c65afdb2e9120ae4764322478ab0ca0d"
        ),
        "container_image_id": (
            "sha256:7bdb827f785b07ceb8b5494de9615d5b4fa90001aa34db4b2ed8070fdbd9e6ca"
        ),
        "container_reference": (
            "deepcritical/grobid:0.9.0-full-p0-c2@"
            "sha256:7bdb827f785b07ceb8b5494de9615d5b"
            "4fa90001aa34db4b2ed8070fdbd9e6ca"
        ),
        "dockerfile_blob": "7a5a7e97e8c25666e6ff4b3378a8813b56fc0abc",
    },
    "ocrmypdf": {
        "container_image_id": None,
        "container_reference": (
            "jbarlow83/ocrmypdf:v17.4.1@"
            "sha256:172736b5dd378a3e1780145b47015f62"
            "504be7b4e0d8cc6f6b5eb0eae8ebd85b"
        ),
    },
}

_EXPECTED_OUTCOMES = {
    "overall_status": "partial",
    "stages": {
        "alignment": {
            "accepted_for_fixture": True,
            "component_id": "docling-grobid-aligner",
            "reason_codes": ["SCHOLARLY_ANNOTATIONS_UNALIGNED"],
            "selected_native_output": False,
            "status": "complete",
        },
        "canonicalize": {
            "accepted_for_fixture": True,
            "component_id": "canonical-document-view",
            "reason_codes": ["UNRESOLVED_SCHOLARLY_ANCHOR"],
            "selected_native_output": False,
            "status": "complete",
        },
        "docling": {
            "accepted_for_fixture": True,
            "component_id": "docling",
            "reason_codes": ["DOCLING_RUNTIME_PROVENANCE_INCOMPLETE"],
            "selected_native_output": True,
            "status": "partial",
        },
        "fallback-grobid": {
            "accepted_for_fixture": True,
            "component_id": "grobid",
            "reason_codes": ["GROBID_RUNTIME_PROVENANCE_INCOMPLETE"],
            "selected_native_output": True,
            "status": "partial",
        },
        "integrity": {
            "accepted_for_fixture": True,
            "component_id": "docling-content-integrity",
            "reason_codes": [],
            "selected_native_output": False,
            "status": "complete",
        },
        "ocr": {
            "accepted_for_fixture": True,
            "component_id": "ocrmypdf",
            "reason_codes": [],
            "selected_native_output": False,
            "status": "complete",
        },
        "primary-grobid": {
            "accepted_for_fixture": True,
            "component_id": "grobid",
            "input_kind": "raster-source-pdf",
            "reason_codes": ["GROBID_CONVERSION_FAILED"],
            "selected_native_output": False,
            "status": "failed",
        },
    },
}

_FORBIDDEN_CAPTURE_KEYS = {
    "api_key",
    "artifacts",
    "captured_at",
    "component_invocation_id",
    "diagnostics",
    "invocation_id",
    "observed_at",
    "pipeline_run_id",
    "processing_runs",
    "remote_task_id",
    "stage_invocation_id",
    "workload_id",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _exact_object(
    value: Any,
    keys: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    assert isinstance(value, dict), f"{label} must be an object"
    assert set(value) == keys, f"{label} keys are not allowlisted"
    return cast("dict[str, Any]", value)


def _positive_int(value: Any, *, label: str) -> int:
    assert type(value) is int, f"{label} must be an integer"
    assert value > 0, f"{label} must be positive"
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    assert type(value) is int, f"{label} must be an integer"
    assert value >= 0, f"{label} must be non-negative"
    return value


def _assert_sanitized(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            assert key not in _FORBIDDEN_CAPTURE_KEYS
            assert "password" not in key.lower()
            assert "secret" not in key.lower()
            _assert_sanitized(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_sanitized(nested)
    elif isinstance(value, str):
        assert _EPHEMERAL_UUID.search(value) is None
        assert re.search(r"artifact-[0-9a-f]{64}", value) is None


def _validate_runtime_attestation(
    value: Any,
    *,
    contract_version: str,
    reporter_id: str,
    source: str,
    component_versions: dict[str, str],
) -> None:
    runtime = _exact_object(
        value,
        {
            "component_versions",
            "contract_version",
            "model_hashes",
            "model_versions",
            "reporter_id",
            "schema_version",
            "source",
        },
        label="runtime attestation",
    )
    assert runtime == {
        "component_versions": component_versions,
        "contract_version": contract_version,
        "model_hashes": {},
        "model_versions": {},
        "reporter_id": reporter_id,
        "schema_version": RUNTIME_ATTESTATION_SCHEMA_VERSION,
        "source": source,
    }


def _validate_service_contracts(value: Any) -> None:
    services = _exact_object(
        value,
        {"docling", "grobid", "ocrmypdf"},
        label="services",
    )

    docling = _exact_object(
        services["docling"],
        {
            "component_id",
            "component_version",
            "container_image_id",
            "container_reference",
            "options",
            "runtime_attestation",
            "serve_version",
        },
        label="Docling service",
    )
    assert docling["component_id"] == "docling"
    assert docling["component_version"] == DOCLING_COMPONENT_VERSION
    assert docling["serve_version"] == DOCLING_SERVE_VERSION
    assert _OCI_REFERENCE.fullmatch(docling["container_reference"])
    assert docling["container_reference"].startswith(f"{DOCLING_CONTAINER_IMAGE}@")
    assert _IMAGE_ID.fullmatch(docling["container_image_id"])
    assert {
        key: docling[key] for key in _EXPECTED_SERVICE_IDENTITIES["docling"]
    } == _EXPECTED_SERVICE_IDENTITIES["docling"]
    assert docling["options"] == {
        "do_ocr": True,
        "image_export_mode": "embedded",
        "table_mode": "accurate",
        "to_formats": ["json"],
    }
    _validate_runtime_attestation(
        docling["runtime_attestation"],
        contract_version=REMOTE_ATTESTATION_CONTRACT_VERSION,
        reporter_id="github-actions-docker-inspector",
        source=RuntimeAttestationSource.AUTHENTICATED_DEPLOYMENT_REPORTER.value,
        component_versions={
            "docling": DOCLING_COMPONENT_VERSION,
            "docling_serve": DOCLING_SERVE_VERSION,
        },
    )

    grobid = _exact_object(
        services["grobid"],
        {
            "base_image_reference",
            "component_id",
            "component_version",
            "container_image_id",
            "container_reference",
            "dockerfile_blob",
            "options",
            "runtime_attestation",
        },
        label="GROBID service",
    )
    assert grobid["component_id"] == "grobid"
    assert grobid["component_version"] == GROBID_COMPONENT_VERSION
    assert _OCI_REFERENCE.fullmatch(grobid["container_reference"])
    assert grobid["container_reference"].startswith(f"{GROBID_CONTAINER_IMAGE}@")
    assert _IMAGE_ID.fullmatch(grobid["container_image_id"])
    assert _OCI_REFERENCE.fullmatch(grobid["base_image_reference"])
    assert grobid["base_image_reference"].startswith("grobid/grobid:0.9.0-full@")
    assert _GIT_SHA.fullmatch(grobid["dockerfile_blob"])
    assert {
        key: grobid[key] for key in _EXPECTED_SERVICE_IDENTITIES["grobid"]
    } == _EXPECTED_SERVICE_IDENTITIES["grobid"]
    assert grobid["options"] == {
        "consolidate_citations": 0,
        "consolidate_header": 0,
        "coordinates": list(GrobidClient.DEFAULT_COORDINATES),
        "minimum_text_characters": 100,
        "segment_sentences": True,
    }
    _validate_runtime_attestation(
        grobid["runtime_attestation"],
        contract_version=REMOTE_ATTESTATION_CONTRACT_VERSION,
        reporter_id="github-actions-docker-inspector",
        source=RuntimeAttestationSource.AUTHENTICATED_DEPLOYMENT_REPORTER.value,
        component_versions={"grobid": GROBID_COMPONENT_VERSION},
    )

    ocrmypdf = _exact_object(
        services["ocrmypdf"],
        {
            "component_id",
            "component_version",
            "container_image_id",
            "container_reference",
            "options",
            "runtime_attestation",
        },
        label="OCRmyPDF service",
    )
    assert ocrmypdf["component_id"] == "ocrmypdf"
    assert ocrmypdf["component_version"] == OCR_COMPONENT_VERSION
    assert _OCI_REFERENCE.fullmatch(ocrmypdf["container_reference"])
    assert ocrmypdf["container_reference"].startswith(f"{OCR_CONTAINER_IMAGE}@")
    assert ocrmypdf["container_image_id"] is None
    assert {
        key: ocrmypdf[key] for key in _EXPECTED_SERVICE_IDENTITIES["ocrmypdf"]
    } == _EXPECTED_SERVICE_IDENTITIES["ocrmypdf"]
    assert ocrmypdf["options"] == {
        "deskew": True,
        "fallback_reason": "image_only_pages+grobid_text_insufficient",
        "jobs": 1,
        "languages": ["eng"],
        "mode": "container_cli",
        "optimize": 1,
        "output_type": "pdf",
        "rotate_pages": True,
        "skip_text": True,
    }
    _validate_runtime_attestation(
        ocrmypdf["runtime_attestation"],
        contract_version=OCR_DIGEST_RUNNER_VERSION,
        reporter_id=OCR_DIGEST_RUNNER_VERSION,
        source=RuntimeAttestationSource.DIGEST_ADDRESSED_OCI_INVOCATION.value,
        component_versions={
            "ocrmypdf": OCR_COMPONENT_VERSION,
            "tesseract": "tesseract 5.5.1",
        },
    )


def _validate_manifest(root: Path, value: Any) -> dict[str, Any]:
    manifest = _exact_object(
        value,
        {
            "accepted_outcomes",
            "capture",
            "files",
            "fixture_version",
            "quality",
            "replay",
            "schema_version",
            "selection",
            "services",
            "synthetic_input",
        },
        label="fixture manifest",
    )
    assert manifest["schema_version"] == ("deepcritical-service-capture-fixture-v1")
    assert manifest["fixture_version"] == "v1"
    assert manifest["synthetic_input"] is True
    _assert_sanitized(manifest)

    capture = _exact_object(
        manifest["capture"],
        {
            "evidence_artifact_digest",
            "evidence_artifact_id",
            "evidence_artifact_name",
            "native_inventory_sha256",
            "raw_manifest_schema",
            "raw_manifest_sha256",
            "repository",
            "tested_revision",
            "workflow_job",
            "workflow_path",
            "workflow_run_attempt",
            "workflow_run_id",
        },
        label="capture provenance",
    )
    assert capture["repository"] == "EmployeeNo427/DeepCritical"
    assert _GIT_SHA.fullmatch(capture["tested_revision"])
    assert capture["raw_manifest_schema"] == ("deepcritical-live-all-real-evidence-v1")
    assert _HEX_SHA256.fullmatch(capture["raw_manifest_sha256"])
    assert _HEX_SHA256.fullmatch(capture["native_inventory_sha256"])
    assert capture["workflow_path"] == (
        ".github/workflows/document-processing-live.yml"
    )
    assert capture["workflow_job"] == "all-real"
    workflow_run_id = _positive_int(
        capture["workflow_run_id"],
        label="workflow_run_id",
    )
    _positive_int(capture["workflow_run_attempt"], label="workflow_run_attempt")
    _positive_int(capture["evidence_artifact_id"], label="evidence_artifact_id")
    assert capture["evidence_artifact_name"] == (
        f"document-processing-all-real-{workflow_run_id}"
    )
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        capture["evidence_artifact_digest"],
    )
    assert capture == _EXPECTED_CAPTURE

    _validate_service_contracts(manifest["services"])
    assert manifest["accepted_outcomes"] == _EXPECTED_OUTCOMES
    assert manifest["selection"] == {
        "docling_document": {
            "file": "docling_document.json",
            "product_schema_version": "docling-document-v1",
            "source_kind": "source-raster-pdf",
            "stage_id": "docling",
        },
        "grobid_tei": {
            "file": "grobid.tei.xml",
            "product_schema_version": "tei-p5",
            "source_kind": "ocr-derivative-pdf",
            "stage_id": "fallback-grobid",
        },
    }

    quality = _exact_object(
        manifest["quality"],
        {"docling", "grobid"},
        label="quality",
    )
    docling_quality = _exact_object(
        quality["docling"],
        {
            "broken_reference_count",
            "issue_codes",
            "located_text_item_count",
            "locator_coverage",
            "reference_count",
            "text_item_count",
        },
        label="Docling quality",
    )
    _positive_int(docling_quality["text_item_count"], label="text_item_count")
    _positive_int(
        docling_quality["located_text_item_count"],
        label="located_text_item_count",
    )
    _positive_int(docling_quality["reference_count"], label="reference_count")
    _nonnegative_int(
        docling_quality["broken_reference_count"],
        label="broken_reference_count",
    )
    locator_coverage = docling_quality["locator_coverage"]
    assert type(locator_coverage) in {int, float}
    assert 0 <= locator_coverage <= 1
    issue_codes = docling_quality["issue_codes"]
    assert isinstance(issue_codes, list)
    assert issue_codes == sorted(set(issue_codes))
    assert all(isinstance(code, str) and code for code in issue_codes)

    grobid_quality = _exact_object(
        quality["grobid"],
        {
            "annotation_count",
            "normalized_text_characters",
            "normalized_text_sha256",
        },
        label="GROBID quality",
    )
    _positive_int(grobid_quality["annotation_count"], label="annotation_count")
    _positive_int(
        grobid_quality["normalized_text_characters"],
        label="normalized_text_characters",
    )
    assert _HEX_SHA256.fullmatch(grobid_quality["normalized_text_sha256"])

    replay = _exact_object(
        manifest["replay"],
        {
            "algorithm",
            "aligned_count",
            "artifact_id",
            "canonical_source_order",
            "content_span_count",
            "expected_diagnostics",
            "expected_scholarly_anchor_count",
            "minimum_alignment_score",
            "producer_run_ids",
            "unaligned_count",
        },
        label="replay",
    )
    assert replay["algorithm"] == "token-sequence-v2"
    assert replay["artifact_id"] == "service-capture-v1"
    assert replay["minimum_alignment_score"] == 0.72
    assert replay["producer_run_ids"] == {
        key: f"service-capture-v1-{key}"
        for key in ("alignment", "docling", "grobid", "integrity")
    }
    assert replay["canonical_source_order"] == [
        "docling_document",
        "content_spans",
        "grobid_tei",
        "alignment_overlay",
        "content_integrity_overlay",
    ]
    _nonnegative_int(replay["aligned_count"], label="aligned_count")
    _nonnegative_int(replay["unaligned_count"], label="unaligned_count")
    _positive_int(replay["content_span_count"], label="content_span_count")
    _nonnegative_int(
        replay["expected_scholarly_anchor_count"],
        label="expected_scholarly_anchor_count",
    )
    assert replay["expected_diagnostics"] == {"UNRESOLVED_SCHOLARLY_ANCHOR": 3}

    file_records = _exact_object(
        manifest["files"],
        set(_FILE_CONTRACTS),
        label="fixture files",
    )
    assert {path.name for path in root.iterdir()} == {
        "manifest.json",
        *_FILE_CONTRACTS,
    }
    for path in root.iterdir():
        assert not path.is_symlink()
        assert path.is_file()
    for relative_path, (role, media_type) in _FILE_CONTRACTS.items():
        assert PurePosixPath(relative_path).parts == (relative_path,)
        assert "\\" not in relative_path
        record = _exact_object(
            file_records[relative_path],
            {"byte_size", "media_type", "role", "sha256"},
            label=f"file record {relative_path}",
        )
        assert record["role"] == role
        assert record["media_type"] == media_type
        byte_size = _positive_int(
            record["byte_size"],
            label=f"{relative_path} byte_size",
        )
        assert _HEX_SHA256.fullmatch(record["sha256"])
        payload = (root / relative_path).read_bytes()
        assert len(payload) == byte_size
        assert sha256_bytes(payload) == record["sha256"]
    return manifest


def _load_manifest(root: Path = FIXTURE_ROOT) -> dict[str, Any]:
    return _validate_manifest(
        root,
        json.loads((root / "manifest.json").read_bytes()),
    )


def _damage_manifest(manifest: dict[str, Any], case: str) -> None:
    if case == "unexpected_top_level_key":
        manifest["processing_runs"] = []
    elif case == "wrong_file_hash":
        manifest["files"]["docling_document.json"]["sha256"] = "0" * 64
    elif case == "wrong_file_size":
        manifest["files"]["grobid.tei.xml"]["byte_size"] = 1
    elif case == "path_traversal":
        record = manifest["files"].pop("docling_document.json")
        manifest["files"]["../docling_document.json"] = record
    elif case == "unselected_grobid_stage":
        manifest["selection"]["grobid_tei"]["stage_id"] = "primary-grobid"
    elif case == "mutable_docling_image":
        manifest["services"]["docling"]["container_reference"] = DOCLING_CONTAINER_IMAGE
    elif case == "substituted_docling_digest":
        manifest["services"]["docling"]["container_reference"] = (
            f"{DOCLING_CONTAINER_IMAGE}@sha256:{'0' * 64}"
        )
    elif case == "invalid_grobid_build_blob":
        manifest["services"]["grobid"]["dockerfile_blob"] = "0" * 39
    elif case == "ephemeral_invocation_identity":
        manifest["services"]["docling"]["runtime_attestation"]["invocation_id"] = (
            "00000000-0000-0000-0000-000000000000"
        )
    elif case == "unstable_replay_artifact":
        manifest["replay"]["artifact_id"] = "artifact-" + "0" * 64
    elif case == "selected_failed_primary_grobid":
        manifest["accepted_outcomes"]["stages"]["primary-grobid"][
            "selected_native_output"
        ] = True
    elif case == "wrong_raw_manifest_hash":
        manifest["capture"]["raw_manifest_sha256"] = "0" * 64
    elif case == "wrong_workflow_job":
        manifest["capture"]["workflow_job"] = "quality"
    elif case == "missing_alignment_stage":
        del manifest["accepted_outcomes"]["stages"]["alignment"]
    elif case == "wrong_stage_reason_code":
        manifest["accepted_outcomes"]["stages"]["docling"]["reason_codes"] = [
            "missing_model_inventory"
        ]
    else:  # pragma: no cover - parametrization owns this closed set
        raise AssertionError(f"unknown manifest damage case: {case}")


@pytest.mark.parametrize(
    "case",
    [
        "unexpected_top_level_key",
        "wrong_file_hash",
        "wrong_file_size",
        "path_traversal",
        "unselected_grobid_stage",
        "mutable_docling_image",
        "substituted_docling_digest",
        "invalid_grobid_build_blob",
        "ephemeral_invocation_identity",
        "unstable_replay_artifact",
        "selected_failed_primary_grobid",
        "wrong_raw_manifest_hash",
        "wrong_workflow_job",
        "missing_alignment_stage",
        "wrong_stage_reason_code",
    ],
)
def test_service_capture_manifest_rejects_tampering(case: str) -> None:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_bytes())
    _damage_manifest(manifest, case)

    with pytest.raises(AssertionError):
        _validate_manifest(FIXTURE_ROOT, manifest)


def test_service_capture_manifest_is_strict_and_hash_complete() -> None:
    manifest = _load_manifest()

    assert manifest["capture"] == _EXPECTED_CAPTURE


def test_service_capture_manifest_rejects_symlinked_payload(tmp_path: Path) -> None:
    shadow_root = tmp_path / "service-capture"
    shutil.copytree(FIXTURE_ROOT, shadow_root)
    shadow_docling = shadow_root / "docling_document.json"
    shadow_docling.unlink()
    shadow_docling.symlink_to(FIXTURE_ROOT / "docling_document.json")

    with pytest.raises(AssertionError):
        _load_manifest(shadow_root)


def test_service_capture_source_matches_generator_and_is_raster_only() -> None:
    _load_manifest()
    source = (FIXTURE_ROOT / "source-rich-raster.pdf").read_bytes()

    assert source == rich_raster_pdf_bytes()
    reader = PdfReader(BytesIO(source))
    assert len(reader.pages) == 1
    page = reader.pages[0]
    assert not (page.extract_text() or "").strip()
    resources = page["/Resources"].get_object()
    assert "/Font" not in resources
    xobjects = resources["/XObject"].get_object()
    assert len(xobjects) == 1
    image = next(iter(xobjects.values())).get_object()
    assert image["/Subtype"] == "/Image"
    assert int(image["/Width"]) == 3200
    assert int(image["/Height"]) == 2100


def test_service_capture_docling_is_canonical_and_reference_complete() -> None:
    manifest = _load_manifest()
    payload = (FIXTURE_ROOT / "docling_document.json").read_bytes()
    document = json.loads(payload)
    assert isinstance(document, dict)

    assert payload == _canonical_json(document)
    report = DoclingQualityValidator(0.95).validate(
        document,
        require_pdf_geometry=True,
    )
    assert report.acceptable
    assert report.text_item_count > 0
    assert report.located_text_item_count > 0
    assert report.locator_coverage >= 0.95
    assert report.reference_count > 0
    assert report.broken_reference_count == 0
    assert (
        docling_document_sha256(document)
        == (manifest["files"]["docling_document.json"]["sha256"])
    )
    assert {
        "broken_reference_count": report.broken_reference_count,
        "issue_codes": sorted({issue.code for issue in report.issues}),
        "located_text_item_count": report.located_text_item_count,
        "locator_coverage": report.locator_coverage,
        "reference_count": report.reference_count,
        "text_item_count": report.text_item_count,
    } == manifest["quality"]["docling"]


def test_service_capture_grobid_is_namespaced_and_textual() -> None:
    manifest = _load_manifest()
    tei = (FIXTURE_ROOT / "grobid.tei.xml").read_bytes()
    root = ElementTree.fromstring(tei)
    namespace = f"{{{_TEI_NAMESPACE}}}"

    assert root.tag == f"{namespace}TEI"
    assert root.find(f"{namespace}teiHeader") is not None
    text = root.find(f"{namespace}text")
    assert text is not None
    assert text.find(f".//{namespace}body") is not None
    normalized_text = " ".join("".join(text.itertext()).split())
    assert len(normalized_text) >= 100
    annotations = DoclingGrobidAligner().extract_annotations(tei)
    assert annotations
    assert all(annotation.text.strip() for annotation in annotations)
    assert all(annotation.coordinate_error is None for annotation in annotations)
    assert {
        "annotation_count": len(annotations),
        "normalized_text_characters": len(normalized_text),
        "normalized_text_sha256": sha256_bytes(normalized_text.encode("utf-8")),
    } == manifest["quality"]["grobid"]


@dataclass(frozen=True, slots=True)
class _ReplayResult:
    view: CanonicalDocumentView
    encoded: bytes
    product_ids: dict[str, str]
    content_span_count: int
    aligned_count: int
    unaligned_count: int


def _put_product(
    store: ContentAddressedStore,
    *,
    name: str,
    payload: bytes,
    producer_run_id: str,
    artifact_id: str,
) -> DataProductRef:
    blob = store.put_blob(payload)
    return store.data_product_ref(
        name=name,
        blob_sha256=blob.sha256,
        producer_run_id=producer_run_id,
        source_artifact_ids=(artifact_id,),
    )


def _replay_service_capture(
    store: ContentAddressedStore,
    manifest: dict[str, Any],
) -> _ReplayResult:
    replay = manifest["replay"]
    artifact_id = replay["artifact_id"]
    source = (FIXTURE_ROOT / "source-rich-raster.pdf").read_bytes()
    source_blob = store.put_blob(source)
    artifact = DocumentArtifact(
        artifact_id=artifact_id,
        source_sha256=source_blob.sha256,
        acquisition_uri=(
            "fixture://document-processing/canonical/service_capture/v1/"
            "source-rich-raster.pdf"
        ),
        identifiers={"filename": "source-rich-raster.pdf"},
        media_type="application/pdf",
        raw_location=source_blob.as_location(
            media_type="application/pdf",
            role=ArtifactLocationRole.RAW,
        ),
    )
    store.save_artifact(artifact)

    producer_ids = replay["producer_run_ids"]
    docling_bytes = (FIXTURE_ROOT / "docling_document.json").read_bytes()
    document = json.loads(docling_bytes)
    assert isinstance(document, dict)
    docling_product = _put_product(
        store,
        name="docling_document",
        payload=docling_bytes,
        producer_run_id=producer_ids["docling"],
        artifact_id=artifact_id,
    )
    content_spans = ContentSpanSet(
        artifact_id=artifact_id,
        processing_run_id=producer_ids["docling"],
        representation_product_id=docling_product.product_id,
        spans=build_pdf_content_spans(
            document,
            artifact_id=artifact_id,
            processing_run_id=producer_ids["docling"],
            representation_product_id=docling_product.product_id,
        ),
    )
    content_spans_product = _put_product(
        store,
        name="content_spans",
        payload=_canonical_json(content_spans.model_dump(mode="json")),
        producer_run_id=producer_ids["docling"],
        artifact_id=artifact_id,
    )

    tei = (FIXTURE_ROOT / "grobid.tei.xml").read_bytes()
    grobid_product = _put_product(
        store,
        name="grobid_tei",
        payload=tei,
        producer_run_id=producer_ids["grobid"],
        artifact_id=artifact_id,
    )
    scholarly_overlay = DoclingGrobidAligner(replay["minimum_alignment_score"]).align(
        document, tei
    )
    alignment_product = _put_product(
        store,
        name="alignment_overlay",
        payload=_canonical_json(scholarly_overlay.to_dict()),
        producer_run_id=producer_ids["alignment"],
        artifact_id=artifact_id,
    )
    integrity_report = validate_content_integrity(
        document,
        scholarly_overlay=scholarly_overlay,
    ).to_dict()
    integrity_product = _put_product(
        store,
        name="content_integrity_overlay",
        payload=_canonical_json(integrity_report),
        producer_run_id=producer_ids["integrity"],
        artifact_id=artifact_id,
    )
    source_products = (
        docling_product,
        content_spans_product,
        grobid_product,
        alignment_product,
        integrity_product,
    )
    view = build_canonical_document_view(
        artifact=artifact,
        docling_document=document,
        docling_product=docling_product,
        content_span_set=content_spans,
        source_products=source_products,
        configuration=CanonicalizationConfig(),
        scholarly_overlay=scholarly_overlay,
        integrity_report=integrity_report,
    )
    encoded = canonical_document_bytes(view)
    canonical_blob = store.put_canonical_document(view)
    stored_bytes = store.read_blob(canonical_blob.sha256)
    assert stored_bytes == encoded
    assert load_canonical_document(stored_bytes) == view
    return _ReplayResult(
        view=view,
        encoded=encoded,
        product_ids={product.name: product.product_id for product in source_products},
        content_span_count=len(content_spans.spans),
        aligned_count=scholarly_overlay.aligned_count,
        unaligned_count=scholarly_overlay.unaligned_count,
    )


def test_service_capture_replays_canonical_deterministically_offline(
    tmp_path: Path,
) -> None:
    manifest = _load_manifest()
    first = _replay_service_capture(
        ContentAddressedStore(tmp_path / "first-store"),
        manifest,
    )
    second = _replay_service_capture(
        ContentAddressedStore(tmp_path / "second-store"),
        manifest,
    )

    assert first == second
    assert first.view.artifact_id == manifest["replay"]["artifact_id"]
    assert first.view.blocks
    assert [product.name for product in first.view.source_products] == (
        manifest["replay"]["canonical_source_order"]
    )
    assert first.content_span_count == manifest["replay"]["content_span_count"]
    assert first.aligned_count == manifest["replay"]["aligned_count"] == 0
    assert first.unaligned_count == manifest["replay"]["unaligned_count"] == 3
    assert all(
        re.fullmatch(r"product-[0-9a-f]{64}", product_id)
        for product_id in first.product_ids.values()
    )

    scholarly_anchors = [
        anchor
        for block in first.view.blocks
        for anchor in block.source_anchors
        if anchor.role is CanonicalAnchorRole.SCHOLARLY
    ] + [
        relationship.source_anchor
        for relationship in first.view.relationships
        if relationship.source_anchor is not None
        and relationship.source_anchor.role is CanonicalAnchorRole.SCHOLARLY
    ]
    assert (
        len(scholarly_anchors)
        == (manifest["replay"]["expected_scholarly_anchor_count"])
    )
    assert (
        Counter(diagnostic.code for diagnostic in first.view.diagnostics)
        == (manifest["replay"]["expected_diagnostics"])
    )
    assert all(
        diagnostic.severity is CanonicalDiagnosticSeverity.WARNING
        and diagnostic.product_id == first.product_ids["grobid_tei"]
        for diagnostic in first.view.diagnostics
    )
