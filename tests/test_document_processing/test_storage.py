"""Tests for immutable content-addressed document-processing storage."""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from DeepResearch.src.document_processing import storage as storage_module
from DeepResearch.src.document_processing.alignment import DoclingGrobidAligner
from DeepResearch.src.document_processing.canonical import (
    CANONICAL_ANCHORING_POLICY,
    CANONICAL_COMPONENT_CAPABILITY,
    CANONICAL_COMPONENT_ID,
    CANONICAL_COMPONENT_VERSION,
    CANONICAL_TEXT_NORMALIZATION,
    CanonicalAnchorRole,
    CanonicalBlock,
    CanonicalBlockKind,
    CanonicalDocumentMetadata,
    CanonicalDocumentView,
    CanonicalizationConfig,
    CanonicalSourceAnchor,
    build_canonical_document_view,
    canonical_block_content_sha256,
    canonical_block_id,
    canonical_document_bytes,
    canonical_invocation_configuration,
    canonical_view_id,
)
from DeepResearch.src.document_processing.models import (
    ArtifactLocationRole,
    ArtifactRelationship,
    ComponentDescriptor,
    ContentSpan,
    ContentSpanSet,
    DataProductRef,
    DiagnosticSeverity,
    DocumentArtifact,
    ExecutionCheckpoint,
    PdfBoundingBox,
    PdfLocator,
    ProcessingDiagnostic,
    ProcessingRun,
    ProcessingRunStatus,
    RepresentationAnchor,
    RuntimeAttestation,
    RuntimeAttestationSource,
    configuration_sha256,
    sha256_bytes,
)
from DeepResearch.src.document_processing.products import build_data_product_ref
from DeepResearch.src.document_processing.storage import (
    BlobNotFoundError,
    BlobTooLargeError,
    ContentAddressedStore,
    CorruptRecordError,
    HashMismatchError,
    RecordConflictError,
    RecordNotFoundError,
    StorageError,
    UnsupportedSchemaVersionError,
)
from DeepResearch.src.document_processing.validation import (
    DoclingQualityValidator,
    build_pdf_content_spans,
    validate_content_integrity,
)


@pytest.fixture
def store(tmp_path: Path) -> ContentAddressedStore:
    return ContentAddressedStore(tmp_path / "document-store")


def save_artifact(
    store: ContentAddressedStore,
    artifact_id: str = "artifact-1",
    *,
    relationship: ArtifactRelationship = ArtifactRelationship.SOURCE,
    parent_artifact_id: str | None = None,
) -> DocumentArtifact:
    blob = store.put_blob(b"%PDF-1.7\nsource document bytes")
    artifact = DocumentArtifact(
        artifact_id=artifact_id,
        source_sha256=blob.sha256,
        acquisition_uri=f"https://example.test/{artifact_id}.pdf",
        identifiers={"pmc": f"PMC-{artifact_id}"},
        media_type="application/pdf",
        relationship=relationship,
        parent_artifact_id=parent_artifact_id,
        raw_location=blob.as_location(
            media_type="application/pdf", role=ArtifactLocationRole.RAW
        ),
    )
    store.save_artifact(artifact)
    return artifact


def make_run(
    store: ContentAddressedStore,
    artifact_id: str,
    run_id: str,
    status: ProcessingRunStatus,
    *,
    minute: int = 0,
    inputs: tuple[DataProductRef, ...] = (),
    outputs: dict[str, str] | None = None,
    output_policy_snapshot: dict[str, object] | None = None,
) -> ProcessingRun:
    config = {"do_ocr": True}
    started_at = datetime(2026, 7, 17, 12, minute, tzinfo=UTC)
    source_artifact_ids = tuple(
        dict.fromkeys(
            (
                artifact_id,
                *(
                    source_artifact_id
                    for product in inputs
                    for source_artifact_id in product.source_artifact_ids
                ),
            )
        )
    )
    product_refs = tuple(
        build_data_product_ref(
            name=name,
            blob_sha256=digest,
            uri=store.blob_uri(digest),
            byte_size=(
                store.blob_path(digest).stat().st_size
                if store.blob_path(digest).is_file()
                else 0
            ),
            producer_run_id=run_id,
            source_artifact_ids=source_artifact_ids,
        )
        for name, digest in (outputs or {}).items()
    )
    return ProcessingRun(
        run_id=run_id,
        artifact_id=artifact_id,
        stage_id="docling",
        component=ComponentDescriptor(
            component_id="docling",
            component_version="2.96.1",
            capability="document-conversion",
        ),
        configuration=config,
        configuration_sha256=configuration_sha256(config),
        output_policy_snapshot=output_policy_snapshot or {},
        output_policy_sha256=(
            configuration_sha256(output_policy_snapshot)
            if output_policy_snapshot
            else None
        ),
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=2),
        status=status,
        inputs=inputs,
        outputs=product_refs,
    )


def make_component_run(
    store: ContentAddressedStore,
    artifact_id: str,
    run_id: str,
    *,
    component: ComponentDescriptor,
    configuration: dict[str, object],
    inputs: tuple[DataProductRef, ...] = (),
    outputs: dict[str, str] | None = None,
    status: ProcessingRunStatus = ProcessingRunStatus.COMPLETE,
    runtime_attestation: RuntimeAttestation | None = None,
    runtime_identity_required: bool = False,
    output_policy_snapshot: dict[str, object] | None = None,
) -> ProcessingRun:
    source_artifact_ids = tuple(
        dict.fromkeys(
            (
                artifact_id,
                *(
                    source_artifact_id
                    for product in inputs
                    for source_artifact_id in product.source_artifact_ids
                ),
            )
        )
    )
    started_at = datetime(2026, 7, 17, 12, tzinfo=UTC)
    output_digests = dict(outputs or {})
    if runtime_attestation is not None:
        attestation_blob = store.put_blob(
            json.dumps(
                runtime_attestation.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        output_digests["runtime_attestation"] = attestation_blob.sha256
    return ProcessingRun(
        run_id=run_id,
        artifact_id=artifact_id,
        stage_id=component.component_id,
        component=component,
        component_invocation_id=(
            runtime_attestation.invocation_id
            if runtime_attestation is not None
            else None
        ),
        runtime_identity_required=runtime_identity_required,
        component_versions=(
            runtime_attestation.component_versions
            if runtime_attestation is not None
            else {}
        ),
        model_versions=(
            runtime_attestation.model_versions
            if runtime_attestation is not None
            else {}
        ),
        model_hashes=(
            runtime_attestation.model_hashes if runtime_attestation is not None else {}
        ),
        container_image=(
            runtime_attestation.container_reference
            if runtime_attestation is not None
            else None
        ),
        container_digest=(
            runtime_attestation.container_digest
            if runtime_attestation is not None
            else None
        ),
        runtime_attestation=runtime_attestation,
        runtime_attestation_sha256=(
            output_digests.get("runtime_attestation")
            if runtime_attestation is not None
            else None
        ),
        configuration=configuration,
        configuration_sha256=configuration_sha256(configuration),
        output_policy_snapshot=output_policy_snapshot or {},
        output_policy_sha256=(
            configuration_sha256(output_policy_snapshot)
            if output_policy_snapshot
            else None
        ),
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=1),
        status=status,
        inputs=inputs,
        outputs=tuple(
            store.data_product_ref(
                name=name,
                blob_sha256=digest,
                producer_run_id=run_id,
                source_artifact_ids=source_artifact_ids,
            )
            for name, digest in output_digests.items()
        ),
    )


def docling_production_configuration(
    artifact: DocumentArtifact,
    *,
    input_format: str = "pdf",
    input_sha256: str | None = None,
) -> dict[str, object]:
    """Return the exact persisted Docling invocation shape used in production."""

    configuration: dict[str, object] = {
        "serve_version": "1.21.0",
        "expected_docling_version": "2.96.1",
        "container_image": "quay.io/docling-project/docling-serve-cpu:v1.21.0",
        "container_digest": None,
        "model_versions": {},
        "model_hashes": {},
        "input_sha256": input_sha256 or artifact.source_sha256,
        "input_format": input_format,
        "options": {
            "to_formats": ["json"],
            "image_export_mode": "embedded",
            "do_ocr": True,
            "table_mode": "accurate",
        },
        "minimum_pdf_locator_coverage": 0.95,
        "quality_validator_version": "docling-quality-v2",
        "content_span_schema_version": "1",
    }
    if input_format == "pdf":
        configuration["pdf_span_algorithm"] = "provenance-charspan-v2"
    return configuration


def grobid_production_configuration(
    artifact: DocumentArtifact,
    *,
    minimum_text_characters: int = 1,
) -> dict[str, object]:
    """Return the exact persisted GROBID invocation shape used in production."""

    return {
        "input_sha256": artifact.source_sha256,
        "expected_grobid_version": "0.9.0",
        "container_image": "deepcritical/grobid:0.9.0-full-p0-c2",
        "container_digest": None,
        "model_versions": {},
        "model_hashes": {},
        "coordinates": ["persName", "ref", "biblStruct", "formula", "figure"],
        "consolidate_header": 0,
        "consolidate_citations": 0,
        "segment_sentences": True,
        "minimum_text_characters": minimum_text_characters,
    }


def ocr_production_configuration(
    artifact: DocumentArtifact,
    *,
    mode: str = "container_cli",
) -> dict[str, object]:
    return {
        "input_sha256": artifact.source_sha256,
        "fallback_reason": "scan_detection",
        "expected_ocrmypdf_version": "17.4.1",
        "mode": mode,
        "container_image": (
            "jbarlow83/ocrmypdf:v17.4.1" if mode == "container_cli" else None
        ),
        "container_digest": None,
        "languages": ["eng"],
        "rotate_pages": True,
        "deskew": True,
        "jobs": 1,
        "optimize": 1,
        "skip_text": True,
        "output_type": "pdf",
    }


def docling_runtime_attestation(
    run_id: str,
    *,
    container_digest: str = f"sha256:{'a' * 64}",
    container_image: str = "quay.io/docling-project/docling-serve-cpu:v1.21.0",
    reporter_id: str = "test-deployment-reporter",
    source: RuntimeAttestationSource = (
        RuntimeAttestationSource.AUTHENTICATED_DEPLOYMENT_REPORTER
    ),
    component_versions: dict[str, str] | None = None,
) -> RuntimeAttestation:
    return RuntimeAttestation(
        component_id="docling",
        component_version="2.96.1",
        invocation_id=f"{run_id}-invocation",
        source=source,
        reporter_id=reporter_id,
        observed_at=datetime(2026, 7, 17, 12, 0, 0, 500_000, tzinfo=UTC),
        workload_id=f"workload-{run_id}",
        container_reference=f"{container_image}@{container_digest}",
        container_digest=container_digest,
        component_versions=(
            component_versions
            if component_versions is not None
            else {"docling": "2.96.1", "docling_serve": "1.21.0"}
        ),
        model_versions={"layout": "fixture-v1"},
        model_hashes={"layout": "b" * 64},
    )


def grobid_runtime_attestation(
    run_id: str,
    *,
    container_digest: str = f"sha256:{'d' * 64}",
    reporter_id: str = "test-deployment-reporter",
    source: RuntimeAttestationSource = (
        RuntimeAttestationSource.AUTHENTICATED_DEPLOYMENT_REPORTER
    ),
) -> RuntimeAttestation:
    return RuntimeAttestation(
        component_id="grobid",
        component_version="0.9.0",
        invocation_id=f"{run_id}-invocation",
        source=source,
        reporter_id=reporter_id,
        observed_at=datetime(2026, 7, 17, 12, 0, 0, 500_000, tzinfo=UTC),
        workload_id=f"workload-{run_id}",
        container_reference=(
            f"deepcritical/grobid:0.9.0-full-p0-c2@{container_digest}"
        ),
        container_digest=container_digest,
        component_versions={"grobid": "0.9.0"},
        model_versions={"citation": "fixture-v1"},
        model_hashes={"citation": "e" * 64},
    )


def ocr_runtime_attestation(
    run_id: str,
    *,
    component_versions: dict[str, str] | None = None,
    container_digest: str = f"sha256:{'c' * 64}",
    reporter_id: str = "deepcritical-container-ocr-runner-v1",
    source: RuntimeAttestationSource = (
        RuntimeAttestationSource.DIGEST_ADDRESSED_OCI_INVOCATION
    ),
) -> RuntimeAttestation:
    return RuntimeAttestation(
        component_id="ocrmypdf",
        component_version="17.4.1",
        invocation_id=f"{run_id}-invocation",
        source=source,
        reporter_id=reporter_id,
        observed_at=datetime(2026, 7, 17, 12, 0, 0, 500_000, tzinfo=UTC),
        workload_id=f"workload-{run_id}",
        container_reference=(f"jbarlow83/ocrmypdf:v17.4.1@{container_digest}"),
        container_digest=container_digest,
        component_versions=(
            component_versions
            if component_versions is not None
            else {"ocrmypdf": "17.4.1", "tesseract": "5.3.4"}
        ),
    )


def runtime_output_policy(
    *,
    component_id: str,
    configuration: dict[str, object],
    attestation: RuntimeAttestation | None,
    runtime_identity_required: bool,
) -> dict[str, object]:
    document_config: dict[str, object] = {
        "require_runtime_identity": runtime_identity_required,
    }
    if component_id == "docling":
        document_config.update(
            {
                "docling_version": "2.96.1",
                "docling_serve_version": "1.21.0",
                "docling_container_image": configuration["container_image"],
                "docling_container_digest": configuration["container_digest"],
                "docling_model_versions": configuration["model_versions"],
                "docling_model_hashes": configuration["model_hashes"],
            }
        )
        policy = {
            "reporter_configured": attestation is not None,
            "expected_reporter_id": (
                attestation.reporter_id if attestation is not None else None
            ),
            "expected_source": "authenticated_deployment_reporter",
            "attestation_contract_version": (
                "deepcritical-authenticated-runtime-attestation-reporter-v1"
            ),
            "attestation_schema_version": "deepcritical-runtime-attestation-v1",
        }
    elif component_id == "grobid":
        document_config.update(
            {
                "grobid_version": "0.9.0",
                "grobid_container_image": configuration["container_image"],
                "grobid_container_digest": configuration["container_digest"],
                "grobid_model_versions": configuration["model_versions"],
                "grobid_model_hashes": configuration["model_hashes"],
            }
        )
        policy = {
            "reporter_configured": attestation is not None,
            "expected_reporter_id": (
                attestation.reporter_id if attestation is not None else None
            ),
            "expected_source": "authenticated_deployment_reporter",
            "attestation_contract_version": (
                "deepcritical-authenticated-runtime-attestation-reporter-v1"
            ),
            "attestation_schema_version": "deepcritical-runtime-attestation-v1",
        }
    else:
        document_config.update(
            {
                "ocrmypdf_version": "17.4.1",
                "ocr_mode": configuration["mode"],
                "ocr_container_image": configuration["container_image"],
                "ocr_container_digest": configuration["container_digest"],
            }
        )
        policy = {
            "reporter_configured": configuration["mode"] == "container_cli",
            "expected_reporter_id": "deepcritical-container-ocr-runner-v1",
            "expected_source": "digest_addressed_oci_invocation",
            "attestation_contract_version": ("deepcritical-container-ocr-runner-v1"),
            "attestation_schema_version": "deepcritical-runtime-attestation-v1",
            "local_digest_runner_version": "deepcritical-container-ocr-runner-v1",
        }
    return {
        "schema": "deepcritical-document-output-policy-v2",
        "document_processing_config": document_config,
        "runtime_trust_policy": {component_id: policy},
        "routing": {
            "algorithm_version": "deterministic-document-router-v1",
            "reject_extension_only_detection": True,
        },
    }


def add_uncaptioned_valid_table(document: dict[str, Any]) -> None:
    """Add a structurally valid table that yields only an integrity issue."""

    document["tables"] = [
        {
            "self_ref": "#/tables/0",
            "label": "table",
            "data": {
                "num_rows": 1,
                "num_cols": 1,
                "table_cells": [
                    {
                        "text": "APOE4",
                        "start_row_offset_idx": 0,
                        "end_row_offset_idx": 1,
                        "start_col_offset_idx": 0,
                        "end_col_offset_idx": 1,
                    }
                ],
            },
        }
    ]


def make_canonical_view(
    *,
    source_artifact_id: str = "canonical-source",
    source_sha256: str = "a" * 64,
    source_products: tuple[DataProductRef, DataProductRef] | None = None,
) -> CanonicalDocumentView:
    """Build the smallest valid canonical view for persistence tests."""

    if source_products is None:
        docling_product = build_data_product_ref(
            name="docling_document",
            blob_sha256="b" * 64,
            uri=f"cas://sha256/{'b' * 64}",
            byte_size=10,
            producer_run_id="docling-canonical",
            source_artifact_ids=(source_artifact_id,),
        )
        spans_product = build_data_product_ref(
            name="content_spans",
            blob_sha256="c" * 64,
            uri=f"cas://sha256/{'c' * 64}",
            byte_size=10,
            producer_run_id="docling-canonical",
            source_artifact_ids=(source_artifact_id,),
        )
    else:
        docling_product, spans_product = source_products
    content_sha256 = canonical_block_content_sha256(
        kind=CanonicalBlockKind.PARAGRAPH,
        text="Canonical content",
        table=None,
    )
    block = CanonicalBlock(
        block_id=canonical_block_id(
            native_node_id="#/texts/0",
            kind=CanonicalBlockKind.PARAGRAPH,
            content_sha256=content_sha256,
        ),
        native_node_id="#/texts/0",
        native_label="paragraph",
        kind=CanonicalBlockKind.PARAGRAPH,
        ordinal=0,
        text="Canonical content",
        content_sha256=content_sha256,
        source_anchors=(
            CanonicalSourceAnchor(
                role=CanonicalAnchorRole.PRIMARY,
                product_id=docling_product.product_id,
                node_id="#/texts/0",
            ),
        ),
    )
    payload = {
        "schema_version": "deepcritical-canonical-document-view-v1",
        "view_id": "pending",
        "artifact_id": source_artifact_id,
        "source_sha256": source_sha256,
        "normalization_policy": CANONICAL_TEXT_NORMALIZATION,
        "anchoring_policy": CANONICAL_ANCHORING_POLICY,
        "metadata": CanonicalDocumentMetadata(
            title="Canonical persistence",
            media_type="application/pdf",
            identifiers={"pmc": "PMC-CANONICAL"},
        ).model_dump(mode="json"),
        "source_products": [
            docling_product.model_dump(mode="json"),
            spans_product.model_dump(mode="json"),
        ],
        "root_block_ids": [block.block_id],
        "blocks": [block.model_dump(mode="json")],
        "relationships": [],
        "diagnostics": [],
    }
    payload["view_id"] = canonical_view_id(payload)
    return CanonicalDocumentView.model_validate(payload)


def make_durable_canonical_view(
    store: ContentAddressedStore,
    artifact: DocumentArtifact,
    *,
    document_mutator: Callable[[dict[str, Any]], None] | None = None,
    docling_bytes_mutator: Callable[[bytes], bytes] | None = None,
    source_component: ComponentDescriptor | None = None,
    source_configuration: dict[str, object] | None = None,
    source_status: ProcessingRunStatus = ProcessingRunStatus.COMPLETE,
    source_attestation: RuntimeAttestation | None = None,
    source_runtime_identity_required: bool = False,
    source_output_policy_snapshot: dict[str, object] | None = None,
) -> tuple[CanonicalDocumentView, ProcessingRun]:
    """Persist both source products used by one canonical test view."""

    document = {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "Canonical persistence",
        "body": {
            "self_ref": "#/body",
            "children": [{"$ref": "#/texts/0"}],
        },
        "furniture": {"self_ref": "#/furniture", "children": []},
        "groups": [],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "parent": {"$ref": "#/body"},
                "label": "paragraph",
                "text": "Canonical content",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {
                            "l": 0,
                            "t": 20,
                            "r": 100,
                            "b": 0,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    }
                ],
            }
        ],
        "tables": [],
        "pictures": [],
        "key_value_items": [],
        "pages": {
            "1": {
                "page_no": 1,
                "size": {"width": 100, "height": 100},
            }
        },
    }
    if document_mutator is not None:
        document_mutator(document)
    docling_bytes = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    if docling_bytes_mutator is not None:
        docling_bytes = docling_bytes_mutator(docling_bytes)
    docling_blob = store.put_blob(docling_bytes)
    source_run_id = f"{artifact.artifact_id}-canonical-source-run"
    docling_product = store.data_product_ref(
        name="docling_document",
        blob_sha256=docling_blob.sha256,
        producer_run_id=source_run_id,
        source_artifact_ids=(artifact.artifact_id,),
    )
    spans = build_pdf_content_spans(
        document,
        artifact_id=artifact.artifact_id,
        processing_run_id=source_run_id,
        representation_product_id=docling_product.product_id,
    )
    span_set = ContentSpanSet(
        artifact_id=artifact.artifact_id,
        processing_run_id=source_run_id,
        representation_product_id=docling_product.product_id,
        spans=spans,
    )
    spans_blob = store.put_blob(
        json.dumps(
            span_set.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    source_run = make_component_run(
        store,
        artifact.artifact_id,
        source_run_id,
        component=source_component
        or ComponentDescriptor(
            component_id="docling",
            component_version="2.96.1",
            capability="document.parse",
        ),
        configuration=(
            docling_production_configuration(artifact)
            if source_configuration is None
            else source_configuration
        ),
        outputs={
            "docling_document": docling_blob.sha256,
            "content_spans": spans_blob.sha256,
        },
        status=source_status,
        runtime_attestation=source_attestation,
        runtime_identity_required=source_runtime_identity_required,
        output_policy_snapshot=source_output_policy_snapshot,
    )
    store.save_processing_run(source_run)
    source_products = (
        source_run.require_output("docling_document"),
        source_run.require_output("content_spans"),
    )
    return (
        build_canonical_document_view(
            artifact=artifact,
            docling_document=document,
            docling_product=source_products[0],
            content_span_set=span_set,
            source_products=source_products,
            configuration=CanonicalizationConfig(),
        ),
        source_run,
    )


def make_colliding_docling_view(
    store: ContentAddressedStore,
    artifact: DocumentArtifact,
    *,
    case_id: str,
) -> tuple[CanonicalDocumentView, ProcessingRun]:
    """Persist the two-node self-ref collision that previously bypassed replay."""

    _, clean_run = make_durable_canonical_view(store, artifact)
    document = json.loads(
        store.read_blob(clean_run.require_output("docling_document").blob_sha256)
    )
    document["texts"].append(
        {
            "self_ref": "#/texts/0",
            "parent": {"$ref": "#/body"},
            "label": "paragraph",
            "text": "Colliding native content",
            "prov": [
                {
                    "page_no": 1,
                    "bbox": {
                        "l": 0,
                        "t": 40,
                        "r": 100,
                        "b": 20,
                        "coord_origin": "BOTTOMLEFT",
                    },
                }
            ],
        }
    )
    docling_blob = store.put_blob(
        json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    run_id = f"{artifact.artifact_id}-{case_id}-docling-run"
    docling_product = store.data_product_ref(
        name="docling_document",
        blob_sha256=docling_blob.sha256,
        producer_run_id=run_id,
        source_artifact_ids=(artifact.artifact_id,),
    )
    span_set = ContentSpanSet(
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        representation_product_id=docling_product.product_id,
        spans=build_pdf_content_spans(
            document,
            artifact_id=artifact.artifact_id,
            processing_run_id=run_id,
            representation_product_id=docling_product.product_id,
        ),
    )
    spans_blob = store.put_blob(
        json.dumps(
            span_set.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    source_run = make_component_run(
        store,
        artifact.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="docling",
            component_version="2.96.1",
            capability="document.parse",
        ),
        configuration=docling_production_configuration(artifact),
        outputs={
            "docling_document": docling_blob.sha256,
            "content_spans": spans_blob.sha256,
        },
        status=ProcessingRunStatus.PARTIAL,
    )
    store.save_processing_run(source_run)
    products = (
        source_run.require_output("docling_document"),
        source_run.require_output("content_spans"),
    )
    return (
        make_canonical_view(
            source_artifact_id=artifact.artifact_id,
            source_sha256=artifact.source_sha256,
            source_products=products,
        ),
        source_run,
    )


def make_irreproducible_canonical_view(
    store: ContentAddressedStore,
    artifact: DocumentArtifact,
    *,
    damage: str,
) -> CanonicalDocumentView:
    """Persist a schema-valid source run whose spans do not exactly replay."""

    _, valid_source_run = make_durable_canonical_view(store, artifact)
    old_docling = valid_source_run.require_output("docling_document")
    old_spans = valid_source_run.require_output("content_spans")
    run_id = f"{artifact.artifact_id}-{damage}-source-run"
    docling_product = store.data_product_ref(
        name="docling_document",
        blob_sha256=old_docling.blob_sha256,
        producer_run_id=run_id,
        source_artifact_ids=(artifact.artifact_id,),
    )
    span_payload = json.loads(store.read_blob(old_spans.blob_sha256))
    span_payload["processing_run_id"] = run_id
    span_payload["representation_product_id"] = docling_product.product_id
    for span in span_payload["spans"]:
        span["processing_run_id"] = run_id
        span["representation_anchor"]["product_id"] = docling_product.product_id
    if damage == "omitted":
        span_payload.pop("spans")
    elif damage == "locator-drift":
        span_payload["spans"][0]["source_locator"]["bounding_box"]["left"] = 1
    else:  # pragma: no cover - test helper guard
        raise AssertionError(f"unsupported damage: {damage}")
    span_set = ContentSpanSet.model_validate(span_payload)
    span_blob = store.put_blob(
        json.dumps(
            span_payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    source_run = make_component_run(
        store,
        artifact.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="docling",
            component_version="2.96.1",
            capability="document.parse",
        ),
        configuration={
            "content_span_schema_version": "1",
            "input_format": "pdf",
            "input_sha256": artifact.source_sha256,
            "pdf_span_algorithm": "provenance-charspan-v2",
        },
        outputs={
            "docling_document": old_docling.blob_sha256,
            "content_spans": span_blob.sha256,
        },
    )
    store.save_processing_run(source_run)
    source_products = (
        source_run.require_output("docling_document"),
        source_run.require_output("content_spans"),
    )
    return build_canonical_document_view(
        artifact=artifact,
        docling_document=json.loads(store.read_blob(old_docling.blob_sha256)),
        docling_product=source_products[0],
        content_span_set=span_set,
        source_products=source_products,
        configuration=CanonicalizationConfig(),
    )


def save_canonical_view_product(
    store: ContentAddressedStore,
    artifact: DocumentArtifact,
    view: CanonicalDocumentView,
    *,
    inputs: tuple[DataProductRef, ...] | None = None,
    run_id: str = "canonical-view-run",
    status: ProcessingRunStatus = ProcessingRunStatus.COMPLETE,
    component: ComponentDescriptor | None = None,
    configuration: dict[str, object] | None = None,
    bypass_admission: bool = False,
) -> tuple[ProcessingRun, DataProductRef]:
    """Persist a canonical view and its producer record with durable inputs."""

    blob = store.put_canonical_document(view)
    source_products = view.source_products if inputs is None else inputs
    invocation_configuration = (
        canonical_invocation_configuration(
            CanonicalizationConfig(),
            source_products,
        )
        if configuration is None
        else configuration
    )
    started_at = datetime(2026, 7, 17, 13, tzinfo=UTC)
    output = store.data_product_ref(
        name="canonical_document_view",
        blob_sha256=blob.sha256,
        producer_run_id=run_id,
        source_artifact_ids=tuple(
            dict.fromkeys(
                (
                    artifact.artifact_id,
                    *(
                        source_artifact_id
                        for product in source_products
                        for source_artifact_id in product.source_artifact_ids
                    ),
                )
            )
        ),
    )
    producer = ProcessingRun(
        run_id=run_id,
        artifact_id=artifact.artifact_id,
        stage_id="canonicalize",
        component=component
        or ComponentDescriptor(
            component_id=CANONICAL_COMPONENT_ID,
            component_version=CANONICAL_COMPONENT_VERSION,
            capability=CANONICAL_COMPONENT_CAPABILITY,
        ),
        configuration=invocation_configuration,
        configuration_sha256=configuration_sha256(invocation_configuration),
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=1),
        status=status,
        inputs=source_products,
        outputs=(output,),
    )
    if bypass_admission:
        store._save_record(
            "processing_runs",
            producer.run_id,
            producer,
        )
    else:
        store.save_processing_run(producer)
    return producer, producer.require_output("canonical_document_view")


def mutate_canonical_view(
    view: CanonicalDocumentView,
    mutator: Callable[[dict[str, Any]], None],
) -> CanonicalDocumentView:
    payload = json.loads(canonical_document_bytes(view))
    mutator(payload)
    payload["view_id"] = "pending"
    payload["view_id"] = canonical_view_id(payload)
    return CanonicalDocumentView.model_validate(payload)


def test_blob_write_is_content_addressed_verified_and_idempotent(
    store: ContentAddressedStore,
) -> None:
    first = store.put_blob(b"same bytes")
    second = store.put_blob(b"same bytes", expected_sha256=first.sha256)

    assert first == second
    assert first.uri == f"cas://sha256/{first.sha256}"
    assert first.path == store.blob_path(first.sha256)
    assert store.read_blob(first.sha256) == b"same bytes"
    assert len(list((store.root / "blobs" / "sha256").rglob("*"))) == 2


def test_blob_and_record_publication_fsync_destination_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsynced_directories: list[Path] = []
    monkeypatch.setattr(
        storage_module,
        "_fsync_directory",
        lambda directory: fsynced_directories.append(Path(directory)),
    )
    durable_store = ContentAddressedStore(tmp_path / "durable-store")
    fsynced_directories.clear()

    blob = durable_store.put_blob(b"durably published")

    assert fsynced_directories.count(blob.path.parent) >= 2
    assert blob.path.parent.parent in fsynced_directories

    artifact = DocumentArtifact(
        artifact_id="durable-artifact",
        source_sha256=blob.sha256,
        acquisition_uri="https://example.test/durable.pdf",
        media_type="application/pdf",
        raw_location=blob.as_location(
            media_type="application/pdf",
            role=ArtifactLocationRole.RAW,
        ),
    )
    fsynced_directories.clear()
    record_path = durable_store.save_artifact(artifact)

    assert fsynced_directories == [record_path.parent]


def test_directory_fsync_supports_the_host_platform(tmp_path: Path) -> None:
    storage_module._fsync_directory(tmp_path)


def test_blob_expected_hash_and_tamper_are_detected(
    store: ContentAddressedStore,
) -> None:
    with pytest.raises(HashMismatchError, match="expected blob"):
        store.put_blob(b"actual", expected_sha256="0" * 64)

    blob = store.put_blob(b"untampered")
    blob.path.write_bytes(b"tampered")
    with pytest.raises(HashMismatchError, match="hashes to"):
        store.read_blob(blob.sha256)


def test_streaming_blob_limit_never_publishes_oversized_content(
    store: ContentAddressedStore,
) -> None:
    with pytest.raises(BlobTooLargeError) as error:
        store.put_blob(BytesIO(b"012345"), max_bytes=5)

    assert error.value.max_bytes == 5
    assert error.value.observed_bytes == 6
    assert not any(path.is_file() for path in (store.root / "blobs").rglob("*"))
    assert not any((store.root / ".tmp").iterdir())

    exact = store.put_blob(BytesIO(b"012345"), max_bytes=6)
    assert exact.byte_size == 6


def test_missing_blob_is_explicit(store: ContentAddressedStore) -> None:
    with pytest.raises(BlobNotFoundError, match="blob not found"):
        store.read_blob("f" * 64)


def test_verified_data_product_reader_returns_exact_declared_bytes(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "verified-product")
    blob = store.put_blob(b'{"document": "verified"}')
    producer = make_run(
        store,
        artifact.artifact_id,
        "verified-product-run",
        ProcessingRunStatus.COMPLETE,
        outputs={"docling_document": blob.sha256},
    )
    store.save_processing_run(producer)

    assert (
        store.read_data_product_bytes(producer.require_output("docling_document"))
        == b'{"document": "verified"}'
    )


def test_output_lineage_deduplicates_inherited_artifacts_in_stable_order(
    store: ContentAddressedStore,
) -> None:
    main = save_artifact(store, "deduplicated-main")
    inherited = save_artifact(store, "deduplicated-inherited")
    main_source_blob = store.put_blob(b"main source product")
    main_source_run = make_run(
        store,
        main.artifact_id,
        "deduplicated-main-source-run",
        ProcessingRunStatus.COMPLETE,
        outputs={"docling_document": main_source_blob.sha256},
    )
    store.save_processing_run(main_source_run)
    inherited_blob = store.put_blob(b"inherited product")
    inherited_run = make_run(
        store,
        inherited.artifact_id,
        "deduplicated-inherited-run",
        ProcessingRunStatus.COMPLETE,
        inputs=(main_source_run.require_output("docling_document"),),
        outputs={"content_spans": inherited_blob.sha256},
    )
    store.save_processing_run(inherited_run)
    final_blob = store.put_blob(b"deduplicated final product")
    final_run = make_run(
        store,
        main.artifact_id,
        "deduplicated-final-run",
        ProcessingRunStatus.COMPLETE,
        inputs=(inherited_run.require_output("content_spans"),),
        outputs={"content_integrity_overlay": final_blob.sha256},
    )
    store.save_processing_run(final_run)
    final_product = final_run.require_output("content_integrity_overlay")

    assert inherited_run.outputs[0].source_artifact_ids == (
        inherited.artifact_id,
        main.artifact_id,
    )
    assert final_product.source_artifact_ids == (
        main.artifact_id,
        inherited.artifact_id,
    )
    assert store.read_data_product_bytes(final_product) == final_blob.path.read_bytes()


def test_transitive_product_verification_rejects_missing_input_producer(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "missing-transitive-producer")
    source_blob = store.put_blob(b"transitive source")
    source_run = make_run(
        store,
        artifact.artifact_id,
        "transitive-source-run",
        ProcessingRunStatus.COMPLETE,
        outputs={"docling_document": source_blob.sha256},
    )
    store.save_processing_run(source_run)
    output_blob = store.put_blob(b"transitive output")
    producer = make_run(
        store,
        artifact.artifact_id,
        "transitive-output-run",
        ProcessingRunStatus.COMPLETE,
        inputs=(source_run.require_output("docling_document"),),
        outputs={"content_spans": output_blob.sha256},
    )
    producer_path = store.save_processing_run(producer)
    missing_input = build_data_product_ref(
        name="docling_document",
        blob_sha256=source_blob.sha256,
        uri=source_blob.uri,
        byte_size=source_blob.byte_size,
        producer_run_id="missing-input-producer-run",
        source_artifact_ids=(artifact.artifact_id,),
    )
    producer_record = json.loads(producer_path.read_text(encoding="utf-8"))
    producer_record["inputs"] = [missing_input.model_dump(mode="json")]
    producer_path.write_text(json.dumps(producer_record), encoding="utf-8")
    output = producer.require_output("content_spans")

    with pytest.raises(RecordNotFoundError, match="missing-input-producer-run"):
        store.read_data_product_bytes(output)

    consumer = make_run(
        store,
        artifact.artifact_id,
        "transitive-consumer-run",
        ProcessingRunStatus.PARTIAL,
        inputs=(output,),
    )
    with pytest.raises(RecordNotFoundError, match="missing-input-producer-run"):
        store.save_processing_run(consumer)


def test_transitive_product_verification_rejects_undeclared_input(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "undeclared-transitive-input")
    source_blob = store.put_blob(b"declared transitive source")
    source_run = make_run(
        store,
        artifact.artifact_id,
        "declared-transitive-source-run",
        ProcessingRunStatus.COMPLETE,
        outputs={"docling_document": source_blob.sha256},
    )
    source_path = store.save_processing_run(source_run)
    output_blob = store.put_blob(b"dependent transitive output")
    producer = make_run(
        store,
        artifact.artifact_id,
        "dependent-transitive-run",
        ProcessingRunStatus.COMPLETE,
        inputs=(source_run.require_output("docling_document"),),
        outputs={"content_spans": output_blob.sha256},
    )
    store.save_processing_run(producer)
    source_record = json.loads(source_path.read_text(encoding="utf-8"))
    source_record["outputs"] = []
    source_path.write_text(json.dumps(source_record), encoding="utf-8")

    with pytest.raises(RecordConflictError, match="is not declared"):
        store.read_data_product_bytes(producer.require_output("content_spans"))


def test_transitive_product_verification_rejects_cycles_without_recursion(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "cyclic-transitive-inputs")
    first_blob = store.put_blob(b"first cyclic output")
    first_run = make_run(
        store,
        artifact.artifact_id,
        "first-cyclic-run",
        ProcessingRunStatus.COMPLETE,
        outputs={"docling_document": first_blob.sha256},
    )
    first_path = store.save_processing_run(first_run)
    second_blob = store.put_blob(b"second cyclic output")
    second_run = make_run(
        store,
        artifact.artifact_id,
        "second-cyclic-run",
        ProcessingRunStatus.COMPLETE,
        outputs={"content_spans": second_blob.sha256},
    )
    second_path = store.save_processing_run(second_run)
    first_record = json.loads(first_path.read_text(encoding="utf-8"))
    first_record["inputs"] = [
        second_run.require_output("content_spans").model_dump(mode="json")
    ]
    first_path.write_text(json.dumps(first_record), encoding="utf-8")
    second_record = json.loads(second_path.read_text(encoding="utf-8"))
    second_record["inputs"] = [
        first_run.require_output("docling_document").model_dump(mode="json")
    ]
    second_path.write_text(json.dumps(second_record), encoding="utf-8")

    with pytest.raises(RecordConflictError, match="provenance contains a cycle"):
        store.read_data_product_bytes(first_run.require_output("docling_document"))


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        (
            "uri",
            f"cas://sha256/{'f' * 64}",
            HashMismatchError,
            "URI does not match",
        ),
        ("byte_size", 999, HashMismatchError, "size does not match"),
        (
            "media_type",
            "text/plain",
            RecordConflictError,
            "registered contract",
        ),
        (
            "blob_sha256",
            "not-a-sha256",
            RecordConflictError,
            "reference is invalid",
        ),
    ],
)
def test_verified_data_product_reader_rejects_invalid_reference_fields(
    store: ContentAddressedStore,
    field: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    artifact = save_artifact(store, f"invalid-{field}")
    blob = store.put_blob(b"declared bytes")
    producer = make_run(
        store,
        artifact.artifact_id,
        f"invalid-{field}-run",
        ProcessingRunStatus.COMPLETE,
        outputs={"docling_document": blob.sha256},
    )
    store.save_processing_run(producer)
    invalid = producer.require_output("docling_document").model_copy(
        update={field: value}
    )

    with pytest.raises(error_type, match=message):
        store.read_data_product_bytes(invalid)


def test_verified_data_product_reader_detects_blob_hash_corruption(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "corrupt-product")
    blob = store.put_blob(b"original product bytes")
    producer = make_run(
        store,
        artifact.artifact_id,
        "corrupt-product-run",
        ProcessingRunStatus.COMPLETE,
        outputs={"docling_document": blob.sha256},
    )
    store.save_processing_run(producer)
    blob.path.write_bytes(b"corrupted product bytes")

    with pytest.raises(HashMismatchError, match="hashes to"):
        store.read_data_product_bytes(producer.require_output("docling_document"))


def test_verified_data_product_reader_requires_artifact_and_producer_lineage(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "product-lineage")
    other_artifact = save_artifact(store, "other-product-lineage")
    blob = store.put_blob(b"lineage product")
    producer = make_run(
        store,
        artifact.artifact_id,
        "product-lineage-run",
        ProcessingRunStatus.COMPLETE,
        outputs={"docling_document": blob.sha256},
    )
    store.save_processing_run(producer)

    missing_artifact = build_data_product_ref(
        name="docling_document",
        blob_sha256=blob.sha256,
        uri=blob.uri,
        byte_size=blob.byte_size,
        producer_run_id=producer.run_id,
        source_artifact_ids=("missing-lineage-artifact",),
    )
    with pytest.raises(RecordNotFoundError, match="missing-lineage-artifact"):
        store.read_data_product_bytes(missing_artifact)

    missing_producer = build_data_product_ref(
        name="docling_document",
        blob_sha256=blob.sha256,
        uri=blob.uri,
        byte_size=blob.byte_size,
        producer_run_id="missing-product-run",
        source_artifact_ids=(artifact.artifact_id,),
    )
    with pytest.raises(RecordNotFoundError, match="missing-product-run"):
        store.read_data_product_bytes(missing_producer)

    wrong_artifact = build_data_product_ref(
        name="docling_document",
        blob_sha256=blob.sha256,
        uri=blob.uri,
        byte_size=blob.byte_size,
        producer_run_id=producer.run_id,
        source_artifact_ids=(other_artifact.artifact_id,),
    )
    with pytest.raises(RecordConflictError, match="lineage must exactly equal"):
        store.read_data_product_bytes(wrong_artifact)


@pytest.mark.parametrize(
    "invalid_lineage",
    [
        ("lineage-main",),
        ("lineage-main", "lineage-source", "lineage-extra"),
        ("lineage-source", "lineage-main"),
    ],
    ids=("missing-inherited", "extra", "wrong-order"),
)
def test_processing_run_save_requires_exact_ordered_inherited_output_lineage(
    store: ContentAddressedStore,
    invalid_lineage: tuple[str, ...],
) -> None:
    main = save_artifact(store, "lineage-main")
    source = save_artifact(store, "lineage-source")
    save_artifact(store, "lineage-extra")
    source_blob = store.put_blob(b"inherited source product")
    source_run = make_run(
        store,
        source.artifact_id,
        "lineage-source-run",
        ProcessingRunStatus.COMPLETE,
        outputs={"content_spans": source_blob.sha256},
    )
    store.save_processing_run(source_run)
    output_blob = store.put_blob(b"product with inherited lineage")
    valid_run = make_run(
        store,
        main.artifact_id,
        "lineage-main-run",
        ProcessingRunStatus.COMPLETE,
        inputs=(source_run.require_output("content_spans"),),
        outputs={"docling_document": output_blob.sha256},
    )
    invalid_output = valid_run.require_output("docling_document").model_copy(
        update={"source_artifact_ids": invalid_lineage}
    )

    with pytest.raises(RecordConflictError, match="lineage must exactly equal"):
        store.save_processing_run(
            valid_run.model_copy(update={"outputs": (invalid_output,)})
        )


@pytest.mark.parametrize(
    "invalid_lineage",
    [
        ("read-lineage-main",),
        ("read-lineage-main", "read-lineage-source", "read-lineage-extra"),
    ],
    ids=("missing-inherited", "extra"),
)
def test_verified_reader_rejects_corrupt_durable_inherited_lineage(
    store: ContentAddressedStore,
    invalid_lineage: tuple[str, ...],
) -> None:
    main = save_artifact(store, "read-lineage-main")
    source = save_artifact(store, "read-lineage-source")
    save_artifact(store, "read-lineage-extra")
    source_blob = store.put_blob(b"durable inherited source")
    source_run = make_run(
        store,
        source.artifact_id,
        "read-lineage-source-run",
        ProcessingRunStatus.COMPLETE,
        outputs={"content_spans": source_blob.sha256},
    )
    store.save_processing_run(source_run)
    output_blob = store.put_blob(b"durable inherited output")
    producer = make_run(
        store,
        main.artifact_id,
        "read-lineage-main-run",
        ProcessingRunStatus.COMPLETE,
        inputs=(source_run.require_output("content_spans"),),
        outputs={"docling_document": output_blob.sha256},
    )
    record_path = store.save_processing_run(producer)
    invalid_output = producer.require_output("docling_document").model_copy(
        update={"source_artifact_ids": invalid_lineage}
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["outputs"] = [invalid_output.model_dump(mode="json")]
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RecordConflictError, match="lineage must exactly equal"):
        store.read_data_product_bytes(invalid_output)


def test_verified_data_product_reader_requires_exact_output_declaration(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "undeclared-product")
    blob = store.put_blob(b"declared under a different product name")
    producer = make_run(
        store,
        artifact.artifact_id,
        "undeclared-product-run",
        ProcessingRunStatus.COMPLETE,
        outputs={"docling_document": blob.sha256},
    )
    store.save_processing_run(producer)
    undeclared = build_data_product_ref(
        name="content_spans",
        blob_sha256=blob.sha256,
        uri=blob.uri,
        byte_size=blob.byte_size,
        producer_run_id=producer.run_id,
        source_artifact_ids=(artifact.artifact_id,),
    )

    with pytest.raises(RecordConflictError, match="is not declared"):
        store.read_data_product_bytes(undeclared)


def test_canonical_persistence_revalidates_model_copy(
    store: ContentAddressedStore,
) -> None:
    valid = make_canonical_view()
    invalid = valid.model_copy(update={"root_block_ids": ("missing-block",)})

    with pytest.raises(ValidationError, match="canonical roots"):
        store.put_canonical_document(invalid)
    assert not any(
        path.is_file() for path in (store.root / "blobs" / "sha256").rglob("*")
    )


def test_canonical_persistence_and_verified_read_round_trip(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-round-trip")
    view, source_run = make_durable_canonical_view(store, artifact)
    producer, product = save_canonical_view_product(
        store,
        artifact,
        view,
        run_id="canonical-round-trip-run",
    )

    assert producer.inputs == source_run.outputs == view.source_products
    assert store.read_canonical_document(product) == view


def test_canonical_admission_accepts_clean_partial_docling_producer(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-partial-docling")
    view, source_run = make_durable_canonical_view(
        store,
        artifact,
        source_status=ProcessingRunStatus.PARTIAL,
    )

    _, product = save_canonical_view_product(
        store,
        artifact,
        view,
        run_id="canonical-partial-docling-view-run",
    )

    assert source_run.status is ProcessingRunStatus.PARTIAL
    assert store.read_canonical_document(product) == view


@pytest.mark.parametrize(
    ("component", "status", "message"),
    [
        (
            ComponentDescriptor(
                component_id="unapproved-docling",
                component_version="2.96.1",
                capability="document.parse",
            ),
            ProcessingRunStatus.COMPLETE,
            "identity",
        ),
        (
            ComponentDescriptor(
                component_id="docling",
                component_version="2.999.0",
                capability="document.parse",
            ),
            ProcessingRunStatus.COMPLETE,
            "version|identity",
        ),
        (
            ComponentDescriptor(
                component_id="docling",
                component_version="2.113.0",
                capability="document.parse",
            ),
            ProcessingRunStatus.COMPLETE,
            "version|identity",
        ),
        (
            ComponentDescriptor(
                component_id="docling",
                component_version="2.96.1",
                capability="document.parse",
            ),
            ProcessingRunStatus.FAILED,
            "status",
        ),
    ],
    ids=("wrong-component", "wrong-version", "stale-draft-version", "failed"),
)
def test_canonical_admission_rejects_unapproved_docling_producer(
    store: ContentAddressedStore,
    component: ComponentDescriptor,
    status: ProcessingRunStatus,
    message: str,
) -> None:
    artifact = save_artifact(store, f"native-{component.component_id}-{status.value}")
    view, _ = make_durable_canonical_view(
        store,
        artifact,
        source_component=component,
        source_status=status,
    )

    with pytest.raises(RecordConflictError, match=message):
        save_canonical_view_product(
            store,
            artifact,
            view,
            run_id=f"native-{component.component_id}-{status.value}-view-run",
        )


@pytest.mark.parametrize(
    "damage",
    ["extra-field", "wrong-input-hash", "wrong-image", "wrong-pdf-algorithm"],
)
def test_canonical_admission_rejects_fake_docling_configuration(
    store: ContentAddressedStore,
    damage: str,
) -> None:
    artifact = save_artifact(store, f"native-config-{damage}")
    configuration = docling_production_configuration(artifact)
    if damage == "extra-field":
        configuration["unregistered_option"] = True
        message = "production schema"
    elif damage == "wrong-input-hash":
        configuration["input_sha256"] = "f" * 64
        message = "input hash"
    elif damage == "wrong-image":
        configuration["container_image"] = "example.test/not-docling:latest"
        message = "container image"
    else:
        configuration["pdf_span_algorithm"] = "unapproved"
        message = "PDF span algorithm"
    view, _ = make_durable_canonical_view(
        store,
        artifact,
        source_configuration=configuration,
    )

    with pytest.raises(RecordConflictError, match=message):
        save_canonical_view_product(
            store,
            artifact,
            view,
            run_id=f"native-config-{damage}-view-run",
        )


def test_canonical_admission_binds_docling_runtime_attestation_to_configuration(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "native-attestation-binding")
    run_id = f"{artifact.artifact_id}-canonical-source-run"
    attestation = docling_runtime_attestation(run_id)
    configuration = docling_production_configuration(artifact)
    configuration.update(
        {
            "container_digest": f"sha256:{'c' * 64}",
            "model_versions": dict(attestation.model_versions),
            "model_hashes": dict(attestation.model_hashes),
        }
    )
    view, _ = make_durable_canonical_view(
        store,
        artifact,
        source_configuration=configuration,
        source_attestation=attestation,
        source_runtime_identity_required=True,
        source_output_policy_snapshot=runtime_output_policy(
            component_id="docling",
            configuration=configuration,
            attestation=attestation,
            runtime_identity_required=True,
        ),
    )

    with pytest.raises(RecordConflictError, match="attestation conflicts"):
        save_canonical_view_product(
            store,
            artifact,
            view,
            run_id="native-attestation-binding-view-run",
        )


def test_canonical_admission_accepts_exact_docling_runtime_attestation(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "native-exact-attestation")
    run_id = f"{artifact.artifact_id}-canonical-source-run"
    attestation = docling_runtime_attestation(run_id)
    configuration = docling_production_configuration(artifact)
    configuration.update(
        {
            "container_digest": attestation.container_digest,
            "model_versions": dict(attestation.model_versions),
            "model_hashes": dict(attestation.model_hashes),
        }
    )
    view, _ = make_durable_canonical_view(
        store,
        artifact,
        source_configuration=configuration,
        source_attestation=attestation,
        source_runtime_identity_required=True,
        source_output_policy_snapshot=runtime_output_policy(
            component_id="docling",
            configuration=configuration,
            attestation=attestation,
            runtime_identity_required=True,
        ),
    )

    _, product = save_canonical_view_product(
        store,
        artifact,
        view,
        run_id="native-exact-attestation-view-run",
    )
    assert store.read_canonical_document(product) == view


@pytest.mark.parametrize("bypass_admission", [False, True], ids=("admission", "read"))
def test_complete_docling_rejects_empty_required_model_identity(
    store: ContentAddressedStore,
    bypass_admission: bool,
) -> None:
    artifact = save_artifact(
        store,
        f"native-empty-model-identity-{'read' if bypass_admission else 'admission'}",
    )
    run_id = f"{artifact.artifact_id}-canonical-source-run"
    attestation = docling_runtime_attestation(run_id).model_copy(
        update={"model_versions": {}, "model_hashes": {}}
    )
    configuration = docling_production_configuration(artifact)
    configuration.update(
        {
            "container_digest": attestation.container_digest,
            "model_versions": {},
            "model_hashes": {},
        }
    )
    view, _ = make_durable_canonical_view(
        store,
        artifact,
        source_configuration=configuration,
        source_attestation=attestation,
        source_runtime_identity_required=True,
        source_output_policy_snapshot=runtime_output_policy(
            component_id="docling",
            configuration=configuration,
            attestation=attestation,
            runtime_identity_required=True,
        ),
    )

    if bypass_admission:
        _, product = save_canonical_view_product(
            store,
            artifact,
            view,
            run_id=f"{artifact.artifact_id}-legacy-view-run",
            bypass_admission=True,
        )
        with pytest.raises(RecordConflictError, match="model identity evidence"):
            store.read_canonical_document(product)
        return

    with pytest.raises(RecordConflictError, match="model identity evidence"):
        save_canonical_view_product(
            store,
            artifact,
            view,
            run_id=f"{artifact.artifact_id}-view-run",
        )


def test_partial_docling_accepts_exact_empty_model_identity(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "native-partial-empty-model-identity")
    run_id = f"{artifact.artifact_id}-canonical-source-run"
    attestation = docling_runtime_attestation(run_id).model_copy(
        update={"model_versions": {}, "model_hashes": {}}
    )
    configuration = docling_production_configuration(artifact)
    configuration.update(
        {
            "container_digest": attestation.container_digest,
            "model_versions": {},
            "model_hashes": {},
        }
    )
    view, source_run = make_durable_canonical_view(
        store,
        artifact,
        source_configuration=configuration,
        source_status=ProcessingRunStatus.PARTIAL,
        source_attestation=attestation,
        source_runtime_identity_required=True,
        source_output_policy_snapshot=runtime_output_policy(
            component_id="docling",
            configuration=configuration,
            attestation=attestation,
            runtime_identity_required=True,
        ),
    )

    _, product = save_canonical_view_product(
        store,
        artifact,
        view,
        run_id="native-partial-empty-model-identity-view-run",
    )
    assert source_run.status is ProcessingRunStatus.PARTIAL
    assert store.read_canonical_document(product) == view


def test_canonical_admission_accepts_clean_partial_docling_without_reporter(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "native-partial-missing-runtime-evidence")
    configuration = docling_production_configuration(artifact)
    view, source_run = make_durable_canonical_view(
        store,
        artifact,
        source_configuration=configuration,
        source_status=ProcessingRunStatus.PARTIAL,
        source_runtime_identity_required=True,
        source_output_policy_snapshot=runtime_output_policy(
            component_id="docling",
            configuration=configuration,
            attestation=None,
            runtime_identity_required=True,
        ),
    )
    assert source_run.runtime_attestation is None

    _, product = save_canonical_view_product(
        store,
        artifact,
        view,
        run_id="native-partial-missing-runtime-evidence-view-run",
    )
    assert store.read_canonical_document(product) == view


@pytest.mark.parametrize("bypass_admission", [False, True], ids=("admission", "read"))
@pytest.mark.parametrize(
    ("damage", "runtime_identity_required", "message"),
    [
        ("wrong-reporter", True, "persisted trust policy"),
        ("optional-wrong-reporter", False, "persisted trust policy"),
        ("wrong-source", True, "persisted trust policy"),
        ("registry-suffix", True, "container image"),
        ("digest-appended-image", True, "container image"),
        ("missing-policy", True, "no persisted trust policy"),
        ("wrong-policy-schema", True, "output policy schema"),
        ("missing-trust-configuration", True, "runtime trust configuration"),
        ("missing-component-policy", True, "component trust policy"),
        ("wrong-contract", True, "trust policy values"),
        ("wrong-attestation-schema-policy", True, "trust policy values"),
        ("extra-trust-field", True, "trust policy shape"),
        ("unconfigured-reporter", True, "persisted trust policy"),
        ("invalid-reporter", True, "reporter is not approved"),
        ("unconfigured-named-reporter", True, "names a reporter"),
        ("required-identity-mismatch", True, "required runtime identity"),
        ("wrong-policy-image", True, "persisted parser policy"),
        ("wrong-policy-digest", True, "persisted parser policy"),
        ("wrong-policy-model-versions", True, "persisted parser policy"),
        ("wrong-policy-model-hashes", True, "persisted parser policy"),
    ],
)
def test_canonical_docling_runtime_trust_is_replayed_exactly(
    store: ContentAddressedStore,
    bypass_admission: bool,
    damage: str,
    runtime_identity_required: bool,
    message: str,
) -> None:
    artifact = save_artifact(
        store,
        f"docling-trust-{damage}-{'read' if bypass_admission else 'admission'}",
    )
    run_id = f"{artifact.artifact_id}-canonical-source-run"
    reporter_id = (
        "wrong-reporter"
        if damage in {"wrong-reporter", "optional-wrong-reporter"}
        else "expected-reporter"
    )
    attestation_image = (
        "registry.example.test/docling-serve-cpu:v1.21.0"
        if damage == "registry-suffix"
        else "quay.io/docling-project/docling-serve-cpu:v1.21.0"
    )
    attestation = docling_runtime_attestation(
        run_id,
        reporter_id=reporter_id,
        container_image=attestation_image,
        source=(
            RuntimeAttestationSource.DIGEST_ADDRESSED_OCI_INVOCATION
            if damage == "wrong-source"
            else RuntimeAttestationSource.AUTHENTICATED_DEPLOYMENT_REPORTER
        ),
    )
    configuration_image = (
        f"{attestation_image}@{attestation.container_digest}"
        if damage == "digest-appended-image"
        else attestation_image
    )
    configuration = docling_production_configuration(artifact)
    configuration.update(
        {
            "container_image": configuration_image,
            "container_digest": attestation.container_digest,
            "model_versions": dict(attestation.model_versions),
            "model_hashes": dict(attestation.model_hashes),
        }
    )
    policy = runtime_output_policy(
        component_id="docling",
        configuration=configuration,
        attestation=attestation,
        runtime_identity_required=runtime_identity_required,
    )
    trust_policies = policy["runtime_trust_policy"]
    assert isinstance(trust_policies, dict)
    docling_policy = trust_policies["docling"]
    assert isinstance(docling_policy, dict)
    document_policy = policy["document_processing_config"]
    assert isinstance(document_policy, dict)
    if damage in {"wrong-reporter", "optional-wrong-reporter"}:
        docling_policy["expected_reporter_id"] = "expected-reporter"
    elif damage == "wrong-policy-schema":
        policy["schema"] = "foreign-output-policy"
    elif damage == "missing-trust-configuration":
        policy["runtime_trust_policy"] = []
    elif damage == "missing-component-policy":
        trust_policies["docling"] = []
    elif damage == "wrong-contract":
        docling_policy["attestation_contract_version"] = "foreign-contract"
    elif damage == "wrong-attestation-schema-policy":
        docling_policy["attestation_schema_version"] = "foreign-schema"
    elif damage == "extra-trust-field":
        docling_policy["unexpected"] = True
    elif damage == "unconfigured-reporter":
        docling_policy["reporter_configured"] = False
        docling_policy["expected_reporter_id"] = None
    elif damage == "invalid-reporter":
        docling_policy["expected_reporter_id"] = None
    elif damage == "unconfigured-named-reporter":
        docling_policy["reporter_configured"] = False
        docling_policy["expected_reporter_id"] = "unexpected-reporter"
    elif damage == "required-identity-mismatch":
        document_policy["require_runtime_identity"] = False
    elif damage == "wrong-policy-image":
        document_policy["docling_container_image"] = (
            "registry.example.test/docling-serve-cpu:v1.21.0"
        )
    elif damage == "wrong-policy-digest":
        document_policy["docling_container_digest"] = f"sha256:{'f' * 64}"
    elif damage == "wrong-policy-model-versions":
        document_policy["docling_model_versions"] = {"layout": "foreign"}
    elif damage == "wrong-policy-model-hashes":
        document_policy["docling_model_hashes"] = {"layout": "f" * 64}
    output_policy = None if damage == "missing-policy" else policy
    view, _ = make_durable_canonical_view(
        store,
        artifact,
        source_configuration=configuration,
        source_status=(
            ProcessingRunStatus.COMPLETE
            if damage == "optional-wrong-reporter"
            else ProcessingRunStatus.PARTIAL
        ),
        source_attestation=attestation,
        source_runtime_identity_required=runtime_identity_required,
        source_output_policy_snapshot=output_policy,
    )

    if bypass_admission:
        _, product = save_canonical_view_product(
            store,
            artifact,
            view,
            run_id=f"docling-trust-{damage}-legacy-view-run",
            bypass_admission=True,
        )
        with pytest.raises(RecordConflictError, match=message):
            store.read_canonical_document(product)
        return

    with pytest.raises(RecordConflictError, match=message):
        save_canonical_view_product(
            store,
            artifact,
            view,
            run_id=f"docling-trust-{damage}-view-run",
        )


def test_canonical_admission_requires_one_docling_pair_producer(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "native-split-producer")
    view, source_run = make_durable_canonical_view(store, artifact)
    old_spans = source_run.require_output("content_spans")
    split_run = make_component_run(
        store,
        artifact.artifact_id,
        "native-split-span-run",
        component=ComponentDescriptor(
            component_id="docling",
            component_version="2.96.1",
            capability="document.parse",
        ),
        configuration=docling_production_configuration(artifact),
        outputs={"content_spans": old_spans.blob_sha256},
    )
    store.save_processing_run(split_run)
    replacement = split_run.require_output("content_spans")

    forged = mutate_canonical_view(
        view,
        lambda payload: payload["source_products"].__setitem__(
            1, replacement.model_dump(mode="json")
        ),
    )
    with pytest.raises(RecordConflictError, match="must share one producer"):
        save_canonical_view_product(
            store,
            artifact,
            forged,
            run_id="native-split-producer-view-run",
        )


@pytest.mark.parametrize(
    ("component", "status", "configuration_damage", "message"),
    [
        (
            ComponentDescriptor(
                component_id="fake-grobid",
                component_version="0.9.0",
                capability="document.parse.scholarly",
            ),
            ProcessingRunStatus.COMPLETE,
            None,
            "identity",
        ),
        (
            ComponentDescriptor(
                component_id="grobid",
                component_version="0.9.0",
                capability="document.parse.scholarly",
            ),
            ProcessingRunStatus.FAILED,
            None,
            "status",
        ),
        (
            ComponentDescriptor(
                component_id="grobid",
                component_version="0.9.0",
                capability="document.parse.scholarly",
            ),
            ProcessingRunStatus.COMPLETE,
            "input-hash",
            "input hash",
        ),
        (
            ComponentDescriptor(
                component_id="grobid",
                component_version="9.9.9",
                capability="document.parse.scholarly",
            ),
            ProcessingRunStatus.COMPLETE,
            None,
            "identity",
        ),
        (
            ComponentDescriptor(
                component_id="grobid",
                component_version="0.9.0",
                capability="document.parse.scholarly",
            ),
            ProcessingRunStatus.COMPLETE,
            "image",
            "container image",
        ),
    ],
)
def test_grobid_native_admission_rejects_unapproved_producer(
    store: ContentAddressedStore,
    component: ComponentDescriptor,
    status: ProcessingRunStatus,
    configuration_damage: str | None,
    message: str,
) -> None:
    artifact = save_artifact(
        store,
        f"grobid-native-{component.component_id}-{status.value}-{configuration_damage}",
    )
    tei = b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text>usable</text></TEI>'
    blob = store.put_blob(tei)
    configuration = grobid_production_configuration(artifact)
    if configuration_damage == "input-hash":
        configuration["input_sha256"] = "f" * 64
    elif configuration_damage == "image":
        configuration["container_image"] = "example.test/not-grobid:latest"
    run = make_component_run(
        store,
        artifact.artifact_id,
        f"{artifact.artifact_id}-run",
        component=component,
        configuration=configuration,
        outputs={"grobid_tei": blob.sha256},
        status=status,
    )
    store.save_processing_run(run)

    with pytest.raises(RecordConflictError, match=message):
        store._verify_grobid_native_source(
            canonical_artifact=artifact,
            product=run.require_output("grobid_tei"),
            tei_xml=tei,
        )


@pytest.mark.parametrize(
    "tei",
    [
        b"<not-tei />",
        b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text /></TEI>',
        (
            b'<!DOCTYPE TEI [<!ENTITY x "boom">]>'
            b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text>&x;</text></TEI>'
        ),
    ],
    ids=("wrong-root", "below-threshold", "forbidden-entity"),
)
def test_grobid_native_admission_requires_usable_tei(
    store: ContentAddressedStore,
    tei: bytes,
) -> None:
    artifact = save_artifact(store, f"grobid-unusable-{sha256_bytes(tei)[:8]}")
    blob = store.put_blob(tei)
    run = make_component_run(
        store,
        artifact.artifact_id,
        f"{artifact.artifact_id}-run",
        component=ComponentDescriptor(
            component_id="grobid",
            component_version="0.9.0",
            capability="document.parse.scholarly",
        ),
        configuration=grobid_production_configuration(artifact),
        outputs={"grobid_tei": blob.sha256},
    )
    store.save_processing_run(run)

    with pytest.raises(RecordConflictError, match=r"TEI|usability"):
        store._verify_grobid_native_source(
            canonical_artifact=artifact,
            product=run.require_output("grobid_tei"),
            tei_xml=tei,
        )


def test_grobid_native_source_accepts_exact_runtime_trust_evidence(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "grobid-runtime-trust-valid")
    tei = b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text>usable</text></TEI>'
    tei_blob = store.put_blob(tei)
    run_id = f"{artifact.artifact_id}-run"
    attestation = grobid_runtime_attestation(run_id)
    configuration = grobid_production_configuration(artifact)
    configuration.update(
        {
            "container_digest": attestation.container_digest,
            "model_versions": dict(attestation.model_versions),
            "model_hashes": dict(attestation.model_hashes),
        }
    )
    run = make_component_run(
        store,
        artifact.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="grobid",
            component_version="0.9.0",
            capability="document.parse.scholarly",
        ),
        configuration=configuration,
        outputs={"grobid_tei": tei_blob.sha256},
        runtime_attestation=attestation,
        runtime_identity_required=True,
        output_policy_snapshot=runtime_output_policy(
            component_id="grobid",
            configuration=configuration,
            attestation=attestation,
            runtime_identity_required=True,
        ),
    )
    store.save_processing_run(run)

    assert (
        store._verify_grobid_native_source(
            canonical_artifact=artifact,
            product=run.require_output("grobid_tei"),
            tei_xml=tei,
        )
        == run
    )


@pytest.mark.parametrize(
    ("damage", "runtime_identity_required", "message"),
    [
        ("wrong-reporter", True, "persisted trust policy"),
        ("optional-wrong-reporter", False, "persisted trust policy"),
        ("missing-policy", True, "no persisted trust policy"),
        ("malformed-policy", True, "trust policy shape"),
        ("wrong-policy-digest", True, "persisted parser policy"),
    ],
)
def test_grobid_native_runtime_trust_is_replayed_directly(
    store: ContentAddressedStore,
    damage: str,
    runtime_identity_required: bool,
    message: str,
) -> None:
    artifact = save_artifact(store, f"grobid-runtime-trust-{damage}")
    tei = b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text>usable</text></TEI>'
    tei_blob = store.put_blob(tei)
    run_id = f"{artifact.artifact_id}-run"
    attestation = grobid_runtime_attestation(run_id, reporter_id="wrong-reporter")
    configuration = grobid_production_configuration(artifact)
    configuration.update(
        {
            "container_digest": attestation.container_digest,
            "model_versions": dict(attestation.model_versions),
            "model_hashes": dict(attestation.model_hashes),
        }
    )
    policy = runtime_output_policy(
        component_id="grobid",
        configuration=configuration,
        attestation=attestation,
        runtime_identity_required=runtime_identity_required,
    )
    trust_policies = policy["runtime_trust_policy"]
    document_policy = policy["document_processing_config"]
    assert isinstance(trust_policies, dict)
    assert isinstance(document_policy, dict)
    grobid_policy = trust_policies["grobid"]
    assert isinstance(grobid_policy, dict)
    if damage in {"wrong-reporter", "optional-wrong-reporter"}:
        grobid_policy["expected_reporter_id"] = "expected-reporter"
    elif damage == "malformed-policy":
        grobid_policy.pop("attestation_contract_version")
    elif damage == "wrong-policy-digest":
        document_policy["grobid_container_digest"] = f"sha256:{'f' * 64}"
    output_policy = None if damage == "missing-policy" else policy
    run = make_component_run(
        store,
        artifact.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="grobid",
            component_version="0.9.0",
            capability="document.parse.scholarly",
        ),
        configuration=configuration,
        outputs={"grobid_tei": tei_blob.sha256},
        runtime_attestation=attestation,
        runtime_identity_required=runtime_identity_required,
        output_policy_snapshot=output_policy,
    )
    store.save_processing_run(run)

    with pytest.raises(RecordConflictError, match=message):
        store._verify_grobid_native_source(
            canonical_artifact=artifact,
            product=run.require_output("grobid_tei"),
            tei_xml=tei,
        )


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("undeclared-product", "does not declare"),
        ("extra-configuration", "production schema"),
        ("configured-version", "version configuration"),
        ("input-lineage", "inputs do not match"),
        ("consolidation-option", "consolidate_header"),
        ("segment-sentences", "segment_sentences"),
        ("coordinates", "coordinates"),
        ("minimum-characters", "minimum text characters"),
    ],
)
def test_grobid_native_admission_rejects_malformed_production_contract(
    store: ContentAddressedStore,
    damage: str,
    message: str,
) -> None:
    artifact = save_artifact(store, f"grobid-contract-{damage}")
    tei = b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text>usable</text></TEI>'
    tei_blob = store.put_blob(tei)
    configuration = grobid_production_configuration(artifact)
    inputs: tuple[DataProductRef, ...] = ()
    if damage == "extra-configuration":
        configuration["unapproved"] = True
    elif damage == "configured-version":
        configuration["expected_grobid_version"] = "99"
    elif damage == "input-lineage":
        upstream_blob = store.put_blob(b"unrelated GROBID input")
        upstream = make_component_run(
            store,
            artifact.artifact_id,
            f"{artifact.artifact_id}-upstream-run",
            component=ComponentDescriptor(
                component_id="unrelated-preprocessor",
                component_version="1",
                capability="document.prepare",
            ),
            configuration={"version": "1"},
            outputs={"ocr_sidecar": upstream_blob.sha256},
        )
        store.save_processing_run(upstream)
        inputs = (upstream.require_output("ocr_sidecar"),)
    elif damage == "consolidation-option":
        configuration["consolidate_header"] = True
    elif damage == "segment-sentences":
        configuration["segment_sentences"] = 1
    elif damage == "coordinates":
        configuration["coordinates"] = [""]
    elif damage == "minimum-characters":
        configuration["minimum_text_characters"] = True

    run = make_component_run(
        store,
        artifact.artifact_id,
        f"{artifact.artifact_id}-run",
        component=ComponentDescriptor(
            component_id="grobid",
            component_version="0.9.0",
            capability="document.parse.scholarly",
        ),
        configuration=configuration,
        inputs=inputs,
        outputs={"grobid_tei": tei_blob.sha256},
    )
    store.save_processing_run(run)
    product = run.require_output("grobid_tei")
    if damage == "undeclared-product":
        product = product.model_copy(update={"name": "ocr_sidecar"})

    with pytest.raises(RecordConflictError, match=message):
        store._verify_grobid_native_source(
            canonical_artifact=artifact,
            product=product,
            tei_xml=tei,
        )


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("supplement", "direct PDF derivative"),
        ("wrong-component", "identity"),
        ("wrong-version", "identity"),
        ("failed", "status"),
        ("wrong-output", "searchable-PDF"),
        ("extra-config", "production schema"),
        ("wrong-source", "exact source"),
        ("wrong-config-version", "exact source"),
        ("empty-reason", "fallback reason"),
        ("wrong-image", "container identity"),
        ("wrong-digest", "container identity"),
        ("local-image", "cannot claim"),
        ("wrong-mode", "mode"),
        ("bad-languages-type", "languages"),
        ("empty-languages", "languages"),
        ("bad-language", "languages"),
        ("bad-bool", "must be a boolean"),
        ("skip-text", "preserve existing text"),
        ("bad-jobs-bool", "jobs"),
        ("bad-jobs", "jobs"),
        ("bad-optimize-bool", "optimize"),
        ("bad-optimize", "optimize"),
        ("bad-output", "output type"),
    ],
)
def test_grobid_derivative_requires_exact_ocr_lineage(
    store: ContentAddressedStore,
    damage: str,
    message: str,
) -> None:
    parent = save_artifact(store, f"ocr-lineage-{damage}")
    derivative_blob = store.put_blob(b"%PDF-1.7\nsearchable derivative")
    configuration = ocr_production_configuration(parent)
    component = ComponentDescriptor(
        component_id="ocrmypdf",
        component_version="17.4.1",
        capability="document.ocr",
    )
    status = ProcessingRunStatus.COMPLETE
    output_name = "searchable_pdf"
    relationship = ArtifactRelationship.DERIVATIVE

    if damage == "supplement":
        relationship = ArtifactRelationship.SUPPLEMENT
    elif damage == "wrong-component":
        component = component.model_copy(update={"component_id": "unapproved-ocr"})
    elif damage == "wrong-version":
        component = component.model_copy(update={"component_version": "99"})
    elif damage == "failed":
        status = ProcessingRunStatus.FAILED
    elif damage == "wrong-output":
        output_name = "ocr_sidecar"
    elif damage == "extra-config":
        configuration["unregistered"] = True
    elif damage == "wrong-source":
        configuration["input_sha256"] = "f" * 64
    elif damage == "wrong-config-version":
        configuration["expected_ocrmypdf_version"] = "99"
    elif damage == "empty-reason":
        configuration["fallback_reason"] = " "
    elif damage == "wrong-image":
        configuration["container_image"] = "example.test/not-ocr:latest"
    elif damage == "wrong-digest":
        configuration["container_digest"] = "sha256:not-a-digest"
    elif damage == "local-image":
        configuration = ocr_production_configuration(parent, mode="local_cli")
        configuration["container_image"] = "jbarlow83/ocrmypdf:v17.4.1"
    elif damage == "wrong-mode":
        configuration["mode"] = "remote"
    elif damage == "bad-languages-type":
        configuration["languages"] = "eng"
    elif damage == "empty-languages":
        configuration["languages"] = []
    elif damage == "bad-language":
        configuration["languages"] = [""]
    elif damage == "bad-bool":
        configuration["rotate_pages"] = 1
    elif damage == "skip-text":
        configuration["skip_text"] = False
    elif damage == "bad-jobs-bool":
        configuration["jobs"] = True
    elif damage == "bad-jobs":
        configuration["jobs"] = 0
    elif damage == "bad-optimize-bool":
        configuration["optimize"] = False
    elif damage == "bad-optimize":
        configuration["optimize"] = 4
    elif damage == "bad-output":
        configuration["output_type"] = "txt"

    creator = make_component_run(
        store,
        parent.artifact_id,
        f"ocr-lineage-{damage}-run",
        component=component,
        configuration=configuration,
        outputs={output_name: derivative_blob.sha256},
        status=status,
    )
    store.save_processing_run(creator)
    derivative = DocumentArtifact(
        artifact_id=f"ocr-lineage-{damage}-derivative",
        source_sha256=derivative_blob.sha256,
        acquisition_uri=f"derived://ocr/{damage}",
        media_type="application/pdf",
        relationship=relationship,
        parent_artifact_id=parent.artifact_id,
        raw_location=derivative_blob.as_location(
            media_type="application/pdf",
            role=ArtifactLocationRole.RAW,
            created_by_run_id=creator.run_id,
        ),
    )
    store.save_artifact(derivative)

    with pytest.raises(RecordConflictError, match=message):
        store._verify_grobid_artifact_lineage(
            canonical_artifact=parent,
            grobid_artifact=derivative,
        )


@pytest.mark.parametrize("mode", ["container_cli", "local_cli"])
def test_grobid_derivative_accepts_exact_ocr_lineage(
    store: ContentAddressedStore,
    mode: str,
) -> None:
    parent = save_artifact(store, f"ocr-lineage-valid-{mode}")
    derivative_blob = store.put_blob(b"%PDF-1.7\nvalid searchable derivative")
    creator = make_component_run(
        store,
        parent.artifact_id,
        f"ocr-lineage-valid-{mode}-run",
        component=ComponentDescriptor(
            component_id="ocrmypdf",
            component_version="17.4.1",
            capability="document.ocr",
        ),
        configuration=ocr_production_configuration(parent, mode=mode),
        outputs={"searchable_pdf": derivative_blob.sha256},
    )
    store.save_processing_run(creator)
    derivative = DocumentArtifact(
        artifact_id=f"ocr-lineage-valid-{mode}-derivative",
        source_sha256=derivative_blob.sha256,
        acquisition_uri=f"derived://ocr/valid-{mode}",
        media_type="application/pdf",
        relationship=ArtifactRelationship.DERIVATIVE,
        parent_artifact_id=parent.artifact_id,
        raw_location=derivative_blob.as_location(
            media_type="application/pdf",
            role=ArtifactLocationRole.RAW,
            created_by_run_id=creator.run_id,
        ),
    )
    store.save_artifact(derivative)

    assert store._verify_grobid_artifact_lineage(
        canonical_artifact=parent,
        grobid_artifact=derivative,
    ) == (creator.require_output("searchable_pdf"),)


def test_grobid_derivative_accepts_production_ocr_runtime_attestation(
    store: ContentAddressedStore,
) -> None:
    parent = save_artifact(store, "ocr-lineage-valid-runtime-attestation")
    derivative_blob = store.put_blob(b"%PDF-1.7\nattested searchable derivative")
    run_id = "ocr-lineage-valid-runtime-attestation-run"
    attestation = ocr_runtime_attestation(run_id)
    configuration = ocr_production_configuration(parent)
    configuration["container_digest"] = attestation.container_digest
    creator = make_component_run(
        store,
        parent.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="ocrmypdf",
            component_version="17.4.1",
            capability="document.ocr",
        ),
        configuration=configuration,
        outputs={"searchable_pdf": derivative_blob.sha256},
        runtime_attestation=attestation,
        runtime_identity_required=True,
        output_policy_snapshot=runtime_output_policy(
            component_id="ocrmypdf",
            configuration=configuration,
            attestation=attestation,
            runtime_identity_required=True,
        ),
    )
    store.save_processing_run(creator)
    derivative = DocumentArtifact(
        artifact_id="ocr-lineage-valid-runtime-attestation-derivative",
        source_sha256=derivative_blob.sha256,
        acquisition_uri="derived://ocr/valid-runtime-attestation",
        media_type="application/pdf",
        relationship=ArtifactRelationship.DERIVATIVE,
        parent_artifact_id=parent.artifact_id,
        raw_location=derivative_blob.as_location(
            media_type="application/pdf",
            role=ArtifactLocationRole.RAW,
            created_by_run_id=creator.run_id,
        ),
    )
    store.save_artifact(derivative)

    assert store._verify_grobid_artifact_lineage(
        canonical_artifact=parent,
        grobid_artifact=derivative,
    ) == (creator.require_output("searchable_pdf"),)


@pytest.mark.parametrize(
    ("damage", "runtime_identity_required", "message"),
    [
        ("wrong-reporter", True, "persisted trust policy"),
        ("optional-wrong-reporter", False, "persisted trust policy"),
        ("missing-policy", True, "no persisted trust policy"),
        ("malformed-policy", True, "trust policy shape"),
        ("wrong-runner-contract", True, "OCR runtime trust policy values"),
        ("wrong-policy-digest", True, "persisted parser policy"),
    ],
)
def test_grobid_lineage_replays_transitive_ocr_runtime_trust(
    store: ContentAddressedStore,
    damage: str,
    runtime_identity_required: bool,
    message: str,
) -> None:
    parent = save_artifact(store, f"ocr-transitive-runtime-trust-{damage}")
    derivative_blob = store.put_blob(b"%PDF-1.7\ntransitively attested derivative")
    run_id = f"{parent.artifact_id}-run"
    attestation = ocr_runtime_attestation(
        run_id,
        reporter_id=(
            "wrong-runner"
            if damage in {"wrong-reporter", "optional-wrong-reporter"}
            else "deepcritical-container-ocr-runner-v1"
        ),
    )
    configuration = ocr_production_configuration(parent)
    configuration["container_digest"] = attestation.container_digest
    policy = runtime_output_policy(
        component_id="ocrmypdf",
        configuration=configuration,
        attestation=attestation,
        runtime_identity_required=runtime_identity_required,
    )
    trust_policies = policy["runtime_trust_policy"]
    document_policy = policy["document_processing_config"]
    assert isinstance(trust_policies, dict)
    assert isinstance(document_policy, dict)
    ocr_policy = trust_policies["ocrmypdf"]
    assert isinstance(ocr_policy, dict)
    if damage == "malformed-policy":
        ocr_policy.pop("local_digest_runner_version")
    elif damage == "wrong-runner-contract":
        ocr_policy["local_digest_runner_version"] = "foreign-runner"
    elif damage == "wrong-policy-digest":
        document_policy["ocr_container_digest"] = f"sha256:{'f' * 64}"
    output_policy = None if damage == "missing-policy" else policy
    creator = make_component_run(
        store,
        parent.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="ocrmypdf",
            component_version="17.4.1",
            capability="document.ocr",
        ),
        configuration=configuration,
        outputs={"searchable_pdf": derivative_blob.sha256},
        runtime_attestation=attestation,
        runtime_identity_required=runtime_identity_required,
        output_policy_snapshot=output_policy,
    )
    store.save_processing_run(creator)
    derivative = DocumentArtifact(
        artifact_id=f"{parent.artifact_id}-derivative",
        source_sha256=derivative_blob.sha256,
        acquisition_uri=f"derived://ocr/{damage}",
        media_type="application/pdf",
        relationship=ArtifactRelationship.DERIVATIVE,
        parent_artifact_id=parent.artifact_id,
        raw_location=derivative_blob.as_location(
            media_type="application/pdf",
            role=ArtifactLocationRole.RAW,
            created_by_run_id=creator.run_id,
        ),
    )
    store.save_artifact(derivative)

    with pytest.raises(RecordConflictError, match=message):
        store._verify_grobid_artifact_lineage(
            canonical_artifact=parent,
            grobid_artifact=derivative,
        )


@pytest.mark.parametrize(
    "component_versions",
    [
        {"ocrmypdf": "17.4.1"},
        {"ocrmypdf": "17.4.1", "tesseract": "5.3.4", "ghostscript": "10.0"},
        {"ocrmypdf": "99", "tesseract": "5.3.4"},
    ],
    ids=("missing-tesseract", "extra-component", "wrong-ocrmypdf"),
)
def test_grobid_derivative_rejects_unapproved_ocr_runtime_components(
    store: ContentAddressedStore,
    component_versions: dict[str, str],
) -> None:
    parent = save_artifact(
        store,
        f"ocr-lineage-runtime-components-{'-'.join(component_versions)}",
    )
    run_id = f"{parent.artifact_id}-run"
    attestation = ocr_runtime_attestation(
        run_id,
        component_versions=component_versions,
    )
    configuration = ocr_production_configuration(parent)
    configuration["container_digest"] = attestation.container_digest
    creator = make_component_run(
        store,
        parent.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="ocrmypdf",
            component_version="17.4.1",
            capability="document.ocr",
        ),
        configuration=configuration,
        outputs={"searchable_pdf": store.put_blob(b"%PDF attested").sha256},
        runtime_attestation=attestation,
        runtime_identity_required=True,
        output_policy_snapshot=runtime_output_policy(
            component_id="ocrmypdf",
            configuration=configuration,
            attestation=attestation,
            runtime_identity_required=True,
        ),
    )
    store.save_processing_run(creator)

    with pytest.raises(RecordConflictError, match="runtime attestation conflicts"):
        store._verify_ocr_configuration(creator, parent)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("attestation-source", "runtime attestation conflicts"),
        ("configuration-digest", "runtime attestation conflicts"),
        ("configuration-image", "container identity"),
        ("digest-appended-image", "container identity"),
    ],
)
def test_grobid_derivative_rejects_ocr_attestation_configuration_conflicts(
    store: ContentAddressedStore,
    damage: str,
    message: str,
) -> None:
    parent = save_artifact(store, f"ocr-lineage-runtime-conflict-{damage}")
    run_id = f"{parent.artifact_id}-run"
    attestation = ocr_runtime_attestation(
        run_id,
        source=(
            RuntimeAttestationSource.AUTHENTICATED_DEPLOYMENT_REPORTER
            if damage == "attestation-source"
            else RuntimeAttestationSource.DIGEST_ADDRESSED_OCI_INVOCATION
        ),
    )
    configuration = ocr_production_configuration(parent)
    configuration["container_digest"] = (
        f"sha256:{'d' * 64}"
        if damage == "configuration-digest"
        else attestation.container_digest
    )
    if damage == "configuration-image":
        configuration["container_image"] = "registry.example.test/ocrmypdf:v17.4.1"
    elif damage == "digest-appended-image":
        configuration["container_image"] = (
            f"jbarlow83/ocrmypdf:v17.4.1@{attestation.container_digest}"
        )
    creator = make_component_run(
        store,
        parent.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="ocrmypdf",
            component_version="17.4.1",
            capability="document.ocr",
        ),
        configuration=configuration,
        outputs={"searchable_pdf": store.put_blob(b"%PDF attested").sha256},
        runtime_attestation=attestation,
        runtime_identity_required=True,
        output_policy_snapshot=runtime_output_policy(
            component_id="ocrmypdf",
            configuration=configuration,
            attestation=attestation,
            runtime_identity_required=True,
        ),
    )
    store.save_processing_run(creator)

    with pytest.raises(RecordConflictError, match=message):
        store._verify_ocr_configuration(creator, parent)


def test_grobid_derivative_rejects_ocr_creator_with_unrelated_input(
    store: ContentAddressedStore,
) -> None:
    parent = save_artifact(store, "ocr-lineage-unrelated-creator-input")
    upstream_blob = store.put_blob(b"unrelated preprocessing output")
    upstream = make_component_run(
        store,
        parent.artifact_id,
        "ocr-lineage-unrelated-upstream-run",
        component=ComponentDescriptor(
            component_id="unrelated-preprocessor",
            component_version="1",
            capability="document.prepare",
        ),
        configuration={"version": "1"},
        outputs={"ocr_sidecar": upstream_blob.sha256},
    )
    store.save_processing_run(upstream)
    derivative_blob = store.put_blob(b"%PDF searchable")
    creator = make_component_run(
        store,
        parent.artifact_id,
        "ocr-lineage-unrelated-creator-input-run",
        component=ComponentDescriptor(
            component_id="ocrmypdf",
            component_version="17.4.1",
            capability="document.ocr",
        ),
        configuration=ocr_production_configuration(parent),
        inputs=(upstream.require_output("ocr_sidecar"),),
        outputs={"searchable_pdf": derivative_blob.sha256},
    )
    store.save_processing_run(creator)
    derivative = DocumentArtifact(
        artifact_id="ocr-lineage-unrelated-creator-input-derivative",
        source_sha256=derivative_blob.sha256,
        acquisition_uri="derived://ocr/unrelated-creator-input",
        media_type="application/pdf",
        relationship=ArtifactRelationship.DERIVATIVE,
        parent_artifact_id=parent.artifact_id,
        raw_location=derivative_blob.as_location(
            media_type="application/pdf",
            role=ArtifactLocationRole.RAW,
            created_by_run_id=creator.run_id,
        ),
    )
    store.save_artifact(derivative)

    with pytest.raises(RecordConflictError, match="inputs do not match"):
        store._verify_grobid_artifact_lineage(
            canonical_artifact=parent,
            grobid_artifact=derivative,
        )


def test_grobid_derivative_rejects_missing_creator_lineage(
    store: ContentAddressedStore,
) -> None:
    parent = save_artifact(store, "ocr-lineage-missing-creator")
    derivative_blob = store.put_blob(b"%PDF derivative without creator")
    derivative = DocumentArtifact(
        artifact_id="ocr-lineage-missing-creator-derivative",
        source_sha256=derivative_blob.sha256,
        acquisition_uri="derived://ocr/missing-creator",
        media_type="application/pdf",
        relationship=ArtifactRelationship.DERIVATIVE,
        parent_artifact_id=parent.artifact_id,
        raw_location=derivative_blob.as_location(
            media_type="application/pdf",
            role=ArtifactLocationRole.RAW,
        ),
    )
    store.save_artifact(derivative)

    with pytest.raises(RecordConflictError, match="no creator run"):
        store._verify_grobid_artifact_lineage(
            canonical_artifact=parent,
            grobid_artifact=derivative,
        )


def test_grobid_derivative_rejects_creator_owned_by_another_artifact(
    store: ContentAddressedStore,
) -> None:
    parent = save_artifact(store, "ocr-lineage-wrong-owner-parent")
    other = save_artifact(store, "ocr-lineage-wrong-owner-other")
    derivative_blob = store.put_blob(b"%PDF wrong-owner derivative")
    creator = make_component_run(
        store,
        other.artifact_id,
        "ocr-lineage-wrong-owner-run",
        component=ComponentDescriptor(
            component_id="ocrmypdf",
            component_version="17.4.1",
            capability="document.ocr",
        ),
        configuration=ocr_production_configuration(other),
        outputs={"searchable_pdf": derivative_blob.sha256},
    )
    store.save_processing_run(creator)
    derivative = DocumentArtifact(
        artifact_id="ocr-lineage-wrong-owner-derivative",
        source_sha256=derivative_blob.sha256,
        acquisition_uri="derived://ocr/wrong-owner",
        media_type="application/pdf",
        relationship=ArtifactRelationship.DERIVATIVE,
        parent_artifact_id=parent.artifact_id,
        raw_location=derivative_blob.as_location(
            media_type="application/pdf",
            role=ArtifactLocationRole.RAW,
            created_by_run_id=creator.run_id,
        ),
    )

    with pytest.raises(RecordConflictError, match="does not own"):
        store._verify_grobid_artifact_lineage(
            canonical_artifact=parent,
            grobid_artifact=derivative,
        )


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("digest", "digest is invalid"),
        ("model-inventory", "model inventory is invalid"),
        ("inventory-keys", "inventories do not match"),
        ("required-attestation", "lacks required runtime attestation"),
    ],
)
def test_container_verifier_rejects_unapproved_configuration(
    store: ContentAddressedStore,
    damage: str,
    message: str,
) -> None:
    artifact = save_artifact(store, f"container-contract-{damage}")
    configuration = grobid_production_configuration(artifact)
    if damage == "digest":
        configuration["container_digest"] = "not-an-oci-digest"
    elif damage == "model-inventory":
        configuration["model_versions"] = []
    elif damage == "inventory-keys":
        configuration["model_versions"] = {"layout": "1"}
    run = make_component_run(
        store,
        artifact.artifact_id,
        f"{artifact.artifact_id}-run",
        component=ComponentDescriptor(
            component_id="grobid",
            component_version="0.9.0",
            capability="document.parse.scholarly",
        ),
        configuration=configuration,
    )
    if damage == "required-attestation":
        run = run.model_copy(update={"runtime_identity_required": True})

    with pytest.raises(RecordConflictError, match=message):
        store._verify_container_configuration(
            run,
            purpose="GROBID",
            component_id="grobid",
            expected_image="deepcritical/grobid:0.9.0-full-p0-c2",
            expected_component_versions={"grobid": "0.9.0"},
        )


@pytest.mark.parametrize(
    "damage",
    ["attestation-source", "component-versions"],
)
def test_container_verifier_rejects_unapproved_attested_identity(
    store: ContentAddressedStore,
    damage: str,
) -> None:
    artifact = save_artifact(store, f"container-attestation-{damage}")
    run_id = f"{artifact.artifact_id}-run"
    attestation = docling_runtime_attestation(
        run_id,
        source=(
            RuntimeAttestationSource.DIGEST_ADDRESSED_OCI_INVOCATION
            if damage == "attestation-source"
            else RuntimeAttestationSource.AUTHENTICATED_DEPLOYMENT_REPORTER
        ),
        component_versions=(
            {"docling": "99", "docling_serve": "1.21.0"}
            if damage == "component-versions"
            else None
        ),
    )
    configuration = docling_production_configuration(artifact)
    configuration.update(
        {
            "container_digest": attestation.container_digest,
            "model_versions": attestation.model_versions,
            "model_hashes": attestation.model_hashes,
        }
    )
    run = make_component_run(
        store,
        artifact.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="docling",
            component_version="2.96.1",
            capability="document.parse",
        ),
        configuration=configuration,
        runtime_attestation=attestation,
        runtime_identity_required=True,
        output_policy_snapshot=runtime_output_policy(
            component_id="docling",
            configuration=configuration,
            attestation=attestation,
            runtime_identity_required=True,
        ),
    )
    store.save_processing_run(run)

    with pytest.raises(RecordConflictError, match=r"runtime attestation|versions"):
        store._verify_container_configuration(
            run,
            purpose="Docling",
            component_id="docling",
            expected_image="quay.io/docling-project/docling-serve-cpu:v1.21.0",
            expected_component_versions={
                "docling": "2.96.1",
                "docling_serve": "1.21.0",
            },
        )


@pytest.mark.parametrize(
    ("status", "accepted"),
    [
        (ProcessingRunStatus.COMPLETE, False),
        (ProcessingRunStatus.PARTIAL, True),
    ],
    ids=("complete", "partial"),
)
def test_grobid_required_model_identity_status_is_replayed(
    store: ContentAddressedStore,
    status: ProcessingRunStatus,
    accepted: bool,
) -> None:
    artifact = save_artifact(store, f"grobid-empty-model-identity-{status.value}")
    run_id = f"{artifact.artifact_id}-run"
    attestation = grobid_runtime_attestation(run_id).model_copy(
        update={"model_versions": {}, "model_hashes": {}}
    )
    configuration = grobid_production_configuration(artifact)
    configuration.update(
        {
            "container_digest": attestation.container_digest,
            "model_versions": {},
            "model_hashes": {},
        }
    )
    run = make_component_run(
        store,
        artifact.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="grobid",
            component_version="0.9.0",
            capability="document.parse.scholarly",
        ),
        configuration=configuration,
        status=status,
        runtime_attestation=attestation,
        runtime_identity_required=True,
        output_policy_snapshot=runtime_output_policy(
            component_id="grobid",
            configuration=configuration,
            attestation=attestation,
            runtime_identity_required=True,
        ),
    )

    def verify() -> None:
        store._verify_container_configuration(
            run,
            purpose="GROBID",
            component_id="grobid",
            expected_image="deepcritical/grobid:0.9.0-full-p0-c2",
            expected_component_versions={"grobid": "0.9.0"},
        )

    if accepted:
        verify()
    else:
        with pytest.raises(RecordConflictError, match="model identity evidence"):
            verify()


@pytest.mark.parametrize("matching_output_count", [1, 2])
def test_native_artifact_inputs_require_one_exact_creator_product(
    store: ContentAddressedStore,
    matching_output_count: int,
) -> None:
    parent = save_artifact(store, f"native-input-parent-{matching_output_count}")
    child_blob = store.put_blob(b"derived native bytes")
    output_names = (
        {"searchable_pdf": child_blob.sha256}
        if matching_output_count == 1
        else {
            "searchable_pdf": child_blob.sha256,
            "ocr_sidecar": child_blob.sha256,
        }
    )
    creator = make_component_run(
        store,
        parent.artifact_id,
        f"native-input-creator-{matching_output_count}",
        component=ComponentDescriptor(
            component_id="artifact-creator",
            component_version="1",
            capability="document.derive",
        ),
        configuration={"version": "1"},
        outputs=output_names,
    )
    store.save_processing_run(creator)
    child = DocumentArtifact(
        artifact_id=f"native-input-child-{matching_output_count}",
        source_sha256=child_blob.sha256,
        acquisition_uri=f"derived://native/{matching_output_count}",
        media_type="application/pdf",
        relationship=ArtifactRelationship.DERIVATIVE,
        parent_artifact_id=parent.artifact_id,
        raw_location=child_blob.as_location(
            media_type="application/pdf",
            role=ArtifactLocationRole.RAW,
            created_by_run_id=creator.run_id,
        ),
    )
    store.save_artifact(child)

    if matching_output_count == 1:
        assert store._native_artifact_inputs(child) == (
            creator.require_output("searchable_pdf"),
        )
    else:
        with pytest.raises(RecordConflictError, match="exactly one creator product"):
            store._native_artifact_inputs(child)


def test_storage_verifier_helpers_reject_invalid_shapes() -> None:
    with pytest.raises(RecordConflictError, match="must be a JSON object"):
        storage_module._strict_canonical_json_object(b"[]", purpose="test payload")
    assert storage_module._is_oci_sha256(None) is False
    with pytest.raises(RecordConflictError, match="exactly one"):
        storage_module._require_single_product({}, "docling_document")


def test_location_verification_rejects_a_mismatched_cas_uri(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "location-uri-mismatch")
    mismatched = artifact.raw_location.model_copy(
        update={"uri": f"cas://sha256/{'0' * 64}"}
    )

    with pytest.raises(HashMismatchError, match="location URI"):
        store._verify_location(mismatched)


def test_runtime_trust_evidence_defensive_guards(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "runtime-trust-defensive-guards")
    clean_run = make_component_run(
        store,
        artifact.artifact_id,
        "runtime-trust-no-attestation-run",
        component=ComponentDescriptor(
            component_id="docling",
            component_version="2.96.1",
            capability="document.parse",
        ),
        configuration=docling_production_configuration(artifact),
    )
    store._verify_runtime_trust_evidence(
        clean_run,
        component_id="docling",
        purpose="test Docling",
        trust_policy=None,
        document_config=None,
        expected_image="quay.io/docling-project/docling-serve-cpu:v1.21.0",
        expected_component_versions={
            "docling": "2.96.1",
            "docling_serve": "1.21.0",
        },
    )

    attestation = docling_runtime_attestation("runtime-trust-unbound-run")
    configuration = docling_production_configuration(artifact)
    configuration.update(
        {
            "container_digest": attestation.container_digest,
            "model_versions": dict(attestation.model_versions),
            "model_hashes": dict(attestation.model_hashes),
        }
    )
    attested_run = make_component_run(
        store,
        artifact.artifact_id,
        "runtime-trust-unbound-run",
        component=ComponentDescriptor(
            component_id="docling",
            component_version="2.96.1",
            capability="document.parse",
        ),
        configuration=configuration,
        runtime_attestation=attestation,
    )
    with pytest.raises(RecordConflictError, match="no persisted trust policy"):
        store._verify_runtime_trust_evidence(
            attested_run,
            component_id="docling",
            purpose="test Docling",
            trust_policy=None,
            document_config=None,
            expected_image="quay.io/docling-project/docling-serve-cpu:v1.21.0",
            expected_component_versions={
                "docling": "2.96.1",
                "docling_serve": "1.21.0",
            },
        )


def test_runtime_trust_evidence_rejects_invalid_ocr_component_inventory(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "runtime-trust-ocr-components")
    run_id = "runtime-trust-ocr-components-run"
    attestation = ocr_runtime_attestation(
        run_id,
        component_versions={"ocrmypdf": "17.4.1"},
    )
    configuration = ocr_production_configuration(artifact)
    configuration["container_digest"] = attestation.container_digest
    policy = runtime_output_policy(
        component_id="ocrmypdf",
        configuration=configuration,
        attestation=attestation,
        runtime_identity_required=True,
    )
    run = make_component_run(
        store,
        artifact.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="ocrmypdf",
            component_version="17.4.1",
            capability="document.ocr",
        ),
        configuration=configuration,
        runtime_attestation=attestation,
        runtime_identity_required=True,
        output_policy_snapshot=policy,
    )
    trust_policy = policy["runtime_trust_policy"]["ocrmypdf"]
    document_config = policy["document_processing_config"]
    assert isinstance(trust_policy, dict)
    assert isinstance(document_config, dict)

    with pytest.raises(RecordConflictError, match="component versions"):
        store._verify_runtime_trust_evidence(
            run,
            component_id="ocrmypdf",
            purpose="test OCR",
            trust_policy=trust_policy,
            document_config=document_config,
            expected_image="jbarlow83/ocrmypdf:v17.4.1",
            expected_component_versions={"ocrmypdf": "17.4.1"},
        )


@pytest.mark.parametrize(
    ("damage", "mutator"),
    [
        ("pretty", lambda payload: json.dumps(json.loads(payload), indent=2).encode()),
        (
            "duplicate-key",
            lambda payload: payload.replace(
                b'"name":"Canonical persistence"',
                b'"name":"forged","name":"Canonical persistence"',
                1,
            ),
        ),
        (
            "non-finite",
            lambda payload: payload.replace(b'"page_no":1', b'"page_no":NaN', 1),
        ),
    ],
)
def test_canonical_admission_rejects_ambiguous_docling_json(
    store: ContentAddressedStore,
    damage: str,
    mutator: Callable[[bytes], bytes],
) -> None:
    artifact = save_artifact(store, f"native-json-{damage}")
    view, _ = make_durable_canonical_view(
        store,
        artifact,
        docling_bytes_mutator=mutator,
    )

    with pytest.raises(RecordConflictError, match=r"strict JSON|serialized"):
        save_canonical_view_product(
            store,
            artifact,
            view,
            run_id=f"native-json-{damage}-view-run",
        )


@pytest.mark.parametrize("bypass_admission", [False, True], ids=("admission", "read"))
def test_partial_docling_reference_collision_is_never_canonical_evidence(
    store: ContentAddressedStore,
    bypass_admission: bool,
) -> None:
    artifact = save_artifact(
        store,
        f"native-reference-collision-{'read' if bypass_admission else 'admission'}",
    )
    view, source_run = make_colliding_docling_view(
        store,
        artifact,
        case_id="collision",
    )
    document = json.loads(
        store.read_blob(source_run.require_output("docling_document").blob_sha256)
    )
    issue_codes = {
        issue.code
        for issue in DoclingQualityValidator()
        .validate(
            document,
            require_pdf_geometry=True,
        )
        .issues
    }
    assert {
        "DUPLICATE_DOCLING_SELF_REF",
        "DOCLING_CANONICAL_REFERENCE_COLLISION",
    } <= issue_codes

    if bypass_admission:
        _, product = save_canonical_view_product(
            store,
            artifact,
            view,
            run_id="native-reference-collision-legacy-view-run",
            bypass_admission=True,
        )
        with pytest.raises(RecordConflictError, match="semantic validation errors"):
            store.read_canonical_document(product)
        return

    with pytest.raises(RecordConflictError, match="semantic validation errors"):
        save_canonical_view_product(
            store,
            artifact,
            view,
            run_id="native-reference-collision-admission-view-run",
        )


def test_complete_docling_producer_cannot_mask_semantic_errors(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "native-complete-semantic-error")
    view, _ = make_durable_canonical_view(
        store,
        artifact,
        document_mutator=lambda document: document.__setitem__("tables", "invalid"),
    )

    with pytest.raises(RecordConflictError, match="semantic validation errors"):
        save_canonical_view_product(
            store,
            artifact,
            view,
            run_id="native-complete-semantic-error-view-run",
        )


def test_complete_canonical_producer_cannot_mask_replayed_errors(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-complete-semantic-error")
    view, _ = make_durable_canonical_view(
        store,
        artifact,
        document_mutator=lambda document: document.__setitem__("body", "invalid"),
        source_status=ProcessingRunStatus.PARTIAL,
    )

    with pytest.raises(RecordConflictError, match="canonical producer masks"):
        save_canonical_view_product(
            store,
            artifact,
            view,
            run_id="canonical-complete-semantic-error-run",
        )

    _, product = save_canonical_view_product(
        store,
        artifact,
        view,
        run_id="canonical-partial-semantic-error-run",
        status=ProcessingRunStatus.PARTIAL,
    )
    assert store.read_canonical_document(product) == view


@pytest.mark.parametrize(
    ("integrity_status", "accepted"),
    [
        (ProcessingRunStatus.COMPLETE, False),
        (ProcessingRunStatus.PARTIAL, True),
    ],
)
def test_integrity_producer_status_must_match_replayed_issues(
    store: ContentAddressedStore,
    integrity_status: ProcessingRunStatus,
    accepted: bool,
) -> None:
    artifact = save_artifact(store, f"integrity-status-{integrity_status.value}")
    _, source_run = make_durable_canonical_view(
        store,
        artifact,
        document_mutator=add_uncaptioned_valid_table,
    )
    docling_product = source_run.require_output("docling_document")
    spans_product = source_run.require_output("content_spans")
    document = json.loads(store.read_blob(docling_product.blob_sha256))
    span_set = ContentSpanSet.model_validate_json(
        store.read_blob(spans_product.blob_sha256)
    )
    report = validate_content_integrity(document).to_dict()
    report_blob = store.put_blob(
        json.dumps(
            report,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    integrity_configuration = {
        "algorithm": "explicit-content-integrity-v1",
        "docling_document_sha256": docling_product.blob_sha256,
        "scholarly_alignment_sha256": None,
    }
    integrity_run = make_component_run(
        store,
        artifact.artifact_id,
        f"integrity-status-{integrity_status.value}-run",
        component=ComponentDescriptor(
            component_id="docling-content-integrity",
            component_version="1",
            capability="document.validate",
        ),
        configuration=integrity_configuration,
        inputs=(docling_product,),
        outputs={"content_integrity_overlay": report_blob.sha256},
        status=integrity_status,
    )
    store.save_processing_run(integrity_run)
    sources = (
        docling_product,
        spans_product,
        integrity_run.require_output("content_integrity_overlay"),
    )
    view = build_canonical_document_view(
        artifact=artifact,
        docling_document=document,
        docling_product=docling_product,
        content_span_set=span_set,
        source_products=sources,
        configuration=CanonicalizationConfig(),
        integrity_report=report,
    )

    if not accepted:
        with pytest.raises(RecordConflictError, match="integrity producer masks"):
            save_canonical_view_product(
                store,
                artifact,
                view,
                run_id="complete-integrity-mask-view-run",
                status=ProcessingRunStatus.PARTIAL,
            )
        return

    _, product = save_canonical_view_product(
        store,
        artifact,
        view,
        run_id="partial-integrity-issues-view-run",
        status=ProcessingRunStatus.PARTIAL,
    )
    assert store.read_canonical_document(product) == view


def test_canonical_read_rechecks_legacy_failed_native_producer(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-legacy-failed-native")
    view, source_run = make_durable_canonical_view(store, artifact)
    _, product = save_canonical_view_product(
        store,
        artifact,
        view,
        run_id="canonical-legacy-failed-native-view-run",
    )
    record_path = store._record_path("processing_runs", source_run.run_id)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["status"] = ProcessingRunStatus.FAILED.value
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RecordConflictError, match="status"):
        store.read_canonical_document(product)


def test_canonical_partial_producer_is_admitted_and_reproducible(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-partial")
    view, _ = make_durable_canonical_view(store, artifact)

    producer, product = save_canonical_view_product(
        store,
        artifact,
        view,
        run_id="canonical-partial-run",
        status=ProcessingRunStatus.PARTIAL,
    )

    assert producer.status is ProcessingRunStatus.PARTIAL
    assert store.read_canonical_document(product) == view


@pytest.mark.parametrize(
    "component",
    [
        ComponentDescriptor(
            component_id="docling",
            component_version=CANONICAL_COMPONENT_VERSION,
            capability=CANONICAL_COMPONENT_CAPABILITY,
        ),
        ComponentDescriptor(
            component_id=CANONICAL_COMPONENT_ID,
            component_version="2",
            capability=CANONICAL_COMPONENT_CAPABILITY,
        ),
        ComponentDescriptor(
            component_id=CANONICAL_COMPONENT_ID,
            component_version=CANONICAL_COMPONENT_VERSION,
            capability="document.parse",
        ),
    ],
    ids=("wrong-component", "wrong-version", "wrong-capability"),
)
def test_canonical_admission_requires_exact_producer_identity(
    store: ContentAddressedStore,
    component: ComponentDescriptor,
) -> None:
    artifact = save_artifact(store, f"canonical-{component.component_id}")
    view, _ = make_durable_canonical_view(store, artifact)

    with pytest.raises(RecordConflictError, match="producer identity"):
        save_canonical_view_product(
            store,
            artifact,
            view,
            component=component,
            run_id=f"canonical-{component.component_version}-identity-run",
        )


@pytest.mark.parametrize(
    "status",
    [ProcessingRunStatus.FAILED, ProcessingRunStatus.QUARANTINED],
)
def test_canonical_admission_rejects_unsuccessful_terminal_status(
    store: ContentAddressedStore,
    status: ProcessingRunStatus,
) -> None:
    artifact = save_artifact(store, f"canonical-{status.value}")
    view, _ = make_durable_canonical_view(store, artifact)

    with pytest.raises(RecordConflictError, match="complete or partial"):
        save_canonical_view_product(
            store,
            artifact,
            view,
            status=status,
            run_id=f"canonical-{status.value}-run",
        )


def test_canonical_read_rechecks_legacy_wrong_producer_identity(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-legacy-wrong-producer")
    view, _ = make_durable_canonical_view(store, artifact)
    wrong_component = ComponentDescriptor(
        component_id="docling",
        component_version=CANONICAL_COMPONENT_VERSION,
        capability=CANONICAL_COMPONENT_CAPABILITY,
    )
    _, product = save_canonical_view_product(
        store,
        artifact,
        view,
        component=wrong_component,
        run_id="canonical-legacy-wrong-producer-run",
        bypass_admission=True,
    )

    with pytest.raises(RecordConflictError, match="producer identity"):
        store.read_canonical_document(product)


def test_canonical_admission_rejects_configuration_drift(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-config-drift")
    view, _ = make_durable_canonical_view(store, artifact)
    configuration = canonical_invocation_configuration(
        CanonicalizationConfig(),
        view.source_products,
    )
    configuration["adapter_version"] = "drifted"

    with pytest.raises(RecordConflictError, match="configuration"):
        save_canonical_view_product(
            store,
            artifact,
            view,
            configuration=configuration,
            run_id="canonical-config-drift-run",
        )


def test_canonical_admission_rejects_nonexistent_native_node(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-nonexistent-node")
    view, _ = make_durable_canonical_view(store, artifact)

    def replace_node(payload: dict[str, Any]) -> None:
        block = payload["blocks"][0]
        block["native_node_id"] = "#/texts/404"
        next(
            anchor for anchor in block["source_anchors"] if anchor["role"] == "primary"
        )["node_id"] = "#/texts/404"
        block["block_id"] = canonical_block_id(
            native_node_id=block["native_node_id"],
            kind=CanonicalBlockKind(block["kind"]),
            content_sha256=block["content_sha256"],
        )
        payload["root_block_ids"] = [block["block_id"]]

    forged = mutate_canonical_view(view, replace_node)

    with pytest.raises(RecordConflictError, match="identity does not match"):
        save_canonical_view_product(
            store,
            artifact,
            forged,
            run_id="canonical-nonexistent-node-run",
        )


def test_canonical_admission_rejects_out_of_bounds_anchor(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-anchor-bounds")
    view, _ = make_durable_canonical_view(store, artifact)

    def expand_anchor(payload: dict[str, Any]) -> None:
        primary = next(
            anchor
            for anchor in payload["blocks"][0]["source_anchors"]
            if anchor["role"] == "primary"
        )
        primary["char_start"] = 0
        primary["char_end"] = 10_000

    forged = mutate_canonical_view(view, expand_anchor)

    with pytest.raises(RecordConflictError, match="identity does not match"):
        save_canonical_view_product(
            store,
            artifact,
            forged,
            run_id="canonical-anchor-bounds-run",
        )


def test_canonical_admission_rejects_metadata_drift(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-metadata-drift")
    view, _ = make_durable_canonical_view(store, artifact)

    def drift_metadata(payload: dict[str, Any]) -> None:
        payload["metadata"]["title"] = "Invented title"
        payload["metadata"]["identifiers"] = {"pmc": "PMC-INVENTED"}

    forged = mutate_canonical_view(view, drift_metadata)

    with pytest.raises(RecordConflictError, match="identity does not match"):
        save_canonical_view_product(
            store,
            artifact,
            forged,
            run_id="canonical-metadata-drift-run",
        )


def test_canonical_read_rebuilds_legacy_metadata_before_returning(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-legacy-metadata-drift")
    view, _ = make_durable_canonical_view(store, artifact)

    def drift_metadata(payload: dict[str, Any]) -> None:
        payload["metadata"]["title"] = "Legacy invented title"

    forged = mutate_canonical_view(view, drift_metadata)
    _, product = save_canonical_view_product(
        store,
        artifact,
        forged,
        run_id="canonical-legacy-metadata-drift-run",
        bypass_admission=True,
    )

    with pytest.raises(RecordConflictError, match="identity does not match"):
        store.read_canonical_document(product)


def test_canonical_admission_rejects_content_span_hash_drift(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-span-hash-drift")
    valid_view, valid_source_run = make_durable_canonical_view(store, artifact)
    old_docling = valid_source_run.require_output("docling_document")
    old_spans = valid_source_run.require_output("content_spans")
    invalid_run_id = "canonical-invalid-span-source-run"
    new_docling = store.data_product_ref(
        name="docling_document",
        blob_sha256=old_docling.blob_sha256,
        producer_run_id=invalid_run_id,
        source_artifact_ids=(artifact.artifact_id,),
    )
    span_payload = json.loads(store.read_blob(old_spans.blob_sha256))
    span_payload["processing_run_id"] = invalid_run_id
    span_payload["representation_product_id"] = new_docling.product_id
    for span in span_payload["spans"]:
        span["processing_run_id"] = invalid_run_id
        span["representation_anchor"]["product_id"] = new_docling.product_id
        span["content_sha256"] = "f" * 64
    invalid_spans = ContentSpanSet.model_validate(span_payload)
    invalid_spans_blob = store.put_blob(
        json.dumps(
            invalid_spans.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    invalid_source_run = make_component_run(
        store,
        artifact.artifact_id,
        invalid_run_id,
        component=ComponentDescriptor(
            component_id="docling",
            component_version="2.96.1",
            capability="document.parse",
        ),
        configuration=docling_production_configuration(artifact),
        outputs={
            "docling_document": old_docling.blob_sha256,
            "content_spans": invalid_spans_blob.sha256,
        },
    )
    store.save_processing_run(invalid_source_run)
    new_sources = (
        invalid_source_run.require_output("docling_document"),
        invalid_source_run.require_output("content_spans"),
    )

    def replace_sources(payload: dict[str, Any]) -> None:
        payload["source_products"] = [
            product.model_dump(mode="json") for product in new_sources
        ]
        for block in payload["blocks"]:
            for anchor in block["source_anchors"]:
                if anchor["product_id"] == old_docling.product_id:
                    anchor["product_id"] = new_sources[0].product_id

    forged = mutate_canonical_view(valid_view, replace_sources)

    with pytest.raises(RecordConflictError, match="content-span source"):
        save_canonical_view_product(
            store,
            artifact,
            forged,
            run_id="canonical-invalid-span-run",
        )


@pytest.mark.parametrize("damage", ["omitted", "locator-drift"])
def test_canonical_admission_rejects_irreproducible_content_span_sets(
    store: ContentAddressedStore,
    damage: str,
) -> None:
    artifact = save_artifact(store, f"canonical-admission-{damage}")
    forged = make_irreproducible_canonical_view(
        store,
        artifact,
        damage=damage,
    )

    with pytest.raises(RecordConflictError, match="content-span source"):
        save_canonical_view_product(
            store,
            artifact,
            forged,
            run_id=f"canonical-admission-{damage}-run",
        )


@pytest.mark.parametrize("damage", ["omitted", "locator-drift"])
def test_canonical_legacy_read_rejects_irreproducible_content_span_sets(
    store: ContentAddressedStore,
    damage: str,
) -> None:
    artifact = save_artifact(store, f"canonical-legacy-{damage}")
    forged = make_irreproducible_canonical_view(
        store,
        artifact,
        damage=damage,
    )
    _, product = save_canonical_view_product(
        store,
        artifact,
        forged,
        run_id=f"canonical-legacy-{damage}-run",
        bypass_admission=True,
    )

    with pytest.raises(RecordConflictError, match="content-span source"):
        store.read_canonical_document(product)


def test_canonical_admission_rejects_stale_scholarly_tei_evidence(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-stale-tei")
    _, source_run = make_durable_canonical_view(store, artifact)
    docling_product = source_run.require_output("docling_document")
    spans_product = source_run.require_output("content_spans")
    docling_document = json.loads(store.read_blob(docling_product.blob_sha256))
    span_set = ContentSpanSet.model_validate_json(
        store.read_blob(spans_product.blob_sha256)
    )
    durable_tei = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
    <p>Canonical content</p>
    </body></text></TEI>"""
    stale_tei = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
    <p>Unrelated stale evidence</p>
    </body></text></TEI>"""
    grobid_blob = store.put_blob(durable_tei)
    grobid_run = make_component_run(
        store,
        artifact.artifact_id,
        "canonical-stale-tei-grobid-run",
        component=ComponentDescriptor(
            component_id="grobid",
            component_version="0.9.0",
            capability="document.parse.scholarly",
        ),
        configuration=grobid_production_configuration(artifact),
        outputs={"grobid_tei": grobid_blob.sha256},
    )
    store.save_processing_run(grobid_run)
    grobid_product = grobid_run.require_output("grobid_tei")
    stale_overlay = DoclingGrobidAligner(minimum_score=0.72).align(
        docling_document,
        stale_tei,
    )
    alignment_blob = store.put_blob(
        json.dumps(
            stale_overlay.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    alignment_configuration = {
        "algorithm": "token-sequence-v2",
        "minimum_score": 0.72,
        "docling_document_sha256": docling_product.blob_sha256,
        "grobid_tei_sha256": grobid_product.blob_sha256,
    }
    alignment_run = make_component_run(
        store,
        artifact.artifact_id,
        "canonical-stale-tei-alignment-run",
        component=ComponentDescriptor(
            component_id="docling-grobid-aligner",
            component_version="2",
            capability="document.align",
        ),
        configuration=alignment_configuration,
        inputs=(docling_product, grobid_product),
        outputs={"alignment_overlay": alignment_blob.sha256},
    )
    store.save_processing_run(alignment_run)
    alignment_product = alignment_run.require_output("alignment_overlay")
    forged = build_canonical_document_view(
        artifact=artifact,
        docling_document=docling_document,
        docling_product=docling_product,
        content_span_set=span_set,
        source_products=(
            docling_product,
            spans_product,
            grobid_product,
            alignment_product,
        ),
        configuration=CanonicalizationConfig(),
        scholarly_overlay=stale_overlay,
    )

    with pytest.raises(RecordConflictError, match="does not reproduce"):
        save_canonical_view_product(
            store,
            artifact,
            forged,
            run_id="canonical-stale-tei-run",
        )


def test_canonical_read_requires_exact_producer_inputs(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-input-mismatch")
    view, _ = make_durable_canonical_view(store, artifact)
    _, product = save_canonical_view_product(
        store,
        artifact,
        view,
        inputs=tuple(reversed(view.source_products)),
        run_id="canonical-input-mismatch-run",
        bypass_admission=True,
    )

    with pytest.raises(RecordConflictError, match="source products do not exactly"):
        store.read_canonical_document(product)


def test_canonical_read_binds_view_to_producer_artifact(
    store: ContentAddressedStore,
) -> None:
    producer_artifact = save_artifact(store, "canonical-producer-artifact")
    view_artifact = save_artifact(store, "canonical-view-artifact")
    view, _ = make_durable_canonical_view(store, view_artifact)
    _, product = save_canonical_view_product(
        store,
        producer_artifact,
        view,
        run_id="canonical-wrong-artifact-run",
        bypass_admission=True,
    )

    with pytest.raises(RecordConflictError, match="artifact does not match"):
        store.read_canonical_document(product)


def test_canonical_read_binds_source_hash_to_producer_artifact(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "canonical-forged-source-hash")
    _, source_run = make_durable_canonical_view(store, artifact)
    forged_view = make_canonical_view(
        source_artifact_id=artifact.artifact_id,
        source_sha256="f" * 64,
        source_products=(
            source_run.require_output("docling_document"),
            source_run.require_output("content_spans"),
        ),
    )
    _, product = save_canonical_view_product(
        store,
        artifact,
        forged_view,
        run_id="canonical-forged-source-hash-run",
        bypass_admission=True,
    )

    with pytest.raises(RecordConflictError, match="source hash does not match"):
        store.read_canonical_document(product)


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_canonical_read_fully_verifies_every_embedded_source_product(
    store: ContentAddressedStore,
    damage: str,
) -> None:
    artifact = save_artifact(store, f"canonical-embedded-{damage}")
    view, _ = make_durable_canonical_view(store, artifact)
    _, product = save_canonical_view_product(
        store,
        artifact,
        view,
        run_id=f"canonical-embedded-{damage}-run",
    )
    embedded_path = store.blob_path(view.source_products[-1].blob_sha256)
    if damage == "corrupt":
        embedded_path.write_bytes(b"corrupt embedded source")
        expected_error: type[StorageError] = HashMismatchError
        message = "hashes to"
    else:
        embedded_path.unlink()
        expected_error = BlobNotFoundError
        message = "blob not found"

    with pytest.raises(expected_error, match=message):
        store.read_canonical_document(product)


def test_artifact_records_are_idempotent_and_conflicts_never_overwrite(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store)
    first_path = store.save_artifact(artifact)
    second_path = store.save_artifact(artifact)

    assert first_path == second_path
    assert store.get_artifact(artifact.artifact_id) == artifact

    conflicting = artifact.model_copy(
        update={"acquisition_uri": "https://different.test/source.pdf"}
    )
    with pytest.raises(RecordConflictError, match="different content"):
        store.save_artifact(conflicting)
    assert store.get_artifact(artifact.artifact_id) == artifact


def test_legacy_processing_run_without_stage_invocation_id_remains_idempotent(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "legacy-stage-invocation")
    run = make_run(
        store,
        artifact.artifact_id,
        "run-legacy-stage-invocation",
        ProcessingRunStatus.COMPLETE,
    )

    record_path = store.save_processing_run(run)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "deepcritical-processing-run-v1"
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    restored = store.get_processing_run(run.run_id)

    assert "stage_invocation_id" not in payload
    assert restored.schema_version == "deepcritical-processing-run-v2"
    assert restored.stage_invocation_id is None
    assert store.save_processing_run(restored) == record_path


def test_transitional_v1_processing_run_with_paired_invocation_ids_migrates(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "transitional-v1-invocation")
    run = make_run(
        store,
        artifact.artifact_id,
        "run-transitional-v1-invocation",
        ProcessingRunStatus.FAILED,
    ).model_copy(
        update={
            "pipeline_run_id": "pipeline-transitional-v1",
            "stage_invocation_id": "stage-invocation-transitional-v1",
        }
    )
    record_path = store.save_processing_run(run)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "deepcritical-processing-run-v1"
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = store.get_processing_run(run.run_id)

    assert restored == run
    assert restored.schema_version == "deepcritical-processing-run-v2"
    assert store.save_processing_run(restored) == record_path


def test_transitional_v1_processing_run_rejects_orphan_invocation_identity(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "transitional-v1-orphan")
    run = make_run(
        store,
        artifact.artifact_id,
        "run-transitional-v1-orphan",
        ProcessingRunStatus.FAILED,
    )
    record_path = store.save_processing_run(run)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "schema_version": "deepcritical-processing-run-v1",
            "stage_invocation_id": "stage-invocation-orphan",
        }
    )
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorruptRecordError, match="invalid record"):
        store.get_processing_run(run.run_id)


def test_unknown_processing_run_schema_is_rejected_before_validation(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "unknown-processing-run-schema")
    run = make_run(
        store,
        artifact.artifact_id,
        "run-unknown-processing-run-schema",
        ProcessingRunStatus.FAILED,
    )
    record_path = store.save_processing_run(run)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "deepcritical-processing-run-v99"
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnsupportedSchemaVersionError, match="v99"):
        store.get_processing_run(run.run_id)


@pytest.mark.parametrize("conflicting", [False, True])
def test_concurrent_v1_processing_run_publish_is_compared_after_migration(
    store: ContentAddressedStore,
    monkeypatch: pytest.MonkeyPatch,
    conflicting: bool,
) -> None:
    artifact = save_artifact(store, f"concurrent-v1-{conflicting}")
    run = make_run(
        store,
        artifact.artifact_id,
        f"run-concurrent-v1-{conflicting}",
        ProcessingRunStatus.FAILED,
    )

    def publish_v1_then_report_race(source: Path, destination: Path) -> None:
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["schema_version"] = "deepcritical-processing-run-v1"
        if conflicting:
            payload["status"] = ProcessingRunStatus.PARTIAL.value
        destination.write_text(json.dumps(payload), encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr(store, "_publish_no_replace", publish_v1_then_report_race)

    if conflicting:
        with pytest.raises(RecordConflictError, match="concurrent conflicting"):
            store.save_processing_run(run)
    else:
        record_path = store.save_processing_run(run)
        assert store.get_processing_run(run.run_id) == run
        assert record_path.is_file()


def test_record_loading_dispatches_schema_versions_before_validation(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "schema-dispatch")
    record_path = store.save_artifact(artifact)
    payload = json.loads(record_path.read_text(encoding="utf-8"))

    payload["schema_version"] = "deepcritical-document-artifact-v99"
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UnsupportedSchemaVersionError, match="v99"):
        store.get_artifact(artifact.artifact_id)

    payload.pop("schema_version")
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UnsupportedSchemaVersionError, match="None"):
        store.get_artifact(artifact.artifact_id)

    record_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CorruptRecordError, match="invalid record"):
        store.get_artifact(artifact.artifact_id)


def test_legacy_unversioned_processing_run_directory_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy-store"
    legacy = root / "records" / "parser_runs"
    legacy.mkdir(parents=True)
    (legacy / "prototype.json").write_text("{}", encoding="utf-8")

    with pytest.raises(UnsupportedSchemaVersionError, match="migrate or remove"):
        ContentAddressedStore(root)


def test_save_revalidates_model_copy_before_persisting(
    store: ContentAddressedStore,
) -> None:
    blob = store.put_blob(b"source document bytes")
    artifact = DocumentArtifact(
        artifact_id="artifact-invalid-copy",
        source_sha256=blob.sha256,
        acquisition_uri="https://example.test/invalid-copy.pdf",
        media_type="application/pdf",
        raw_location=blob.as_location(
            media_type="application/pdf", role=ArtifactLocationRole.RAW
        ),
    )
    invalid = artifact.model_copy(update={"source_sha256": "0" * 64})

    with pytest.raises(
        ValidationError, match=r"source_sha256 must match raw_location\.sha256"
    ):
        store.save_artifact(invalid)
    with pytest.raises(RecordNotFoundError, match="artifact-invalid-copy"):
        store.get_artifact(artifact.artifact_id)


def test_artifact_save_verifies_location_hash_size_and_parent(
    store: ContentAddressedStore,
) -> None:
    blob = store.put_blob(b"child")
    child = DocumentArtifact(
        artifact_id="supplement-1",
        source_sha256=blob.sha256,
        acquisition_uri="https://example.test/supplement-1.pdf",
        media_type="application/pdf",
        relationship=ArtifactRelationship.SUPPLEMENT,
        parent_artifact_id="missing-parent",
        raw_location=blob.as_location(
            media_type="application/pdf", role=ArtifactLocationRole.RAW
        ),
    )
    with pytest.raises(RecordNotFoundError, match="missing-parent"):
        store.save_artifact(child)

    save_artifact(store, "parent")
    wrong_size = child.model_copy(
        update={
            "parent_artifact_id": "parent",
            "raw_location": child.raw_location.model_copy(update={"byte_size": 999}),
        }
    )
    with pytest.raises(HashMismatchError, match="size"):
        store.save_artifact(wrong_size)


def test_derived_artifact_creator_lineage_requires_matching_durable_run(
    store: ContentAddressedStore,
) -> None:
    parent = save_artifact(store, "parent-lineage")
    derivative_blob = store.put_blob(b"searchable derivative")

    def derivative_for(run_id: str) -> DocumentArtifact:
        return DocumentArtifact(
            artifact_id=f"derivative-{run_id}",
            source_sha256=derivative_blob.sha256,
            acquisition_uri=f"derived://ocr/{run_id}",
            media_type="application/pdf",
            relationship=ArtifactRelationship.DERIVATIVE,
            parent_artifact_id=parent.artifact_id,
            raw_location=derivative_blob.as_location(
                media_type="application/pdf",
                role=ArtifactLocationRole.RAW,
                created_by_run_id=run_id,
            ),
        )

    with pytest.raises(RecordNotFoundError, match="run-missing-creator"):
        store.save_artifact(derivative_for("run-missing-creator"))

    unrelated_output = store.put_blob(b"different output")
    wrong_output_run = make_run(
        store,
        parent.artifact_id,
        "run-wrong-output",
        ProcessingRunStatus.COMPLETE,
        outputs={"searchable_pdf": unrelated_output.sha256},
    )
    store.save_processing_run(wrong_output_run)
    with pytest.raises(RecordConflictError, match="does not declare"):
        store.save_artifact(derivative_for(wrong_output_run.run_id))

    other_parent = save_artifact(store, "other-parent-lineage")
    wrong_parent_run = make_run(
        store,
        other_parent.artifact_id,
        "run-wrong-parent",
        ProcessingRunStatus.COMPLETE,
        outputs={"searchable_pdf": derivative_blob.sha256},
    )
    store.save_processing_run(wrong_parent_run)
    with pytest.raises(RecordConflictError, match="different artifact"):
        store.save_artifact(derivative_for(wrong_parent_run.run_id))

    correct_run = make_run(
        store,
        parent.artifact_id,
        "run-correct-output",
        ProcessingRunStatus.COMPLETE,
        outputs={"searchable_pdf": derivative_blob.sha256},
    )
    store.save_processing_run(correct_run)
    derivative = derivative_for(correct_run.run_id)

    store.save_artifact(derivative)

    assert store.get_artifact(derivative.artifact_id) == derivative


def test_processing_run_requires_artifact_and_verified_outputs(
    store: ContentAddressedStore,
) -> None:
    with pytest.raises(RecordNotFoundError, match="artifact-1"):
        store.save_processing_run(
            make_run(
                store,
                "artifact-1",
                "run-before-artifact",
                ProcessingRunStatus.FAILED,
            )
        )

    artifact = save_artifact(store)
    output = store.put_blob(b'{"docling": "document"}')
    run = make_run(
        store,
        artifact.artifact_id,
        "run-complete",
        ProcessingRunStatus.COMPLETE,
        outputs={"docling_document": output.sha256},
    )
    store.save_processing_run(run)

    assert store.get_processing_run(run.run_id) == run
    assert store.has_complete_run(artifact.artifact_id, "docling")

    missing_output = make_run(
        store,
        artifact.artifact_id,
        "run-missing-output",
        ProcessingRunStatus.PARTIAL,
        outputs={"grobid_tei": "e" * 64},
    )
    with pytest.raises(BlobNotFoundError, match="blob not found"):
        store.save_processing_run(missing_output)


def test_save_revalidates_mutated_nested_configuration(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store)
    run = make_run(
        store,
        artifact.artifact_id,
        "run-mutated-configuration",
        ProcessingRunStatus.PARTIAL,
    )
    run.configuration["do_ocr"] = False

    with pytest.raises(
        ValidationError, match="configuration_sha256 does not match configuration"
    ):
        store.save_processing_run(run)
    with pytest.raises(RecordNotFoundError, match="run-mutated-configuration"):
        store.get_processing_run(run.run_id)


def test_run_queries_make_interrupted_work_resumable(
    store: ContentAddressedStore,
) -> None:
    artifact_ids = ("never-run", "partial", "failed", "complete", "quarantined")
    for artifact_id in artifact_ids:
        save_artifact(store, artifact_id)

    statuses = {
        "partial": ProcessingRunStatus.PARTIAL,
        "failed": ProcessingRunStatus.FAILED,
        "complete": ProcessingRunStatus.COMPLETE,
        "quarantined": ProcessingRunStatus.QUARANTINED,
    }
    for minute, (artifact_id, status) in enumerate(statuses.items()):
        store.save_processing_run(
            make_run(
                store,
                artifact_id,
                f"run-{artifact_id}",
                status,
                minute=minute,
            )
        )

    candidates = store.list_resume_candidates("docling")
    assert {artifact.artifact_id for artifact in candidates} == {
        "never-run",
        "partial",
        "failed",
    }
    assert store.latest_processing_run("partial", "docling") is not None
    assert len(store.list_processing_runs(status=ProcessingRunStatus.FAILED)) == 1


def test_run_queries_can_isolate_output_policies(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store, "policy-filter")
    policy_a: dict[str, object] = {"pipeline": "a"}
    policy_b: dict[str, object] = {"pipeline": "b"}
    policy_c: dict[str, object] = {"pipeline": "c"}
    policy_a_hash = configuration_sha256(policy_a)
    policy_b_hash = configuration_sha256(policy_b)
    policy_c_hash = configuration_sha256(policy_c)
    complete_a = make_run(
        store,
        artifact.artifact_id,
        "run-policy-a",
        ProcessingRunStatus.COMPLETE,
        output_policy_snapshot=policy_a,
    )
    failed_b = make_run(
        store,
        artifact.artifact_id,
        "run-policy-b",
        ProcessingRunStatus.FAILED,
        minute=1,
        output_policy_snapshot=policy_b,
    )
    legacy_partial = make_run(
        store,
        artifact.artifact_id,
        "run-without-policy",
        ProcessingRunStatus.PARTIAL,
        minute=2,
    )
    for run in (complete_a, failed_b, legacy_partial):
        store.save_processing_run(run)

    assert store.list_processing_runs(
        artifact_id=artifact.artifact_id,
        output_policy_sha256=policy_a_hash,
    ) == (complete_a,)
    assert (
        store.latest_processing_run(
            artifact.artifact_id,
            "docling",
            output_policy_sha256=policy_a_hash,
        )
        == complete_a
    )
    assert store.has_complete_run(
        artifact.artifact_id,
        "docling",
        output_policy_sha256=policy_a_hash,
    )
    assert not store.has_complete_run(
        artifact.artifact_id,
        "docling",
        output_policy_sha256=policy_b_hash,
    )
    assert (
        store.list_resume_candidates(
            "docling",
            output_policy_sha256=policy_a_hash,
        )
        == ()
    )
    assert store.list_resume_candidates(
        "docling",
        output_policy_sha256=policy_b_hash,
    ) == (artifact,)
    assert store.list_resume_candidates(
        "docling",
        output_policy_sha256=policy_c_hash,
    ) == (artifact,)
    assert (
        store.latest_processing_run(artifact.artifact_id, "docling") == legacy_partial
    )


def test_run_query_output_policy_filters_validate_hashes(
    store: ContentAddressedStore,
) -> None:
    invalid_hash = "not-a-sha256"

    with pytest.raises(ValueError, match="SHA-256 hashes"):
        store.list_processing_runs(output_policy_sha256=invalid_hash)
    with pytest.raises(ValueError, match="SHA-256 hashes"):
        store.latest_processing_run(
            "artifact",
            "docling",
            output_policy_sha256=invalid_hash,
        )
    with pytest.raises(ValueError, match="SHA-256 hashes"):
        store.has_complete_run(
            "artifact",
            "docling",
            output_policy_sha256=invalid_hash,
        )
    with pytest.raises(ValueError, match="SHA-256 hashes"):
        store.list_resume_candidates(
            "docling",
            output_policy_sha256=invalid_hash,
        )


def test_checkpoint_deletion_fsyncs_only_after_actual_deletion(
    store: ContentAddressedStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = save_artifact(store, "checkpoint-durability")
    checkpoint = ExecutionCheckpoint(
        checkpoint_id="durable-checkpoint",
        artifact_id=artifact.artifact_id,
        component_id="docling",
        configuration_sha256=configuration_sha256({"do_ocr": True}),
        output_policy_sha256=configuration_sha256({"pipeline": "durable"}),
        remote_task_id="remote-task-1",
    )
    store.save_execution_checkpoint(checkpoint)
    fsynced_directories: list[Path] = []
    monkeypatch.setattr(
        storage_module,
        "_fsync_directory",
        lambda directory: fsynced_directories.append(Path(directory)),
    )

    store.delete_execution_checkpoint(checkpoint.checkpoint_id)

    assert fsynced_directories == [store.root / "records" / "execution_checkpoints"]
    with pytest.raises(RecordNotFoundError, match="durable-checkpoint"):
        store.get_execution_checkpoint(checkpoint.checkpoint_id)

    store.delete_execution_checkpoint(checkpoint.checkpoint_id)
    assert fsynced_directories == [store.root / "records" / "execution_checkpoints"]


def test_newer_failed_run_remains_resumable_after_an_older_complete_run(
    store: ContentAddressedStore,
) -> None:
    artifact = save_artifact(store)
    store.save_processing_run(
        make_run(
            store,
            artifact.artifact_id,
            "run-old-complete",
            ProcessingRunStatus.COMPLETE,
        )
    )
    store.save_processing_run(
        make_run(
            store,
            artifact.artifact_id,
            "run-new-failed",
            ProcessingRunStatus.FAILED,
            minute=1,
        )
    )

    assert store.list_resume_candidates("docling") == (artifact,)
    assert store.has_complete_run(artifact.artifact_id, "docling")


def test_diagnostics_validate_run_lineage_and_are_queryable(
    store: ContentAddressedStore,
) -> None:
    first_artifact = save_artifact(store, "artifact-1")
    second_artifact = save_artifact(store, "artifact-2")
    run = make_run(
        store,
        first_artifact.artifact_id,
        "run-partial",
        ProcessingRunStatus.PARTIAL,
    )
    store.save_processing_run(run)

    wrong_artifact = ProcessingDiagnostic(
        diagnostic_id="diagnostic-wrong",
        artifact_id=second_artifact.artifact_id,
        processing_run_id=run.run_id,
        severity=DiagnosticSeverity.ERROR,
        stage="alignment",
        code="ALIGNMENT.UNRESOLVED_CITATION",
        message="Citation did not resolve.",
    )
    with pytest.raises(RecordConflictError, match="does not match"):
        store.save_diagnostic(wrong_artifact)

    diagnostic = wrong_artifact.model_copy(
        update={
            "diagnostic_id": "diagnostic-1",
            "artifact_id": first_artifact.artifact_id,
        }
    )
    store.save_diagnostic(diagnostic)

    assert store.get_diagnostic(diagnostic.diagnostic_id) == diagnostic
    assert store.list_diagnostics(processing_run_id=run.run_id) == (diagnostic,)
