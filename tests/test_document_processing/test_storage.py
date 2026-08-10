"""Tests for immutable content-addressed document-processing storage."""

import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from DeepResearch.src.document_processing import storage as storage_module
from DeepResearch.src.document_processing.canonical import (
    CANONICAL_ANCHORING_POLICY,
    CANONICAL_TEXT_NORMALIZATION,
    CanonicalAnchorRole,
    CanonicalBlock,
    CanonicalBlockKind,
    CanonicalDocumentMetadata,
    CanonicalDocumentView,
    CanonicalSourceAnchor,
    canonical_block_content_sha256,
    canonical_block_id,
    canonical_view_id,
)
from DeepResearch.src.document_processing.models import (
    ArtifactLocationRole,
    ArtifactRelationship,
    ComponentDescriptor,
    DataProductRef,
    DiagnosticSeverity,
    DocumentArtifact,
    ExecutionCheckpoint,
    ProcessingDiagnostic,
    ProcessingRun,
    ProcessingRunStatus,
    configuration_sha256,
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
    blob = store.put_blob(b"source document bytes")
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
            component_version="2.113.0",
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
) -> tuple[CanonicalDocumentView, ProcessingRun]:
    """Persist both source products used by one canonical test view."""

    docling_blob = store.put_blob(b'{"docling": "canonical source"}')
    spans_blob = store.put_blob(b'{"spans": []}')
    source_run = make_run(
        store,
        artifact.artifact_id,
        f"{artifact.artifact_id}-canonical-source-run",
        ProcessingRunStatus.COMPLETE,
        outputs={
            "docling_document": docling_blob.sha256,
            "content_spans": spans_blob.sha256,
        },
    )
    store.save_processing_run(source_run)
    source_products = (
        source_run.require_output("docling_document"),
        source_run.require_output("content_spans"),
    )
    return (
        make_canonical_view(
            source_artifact_id=artifact.artifact_id,
            source_sha256=artifact.source_sha256,
            source_products=source_products,
        ),
        source_run,
    )


def save_canonical_view_product(
    store: ContentAddressedStore,
    artifact: DocumentArtifact,
    view: CanonicalDocumentView,
    *,
    inputs: tuple[DataProductRef, ...] | None = None,
    run_id: str = "canonical-view-run",
) -> tuple[ProcessingRun, DataProductRef]:
    """Persist a canonical view and its producer record with durable inputs."""

    blob = store.put_canonical_document(view)
    producer = make_run(
        store,
        artifact.artifact_id,
        run_id,
        ProcessingRunStatus.COMPLETE,
        inputs=view.source_products if inputs is None else inputs,
        outputs={"canonical_document_view": blob.sha256},
    )
    store.save_processing_run(producer)
    return producer, producer.require_output("canonical_document_view")


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
