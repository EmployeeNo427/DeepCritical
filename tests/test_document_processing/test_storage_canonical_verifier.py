"""Adversarial branch coverage for canonical storage admission replay."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from DeepResearch.src.document_processing import storage as storage_module
from DeepResearch.src.document_processing.alignment import DoclingGrobidAligner
from DeepResearch.src.document_processing.canonical import (
    CanonicalDocumentView,
    CanonicalizationConfig,
    build_canonical_document_view,
    canonical_document_bytes,
    canonical_invocation_configuration,
)
from DeepResearch.src.document_processing.models import (
    ComponentDescriptor,
    ContentSpanSet,
    DataProductRef,
    DocumentArtifact,
    ProcessingRun,
    ProcessingRunStatus,
    configuration_sha256,
)
from DeepResearch.src.document_processing.storage import (
    ContentAddressedStore,
    RecordConflictError,
)
from DeepResearch.src.document_processing.validation import validate_content_integrity
from tests.test_document_processing.test_storage import (
    docling_production_configuration,
    grobid_production_configuration,
    make_component_run,
    make_durable_canonical_view,
    mutate_canonical_view,
    save_artifact,
    save_canonical_view_product,
)


def _json_bytes(payload: object, *, indent: int | None = None) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    ).encode()


def _replace_view_sources(
    view: CanonicalDocumentView,
    *,
    old_docling: DataProductRef,
    sources: tuple[DataProductRef, ...],
) -> CanonicalDocumentView:
    def replace(payload: dict[str, Any]) -> None:
        payload["source_products"] = [
            product.model_dump(mode="json") for product in sources
        ]
        for block in payload["blocks"]:
            for anchor in block["source_anchors"]:
                if anchor["product_id"] == old_docling.product_id:
                    anchor["product_id"] = sources[0].product_id

    return mutate_canonical_view(view, replace)


def _clone_docling_sources(
    store: ContentAddressedStore,
    artifact: DocumentArtifact,
    *,
    case_id: str,
    span_payload_mutator: Callable[[dict[str, Any]], None] | None = None,
    pretty_spans: bool = False,
    extra_outputs: dict[str, bytes] | None = None,
) -> tuple[CanonicalDocumentView, ProcessingRun]:
    base_view, base_run = make_durable_canonical_view(store, artifact)
    old_docling = base_run.require_output("docling_document")
    old_spans = base_run.require_output("content_spans")
    run_id = f"{artifact.artifact_id}-{case_id}-docling-run"
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
    if span_payload_mutator is not None:
        span_payload_mutator(span_payload)
    span_blob = store.put_blob(
        _json_bytes(span_payload, indent=2 if pretty_spans else None)
    )
    output_blobs = {
        "docling_document": old_docling.blob_sha256,
        "content_spans": span_blob.sha256,
    }
    for name, content in (extra_outputs or {}).items():
        output_blobs[name] = store.put_blob(content).sha256
    source_run = make_component_run(
        store,
        artifact.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="docling",
            component_version="2.113.0",
            capability="document.parse",
        ),
        configuration=docling_production_configuration(artifact),
        outputs=output_blobs,
    )
    store.save_processing_run(source_run)
    sources = (
        source_run.require_output("docling_document"),
        source_run.require_output("content_spans"),
    )
    return (
        _replace_view_sources(base_view, old_docling=old_docling, sources=sources),
        source_run,
    )


def _minimal_case(
    store: ContentAddressedStore,
    case_id: str,
) -> tuple[DocumentArtifact, CanonicalDocumentView, ProcessingRun, DataProductRef]:
    artifact = save_artifact(store, case_id)
    view, _ = make_durable_canonical_view(store, artifact)
    producer, product = save_canonical_view_product(
        store,
        artifact,
        view,
        run_id=f"{case_id}-canonical-run",
        bypass_admission=True,
    )
    return artifact, view, producer, product


def test_canonical_verifier_rejects_undeclared_output(tmp_path) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    _, _, producer, product = _minimal_case(store, "canonical-undeclared-output")

    with pytest.raises(RecordConflictError, match="not declared"):
        store._verify_canonical_product(
            product,
            producer.model_copy(update={"outputs": ()}),
        )


def test_canonical_verifier_rejects_invalid_policy_payload(tmp_path) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    _, _, producer, product = _minimal_case(store, "canonical-invalid-policy")
    invalid_configuration = dict(producer.configuration)
    invalid_configuration["policy"] = {"unexpected": True}

    with pytest.raises(RecordConflictError, match="policy is invalid"):
        store._verify_canonical_product(
            product,
            producer.model_copy(
                update={
                    "configuration": invalid_configuration,
                    "configuration_sha256": configuration_sha256(invalid_configuration),
                }
            ),
        )


@pytest.mark.parametrize(
    ("damage", "mutator", "pretty", "message"),
    [
        (
            "invalid-model",
            lambda payload: payload.__setitem__("spans", "not-a-list"),
            False,
            "content-span source is invalid",
        ),
        (
            "pretty-json",
            None,
            True,
            "not deterministically serialized",
        ),
    ],
)
def test_canonical_verifier_rejects_invalid_or_ambiguous_span_payload(
    tmp_path,
    damage: str,
    mutator: Callable[[dict[str, Any]], None] | None,
    pretty: bool,
    message: str,
) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    artifact = save_artifact(store, f"canonical-span-{damage}")
    view, _ = _clone_docling_sources(
        store,
        artifact,
        case_id=damage,
        span_payload_mutator=mutator,
        pretty_spans=pretty,
    )

    with pytest.raises(RecordConflictError, match=message):
        save_canonical_view_product(
            store,
            artifact,
            view,
            run_id=f"canonical-span-{damage}-run",
        )


def test_canonical_verifier_checks_optional_native_outputs_before_replay(
    tmp_path,
) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    artifact = save_artifact(store, "canonical-unexpected-native-outputs")
    view, _ = _clone_docling_sources(
        store,
        artifact,
        case_id="unexpected-native-outputs",
        extra_outputs={
            "native_locator_overlay": b"[]",
            "jats_locator_alignment": b"{}",
        },
    )

    with pytest.raises(RecordConflictError, match="does not reproduce"):
        save_canonical_view_product(
            store,
            artifact,
            view,
            run_id="canonical-unexpected-native-outputs-run",
        )


def _scholarly_case(
    store: ContentAddressedStore,
    case_id: str,
    *,
    alignment_damage: str | None = None,
) -> tuple[ProcessingRun, DataProductRef]:
    artifact = save_artifact(store, f"scholarly-{case_id}")
    _, source_run = make_durable_canonical_view(store, artifact)
    docling_product = source_run.require_output("docling_document")
    spans_product = source_run.require_output("content_spans")
    document = json.loads(store.read_blob(docling_product.blob_sha256))
    span_set = ContentSpanSet.model_validate_json(
        store.read_blob(spans_product.blob_sha256)
    )
    tei = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
    <p>Canonical content</p></body></text></TEI>"""
    grobid_blob = store.put_blob(tei)
    grobid_run = make_component_run(
        store,
        artifact.artifact_id,
        f"scholarly-{case_id}-grobid-run",
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
    overlay = DoclingGrobidAligner(minimum_score=0.72).align(document, tei)
    alignment_payload = overlay.to_dict()
    if alignment_damage == "invalid-json":
        alignment_bytes = b"{"
    elif alignment_damage == "non-object":
        alignment_bytes = b"[]"
    elif alignment_damage == "pretty-json":
        alignment_bytes = _json_bytes(alignment_payload, indent=2)
    else:
        alignment_bytes = _json_bytes(alignment_payload)
    alignment_blob = store.put_blob(alignment_bytes)
    alignment_configuration: dict[str, object] = {
        "algorithm": "token-sequence-v2",
        "minimum_score": 0.72,
        "docling_document_sha256": docling_product.blob_sha256,
        "grobid_tei_sha256": grobid_product.blob_sha256,
    }
    alignment_inputs = (docling_product, grobid_product)
    alignment_artifact_id = artifact.artifact_id
    if alignment_damage == "wrong-inputs":
        alignment_inputs = tuple(reversed(alignment_inputs))
    elif alignment_damage == "wrong-artifact":
        alignment_artifact_id = "unrelated-artifact"
    elif alignment_damage == "invalid-score":
        alignment_configuration["minimum_score"] = True
    elif alignment_damage == "configuration-drift":
        alignment_configuration["algorithm"] = "drifted"
    alignment_run = make_component_run(
        store,
        alignment_artifact_id,
        f"scholarly-{case_id}-alignment-run",
        component=ComponentDescriptor(
            component_id="docling-grobid-aligner",
            component_version="2",
            capability="document.align",
        ),
        configuration=alignment_configuration,
        inputs=alignment_inputs,
        outputs={"alignment_overlay": alignment_blob.sha256},
    )
    store._save_record("processing_runs", alignment_run.run_id, alignment_run)
    alignment_product = alignment_run.require_output("alignment_overlay")
    sources = (
        docling_product,
        spans_product,
        grobid_product,
        alignment_product,
    )
    view = build_canonical_document_view(
        artifact=artifact,
        docling_document=document,
        docling_product=docling_product,
        content_span_set=span_set,
        source_products=sources,
        configuration=CanonicalizationConfig(),
        scholarly_overlay=overlay,
    )
    producer, product = save_canonical_view_product(
        store,
        artifact,
        view,
        inputs=sources,
        run_id=f"scholarly-{case_id}-canonical-run",
        bypass_admission=True,
    )
    return producer, product


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("wrong-inputs", "alignment inputs"),
        ("wrong-artifact", "does not own"),
        ("invalid-score", "minimum score is invalid"),
        ("configuration-drift", "configuration does not match"),
        ("invalid-json", "not valid JSON"),
        ("non-object", "must be a JSON object"),
        ("pretty-json", "not deterministically serialized"),
    ],
)
def test_canonical_verifier_rejects_invalid_scholarly_producer_or_bytes(
    tmp_path,
    damage: str,
    message: str,
) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    producer, product = _scholarly_case(
        store,
        damage,
        alignment_damage=damage,
    )

    with pytest.raises(RecordConflictError, match=message):
        store._verify_canonical_product(product, producer)


def test_canonical_verifier_accepts_reproduced_scholarly_sources(tmp_path) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    producer, product = _scholarly_case(store, "valid")

    assert store._verify_canonical_product(product, producer).source_products == (
        producer.inputs
    )


def _integrity_case(
    store: ContentAddressedStore,
    case_id: str,
    *,
    damage: str | None = None,
) -> tuple[ProcessingRun, DataProductRef]:
    artifact = save_artifact(store, f"integrity-{case_id}")
    view, source_run = make_durable_canonical_view(
        store,
        artifact,
        document_mutator=(
            (lambda document: document.__setitem__("tables", "invalid"))
            if damage == "complete-with-issues"
            else None
        ),
        source_status=(
            ProcessingRunStatus.PARTIAL
            if damage == "complete-with-issues"
            else ProcessingRunStatus.COMPLETE
        ),
    )
    docling_product = source_run.require_output("docling_document")
    spans_product = source_run.require_output("content_spans")
    document = json.loads(store.read_blob(docling_product.blob_sha256))
    report = validate_content_integrity(document).to_dict()
    integrity_bytes = b"{}" if damage == "wrong-bytes" else _json_bytes(report)
    integrity_blob = store.put_blob(integrity_bytes)
    integrity_configuration: dict[str, object] = {
        "algorithm": "explicit-content-integrity-v1",
        "docling_document_sha256": docling_product.blob_sha256,
        "scholarly_alignment_sha256": None,
    }
    integrity_inputs = (docling_product,)
    integrity_artifact_id = artifact.artifact_id
    if damage == "wrong-inputs":
        integrity_inputs = ()
    elif damage == "wrong-artifact":
        integrity_artifact_id = "unrelated-artifact"
    elif damage == "configuration-drift":
        integrity_configuration["algorithm"] = "drifted"
    integrity_run = make_component_run(
        store,
        integrity_artifact_id,
        f"integrity-{case_id}-report-run",
        component=ComponentDescriptor(
            component_id="docling-content-integrity",
            component_version="1",
            capability="document.validate",
        ),
        configuration=integrity_configuration,
        inputs=integrity_inputs,
        outputs={"content_integrity_overlay": integrity_blob.sha256},
        status=ProcessingRunStatus.COMPLETE,
    )
    store._save_record("processing_runs", integrity_run.run_id, integrity_run)
    integrity_product = integrity_run.require_output("content_integrity_overlay")
    sources = (docling_product, spans_product, integrity_product)
    view = build_canonical_document_view(
        artifact=artifact,
        docling_document=document,
        docling_product=docling_product,
        content_span_set=ContentSpanSet.model_validate_json(
            store.read_blob(spans_product.blob_sha256)
        ),
        source_products=sources,
        configuration=CanonicalizationConfig(),
        integrity_report=report,
    )
    producer, product = save_canonical_view_product(
        store,
        artifact,
        view,
        inputs=sources,
        run_id=f"integrity-{case_id}-canonical-run",
        bypass_admission=True,
    )
    return producer, product


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("wrong-inputs", "integrity inputs"),
        ("wrong-artifact", "does not own"),
        ("configuration-drift", "configuration does not match"),
        ("complete-with-issues", "masks replayed issues"),
        ("wrong-bytes", "does not reproduce"),
    ],
)
def test_canonical_verifier_rejects_invalid_integrity_producer_or_bytes(
    tmp_path,
    damage: str,
    message: str,
) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    producer, product = _integrity_case(store, damage, damage=damage)

    with pytest.raises(RecordConflictError, match=message):
        store._verify_canonical_product(product, producer)


def test_canonical_verifier_wraps_rebuild_errors(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    _, _, producer, product = _minimal_case(store, "canonical-rebuild-error")

    def reject_rebuild(**_: object) -> CanonicalDocumentView:
        raise ValueError("synthetic rebuild rejection")

    monkeypatch.setattr(
        storage_module,
        "build_canonical_document_view",
        reject_rebuild,
    )

    with pytest.raises(RecordConflictError, match="cannot be rebuilt"):
        store._verify_canonical_product(product, producer)


def test_canonical_verifier_rejects_noncanonical_stored_bytes(tmp_path) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    artifact, view, producer, _ = _minimal_case(
        store,
        "canonical-pretty-stored-view",
    )
    pretty_blob = store.put_blob(
        _json_bytes(json.loads(canonical_document_bytes(view)), indent=2)
    )
    run_id = "canonical-pretty-stored-view-run"
    product = store.data_product_ref(
        name="canonical_document_view",
        blob_sha256=pretty_blob.sha256,
        producer_run_id=run_id,
        source_artifact_ids=(artifact.artifact_id,),
    )
    configuration = canonical_invocation_configuration(
        CanonicalizationConfig(),
        producer.inputs,
    )
    pretty_producer = producer.model_copy(
        update={
            "run_id": run_id,
            "configuration": configuration,
            "configuration_sha256": configuration_sha256(configuration),
            "outputs": (product,),
        }
    )

    with pytest.raises(RecordConflictError, match="bytes do not match"):
        store._verify_canonical_product(product, pretty_producer)
