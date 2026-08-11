"""Adversarial branch coverage for canonical native-source admission."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from DeepResearch.src.document_processing.adapters import (
    BioCAdapter,
    JATSLocatorAdapter,
)
from DeepResearch.src.document_processing.models import (
    ArtifactLocationRole,
    ArtifactRelationship,
    ComponentDescriptor,
    ContentSpanSet,
    DataProductRef,
    DocumentArtifact,
    ProcessingRun,
    ProcessingRunStatus,
    sha256_bytes,
)
from DeepResearch.src.document_processing.routing import InputFormat
from DeepResearch.src.document_processing.storage import (
    ContentAddressedStore,
    RecordConflictError,
)
from tests.test_document_processing.test_storage import (
    docling_production_configuration,
    make_component_run,
    make_durable_canonical_view,
    save_artifact,
)


@pytest.fixture
def store(tmp_path: Path) -> ContentAddressedStore:
    return ContentAddressedStore(tmp_path / "changed-coverage-store")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _pdf_docling_case(
    store: ContentAddressedStore,
) -> tuple[dict[str, Any], ProcessingRun]:
    artifact = save_artifact(store, "docling-native-branches")
    _, producer = make_durable_canonical_view(store, artifact)
    docling_product = producer.require_output("docling_document")
    spans_product = producer.require_output("content_spans")
    return (
        {
            "canonical_artifact": artifact,
            "docling_product": docling_product,
            "spans_product": spans_product,
            "docling_payload": json.loads(store.read_blob(docling_product.blob_sha256)),
            "span_set": ContentSpanSet.model_validate_json(
                store.read_blob(spans_product.blob_sha256)
            ),
        },
        producer,
    )


def _call_with_docling_producer(
    monkeypatch: pytest.MonkeyPatch,
    store: ContentAddressedStore,
    arguments: dict[str, Any],
    producer: ProcessingRun,
) -> ProcessingRun:
    original = ContentAddressedStore.get_processing_run

    def get_processing_run(
        current: ContentAddressedStore, run_id: str
    ) -> ProcessingRun:
        if run_id == producer.run_id:
            return producer
        return original(current, run_id)

    with monkeypatch.context() as patch:
        patch.setattr(ContentAddressedStore, "get_processing_run", get_processing_run)
        return store._verify_docling_native_sources(**arguments)


def test_docling_native_source_rejects_wrong_owner_or_output_pair(
    store: ContentAddressedStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, producer = _pdf_docling_case(store)

    damaged = producer.model_copy(update={"artifact_id": "different-artifact"})
    with pytest.raises(RecordConflictError, match="does not own"):
        _call_with_docling_producer(monkeypatch, store, arguments, damaged)

    damaged = producer.model_copy(
        update={"outputs": (producer.require_output("docling_document"),)}
    )
    with pytest.raises(RecordConflictError, match="document/span pair"):
        _call_with_docling_producer(monkeypatch, store, arguments, damaged)


def test_canonical_reader_rejects_a_different_registered_product(
    store: ContentAddressedStore,
) -> None:
    arguments, _ = _pdf_docling_case(store)

    with pytest.raises(ValueError, match="not a canonical document view"):
        store.read_canonical_document(arguments["docling_product"])


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda config: config.update(input_format=None), "not supported"),
        (lambda config: config.update(input_format="unknown"), "cannot be unknown"),
        (lambda config: config.update(serve_version="99"), "not approved"),
        (lambda config: config.update(options=[]), "options must be an object"),
        (
            lambda config: config.update(options={"to_formats": [1]}),
            "serialized JSON",
        ),
        (
            lambda config: config.update(minimum_pdf_locator_coverage=True),
            "coverage threshold",
        ),
    ],
    ids=(
        "invalid-format",
        "unknown-format",
        "unapproved-version",
        "options-not-object",
        "invalid-output-format",
        "invalid-minimum-coverage",
    ),
)
def test_docling_native_source_rejects_malformed_invocation_fields(
    store: ContentAddressedStore,
    monkeypatch: pytest.MonkeyPatch,
    mutator: Any,
    message: str,
) -> None:
    arguments, producer = _pdf_docling_case(store)
    configuration = dict(producer.configuration)
    mutator(configuration)
    damaged = producer.model_copy(update={"configuration": configuration})

    with pytest.raises(RecordConflictError, match=message):
        _call_with_docling_producer(monkeypatch, store, arguments, damaged)


def test_docling_native_source_rejects_wrong_direct_inputs_and_span_binding(
    store: ContentAddressedStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, producer = _pdf_docling_case(store)
    damaged = producer.model_copy(
        update={"inputs": (producer.require_output("docling_document"),)}
    )
    with pytest.raises(RecordConflictError, match="artifact provenance"):
        _call_with_docling_producer(monkeypatch, store, arguments, damaged)

    damaged_span_set = arguments["span_set"].model_copy(
        update={"artifact_id": "different-artifact"}
    )
    with pytest.raises(RecordConflictError, match="not bound"):
        _call_with_docling_producer(
            monkeypatch,
            store,
            {**arguments, "span_set": damaged_span_set},
            producer,
        )


def test_docling_native_source_rejects_semantically_empty_document(
    store: ContentAddressedStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, producer = _pdf_docling_case(store)
    empty_document = {
        **arguments["docling_payload"],
        "body": {"self_ref": "#/body", "children": []},
        "texts": [],
    }
    with pytest.raises(RecordConflictError, match="not semantically usable"):
        _call_with_docling_producer(
            monkeypatch,
            store,
            {**arguments, "docling_payload": empty_document},
            producer,
        )


def _save_source_artifact(
    store: ContentAddressedStore,
    *,
    artifact_id: str,
    source: bytes,
    media_type: str,
    extension: str,
) -> DocumentArtifact:
    blob = store.put_blob(source)
    artifact = DocumentArtifact(
        artifact_id=artifact_id,
        source_sha256=blob.sha256,
        acquisition_uri=f"https://example.test/{artifact_id}.{extension}",
        media_type=media_type,
        relationship=ArtifactRelationship.SOURCE,
        raw_location=blob.as_location(
            media_type=media_type,
            role=ArtifactLocationRole.RAW,
        ),
    )
    store.save_artifact(artifact)
    return artifact


def _structured_docling_case(
    store: ContentAddressedStore,
    input_format: InputFormat,
) -> tuple[dict[str, Any], ProcessingRun, ProcessingRun]:
    if input_format is InputFormat.JATS:
        source = b'<article><body><p id="p1">Native JATS text</p></body></article>'
        artifact = _save_source_artifact(
            store,
            artifact_id="jats-native-branches",
            source=source,
            media_type="application/jats+xml",
            extension="nxml",
        )
        locators = JATSLocatorAdapter().extract_locators(source)
        html_bytes = None
        component_id = "jats-locator-adapter"
    else:
        source = _canonical_json_bytes(
            {
                "source": "DeepCritical",
                "date": "2026-08-11",
                "key": "native-branches",
                "infons": {},
                "documents": [],
            }
        )
        artifact = _save_source_artifact(
            store,
            artifact_id="bioc-native-branches",
            source=source,
            media_type="application/bioc+json",
            extension="json",
        )
        adapted = BioCAdapter().adapt(source, input_format=input_format)
        locators = adapted.locator_overlay
        html_bytes = adapted.content
        component_id = "bioc-adapter"

    locator_bytes = _canonical_json_bytes([asdict(locator) for locator in locators])
    locator_blob = store.put_blob(locator_bytes)
    adapter_run_id = f"{artifact.artifact_id}-adapter"
    outputs = {"native_locator_overlay": locator_blob.sha256}
    if html_bytes is not None:
        html_blob = store.put_blob(html_bytes)
        outputs = {
            "html_projection": html_blob.sha256,
            **outputs,
        }
    adapter = make_component_run(
        store,
        artifact.artifact_id,
        adapter_run_id,
        component=ComponentDescriptor(
            component_id=component_id,
            component_version="1",
            capability="document.adapt",
        ),
        configuration={
            "adapter_version": "1",
            "input_format": input_format.value,
            "input_sha256": artifact.source_sha256,
        },
        outputs=outputs,
    )
    store.save_processing_run(adapter)

    adapter_locator = adapter.require_output("native_locator_overlay")
    inputs = (adapter_locator,)
    input_sha256 = artifact.source_sha256
    if html_bytes is not None:
        html_product = adapter.require_output("html_projection")
        inputs = (html_product, adapter_locator)
        input_sha256 = html_product.blob_sha256

    document = {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "Structured branches",
        "body": {
            "self_ref": "#/body",
            "children": [{"$ref": "#/texts/0"}],
        },
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "paragraph",
                "text": "Structured native content",
            }
        ],
        "tables": [],
        "pictures": [],
    }
    document_blob = store.put_blob(_canonical_json_bytes(document))
    run_id = f"{artifact.artifact_id}-docling"
    docling_product = store.data_product_ref(
        name="docling_document",
        blob_sha256=document_blob.sha256,
        producer_run_id=run_id,
        source_artifact_ids=(artifact.artifact_id,),
    )
    span_set = ContentSpanSet(
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        representation_product_id=docling_product.product_id,
        spans=(),
    )
    spans_blob = store.put_blob(_canonical_json_bytes(span_set.model_dump(mode="json")))
    configuration = docling_production_configuration(
        artifact,
        input_format=input_format.value,
        input_sha256=input_sha256,
    )
    alignment_key = (
        "jats_locator_alignment_algorithm"
        if input_format is InputFormat.JATS
        else "bioc_locator_alignment_algorithm"
    )
    configuration.update(
        {
            alignment_key: "normalized-exact-v1",
            "native_locator_overlay_sha256": locator_blob.sha256,
        }
    )
    producer = make_component_run(
        store,
        artifact.artifact_id,
        run_id,
        component=ComponentDescriptor(
            component_id="docling",
            component_version="2.113.0",
            capability="document.parse",
        ),
        configuration=configuration,
        inputs=inputs,
        outputs={
            "docling_document": document_blob.sha256,
            "content_spans": spans_blob.sha256,
        },
    )
    store.save_processing_run(producer)
    return (
        {
            "canonical_artifact": artifact,
            "docling_product": producer.require_output("docling_document"),
            "spans_product": producer.require_output("content_spans"),
            "docling_payload": document,
            "span_set": span_set,
        },
        producer,
        adapter,
    )


def test_jats_docling_rejects_wrong_direct_source_input(
    store: ContentAddressedStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, producer, _ = _structured_docling_case(store, InputFormat.JATS)
    direct_blob = store.put_blob(b"expected creator product")
    wrong_blob = store.put_blob(b"wrong creator product")
    direct = store.data_product_ref(
        name="searchable_pdf",
        blob_sha256=direct_blob.sha256,
        producer_run_id="expected-creator",
        source_artifact_ids=(arguments["canonical_artifact"].artifact_id,),
    )
    wrong = store.data_product_ref(
        name="searchable_pdf",
        blob_sha256=wrong_blob.sha256,
        producer_run_id="wrong-creator",
        source_artifact_ids=(arguments["canonical_artifact"].artifact_id,),
    )
    locator = producer.inputs[-1]
    damaged = producer.model_copy(update={"inputs": (wrong, locator)})

    with monkeypatch.context() as patch:
        patch.setattr(
            ContentAddressedStore,
            "_native_artifact_inputs",
            lambda _store, _artifact: (direct,),
        )
        with pytest.raises(RecordConflictError, match="source inputs"):
            _call_with_docling_producer(monkeypatch, store, arguments, damaged)


def test_jats_docling_requires_one_locator_beyond_direct_source_inputs(
    store: ContentAddressedStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, producer, _ = _structured_docling_case(store, InputFormat.JATS)
    damaged = producer.model_copy(update={"inputs": ()})

    with pytest.raises(RecordConflictError, match="inputs do not match"):
        _call_with_docling_producer(monkeypatch, store, arguments, damaged)


@pytest.mark.parametrize("input_format", [InputFormat.JATS, InputFormat.BIOC_JSON])
def test_structured_docling_native_source_accepts_exact_adapter_lineage(
    store: ContentAddressedStore,
    monkeypatch: pytest.MonkeyPatch,
    input_format: InputFormat,
) -> None:
    arguments, producer, _ = _structured_docling_case(store, input_format)

    assert (
        _call_with_docling_producer(monkeypatch, store, arguments, producer) == producer
    )


def test_structured_docling_rejects_drifted_locator_configuration(
    store: ContentAddressedStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, producer, _ = _structured_docling_case(store, InputFormat.JATS)
    configuration = {
        **producer.configuration,
        "jats_locator_alignment_algorithm": "future-v99",
    }
    damaged = producer.model_copy(update={"configuration": configuration})
    with pytest.raises(RecordConflictError, match="locator configuration"):
        _call_with_docling_producer(monkeypatch, store, arguments, damaged)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-json", "not valid JSON"),
        (b"{}", "list of objects"),
        (b"[1]", "list of objects"),
        (b"[ {} ]", "deterministically serialized"),
    ],
)
def test_locator_configuration_requires_canonical_object_array(
    store: ContentAddressedStore,
    payload: bytes,
    message: str,
) -> None:
    blob = store.put_blob(payload)
    product = store.data_product_ref(
        name="native_locator_overlay",
        blob_sha256=blob.sha256,
        producer_run_id="locator-configuration-test",
        source_artifact_ids=("locator-artifact",),
    )
    with pytest.raises(RecordConflictError, match=message):
        store._docling_locator_configuration_sha256(product)


def _bioc_adapter_case(
    store: ContentAddressedStore,
) -> tuple[DocumentArtifact, ProcessingRun, DataProductRef, DataProductRef]:
    _, _, adapter = _structured_docling_case(store, InputFormat.BIOC_JSON)
    artifact = store.get_artifact(adapter.artifact_id)
    return (
        artifact,
        adapter,
        adapter.require_output("html_projection"),
        adapter.require_output("native_locator_overlay"),
    )


def test_bioc_adapter_inputs_require_exact_pair_and_shared_producer(
    store: ContentAddressedStore,
) -> None:
    artifact, adapter, html, locator = _bioc_adapter_case(store)

    with pytest.raises(RecordConflictError, match="HTML and locator"):
        store._verify_adapter_inputs(
            artifact=artifact,
            input_format=InputFormat.BIOC_JSON,
            products=(html,),
            expected_direct_inputs=(),
        )
    with pytest.raises(RecordConflictError, match="not an HTML projection"):
        store._verify_adapter_inputs(
            artifact=artifact,
            input_format=InputFormat.BIOC_JSON,
            products=(locator, html),
            expected_direct_inputs=(),
        )

    foreign_html = html.model_copy(update={"producer_run_id": "foreign-adapter"})
    with pytest.raises(RecordConflictError, match="share the approved"):
        store._verify_adapter_inputs(
            artifact=artifact,
            input_format=InputFormat.BIOC_JSON,
            products=(foreign_html, locator),
            expected_direct_inputs=(),
        )

    assert store._verify_adapter_inputs(
        artifact=artifact,
        input_format=InputFormat.BIOC_JSON,
        products=(html, locator),
        expected_direct_inputs=(),
    ) == (html, locator)
    assert adapter.run_id == html.producer_run_id


def _verify_damaged_adapter(
    monkeypatch: pytest.MonkeyPatch,
    store: ContentAddressedStore,
    *,
    artifact: DocumentArtifact,
    adapter: ProcessingRun,
    product: DataProductRef,
) -> ProcessingRun:
    original = ContentAddressedStore.get_processing_run

    def get_processing_run(
        current: ContentAddressedStore, run_id: str
    ) -> ProcessingRun:
        if run_id == product.producer_run_id:
            return adapter
        return original(current, run_id)

    with monkeypatch.context() as patch:
        patch.setattr(ContentAddressedStore, "get_processing_run", get_processing_run)
        return store._verify_adapter_product(
            artifact=artifact,
            input_format=InputFormat.BIOC_JSON,
            product=product,
            expected_direct_inputs=(),
        )


def test_adapter_product_rejects_wrong_contract_owner_config_or_lineage(
    store: ContentAddressedStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, adapter, html, locator = _bioc_adapter_case(store)

    with pytest.raises(RecordConflictError, match="wrong product contract"):
        store._verify_adapter_product(
            artifact=artifact,
            input_format=InputFormat.BIOC_JSON,
            product=html,
            expected_direct_inputs=(),
        )

    mutations = (
        (adapter.model_copy(update={"artifact_id": "wrong-artifact"}), "does not own"),
        (adapter.model_copy(update={"configuration": {}}), "configuration"),
        (adapter.model_copy(update={"outputs": (html,)}), "lineage"),
        (
            adapter.model_copy(
                update={"inputs": (adapter.require_output("native_locator_overlay"),)}
            ),
            "lineage",
        ),
    )
    for damaged, message in mutations:
        with pytest.raises(RecordConflictError, match=message):
            _verify_damaged_adapter(
                monkeypatch,
                store,
                artifact=artifact,
                adapter=damaged,
                product=locator,
            )
