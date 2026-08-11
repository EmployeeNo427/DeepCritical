"""Exact replay tests for persisted Docling content spans."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from io import BytesIO
from typing import Any
from zipfile import ZipFile

import pytest

from DeepResearch.src.document_processing import span_replay
from DeepResearch.src.document_processing.adapters import (
    BioCAdapter,
    JATSLocatorAdapter,
    NativeTextLocator,
)
from DeepResearch.src.document_processing.models import (
    ArtifactLocation,
    ArtifactLocationRole,
    ComponentDescriptor,
    ContentSpanSet,
    DataProductRef,
    DocumentArtifact,
    ProcessingRun,
    ProcessingRunStatus,
    configuration_sha256,
    sha256_bytes,
)
from DeepResearch.src.document_processing.products import build_data_product_ref
from DeepResearch.src.document_processing.routing import InputFormat
from DeepResearch.src.document_processing.span_replay import (
    ContentSpanReplayError,
    replay_content_span_set,
)
from DeepResearch.src.document_processing.validation import (
    align_bioc_content_spans,
    align_jats_content_spans,
    build_docling_content_spans,
    build_pdf_content_spans,
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _artifact(source: bytes, input_format: InputFormat) -> DocumentArtifact:
    digest = sha256_bytes(source)
    return DocumentArtifact(
        artifact_id=f"span-replay-{input_format.value}",
        source_sha256=digest,
        acquisition_uri=f"https://example.test/source.{input_format.value}",
        media_type="application/octet-stream",
        raw_location=ArtifactLocation(
            uri=f"cas://sha256/{digest}",
            sha256=digest,
            byte_size=len(source),
            media_type="application/octet-stream",
            role=ArtifactLocationRole.RAW,
        ),
    )


def _product(
    *,
    name: str,
    data: bytes,
    producer_run_id: str,
    artifact_id: str,
) -> DataProductRef:
    digest = sha256_bytes(data)
    return build_data_product_ref(
        name=name,
        blob_sha256=digest,
        uri=f"cas://sha256/{digest}",
        byte_size=len(data),
        producer_run_id=producer_run_id,
        source_artifact_ids=(artifact_id,),
    )


def _run(
    *,
    run_id: str,
    artifact_id: str,
    component_id: str,
    component_version: str,
    capability: str,
    configuration: dict[str, Any],
    inputs: tuple[DataProductRef, ...] = (),
    outputs: tuple[DataProductRef, ...] = (),
    status: ProcessingRunStatus = ProcessingRunStatus.COMPLETE,
    output_policy_snapshot: dict[str, Any] | None = None,
) -> ProcessingRun:
    started = datetime(2026, 8, 11, tzinfo=UTC)
    return ProcessingRun(
        run_id=run_id,
        artifact_id=artifact_id,
        stage_id=component_id,
        component=ComponentDescriptor(
            component_id=component_id,
            component_version=component_version,
            capability=capability,
        ),
        configuration=configuration,
        configuration_sha256=configuration_sha256(configuration),
        output_policy_snapshot=output_policy_snapshot or {},
        output_policy_sha256=(
            configuration_sha256(output_policy_snapshot)
            if output_policy_snapshot
            else None
        ),
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        status=status,
        inputs=inputs,
        outputs=outputs,
    )


def _docling_document(texts: list[str], *, pdf: bool = False) -> dict[str, Any]:
    items = []
    for index, text in enumerate(texts):
        item: dict[str, Any] = {
            "self_ref": f"#/texts/{index}",
            "label": "paragraph",
            "text": text,
        }
        if pdf:
            item["prov"] = [
                {
                    "page_no": 1,
                    "bbox": {
                        "l": 1,
                        "t": 10,
                        "r": 90,
                        "b": 1,
                        "coord_origin": "BOTTOMLEFT",
                    },
                }
            ]
        items.append(item)
    document: dict[str, Any] = {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "replay",
        "body": {
            "self_ref": "#/body",
            "children": [{"$ref": f"#/texts/{index}"} for index in range(len(items))],
        },
        "texts": items,
        "tables": [],
        "pictures": [],
        # Formula nodes intentionally have no content spans.  Formula evidence
        # remains an explicit canonical omission until a native locator exists.
        "formulas": [{"self_ref": "#/formulas/0", "text": "E = mc²"}],
    }
    if pdf:
        document["pages"] = {"1": {"page_no": 1, "size": {"width": 100, "height": 100}}}
    return document


def _native_bytes(locators: tuple[NativeTextLocator, ...]) -> bytes:
    return _canonical_json_bytes([asdict(locator) for locator in locators])


def _source_bytes(input_format: InputFormat) -> bytes:
    if input_format is InputFormat.PDF:
        return b"%PDF-1.7\nexact source bytes"
    if input_format is InputFormat.HTML:
        return b"<!doctype html><html><body>Exact source bytes</body></html>"
    if input_format is InputFormat.DOCX:
        payload = BytesIO()
        with ZipFile(payload, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", "<document/>")
        return payload.getvalue()
    raise AssertionError(f"unsupported test input format: {input_format.value}")


def _alignment_bytes(alignment: Any) -> bytes:
    return _canonical_json_bytes(
        {
            "algorithm": "normalized-exact-v1",
            "aligned_count": alignment.aligned_count,
            "unaligned_count": alignment.unaligned_count,
            "records": [record.to_dict() for record in alignment.records],
        }
    )


@pytest.mark.parametrize(
    "input_format",
    [InputFormat.PDF, InputFormat.HTML, InputFormat.DOCX],
)
def test_replay_accepts_exact_pdf_and_docling_native_spans(
    input_format: InputFormat,
) -> None:
    source = _source_bytes(input_format)
    artifact = _artifact(source, input_format)
    run_id = f"docling-{input_format.value}"
    document = _docling_document(["Exact span"], pdf=input_format is InputFormat.PDF)
    docling_bytes = _canonical_json_bytes(document)
    docling_product = _product(
        name="docling_document",
        data=docling_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    if input_format is InputFormat.PDF:
        spans = build_pdf_content_spans(
            document,
            artifact_id=artifact.artifact_id,
            processing_run_id=run_id,
            representation_product_id=docling_product.product_id,
        )
    else:
        spans = build_docling_content_spans(
            document,
            artifact_id=artifact.artifact_id,
            processing_run_id=run_id,
            input_format=input_format.value,  # type: ignore[arg-type]
            representation_product_id=docling_product.product_id,
        )
    span_set = ContentSpanSet(
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        representation_product_id=docling_product.product_id,
        spans=spans,
    )
    span_bytes = _canonical_json_bytes(span_set.model_dump(mode="json"))
    spans_product = _product(
        name="content_spans",
        data=span_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    configuration = {
        "content_span_schema_version": "1",
        "input_format": input_format.value,
        "input_sha256": artifact.source_sha256,
    }
    if input_format is InputFormat.PDF:
        configuration["pdf_span_algorithm"] = "provenance-charspan-v2"
    producer = _run(
        run_id=run_id,
        artifact_id=artifact.artifact_id,
        component_id="docling",
        component_version="2.96.1",
        capability="document.parse",
        configuration=configuration,
        outputs=(docling_product, spans_product),
    )

    replayed = replay_content_span_set(
        artifact=artifact,
        source_bytes=source,
        producer=producer,
        docling_product=docling_product,
        docling_bytes=docling_bytes,
        docling_document=document,
        content_spans_product=spans_product,
        content_spans_bytes=span_bytes,
    )

    assert replayed == span_set
    assert len(replayed.spans) == 1
    assert all(span.representation_anchor.node_id != "#/formulas/0" for span in spans)


@pytest.mark.parametrize("input_format", [InputFormat.BIOC_JSON, InputFormat.BIOC_XML])
def test_replay_accepts_exact_bioc_spans_and_adapter_products(
    input_format: InputFormat,
) -> None:
    if input_format is InputFormat.BIOC_JSON:
        source = _canonical_json_bytes(
            {
                "source": "DeepCritical",
                "date": "2026-08-11",
                "key": "replay",
                "infons": {},
                "documents": [
                    {
                        "id": "D1",
                        "infons": {},
                        "passages": [
                            {
                                "offset": 0,
                                "infons": {"type": "abstract"},
                                "text": "BioC exact content",
                                "sentences": [],
                                "annotations": [],
                                "relations": [],
                            }
                        ],
                        "annotations": [],
                        "relations": [],
                    }
                ],
            }
        )
    else:
        source = b"""<?xml version="1.0" encoding="UTF-8"?>
        <collection><source>DeepCritical</source><date>2026-08-11</date>
        <key>replay</key><document><id>D1</id><passage>
        <infon key="type">abstract</infon><offset>0</offset>
        <text>BioC exact content</text></passage></document></collection>"""
    artifact = _artifact(source, input_format)
    adapted = BioCAdapter().adapt(source, input_format=input_format)
    locator_bytes = _native_bytes(adapted.locator_overlay)
    adapter_run_id = f"adapter-{input_format.value}"
    adapter_native = _product(
        name="native_locator_overlay",
        data=locator_bytes,
        producer_run_id=adapter_run_id,
        artifact_id=artifact.artifact_id,
    )
    html_product = _product(
        name="html_projection",
        data=adapted.content,
        producer_run_id=adapter_run_id,
        artifact_id=artifact.artifact_id,
    )
    adapter_configuration = {
        "adapter_version": "1",
        "input_format": input_format.value,
        "input_sha256": artifact.source_sha256,
    }
    adapter_run = _run(
        run_id=adapter_run_id,
        artifact_id=artifact.artifact_id,
        component_id="bioc-adapter",
        component_version="1",
        capability="document.adapt",
        configuration=adapter_configuration,
        outputs=(html_product, adapter_native),
    )

    document = _docling_document([locator.text for locator in adapted.locator_overlay])
    run_id = f"docling-{input_format.value}"
    docling_bytes = _canonical_json_bytes(document)
    docling_product = _product(
        name="docling_document",
        data=docling_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    alignment = align_bioc_content_spans(
        document,
        adapted.locator_overlay,
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        representation_product_id=docling_product.product_id,
    )
    spans = alignment.spans
    span_set = ContentSpanSet(
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        representation_product_id=docling_product.product_id,
        spans=spans,
    )
    span_bytes = _canonical_json_bytes(span_set.model_dump(mode="json"))
    spans_product = _product(
        name="content_spans",
        data=span_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    docling_native = _product(
        name="native_locator_overlay",
        data=locator_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    alignment_bytes = _alignment_bytes(alignment)
    alignment_product = _product(
        name="bioc_locator_alignment",
        data=alignment_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    configuration = {
        "content_span_schema_version": "1",
        "input_format": input_format.value,
        "input_sha256": html_product.blob_sha256,
        "bioc_locator_alignment_algorithm": "normalized-exact-v1",
        "native_locator_overlay_sha256": docling_native.blob_sha256,
    }
    producer = _run(
        run_id=run_id,
        artifact_id=artifact.artifact_id,
        component_id="docling",
        component_version="2.96.1",
        capability="document.parse",
        configuration=configuration,
        inputs=(html_product, adapter_native),
        outputs=(
            docling_product,
            spans_product,
            docling_native,
            alignment_product,
        ),
    )

    replayed = replay_content_span_set(
        artifact=artifact,
        source_bytes=source,
        producer=producer,
        docling_product=docling_product,
        docling_bytes=docling_bytes,
        docling_document=document,
        content_spans_product=spans_product,
        content_spans_bytes=span_bytes,
        native_locator_product=docling_native,
        native_locator_bytes=locator_bytes,
        adapter_producer=adapter_run,
        html_projection_bytes=adapted.content,
        native_alignment_product=alignment_product,
        native_alignment_bytes=alignment_bytes,
    )

    assert replayed == span_set
    assert replayed.spans


def test_replay_accepts_exact_jats_spans_and_adapter_product() -> None:
    source = b"""<article><body><sec id="s1"><title>Methods</title>
    <p id="p1">JATS exact content</p></sec></body></article>"""
    artifact = _artifact(source, InputFormat.JATS)
    locators = JATSLocatorAdapter().extract_locators(source)
    locator_bytes = _native_bytes(locators)
    adapter_run_id = "adapter-jats"
    adapter_native = _product(
        name="native_locator_overlay",
        data=locator_bytes,
        producer_run_id=adapter_run_id,
        artifact_id=artifact.artifact_id,
    )
    adapter_configuration = {
        "adapter_version": "1",
        "input_format": "jats",
        "input_sha256": artifact.source_sha256,
    }
    adapter_run = _run(
        run_id=adapter_run_id,
        artifact_id=artifact.artifact_id,
        component_id="jats-locator-adapter",
        component_version="1",
        capability="document.adapt",
        configuration=adapter_configuration,
        outputs=(adapter_native,),
    )

    document = _docling_document([locator.text for locator in locators])
    run_id = "docling-jats"
    docling_bytes = _canonical_json_bytes(document)
    docling_product = _product(
        name="docling_document",
        data=docling_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    alignment = align_jats_content_spans(
        document,
        locators,
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        representation_product_id=docling_product.product_id,
    )
    spans = alignment.spans
    span_set = ContentSpanSet(
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        representation_product_id=docling_product.product_id,
        spans=spans,
    )
    span_bytes = _canonical_json_bytes(span_set.model_dump(mode="json"))
    spans_product = _product(
        name="content_spans",
        data=span_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    docling_native = _product(
        name="native_locator_overlay",
        data=locator_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    alignment_bytes = _alignment_bytes(alignment)
    alignment_product = _product(
        name="jats_locator_alignment",
        data=alignment_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    configuration = {
        "content_span_schema_version": "1",
        "input_format": "jats",
        "input_sha256": artifact.source_sha256,
        "jats_locator_alignment_algorithm": "normalized-exact-v1",
        "native_locator_overlay_sha256": docling_native.blob_sha256,
    }
    producer = _run(
        run_id=run_id,
        artifact_id=artifact.artifact_id,
        component_id="docling",
        component_version="2.96.1",
        capability="document.parse",
        configuration=configuration,
        inputs=(adapter_native,),
        outputs=(
            docling_product,
            spans_product,
            docling_native,
            alignment_product,
        ),
    )

    replayed = replay_content_span_set(
        artifact=artifact,
        source_bytes=source,
        producer=producer,
        docling_product=docling_product,
        docling_bytes=docling_bytes,
        docling_document=document,
        content_spans_product=spans_product,
        content_spans_bytes=span_bytes,
        native_locator_product=docling_native,
        native_locator_bytes=locator_bytes,
        adapter_producer=adapter_run,
        native_alignment_product=alignment_product,
        native_alignment_bytes=alignment_bytes,
    )

    assert replayed == span_set
    assert len(replayed.spans) == len(locators)

    forged_payload = json.loads(locator_bytes)
    forged_payload[0]["text"] = "Invented locator text"
    forged_locator_bytes = _canonical_json_bytes(forged_payload)
    forged_adapter_native = _product(
        name="native_locator_overlay",
        data=forged_locator_bytes,
        producer_run_id=adapter_run_id,
        artifact_id=artifact.artifact_id,
    )
    forged_adapter = _run(
        run_id=adapter_run_id,
        artifact_id=artifact.artifact_id,
        component_id="jats-locator-adapter",
        component_version="1",
        capability="document.adapt",
        configuration=adapter_configuration,
        outputs=(forged_adapter_native,),
    )
    forged_docling_native = _product(
        name="native_locator_overlay",
        data=forged_locator_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    forged_configuration = {
        **configuration,
        "native_locator_overlay_sha256": forged_docling_native.blob_sha256,
    }
    forged_producer = _run(
        run_id=run_id,
        artifact_id=artifact.artifact_id,
        component_id="docling",
        component_version="2.96.1",
        capability="document.parse",
        configuration=forged_configuration,
        inputs=(forged_adapter_native,),
        outputs=(
            docling_product,
            spans_product,
            forged_docling_native,
            alignment_product,
        ),
    )

    with pytest.raises(ContentSpanReplayError, match="raw source"):
        replay_content_span_set(
            artifact=artifact,
            source_bytes=source,
            producer=forged_producer,
            docling_product=docling_product,
            docling_bytes=docling_bytes,
            docling_document=document,
            content_spans_product=spans_product,
            content_spans_bytes=span_bytes,
            native_locator_product=forged_docling_native,
            native_locator_bytes=forged_locator_bytes,
            adapter_producer=forged_adapter,
            native_alignment_product=alignment_product,
            native_alignment_bytes=alignment_bytes,
        )


def test_replay_rejects_claimed_format_that_disagrees_with_production_routing() -> None:
    source = _source_bytes(InputFormat.PDF)
    artifact = _artifact(source, InputFormat.HTML)
    run_id = "docling-format-confusion"
    document = _docling_document(["Parser-native forgery"])
    docling_bytes = _canonical_json_bytes(document)
    docling_product = _product(
        name="docling_document",
        data=docling_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    spans = build_docling_content_spans(
        document,
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        input_format="html",
        representation_product_id=docling_product.product_id,
    )
    span_set = ContentSpanSet(
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        representation_product_id=docling_product.product_id,
        spans=spans,
    )
    span_bytes = _canonical_json_bytes(span_set.model_dump(mode="json"))
    spans_product = _product(
        name="content_spans",
        data=span_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    producer = _run(
        run_id=run_id,
        artifact_id=artifact.artifact_id,
        component_id="docling",
        component_version="2.96.1",
        capability="document.parse",
        configuration={
            "content_span_schema_version": "1",
            "input_format": "html",
            "input_sha256": artifact.source_sha256,
        },
        outputs=(docling_product, spans_product),
    )

    with pytest.raises(ContentSpanReplayError, match="production routing"):
        replay_content_span_set(
            artifact=artifact,
            source_bytes=source,
            producer=producer,
            docling_product=docling_product,
            docling_bytes=docling_bytes,
            docling_document=document,
            content_spans_product=spans_product,
            content_spans_bytes=span_bytes,
        )


def test_replay_uses_recorded_extension_only_routing_policy() -> None:
    source = b"legacy extension-only HTML source"
    artifact = _artifact(source, InputFormat.HTML)
    run_id = "docling-extension-only"
    document = _docling_document(["Extension-only content"])
    docling_bytes = _canonical_json_bytes(document)
    docling_product = _product(
        name="docling_document",
        data=docling_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    spans = build_docling_content_spans(
        document,
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        input_format="html",
        representation_product_id=docling_product.product_id,
    )
    span_set = ContentSpanSet(
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        representation_product_id=docling_product.product_id,
        spans=spans,
    )
    span_bytes = _canonical_json_bytes(span_set.model_dump(mode="json"))
    spans_product = _product(
        name="content_spans",
        data=span_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    producer = _run(
        run_id=run_id,
        artifact_id=artifact.artifact_id,
        component_id="docling",
        component_version="2.96.1",
        capability="document.parse",
        configuration={
            "content_span_schema_version": "1",
            "input_format": "html",
            "input_sha256": artifact.source_sha256,
        },
        outputs=(docling_product, spans_product),
        output_policy_snapshot={
            "routing": {
                "algorithm_version": "deterministic-document-router-v1",
                "reject_extension_only_detection": False,
            }
        },
    )

    assert (
        replay_content_span_set(
            artifact=artifact,
            source_bytes=source,
            producer=producer,
            docling_product=docling_product,
            docling_bytes=docling_bytes,
            docling_document=document,
            content_spans_product=spans_product,
            content_spans_bytes=span_bytes,
        )
        == span_set
    )


def test_replay_rejects_forged_native_alignment_and_complete_status() -> None:
    source = b'<article><body><p id="p1">Native truth</p></body></article>'
    artifact = _artifact(source, InputFormat.JATS)
    locators = JATSLocatorAdapter().extract_locators(source)
    locator_bytes = _native_bytes(locators)
    adapter_run_id = "adapter-jats-unaligned"
    adapter_native = _product(
        name="native_locator_overlay",
        data=locator_bytes,
        producer_run_id=adapter_run_id,
        artifact_id=artifact.artifact_id,
    )
    adapter_run = _run(
        run_id=adapter_run_id,
        artifact_id=artifact.artifact_id,
        component_id="jats-locator-adapter",
        component_version="1",
        capability="document.adapt",
        configuration={
            "adapter_version": "1",
            "input_format": "jats",
            "input_sha256": artifact.source_sha256,
        },
        outputs=(adapter_native,),
    )

    run_id = "docling-jats-unaligned"
    document = _docling_document(["Different nonempty Docling output"])
    docling_bytes = _canonical_json_bytes(document)
    docling_product = _product(
        name="docling_document",
        data=docling_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    alignment = align_jats_content_spans(
        document,
        locators,
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        representation_product_id=docling_product.product_id,
    )
    assert alignment.unaligned_count == 1
    span_set = ContentSpanSet(
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        representation_product_id=docling_product.product_id,
        spans=alignment.spans,
    )
    span_bytes = _canonical_json_bytes(span_set.model_dump(mode="json"))
    spans_product = _product(
        name="content_spans",
        data=span_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    docling_native = _product(
        name="native_locator_overlay",
        data=locator_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    alignment_bytes = _alignment_bytes(alignment)
    alignment_product = _product(
        name="jats_locator_alignment",
        data=alignment_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    configuration = {
        "content_span_schema_version": "1",
        "input_format": "jats",
        "input_sha256": artifact.source_sha256,
        "jats_locator_alignment_algorithm": "normalized-exact-v1",
        "native_locator_overlay_sha256": docling_native.blob_sha256,
    }

    def producer(status: ProcessingRunStatus, product: DataProductRef) -> ProcessingRun:
        return _run(
            run_id=run_id,
            artifact_id=artifact.artifact_id,
            component_id="docling",
            component_version="2.96.1",
            capability="document.parse",
            configuration=configuration,
            inputs=(adapter_native,),
            outputs=(docling_product, spans_product, docling_native, product),
            status=status,
        )

    def replay(run: ProcessingRun, product: DataProductRef, payload: bytes) -> None:
        replay_content_span_set(
            artifact=artifact,
            source_bytes=source,
            producer=run,
            docling_product=docling_product,
            docling_bytes=docling_bytes,
            docling_document=document,
            content_spans_product=spans_product,
            content_spans_bytes=span_bytes,
            native_locator_product=docling_native,
            native_locator_bytes=locator_bytes,
            adapter_producer=adapter_run,
            native_alignment_product=product,
            native_alignment_bytes=payload,
        )

    with pytest.raises(ContentSpanReplayError, match="masks unaligned"):
        replay(
            producer(ProcessingRunStatus.COMPLETE, alignment_product),
            alignment_product,
            alignment_bytes,
        )

    replay(
        producer(ProcessingRunStatus.PARTIAL, alignment_product),
        alignment_product,
        alignment_bytes,
    )

    forged_payload = json.loads(alignment_bytes)
    forged_payload["unaligned_count"] = 0
    forged_bytes = _canonical_json_bytes(forged_payload)
    forged_product = _product(
        name="jats_locator_alignment",
        data=forged_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    with pytest.raises(ContentSpanReplayError, match="does not reproduce"):
        replay(
            producer(ProcessingRunStatus.PARTIAL, forged_product),
            forged_product,
            forged_bytes,
        )


def _html_replay_case() -> tuple[dict[str, Any], ProcessingRun, ContentSpanSet]:
    source = _source_bytes(InputFormat.HTML)
    artifact = _artifact(source, InputFormat.HTML)
    run_id = "docling-html-adversarial"
    document = _docling_document(["Exact HTML span"])
    docling_bytes = _canonical_json_bytes(document)
    docling_product = _product(
        name="docling_document",
        data=docling_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    spans = build_docling_content_spans(
        document,
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        input_format="html",
        representation_product_id=docling_product.product_id,
    )
    span_set = ContentSpanSet(
        artifact_id=artifact.artifact_id,
        processing_run_id=run_id,
        representation_product_id=docling_product.product_id,
        spans=spans,
    )
    span_bytes = _canonical_json_bytes(span_set.model_dump(mode="json"))
    spans_product = _product(
        name="content_spans",
        data=span_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    producer = _run(
        run_id=run_id,
        artifact_id=artifact.artifact_id,
        component_id="docling",
        component_version="2.96.1",
        capability="document.parse",
        configuration={
            "content_span_schema_version": "1",
            "input_format": "html",
            "input_sha256": artifact.source_sha256,
        },
        outputs=(docling_product, spans_product),
    )
    return (
        {
            "artifact": artifact,
            "source_bytes": source,
            "producer": producer,
            "docling_product": docling_product,
            "docling_bytes": docling_bytes,
            "docling_document": document,
            "content_spans_product": spans_product,
            "content_spans_bytes": span_bytes,
        },
        producer,
        span_set,
    )


def test_replay_rejects_malformed_primary_evidence() -> None:
    def reject(message: str, **updates: Any) -> None:
        arguments, _, _ = _html_replay_case()
        arguments.update(updates)
        with pytest.raises(ContentSpanReplayError, match=message):
            replay_content_span_set(**arguments)

    reject("durable source hash", source_bytes=b"forged source")

    arguments, _, _ = _html_replay_case()
    reject(
        "deterministically serialized",
        docling_bytes=json.dumps(arguments["docling_document"], indent=2).encode(),
    )
    reject("content-span product is invalid", content_spans_bytes=b"not-json")
    reject(
        "content-span product is not deterministically serialized",
        content_spans_bytes=json.dumps(
            json.loads(arguments["content_spans_bytes"]), indent=2
        ).encode(),
    )

    _, producer, _ = _html_replay_case()
    reject(
        "schema version",
        producer=producer.model_copy(
            update={
                "configuration": {
                    **producer.configuration,
                    "content_span_schema_version": "99",
                }
            }
        ),
    )
    reject("non-native", native_alignment_bytes=b"invented alignment")

    changed_document = _docling_document(["Different replayed content"])
    reject(
        "does not reproduce",
        docling_document=changed_document,
        docling_bytes=_canonical_json_bytes(changed_document),
    )


def test_replay_rejects_invalid_docling_producer_contracts() -> None:
    arguments, producer, _ = _html_replay_case()

    mutations = (
        (
            producer.model_copy(update={"artifact_id": "a-different-artifact"}),
            "producer artifact",
        ),
        (
            producer.model_copy(
                update={
                    "component": ComponentDescriptor(
                        component_id="forged-docling",
                        component_version="2.96.1",
                        capability="document.parse",
                    )
                }
            ),
            "producer identity",
        ),
        (
            producer.model_copy(update={"status": ProcessingRunStatus.FAILED}),
            "complete or partial",
        ),
    )
    for damaged, message in mutations:
        with pytest.raises(ContentSpanReplayError, match=message):
            replay_content_span_set(**{**arguments, "producer": damaged})

    with pytest.raises(ContentSpanReplayError, match="must declare exactly"):
        replay_content_span_set(
            **{
                **arguments,
                "producer": producer.model_copy(update={"outputs": ()}),
            }
        )

    foreign_product = arguments["docling_product"].model_copy(
        update={"producer_run_id": "foreign-docling-run"}
    )
    forged_producer = producer.model_copy(
        update={
            "outputs": (
                foreign_product,
                arguments["content_spans_product"],
            )
        }
    )
    with pytest.raises(ContentSpanReplayError, match="does not belong"):
        span_replay._require_exact_output(
            forged_producer, foreign_product, "docling_document"
        )


@pytest.mark.parametrize(
    ("configuration", "message"),
    [
        ({"input_format": None}, "configuration is invalid"),
        ({"input_format": "not-a-format"}, "unsupported Docling input format"),
        ({"input_format": "unknown"}, "unsupported Docling input format"),
    ],
)
def test_input_format_requires_one_supported_production_route(
    configuration: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ContentSpanReplayError, match=message):
        span_replay._input_format(configuration)


def test_invocation_hash_requires_exact_bioc_projection() -> None:
    arguments, producer, _ = _html_replay_case()
    artifact = arguments["artifact"]

    invalid_type = producer.model_copy(
        update={"configuration": {**producer.configuration, "input_sha256": None}}
    )
    with pytest.raises(ContentSpanReplayError, match="configuration is invalid"):
        span_replay._verify_invocation_input_hash(
            artifact=artifact,
            producer=invalid_type,
            input_format=InputFormat.HTML,
        )

    with pytest.raises(ContentSpanReplayError, match="exactly one HTML projection"):
        span_replay._verify_invocation_input_hash(
            artifact=artifact,
            producer=producer,
            input_format=InputFormat.BIOC_JSON,
        )

    mismatched = producer.model_copy(
        update={
            "configuration": {
                **producer.configuration,
                "input_sha256": "f" * 64,
            }
        }
    )
    with pytest.raises(ContentSpanReplayError, match="durable invocation input"):
        span_replay._verify_invocation_input_hash(
            artifact=artifact,
            producer=mismatched,
            input_format=InputFormat.HTML,
        )


@pytest.mark.parametrize(
    ("routing", "message"),
    [
        ("not-an-object", "production contract"),
        (
            {"algorithm_version": "deterministic-document-router-v1"},
            "production contract",
        ),
        (
            {
                "algorithm_version": "future-router-v99",
                "reject_extension_only_detection": True,
            },
            "routing algorithm",
        ),
        (
            {
                "algorithm_version": "deterministic-document-router-v1",
                "reject_extension_only_detection": 1,
            },
            "extension-only routing policy",
        ),
    ],
)
def test_routing_policy_snapshot_is_exact(
    routing: Any,
    message: str,
) -> None:
    arguments, producer, _ = _html_replay_case()
    damaged = producer.model_copy(
        update={"output_policy_snapshot": {"routing": routing}}
    )
    with pytest.raises(ContentSpanReplayError, match=message):
        span_replay._verify_routed_input_format(
            artifact=arguments["artifact"],
            source_bytes=arguments["source_bytes"],
            producer=damaged,
            input_format=InputFormat.HTML,
        )


def _native_jats_case(
    *,
    source: bytes = b'<article><body><p id="p1">Native text</p></body></article>',
    locator_bytes: bytes | None = None,
    adapter_status: ProcessingRunStatus = ProcessingRunStatus.COMPLETE,
) -> dict[str, Any]:
    artifact = _artifact(source, InputFormat.JATS)
    if locator_bytes is None:
        locator_bytes = _native_bytes(JATSLocatorAdapter().extract_locators(source))
    adapter_run_id = "adapter-jats-branches"
    adapter_native = _product(
        name="native_locator_overlay",
        data=locator_bytes,
        producer_run_id=adapter_run_id,
        artifact_id=artifact.artifact_id,
    )
    adapter = _run(
        run_id=adapter_run_id,
        artifact_id=artifact.artifact_id,
        component_id="jats-locator-adapter",
        component_version="1",
        capability="document.adapt",
        configuration={
            "adapter_version": "1",
            "input_format": "jats",
            "input_sha256": artifact.source_sha256,
        },
        outputs=(adapter_native,),
        status=adapter_status,
    )
    docling_run_id = "docling-jats-branches"
    docling_native = _product(
        name="native_locator_overlay",
        data=locator_bytes,
        producer_run_id=docling_run_id,
        artifact_id=artifact.artifact_id,
    )
    producer = _run(
        run_id=docling_run_id,
        artifact_id=artifact.artifact_id,
        component_id="docling",
        component_version="2.96.1",
        capability="document.parse",
        configuration={
            "input_format": "jats",
            "input_sha256": artifact.source_sha256,
            "native_locator_overlay_sha256": sha256_bytes(locator_bytes),
        },
        inputs=(adapter_native,),
        outputs=(docling_native,),
    )
    return {
        "artifact": artifact,
        "source_bytes": source,
        "producer": producer,
        "input_format": InputFormat.JATS,
        "native_locator_product": docling_native,
        "native_locator_bytes": locator_bytes,
        "adapter_producer": adapter,
        "html_projection_bytes": None,
    }


def test_native_alignment_requires_exact_declared_output_and_bytes() -> None:
    arguments, _, _ = _html_replay_case()
    artifact = arguments["artifact"]
    run_id = "docling-native-alignment-branches"
    expected_bytes = _canonical_json_bytes(
        {
            "algorithm": "normalized-exact-v1",
            "aligned_count": 1,
            "unaligned_count": 0,
            "records": [],
        }
    )
    product = _product(
        name="jats_locator_alignment",
        data=expected_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    producer = _run(
        run_id=run_id,
        artifact_id=artifact.artifact_id,
        component_id="docling",
        component_version="2.96.1",
        capability="document.parse",
        configuration={},
        outputs=(product,),
    )

    common = {
        "input_format": InputFormat.JATS,
        "aligned_count": 1,
        "unaligned_count": 0,
        "records": [],
    }
    with pytest.raises(ContentSpanReplayError, match="exact locator-alignment"):
        span_replay._verify_native_alignment(
            producer=producer.model_copy(update={"outputs": ()}),
            alignment_product=None,
            alignment_bytes=None,
            **common,
        )
    with pytest.raises(ContentSpanReplayError, match="durable product"):
        span_replay._verify_native_alignment(
            producer=producer,
            alignment_product=product,
            alignment_bytes=b"forged bytes",
            **common,
        )


def test_native_locator_envelope_rejects_missing_or_forged_evidence() -> None:
    valid = _native_jats_case()

    for updates in (
        {"native_locator_product": None},
        {"native_locator_bytes": None},
    ):
        with pytest.raises(ContentSpanReplayError, match="exact native-locator"):
            span_replay._decode_native_locators(**{**valid, **updates})

    with pytest.raises(ContentSpanReplayError, match="overlay hash"):
        span_replay._decode_native_locators(
            **{**valid, "native_locator_bytes": b"forged"}
        )

    no_input = valid["producer"].model_copy(update={"inputs": ()})
    with pytest.raises(ContentSpanReplayError, match="durable adapter input"):
        span_replay._decode_native_locators(**{**valid, "producer": no_input})

    with pytest.raises(ContentSpanReplayError, match="durable adapter producer"):
        span_replay._decode_native_locators(**{**valid, "adapter_producer": None})


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-json", "not valid JSON"),
        (b"{}", "must be a JSON array"),
        (b"[{}]", "exact payload schema"),
    ],
)
def test_native_locator_payload_shape_is_strict(payload: bytes, message: str) -> None:
    with pytest.raises(ContentSpanReplayError, match=message):
        span_replay._decode_native_locators(**_native_jats_case(locator_bytes=payload))


def test_native_locator_serialization_and_empty_status_are_exact() -> None:
    valid = _native_jats_case()
    pretty_bytes = json.dumps(
        json.loads(valid["native_locator_bytes"]), indent=2
    ).encode()
    with pytest.raises(ContentSpanReplayError, match="deterministically serialized"):
        span_replay._decode_native_locators(
            **_native_jats_case(locator_bytes=pretty_bytes)
        )

    empty_source = b"<article><body/></article>"
    empty_complete = _native_jats_case(source=empty_source)
    with pytest.raises(ContentSpanReplayError, match="empty replay"):
        span_replay._decode_native_locators(**empty_complete)

    empty_partial = _native_jats_case(
        source=empty_source,
        adapter_status=ProcessingRunStatus.PARTIAL,
    )
    assert span_replay._decode_native_locators(**empty_partial) == ()


def test_native_adapter_contract_rejects_forged_metadata() -> None:
    valid = _native_jats_case()
    adapter = valid["adapter_producer"]
    common = {
        "artifact": valid["artifact"],
        "input_format": InputFormat.JATS,
        "native_locator_product": adapter.require_output("native_locator_overlay"),
        "html_projection_bytes": None,
        "docling_producer": valid["producer"],
    }
    mutations = (
        (adapter.model_copy(update={"artifact_id": "wrong-artifact"}), "artifact"),
        (
            adapter.model_copy(
                update={
                    "component": ComponentDescriptor(
                        component_id="forged-adapter",
                        component_version="1",
                        capability="document.adapt",
                    )
                }
            ),
            "identity",
        ),
        (
            adapter.model_copy(update={"status": ProcessingRunStatus.FAILED}),
            "status",
        ),
        (
            adapter.model_copy(update={"configuration": {}}),
            "configuration",
        ),
    )
    for damaged, message in mutations:
        with pytest.raises(ContentSpanReplayError, match=message):
            span_replay._verify_adapter_producer(
                producer=damaged,
                **common,
            )

    with pytest.raises(ContentSpanReplayError, match="cannot claim an HTML"):
        span_replay._verify_adapter_producer(
            producer=adapter,
            **{**common, "html_projection_bytes": b"invented"},
        )

    html_input = _product(
        name="html_projection",
        data=b"invented",
        producer_run_id="adapter-jats-branches",
        artifact_id=valid["artifact"].artifact_id,
    )
    with pytest.raises(ContentSpanReplayError, match="BioC HTML projection"):
        span_replay._verify_adapter_producer(
            producer=adapter,
            **{
                **common,
                "docling_producer": valid["producer"].model_copy(
                    update={"inputs": (common["native_locator_product"], html_input)}
                ),
            },
        )


def _bioc_adapter_case() -> dict[str, Any]:
    source = _canonical_json_bytes(
        {
            "source": "DeepCritical",
            "date": "2026-08-11",
            "key": "branches",
            "infons": {},
            "documents": [],
        }
    )
    artifact = _artifact(source, InputFormat.BIOC_JSON)
    adapted = BioCAdapter().adapt(source, input_format=InputFormat.BIOC_JSON)
    locator_bytes = _native_bytes(adapted.locator_overlay)
    run_id = "adapter-bioc-branches"
    locator_product = _product(
        name="native_locator_overlay",
        data=locator_bytes,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    html_product = _product(
        name="html_projection",
        data=adapted.content,
        producer_run_id=run_id,
        artifact_id=artifact.artifact_id,
    )
    adapter = _run(
        run_id=run_id,
        artifact_id=artifact.artifact_id,
        component_id="bioc-adapter",
        component_version="1",
        capability="document.adapt",
        configuration={
            "adapter_version": "1",
            "input_format": "bioc_json",
            "input_sha256": artifact.source_sha256,
        },
        outputs=(html_product, locator_product),
    )
    docling = _run(
        run_id="docling-bioc-branches",
        artifact_id=artifact.artifact_id,
        component_id="docling",
        component_version="2.96.1",
        capability="document.parse",
        configuration={},
        inputs=(html_product, locator_product),
    )
    return {
        "artifact": artifact,
        "producer": adapter,
        "input_format": InputFormat.BIOC_JSON,
        "native_locator_product": locator_product,
        "html_projection_bytes": adapted.content,
        "docling_producer": docling,
    }


def test_bioc_adapter_requires_exact_html_projection() -> None:
    valid = _bioc_adapter_case()
    docling = valid["docling_producer"]
    locator = valid["native_locator_product"]

    with pytest.raises(ContentSpanReplayError, match="exact HTML projection"):
        span_replay._verify_adapter_producer(
            **{
                **valid,
                "docling_producer": docling.model_copy(update={"inputs": (locator,)}),
            }
        )
    with pytest.raises(ContentSpanReplayError, match="bytes do not match"):
        span_replay._verify_adapter_producer(
            **{**valid, "html_projection_bytes": b"forged projection"}
        )


@pytest.mark.parametrize(
    ("source", "input_format"),
    [
        (b"not XML", InputFormat.JATS),
        (b"not BioC", InputFormat.BIOC_JSON),
    ],
)
def test_native_source_replay_rejects_unparseable_source(
    source: bytes,
    input_format: InputFormat,
) -> None:
    with pytest.raises(ContentSpanReplayError, match="exact adapter"):
        span_replay._verify_native_source_replay(
            source_bytes=source,
            input_format=input_format,
            locator_bytes=b"[]",
            html_projection_bytes=None,
        )


def test_native_source_replay_rejects_forged_bioc_projection() -> None:
    valid = _bioc_adapter_case()
    with pytest.raises(ContentSpanReplayError, match="does not reproduce"):
        span_replay._verify_native_source_replay(
            source_bytes=_source_for_artifact(valid["artifact"]),
            input_format=InputFormat.BIOC_JSON,
            locator_bytes=_native_bytes(()),
            html_projection_bytes=b"forged projection",
        )


def _source_for_artifact(artifact: DocumentArtifact) -> bytes:
    # The empty BioC fixture is deliberately stable and source-backed.
    source = _canonical_json_bytes(
        {
            "source": "DeepCritical",
            "date": "2026-08-11",
            "key": "branches",
            "infons": {},
            "documents": [],
        }
    )
    assert sha256_bytes(source) == artifact.source_sha256
    return source


def _strict_locator_payload() -> dict[str, Any]:
    return asdict(
        NativeTextLocator(
            text="Native text",
            source_kind="jats",
            xml_id="p1",
            xpath="/article/body/p",
        )
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("text", 1, "text is invalid"),
        ("source_kind", "bioc_json", "source_kind"),
        ("document_index", True, "document_index"),
        ("offset", "0", "offset"),
        ("xml_id", 1, "xml_id"),
        ("infons", [], "infons"),
        ("infons", {"type": 1}, "infons"),
    ],
)
def test_native_locator_fields_are_strictly_typed(
    field: str,
    value: Any,
    message: str,
) -> None:
    payload = _strict_locator_payload()
    payload[field] = value
    with pytest.raises(ContentSpanReplayError, match=message):
        span_replay._strict_native_locator(
            payload,
            index=0,
            input_format=InputFormat.JATS,
        )


def test_span_algorithm_identifier_is_exact() -> None:
    with pytest.raises(ContentSpanReplayError, match="replay contract"):
        span_replay._require_algorithm(
            {"pdf_span_algorithm": "forged-v99"},
            key="pdf_span_algorithm",
            expected="provenance-charspan-v2",
        )
