"""Deterministic replay of Docling content-span products.

The persisted span set is evidence, not an authority.  This module rebuilds it
from the exact immutable Docling document, invocation configuration, and (for
JATS/BioC) native-locator overlay that produced it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from pydantic import ValidationError

from .adapters import BioCAdapter, JATSLocatorAdapter, NativeTextLocator
from .models import (
    ContentSpanSet,
    DataProductRef,
    DoclingInputFormat,
    DocumentArtifact,
    ProcessingRun,
    ProcessingRunStatus,
)
from .routing import DocumentRouter, InputFormat, filename_from_uri
from .validation import (
    align_bioc_content_spans,
    align_jats_content_spans,
    build_docling_content_spans,
    build_pdf_content_spans,
)

_CONTENT_SPAN_SCHEMA_VERSION = "1"
_PDF_CONTENT_SPAN_ALGORITHM = "provenance-charspan-v2"
_NATIVE_ALIGNMENT_ALGORITHM = "normalized-exact-v1"
_ROUTING_ALGORITHM_VERSION = "deterministic-document-router-v1"
_NATIVE_ALIGNMENT_PRODUCT_NAMES = frozenset(
    {"jats_locator_alignment", "bioc_locator_alignment"}
)
_DOCLING_INPUT_FORMATS: dict[InputFormat, DoclingInputFormat] = {
    InputFormat.HTML: "html",
    InputFormat.DOCX: "docx",
    InputFormat.PPTX: "pptx",
    InputFormat.XLSX: "xlsx",
    InputFormat.IMAGE: "image",
}
_NATIVE_LOCATOR_FIELDS = frozenset(
    {
        "text",
        "source_kind",
        "document_index",
        "document_id",
        "passage_index",
        "sentence_index",
        "offset",
        "length",
        "xml_id",
        "xpath",
        "infons",
    }
)


class ContentSpanReplayError(ValueError):
    """Raised when persisted content spans cannot be reproduced exactly."""


def replay_content_span_set(
    *,
    artifact: DocumentArtifact,
    source_bytes: bytes,
    producer: ProcessingRun,
    docling_product: DataProductRef,
    docling_bytes: bytes,
    docling_document: dict[str, Any],
    content_spans_product: DataProductRef,
    content_spans_bytes: bytes,
    native_locator_product: DataProductRef | None = None,
    native_locator_bytes: bytes | None = None,
    adapter_producer: ProcessingRun | None = None,
    html_projection_bytes: bytes | None = None,
    native_alignment_product: DataProductRef | None = None,
    native_alignment_bytes: bytes | None = None,
) -> ContentSpanSet:
    """Return the exact reproduced span set or reject irreproducible evidence.

    The caller remains responsible for verifying each referenced CAS blob and
    durable producer chain.  This function binds those verified products to the
    Docling invocation semantics and byte-for-byte deterministic outputs.
    """

    _verify_docling_producer(
        artifact=artifact,
        producer=producer,
        docling_product=docling_product,
        content_spans_product=content_spans_product,
    )
    if hashlib.sha256(source_bytes).hexdigest() != artifact.source_sha256:
        raise ContentSpanReplayError(
            "artifact source bytes do not match the durable source hash"
        )
    if _canonical_json_bytes(docling_document) != docling_bytes:
        raise ContentSpanReplayError(
            "Docling document is not deterministically serialized"
        )

    try:
        persisted = ContentSpanSet.model_validate_json(content_spans_bytes)
    except ValidationError as exc:
        raise ContentSpanReplayError("content-span product is invalid") from exc
    if _canonical_json_bytes(persisted.model_dump(mode="json")) != content_spans_bytes:
        raise ContentSpanReplayError(
            "content-span product is not deterministically serialized"
        )

    configuration = producer.configuration
    if configuration.get("content_span_schema_version") != _CONTENT_SPAN_SCHEMA_VERSION:
        raise ContentSpanReplayError(
            "Docling content-span schema version does not match the replay contract"
        )
    input_format = _input_format(configuration)
    _verify_routed_input_format(
        artifact=artifact,
        source_bytes=source_bytes,
        producer=producer,
        input_format=input_format,
    )
    _verify_invocation_input_hash(
        artifact=artifact,
        producer=producer,
        input_format=input_format,
    )

    locators: tuple[NativeTextLocator, ...] = ()
    if input_format in {
        InputFormat.JATS,
        InputFormat.BIOC_JSON,
        InputFormat.BIOC_XML,
    }:
        locators = _decode_native_locators(
            artifact=artifact,
            source_bytes=source_bytes,
            producer=producer,
            input_format=input_format,
            native_locator_product=native_locator_product,
            native_locator_bytes=native_locator_bytes,
            adapter_producer=adapter_producer,
            html_projection_bytes=html_projection_bytes,
        )
    elif any(
        value is not None
        for value in (
            native_locator_product,
            native_locator_bytes,
            adapter_producer,
            html_projection_bytes,
            native_alignment_product,
            native_alignment_bytes,
        )
    ):
        raise ContentSpanReplayError(
            "non-native Docling formats cannot claim a native-locator overlay"
        )

    if input_format is InputFormat.PDF:
        _require_algorithm(
            configuration,
            key="pdf_span_algorithm",
            expected=_PDF_CONTENT_SPAN_ALGORITHM,
        )
        spans = build_pdf_content_spans(
            docling_document,
            artifact_id=artifact.artifact_id,
            processing_run_id=producer.run_id,
            representation_product_id=docling_product.product_id,
        )
    elif input_format is InputFormat.JATS:
        _require_algorithm(
            configuration,
            key="jats_locator_alignment_algorithm",
            expected=_NATIVE_ALIGNMENT_ALGORITHM,
        )
        alignment = align_jats_content_spans(
            docling_document,
            locators,
            artifact_id=artifact.artifact_id,
            processing_run_id=producer.run_id,
            representation_product_id=docling_product.product_id,
        )
        spans = alignment.spans
        _verify_native_alignment(
            producer=producer,
            input_format=input_format,
            alignment_product=native_alignment_product,
            alignment_bytes=native_alignment_bytes,
            aligned_count=alignment.aligned_count,
            unaligned_count=alignment.unaligned_count,
            records=[record.to_dict() for record in alignment.records],
        )
    elif input_format in {InputFormat.BIOC_JSON, InputFormat.BIOC_XML}:
        _require_algorithm(
            configuration,
            key="bioc_locator_alignment_algorithm",
            expected=_NATIVE_ALIGNMENT_ALGORITHM,
        )
        alignment = align_bioc_content_spans(
            docling_document,
            locators,
            artifact_id=artifact.artifact_id,
            processing_run_id=producer.run_id,
            representation_product_id=docling_product.product_id,
        )
        spans = alignment.spans
        _verify_native_alignment(
            producer=producer,
            input_format=input_format,
            alignment_product=native_alignment_product,
            alignment_bytes=native_alignment_bytes,
            aligned_count=alignment.aligned_count,
            unaligned_count=alignment.unaligned_count,
            records=[record.to_dict() for record in alignment.records],
        )
    else:
        try:
            docling_input_format = _DOCLING_INPUT_FORMATS[input_format]
        except KeyError as exc:  # pragma: no cover - guarded by _input_format
            raise ContentSpanReplayError(
                f"unsupported Docling input format: {input_format.value}"
            ) from exc
        spans = build_docling_content_spans(
            docling_document,
            artifact_id=artifact.artifact_id,
            processing_run_id=producer.run_id,
            input_format=docling_input_format,
            representation_product_id=docling_product.product_id,
        )

    expected = ContentSpanSet(
        artifact_id=artifact.artifact_id,
        processing_run_id=producer.run_id,
        representation_product_id=docling_product.product_id,
        spans=spans,
    )
    expected_bytes = _canonical_json_bytes(expected.model_dump(mode="json"))
    if persisted != expected or content_spans_bytes != expected_bytes:
        raise ContentSpanReplayError(
            "content-span product does not reproduce from its exact Docling sources"
        )
    return persisted


def _verify_docling_producer(
    *,
    artifact: DocumentArtifact,
    producer: ProcessingRun,
    docling_product: DataProductRef,
    content_spans_product: DataProductRef,
) -> None:
    if producer.artifact_id != artifact.artifact_id:
        raise ContentSpanReplayError(
            "content-span producer artifact does not match the Docling artifact"
        )
    if (
        producer.component_id,
        producer.component.capability,
    ) != ("docling", "document.parse"):
        raise ContentSpanReplayError(
            "content-span producer identity does not match the Docling contract"
        )
    if producer.status not in {
        ProcessingRunStatus.COMPLETE,
        ProcessingRunStatus.PARTIAL,
    }:
        raise ContentSpanReplayError(
            "content-span producer must have a complete or partial status"
        )
    _require_exact_output(producer, docling_product, "docling_document")
    _require_exact_output(producer, content_spans_product, "content_spans")


def _require_exact_output(
    producer: ProcessingRun,
    product: DataProductRef,
    name: str,
) -> None:
    matching = tuple(output for output in producer.outputs if output.name == name)
    if len(matching) != 1 or matching[0] != product:
        raise ContentSpanReplayError(
            f"Docling producer must declare exactly the supplied {name!r} product"
        )
    if product.producer_run_id != producer.run_id:
        raise ContentSpanReplayError(
            f"{name!r} product does not belong to its Docling producer"
        )


def _input_format(configuration: dict[str, Any]) -> InputFormat:
    value = configuration.get("input_format")
    if not isinstance(value, str):
        raise ContentSpanReplayError("Docling input_format configuration is invalid")
    try:
        input_format = InputFormat(value)
    except ValueError as exc:
        raise ContentSpanReplayError(
            f"unsupported Docling input format: {value!r}"
        ) from exc
    if input_format not in {
        InputFormat.PDF,
        InputFormat.JATS,
        InputFormat.BIOC_JSON,
        InputFormat.BIOC_XML,
        *_DOCLING_INPUT_FORMATS,
    }:
        raise ContentSpanReplayError(
            f"unsupported Docling input format: {input_format.value}"
        )
    return input_format


def _verify_invocation_input_hash(
    *,
    artifact: DocumentArtifact,
    producer: ProcessingRun,
    input_format: InputFormat,
) -> None:
    configured_hash = producer.configuration.get("input_sha256")
    if not isinstance(configured_hash, str):
        raise ContentSpanReplayError("Docling input_sha256 configuration is invalid")
    if input_format in {InputFormat.BIOC_JSON, InputFormat.BIOC_XML}:
        projections = tuple(
            product for product in producer.inputs if product.name == "html_projection"
        )
        if len(projections) != 1:
            raise ContentSpanReplayError(
                "BioC Docling replay requires exactly one HTML projection input"
            )
        expected_hash = projections[0].blob_sha256
    else:
        expected_hash = artifact.source_sha256
    if configured_hash != expected_hash:
        raise ContentSpanReplayError(
            "Docling input_sha256 does not match its durable invocation input"
        )


def _verify_routed_input_format(
    *,
    artifact: DocumentArtifact,
    source_bytes: bytes,
    producer: ProcessingRun,
    input_format: InputFormat,
) -> None:
    reject_extension_only = True
    if producer.output_policy_snapshot:
        routing_policy = producer.output_policy_snapshot.get("routing")
        if not isinstance(routing_policy, dict) or set(routing_policy) != {
            "algorithm_version",
            "reject_extension_only_detection",
        }:
            raise ContentSpanReplayError(
                "Docling routing policy does not match the production contract"
            )
        if routing_policy.get("algorithm_version") != _ROUTING_ALGORITHM_VERSION:
            raise ContentSpanReplayError(
                "Docling routing algorithm does not match the replay contract"
            )
        reject_extension_only = routing_policy.get("reject_extension_only_detection")
        if not isinstance(reject_extension_only, bool):
            raise ContentSpanReplayError(
                "Docling extension-only routing policy is invalid"
            )

    detected = DocumentRouter(reject_extension_only=reject_extension_only).detect(
        source_bytes,
        filename=(
            artifact.identifiers.get("filename")
            or filename_from_uri(artifact.acquisition_uri)
        ),
        media_type=artifact.media_type,
    )
    if detected is not input_format:
        raise ContentSpanReplayError(
            "Docling input_format does not match deterministic production routing"
        )


def _verify_native_alignment(
    *,
    producer: ProcessingRun,
    input_format: InputFormat,
    alignment_product: DataProductRef | None,
    alignment_bytes: bytes | None,
    aligned_count: int,
    unaligned_count: int,
    records: list[dict[str, Any]],
) -> None:
    expected_name = (
        "jats_locator_alignment"
        if input_format is InputFormat.JATS
        else "bioc_locator_alignment"
    )
    declared_alignments = tuple(
        output
        for output in producer.outputs
        if output.name in _NATIVE_ALIGNMENT_PRODUCT_NAMES
    )
    if (
        len(declared_alignments) != 1
        or declared_alignments[0].name != expected_name
        or alignment_product is None
        or alignment_bytes is None
    ):
        raise ContentSpanReplayError(
            "native Docling replay requires its exact locator-alignment output"
        )
    _require_exact_output(producer, alignment_product, expected_name)
    actual_hash = hashlib.sha256(alignment_bytes).hexdigest()
    if actual_hash != alignment_product.blob_sha256:
        raise ContentSpanReplayError(
            "native locator-alignment bytes do not match their durable product"
        )
    expected_bytes = _canonical_json_bytes(
        {
            "algorithm": _NATIVE_ALIGNMENT_ALGORITHM,
            "aligned_count": aligned_count,
            "unaligned_count": unaligned_count,
            "records": records,
        }
    )
    if alignment_bytes != expected_bytes:
        raise ContentSpanReplayError(
            "native locator-alignment output does not reproduce from its inputs"
        )
    if unaligned_count and producer.status is ProcessingRunStatus.COMPLETE:
        raise ContentSpanReplayError(
            "complete Docling producer masks unaligned native locators"
        )


def _decode_native_locators(
    *,
    artifact: DocumentArtifact,
    source_bytes: bytes,
    producer: ProcessingRun,
    input_format: InputFormat,
    native_locator_product: DataProductRef | None,
    native_locator_bytes: bytes | None,
    adapter_producer: ProcessingRun | None,
    html_projection_bytes: bytes | None,
) -> tuple[NativeTextLocator, ...]:
    if native_locator_product is None or native_locator_bytes is None:
        raise ContentSpanReplayError(
            "native Docling replay requires its exact native-locator output"
        )
    _require_exact_output(producer, native_locator_product, "native_locator_overlay")
    configured_hash = producer.configuration.get("native_locator_overlay_sha256")
    actual_hash = hashlib.sha256(native_locator_bytes).hexdigest()
    if (
        configured_hash != actual_hash
        or native_locator_product.blob_sha256 != actual_hash
    ):
        raise ContentSpanReplayError(
            "native-locator overlay hash does not match the Docling configuration"
        )
    input_overlays = tuple(
        product
        for product in producer.inputs
        if product.name == "native_locator_overlay"
    )
    if len(input_overlays) != 1 or input_overlays[0].blob_sha256 != actual_hash:
        raise ContentSpanReplayError(
            "native-locator output does not match the durable adapter input"
        )
    adapter_native_product = input_overlays[0]
    if adapter_producer is None:
        raise ContentSpanReplayError(
            "native Docling replay requires the durable adapter producer"
        )
    _verify_adapter_producer(
        artifact=artifact,
        producer=adapter_producer,
        input_format=input_format,
        native_locator_product=adapter_native_product,
        html_projection_bytes=html_projection_bytes,
        docling_producer=producer,
    )

    try:
        payload = json.loads(native_locator_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContentSpanReplayError(
            "native-locator overlay is not valid JSON"
        ) from exc
    if not isinstance(payload, list):
        raise ContentSpanReplayError("native-locator overlay must be a JSON array")

    locators: list[NativeTextLocator] = []
    normalized_payload: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != _NATIVE_LOCATOR_FIELDS:
            raise ContentSpanReplayError(
                f"native-locator entry {index} does not match the exact payload schema"
            )
        normalized = _strict_native_locator(
            cast("dict[str, Any]", item),
            index=index,
            input_format=input_format,
        )
        normalized_payload.append(normalized)
        locators.append(NativeTextLocator(**normalized))
    if _canonical_json_bytes(normalized_payload) != native_locator_bytes:
        raise ContentSpanReplayError(
            "native-locator overlay is not deterministically serialized"
        )
    _verify_native_source_replay(
        source_bytes=source_bytes,
        input_format=input_format,
        locator_bytes=native_locator_bytes,
        html_projection_bytes=html_projection_bytes,
    )
    if not locators and adapter_producer.status is not ProcessingRunStatus.PARTIAL:
        raise ContentSpanReplayError(
            "native-locator adapter cannot report complete for an empty replay"
        )
    return tuple(locators)


def _verify_adapter_producer(
    *,
    artifact: DocumentArtifact,
    producer: ProcessingRun,
    input_format: InputFormat,
    native_locator_product: DataProductRef,
    html_projection_bytes: bytes | None,
    docling_producer: ProcessingRun,
) -> None:
    component_id = (
        "jats-locator-adapter" if input_format is InputFormat.JATS else "bioc-adapter"
    )
    if producer.artifact_id != artifact.artifact_id:
        raise ContentSpanReplayError(
            "native-locator adapter artifact does not match the Docling artifact"
        )
    if (
        producer.component_id,
        producer.component_version,
        producer.component.capability,
    ) != (component_id, "1", "document.adapt"):
        raise ContentSpanReplayError(
            "native-locator adapter identity does not match its contract"
        )
    if producer.status not in {
        ProcessingRunStatus.COMPLETE,
        ProcessingRunStatus.PARTIAL,
    }:
        raise ContentSpanReplayError(
            "native-locator adapter status does not permit durable replay"
        )
    expected_configuration = {
        "adapter_version": "1",
        "input_format": input_format.value,
        "input_sha256": artifact.source_sha256,
    }
    if producer.configuration != expected_configuration:
        raise ContentSpanReplayError(
            "native-locator adapter configuration does not match its source"
        )
    _require_exact_output(producer, native_locator_product, "native_locator_overlay")

    if input_format is InputFormat.JATS:
        if html_projection_bytes is not None:
            raise ContentSpanReplayError(
                "JATS native-locator replay cannot claim an HTML projection"
            )
        if any(
            product.name == "html_projection" for product in docling_producer.inputs
        ):
            raise ContentSpanReplayError(
                "JATS Docling producer cannot claim a BioC HTML projection"
            )
        return

    projections = tuple(
        product
        for product in docling_producer.inputs
        if product.name == "html_projection"
    )
    if len(projections) != 1 or html_projection_bytes is None:
        raise ContentSpanReplayError(
            "BioC Docling replay requires its exact HTML projection"
        )
    _require_exact_output(producer, projections[0], "html_projection")
    if hashlib.sha256(html_projection_bytes).hexdigest() != projections[0].blob_sha256:
        raise ContentSpanReplayError(
            "BioC HTML projection bytes do not match their durable product"
        )


def _verify_native_source_replay(
    *,
    source_bytes: bytes,
    input_format: InputFormat,
    locator_bytes: bytes,
    html_projection_bytes: bytes | None,
) -> None:
    try:
        if input_format is InputFormat.JATS:
            locators = JATSLocatorAdapter().extract_locators(source_bytes)
            expected_projection = None
        else:
            adapted = BioCAdapter().adapt(source_bytes, input_format=input_format)
            locators = adapted.locator_overlay
            expected_projection = adapted.content
    except ValueError as exc:
        raise ContentSpanReplayError(
            "native source cannot be replayed by its exact adapter"
        ) from exc

    expected_locator_bytes = _canonical_json_bytes(
        [_native_locator_payload(locator) for locator in locators]
    )
    if locator_bytes != expected_locator_bytes:
        raise ContentSpanReplayError(
            "native-locator overlay does not reproduce from the raw source"
        )
    if expected_projection is not None and html_projection_bytes != expected_projection:
        raise ContentSpanReplayError(
            "BioC HTML projection does not reproduce from the raw source"
        )


def _native_locator_payload(locator: NativeTextLocator) -> dict[str, Any]:
    return {
        "text": locator.text,
        "source_kind": locator.source_kind,
        "document_index": locator.document_index,
        "document_id": locator.document_id,
        "passage_index": locator.passage_index,
        "sentence_index": locator.sentence_index,
        "offset": locator.offset,
        "length": locator.length,
        "xml_id": locator.xml_id,
        "xpath": locator.xpath,
        "infons": locator.infons,
    }


def _strict_native_locator(
    item: dict[str, Any],
    *,
    index: int,
    input_format: InputFormat,
) -> dict[str, Any]:
    if not isinstance(item["text"], str):
        raise ContentSpanReplayError(f"native-locator entry {index} text is invalid")
    if item["source_kind"] != input_format.value:
        raise ContentSpanReplayError(
            f"native-locator entry {index} source_kind does not match input_format"
        )
    for field in (
        "document_index",
        "passage_index",
        "sentence_index",
        "offset",
        "length",
    ):
        value = item[field]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ContentSpanReplayError(
                f"native-locator entry {index} {field} is invalid"
            )
    for field in ("document_id", "xml_id", "xpath"):
        value = item[field]
        if value is not None and not isinstance(value, str):
            raise ContentSpanReplayError(
                f"native-locator entry {index} {field} is invalid"
            )
    infons = item["infons"]
    if infons is not None and (
        not isinstance(infons, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in infons.items()
        )
    ):
        raise ContentSpanReplayError(f"native-locator entry {index} infons are invalid")
    return dict(item)


def _require_algorithm(
    configuration: dict[str, Any],
    *,
    key: str,
    expected: str,
) -> None:
    if configuration.get(key) != expected:
        raise ContentSpanReplayError(
            f"Docling {key} does not match the replay contract"
        )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = ["ContentSpanReplayError", "replay_content_span_set"]
