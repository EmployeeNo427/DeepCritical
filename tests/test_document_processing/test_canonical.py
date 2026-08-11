from __future__ import annotations

import json
import random
from copy import copy, deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pypdf import PdfReader

from DeepResearch.src.document_processing import canonical as canonical_module
from DeepResearch.src.document_processing.adapters import (
    BioCAdapter,
    JATSLocatorAdapter,
)
from DeepResearch.src.document_processing.alignment import DoclingGrobidAligner
from DeepResearch.src.document_processing.canonical import (
    CANONICAL_COMPONENT_CAPABILITY,
    CANONICAL_COMPONENT_ID,
    CANONICAL_COMPONENT_VERSION,
    CANONICAL_DOCUMENT_SCHEMA_VERSION,
    MAX_CANONICAL_TABLE_AXIS,
    MAX_CANONICAL_TABLE_CELLS,
    CanonicalAnchorRole,
    CanonicalBlock,
    CanonicalBlockKind,
    CanonicalDiagnosticSeverity,
    CanonicalDocumentError,
    CanonicalDocumentMetadata,
    CanonicalDocumentView,
    CanonicalizationConfig,
    CanonicalMappingDiagnostic,
    CanonicalRelationship,
    CanonicalRelationshipKind,
    CanonicalRelationshipStatus,
    CanonicalSourceAnchor,
    CanonicalTable,
    CanonicalTableCell,
    InvalidCanonicalDocumentError,
    UnsupportedCanonicalDocumentVersionError,
    build_canonical_document_view,
    canonical_block_content_sha256,
    canonical_block_id,
    canonical_diagnostic_id,
    canonical_document_bytes,
    canonical_invocation_configuration,
    canonical_relationship_id,
    canonical_relationship_status,
    canonical_view_id,
    load_canonical_document,
    normalize_canonical_text,
)
from DeepResearch.src.document_processing.models import (
    ArtifactLocation,
    ArtifactLocationRole,
    BioCLocator,
    ComponentDescriptor,
    ContentSpan,
    ContentSpanSet,
    DocumentArtifact,
    JatsLocator,
    PdfBoundingBox,
    PdfLocator,
    ProcessingRun,
    ProcessingRunStatus,
    RepresentationAnchor,
    configuration_sha256,
    sha256_bytes,
)
from DeepResearch.src.document_processing.products import build_data_product_ref
from DeepResearch.src.document_processing.routing import InputFormat
from DeepResearch.src.document_processing.storage import ContentAddressedStore
from DeepResearch.src.document_processing.validation import (
    align_bioc_content_spans,
    align_jats_content_spans,
    build_pdf_content_spans,
    docling_document_sha256,
    validate_content_integrity,
)

FIXTURE_ROOT = (
    Path(__file__).parents[1] / "fixtures" / "document_processing" / "canonical"
)
FIXTURE_V1_ROOT = FIXTURE_ROOT / "v1"


def _document() -> dict[str, Any]:
    return json.loads(
        (FIXTURE_V1_ROOT / "docling_document.json").read_text(encoding="utf-8")
    )


def _tei() -> bytes:
    return (FIXTURE_V1_ROOT / "grobid.tei.xml").read_bytes()


def _canonical_json_fixture_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _product_from_bytes(
    name: str,
    payload: bytes,
    *,
    artifact_id: str,
    producer_run_id: str,
):
    digest = sha256_bytes(payload)
    return build_data_product_ref(
        name=name,
        blob_sha256=digest,
        uri=f"cas://sha256/{digest}",
        byte_size=len(payload),
        producer_run_id=producer_run_id,
        source_artifact_ids=(artifact_id,),
    )


def _product(
    name: str,
    digest_character: str,
    *,
    artifact_id: str,
    producer_run_id: str,
):
    return build_data_product_ref(
        name=name,
        blob_sha256=digest_character * 64,
        uri=f"cas://sha256/{digest_character * 64}",
        byte_size=64,
        producer_run_id=producer_run_id,
        source_artifact_ids=(artifact_id,),
    )


def _retarget_span_set(
    span_set: ContentSpanSet,
    *,
    artifact_id: str | None = None,
    processing_run_id: str | None = None,
    representation_product_id: str | None = None,
) -> ContentSpanSet:
    """Return a fully valid span set with consistently replaced lineage."""

    payload = span_set.model_dump(mode="python")
    target_artifact = artifact_id or span_set.artifact_id
    target_run = processing_run_id or span_set.processing_run_id
    target_product = representation_product_id or span_set.representation_product_id
    payload["artifact_id"] = target_artifact
    payload["processing_run_id"] = target_run
    payload["representation_product_id"] = target_product
    for span in payload["spans"]:
        span["artifact_id"] = target_artifact
        span["processing_run_id"] = target_run
        span["representation_anchor"]["product_id"] = target_product
    return ContentSpanSet.model_validate(payload)


def _locator(format_name: str, index: int, length: int):
    if format_name == "pdf":
        return PdfLocator(
            page_number=1,
            bounding_box=PdfBoundingBox(
                left=10,
                top=10 + index,
                right=90,
                bottom=20 + index,
                page_width=100,
                page_height=100,
            ),
        )
    if format_name == "jats":
        return JatsLocator(
            xml_id=f"node-{index}",
            xpath=f"/article[1]/body[1]/p[{index + 1}]",
        )
    return BioCLocator(
        document_index=0,
        document_id="PMC-CANONICAL",
        passage_index=index,
        offset=index * 100,
        length=length,
    )


def _fixture_bundle(
    format_name: str,
) -> tuple[CanonicalDocumentView, dict[str, bytes]]:
    document = _document()
    artifact_id = f"canonical-{format_name}"
    source_bytes = (
        FIXTURE_V1_ROOT
        / {
            "pdf": "article.pdf",
            "jats": "article.jats.xml",
            "bioc": "article.bioc.json",
        }[format_name]
    ).read_bytes()
    source_sha256 = sha256_bytes(source_bytes)
    artifact = DocumentArtifact(
        artifact_id=artifact_id,
        source_sha256=source_sha256,
        acquisition_uri=f"https://example.test/{artifact_id}",
        identifiers={"pmc": "PMC-CANONICAL"},
        media_type={
            "pdf": "application/pdf",
            "jats": "application/xml",
            "bioc": "application/bioc+json",
        }[format_name],
        raw_location=ArtifactLocation(
            uri=f"cas://sha256/{source_sha256}",
            sha256=source_sha256,
            byte_size=len(source_bytes),
            media_type={
                "pdf": "application/pdf",
                "jats": "application/xml",
                "bioc": "application/bioc+json",
            }[format_name],
            role=ArtifactLocationRole.RAW,
        ),
    )
    producer = f"docling-{format_name}"
    docling_bytes = _canonical_json_fixture_bytes(document)
    docling_product = _product_from_bytes(
        "docling_document",
        docling_bytes,
        artifact_id=artifact_id,
        producer_run_id=producer,
    )
    if format_name == "pdf":
        spans = build_pdf_content_spans(
            document,
            artifact_id=artifact_id,
            processing_run_id=producer,
            representation_product_id=docling_product.product_id,
        )
    elif format_name == "jats":
        native_locators = JATSLocatorAdapter().extract_locators(source_bytes)
        spans = align_jats_content_spans(
            document,
            native_locators,
            artifact_id=artifact_id,
            processing_run_id=producer,
            representation_product_id=docling_product.product_id,
        ).spans
    else:
        native_locators = (
            BioCAdapter()
            .adapt(
                source_bytes,
                input_format=InputFormat.BIOC_JSON,
            )
            .locator_overlay
        )
        spans = align_bioc_content_spans(
            document,
            native_locators,
            artifact_id=artifact_id,
            processing_run_id=producer,
            representation_product_id=docling_product.product_id,
        ).spans
    span_set = ContentSpanSet(
        artifact_id=artifact_id,
        processing_run_id=producer,
        representation_product_id=docling_product.product_id,
        spans=spans,
    )
    spans_bytes = _canonical_json_fixture_bytes(span_set.model_dump(mode="json"))
    content_spans_product = _product_from_bytes(
        "content_spans",
        spans_bytes,
        artifact_id=artifact_id,
        producer_run_id=producer,
    )
    overlay = (
        DoclingGrobidAligner(minimum_score=0.7).align(document, _tei())
        if format_name == "pdf"
        else None
    )
    alignment_bytes = (
        _canonical_json_fixture_bytes(overlay.to_dict())
        if overlay is not None
        else None
    )
    integrity = validate_content_integrity(
        document, scholarly_overlay=overlay
    ).to_dict()
    integrity_bytes = _canonical_json_fixture_bytes(integrity)
    source_products = [docling_product, content_spans_product]
    if overlay is not None:
        grobid_product = _product_from_bytes(
            "grobid_tei",
            _tei(),
            artifact_id=artifact_id,
            producer_run_id=f"grobid-{format_name}",
        )
        alignment_product = _product_from_bytes(
            "alignment_overlay",
            alignment_bytes,
            artifact_id=artifact_id,
            producer_run_id=f"alignment-{format_name}",
        )
        source_products.extend((grobid_product, alignment_product))
    integrity_product = _product_from_bytes(
        "content_integrity_overlay",
        integrity_bytes,
        artifact_id=artifact_id,
        producer_run_id=f"integrity-{format_name}",
    )
    source_products.append(integrity_product)
    view = build_canonical_document_view(
        artifact=artifact,
        docling_document=document,
        docling_product=docling_product,
        content_span_set=span_set,
        source_products=tuple(source_products),
        configuration=CanonicalizationConfig(),
        scholarly_overlay=overlay,
        integrity_report=integrity,
    )
    intermediates = {
        "content_integrity_overlay.json": integrity_bytes,
        "content_spans.json": spans_bytes,
    }
    if alignment_bytes is not None:
        intermediates["alignment_overlay.json"] = alignment_bytes
    return view, intermediates


def _fixture(format_name: str) -> CanonicalDocumentView:
    return _fixture_bundle(format_name)[0]


@pytest.mark.parametrize("format_name", ["pdf", "jats", "bioc"])
def test_golden_canonical_views_preserve_real_pipeline_anchoring(
    format_name: str,
) -> None:
    first, first_intermediates = _fixture_bundle(format_name)
    second, second_intermediates = _fixture_bundle(format_name)
    expected_bytes = (FIXTURE_ROOT / f"{format_name}.json").read_bytes()
    expected = load_canonical_document(expected_bytes)

    assert first == second
    assert first_intermediates == second_intermediates
    assert set(first_intermediates) == {
        "content_integrity_overlay.json",
        "content_spans.json",
        *({"alignment_overlay.json"} if format_name == "pdf" else set()),
    }
    for name, actual_bytes in first_intermediates.items():
        assert actual_bytes == (FIXTURE_V1_ROOT / format_name / name).read_bytes()
    assert canonical_document_bytes(first) == canonical_document_bytes(second)
    assert first == expected
    assert canonical_document_bytes(first) == expected_bytes
    assert first.view_id.startswith("canonical-view-")
    assert all(block.block_id.startswith("block-") for block in first.blocks)
    unlocated_text_nodes = {
        block.native_node_id
        for block in first.blocks
        if block.text is not None
        and not any(
            anchor.source_locator is not None for anchor in block.source_anchors
        )
    }
    assert unlocated_text_nodes == {"#/formulas/0"}
    assert [
        (diagnostic.code, diagnostic.native_node_ids)
        for diagnostic in first.diagnostics
    ] == [("MISSING_SOURCE_SPAN", ("#/formulas/0",))]
    assert all(
        relation.status is CanonicalRelationshipStatus.RESOLVED
        for relation in first.relationships
    )
    scholarly_anchors = [
        anchor
        for block in first.blocks
        for anchor in block.source_anchors
        if anchor.role is CanonicalAnchorRole.SCHOLARLY
    ] + [
        relation.source_anchor
        for relation in first.relationships
        if relation.source_anchor is not None
    ]
    assert bool(scholarly_anchors) is (format_name == "pdf")
    assert all(
        anchor.char_start is None and anchor.char_end is None
        for anchor in scholarly_anchors
    )


def test_canonical_sources_contain_the_represented_text() -> None:
    document = _document()
    represented_text = {
        item["text"] for item in document["texts"] if isinstance(item, dict)
    }
    represented_text.update(
        cell["text"]
        for table in document["tables"]
        for cell in table["data"]["table_cells"]
    )
    represented_text.update(
        item["text"] for item in document["formulas"] if isinstance(item, dict)
    )

    pdf_text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(
            BytesIO((FIXTURE_V1_ROOT / "article.pdf").read_bytes())
        ).pages
    )
    assert all(text in pdf_text for text in represented_text)
    scholarly_text = {
        annotation.text
        for annotation in DoclingGrobidAligner().extract_annotations(_tei())
    }
    assert all(text in pdf_text for text in scholarly_text)

    jats_text = {
        locator.text
        for locator in JATSLocatorAdapter().extract_locators(
            (FIXTURE_V1_ROOT / "article.jats.xml").read_bytes()
        )
    }
    assert represented_text <= jats_text

    bioc_text = {
        locator.text
        for locator in BioCAdapter()
        .adapt(
            (FIXTURE_V1_ROOT / "article.bioc.json").read_bytes(),
            input_format=InputFormat.BIOC_JSON,
        )
        .locator_overlay
    }
    assert represented_text <= bioc_text


def test_canonical_fixture_manifest_matches_every_frozen_file() -> None:
    manifest = json.loads((FIXTURE_V1_ROOT / "manifest.json").read_bytes())
    assert manifest["schema_version"] == ("deepcritical-canonical-fixture-manifest-v1")
    assert manifest["external_services"]["captured_from_services"] is False
    assert manifest["external_services"]["container_image_digests"] == {}
    for relative_path, expected in manifest["files"].items():
        payload = (FIXTURE_V1_ROOT / relative_path).resolve().read_bytes()
        assert len(payload) == expected["byte_size"]
        assert sha256_bytes(payload) == expected["sha256"]
        product_bytes = expected.get("product_bytes")
        if product_bytes is None:
            continue
        encoded = _canonical_json_fixture_bytes(json.loads(payload))
        assert len(encoded) == product_bytes["byte_size"]
        assert sha256_bytes(encoded) == product_bytes["sha256"]


def test_normalization_and_table_content_are_processor_independent() -> None:
    view = _fixture("pdf")

    assert normalize_canonical_text(" A\tB\nC ") == "A B C"
    title = next(
        block for block in view.blocks if block.kind is CanonicalBlockKind.TITLE
    )
    table = next(
        block for block in view.blocks if block.kind is CanonicalBlockKind.TABLE
    )
    assert title.text == "APOE4 pathway"
    assert table.table is not None
    assert table.table == CanonicalTable(
        row_count=2,
        column_count=2,
        cells=(
            CanonicalTableCell(
                text="Group",
                row_index=0,
                column_index=0,
                column_header=True,
            ),
            CanonicalTableCell(
                text="N",
                row_index=0,
                column_index=1,
                column_header=True,
            ),
            CanonicalTableCell(
                text="APOE4",
                row_index=1,
                column_index=0,
                row_header=True,
            ),
            CanonicalTableCell(text="12", row_index=1, column_index=1),
        ),
    )


def test_structured_docling_tables_preserve_coordinates_spans_and_headers() -> None:
    document = _document()
    spanning_header = {
        "text": " Cohort\nlabel ",
        "start_row_offset_idx": 0,
        "end_row_offset_idx": 1,
        "start_col_offset_idx": 0,
        "end_col_offset_idx": 2,
        "column_header": True,
    }
    document["tables"][0]["data"] = {
        "num_rows": 2,
        "num_cols": 3,
        "table_cells": [
            spanning_header,
            {
                "text": "N",
                "start_row_offset_idx": 0,
                "end_row_offset_idx": 1,
                "start_col_offset_idx": 2,
                "end_col_offset_idx": 3,
                "column_header": True,
                "fillable": True,
            },
            {
                "text": "APOE4",
                "start_row_offset_idx": 1,
                "start_col_offset_idx": 0,
                "row_header": True,
            },
            {
                "text": "12",
                "start_row_offset_idx": 1,
                "start_col_offset_idx": 1,
                "row_section": True,
            },
            {
                "text": "measured",
                "start_row_offset_idx": 1,
                "start_col_offset_idx": 2,
            },
        ],
        "grid": [["ignored because table_cells is authoritative"]],
    }

    view = build_canonical_document_view(**_fixture_inputs("pdf", document))
    table = next(
        block.table for block in view.blocks if block.kind is CanonicalBlockKind.TABLE
    )

    assert table == CanonicalTable(
        row_count=2,
        column_count=3,
        cells=(
            CanonicalTableCell(
                text="Cohort label",
                row_index=0,
                column_index=0,
                column_span=2,
                column_header=True,
            ),
            CanonicalTableCell(
                text="N",
                row_index=0,
                column_index=2,
                column_header=True,
                fillable=True,
            ),
            CanonicalTableCell(
                text="APOE4",
                row_index=1,
                column_index=0,
                row_header=True,
            ),
            CanonicalTableCell(
                text="12",
                row_index=1,
                column_index=1,
                row_section=True,
            ),
            CanonicalTableCell(
                text="measured",
                row_index=1,
                column_index=2,
            ),
        ),
    )


def test_empty_structured_table_cells_are_authoritative_over_grid() -> None:
    document = _document()
    document["tables"][0]["data"] = {
        "table_cells": [],
        "grid": [["must not be used"]],
    }

    view = build_canonical_document_view(**_fixture_inputs("pdf", document))
    table = next(
        block.table for block in view.blocks if block.kind is CanonicalBlockKind.TABLE
    )

    assert table == CanonicalTable()


@pytest.mark.parametrize(
    ("table", "match"),
    [
        (
            {
                "row_count": 1,
                "column_count": 1,
                "cells": [
                    {"text": "a", "row_index": 0, "column_index": 0},
                    {"text": "b", "row_index": 0, "column_index": 0},
                ],
            },
            "coordinates must be unique",
        ),
        (
            {
                "row_count": 1,
                "column_count": 1,
                "cells": [
                    {
                        "text": "a",
                        "row_index": 0,
                        "column_index": 0,
                        "row_span": 2,
                    }
                ],
            },
            "exceeds row_count",
        ),
        (
            {
                "row_count": 1,
                "column_count": 1,
                "cells": [
                    {
                        "text": "a",
                        "row_index": 0,
                        "column_index": 0,
                        "column_span": 2,
                    }
                ],
            },
            "exceeds column_count",
        ),
        (
            {
                "row_count": 1,
                "column_count": 2,
                "cells": [
                    {
                        "text": "a",
                        "row_index": 0,
                        "column_index": 0,
                        "column_span": 2,
                    },
                    {"text": "b", "row_index": 0, "column_index": 1},
                ],
            },
            "extents must not overlap",
        ),
        ({"row_count": 1, "column_count": 1}, "must have zero dimensions"),
    ],
)
def test_canonical_table_contract_rejects_inconsistent_grids(
    table: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        CanonicalTable.model_validate(table)


@pytest.mark.parametrize(
    ("grid", "match"),
    [
        (
            [[{"text": "a"}, {"text": "b", "start_col_offset_idx": 0}]],
            "conflicting cells",
        ),
        (
            [
                [
                    {"text": "a", "column_span": 2},
                    {"text": "b"},
                ]
            ],
            "overlapping cell extents",
        ),
    ],
)
def test_builder_rejects_conflicting_structured_table_cells(
    grid: list[list[dict[str, Any]]], match: str
) -> None:
    document = _document()
    document["tables"][0]["data"] = {"grid": grid}

    with pytest.raises(CanonicalDocumentError, match=match):
        build_canonical_document_view(**_fixture_inputs("pdf", document))


def test_builder_deduplicates_repeated_spanning_cells_in_grid_fallback() -> None:
    document = _document()
    spanning = {
        "text": "header",
        "start_row_offset_idx": 0,
        "end_row_offset_idx": 1,
        "start_col_offset_idx": 0,
        "end_col_offset_idx": 2,
    }
    document["tables"][0]["data"] = {
        "grid": [[spanning, deepcopy(spanning)]],
    }

    view = build_canonical_document_view(**_fixture_inputs("pdf", document))
    table = next(
        block.table for block in view.blocks if block.kind is CanonicalBlockKind.TABLE
    )
    assert table == CanonicalTable(
        row_count=1,
        column_count=2,
        cells=(
            CanonicalTableCell(
                text="header",
                row_index=0,
                column_index=0,
                column_span=2,
            ),
        ),
    )


def test_builder_rejects_duplicate_authoritative_cells() -> None:
    document = _document()
    cell = {
        "text": "header",
        "start_row_offset_idx": 0,
        "start_col_offset_idx": 0,
    }
    document["tables"][0]["data"] = {
        "table_cells": [cell, deepcopy(cell)],
    }

    with pytest.raises(CanonicalDocumentError, match="conflicting cells"):
        build_canonical_document_view(**_fixture_inputs("pdf", document))


def test_table_overlap_validation_is_span_bounded_and_half_open() -> None:
    disjoint_huge = CanonicalTable(
        row_count=MAX_CANONICAL_TABLE_AXIS,
        column_count=2,
        cells=(
            CanonicalTableCell(
                text="left",
                row_index=0,
                column_index=0,
                row_span=MAX_CANONICAL_TABLE_AXIS,
            ),
            CanonicalTableCell(
                text="right",
                row_index=0,
                column_index=1,
                row_span=MAX_CANONICAL_TABLE_AXIS,
            ),
        ),
    )
    assert len(disjoint_huge.cells) == 2

    edge_touching = CanonicalTable(
        row_count=MAX_CANONICAL_TABLE_AXIS,
        column_count=1,
        cells=(
            CanonicalTableCell(
                text="top",
                row_index=0,
                column_index=0,
                row_span=MAX_CANONICAL_TABLE_AXIS // 2,
            ),
            CanonicalTableCell(
                text="bottom",
                row_index=MAX_CANONICAL_TABLE_AXIS // 2,
                column_index=0,
                row_span=MAX_CANONICAL_TABLE_AXIS // 2,
            ),
        ),
    )
    assert len(edge_touching.cells) == 2

    with pytest.raises(ValidationError, match="extents must not overlap"):
        CanonicalTable(
            row_count=MAX_CANONICAL_TABLE_AXIS,
            column_count=2,
            cells=(
                CanonicalTableCell(
                    text="wide",
                    row_index=0,
                    column_index=0,
                    row_span=MAX_CANONICAL_TABLE_AXIS,
                    column_span=2,
                ),
                CanonicalTableCell(
                    text="overlap",
                    row_index=MAX_CANONICAL_TABLE_AXIS - 1,
                    column_index=1,
                ),
            ),
        )


def test_table_rectangle_sweep_matches_seeded_brute_force() -> None:
    random_source = random.Random(427)

    def overlaps(left: CanonicalTableCell, right: CanonicalTableCell) -> bool:
        return (
            left.row_index < right.row_index + right.row_span
            and right.row_index < left.row_index + left.row_span
            and left.column_index < right.column_index + right.column_span
            and right.column_index < left.column_index + left.column_span
        )

    for sample_index in range(250):
        coordinates: set[tuple[int, int]] = set()
        cells: list[CanonicalTableCell] = []
        for cell_index in range(random_source.randint(1, 8)):
            while True:
                row = random_source.randrange(6)
                column = random_source.randrange(6)
                if (row, column) not in coordinates:
                    coordinates.add((row, column))
                    break
            cells.append(
                CanonicalTableCell(
                    text=f"{sample_index}-{cell_index}",
                    row_index=row,
                    column_index=column,
                    row_span=random_source.randint(1, 6 - row),
                    column_span=random_source.randint(1, 6 - column),
                )
            )
        expected_overlap = any(
            overlaps(left, right)
            for left_index, left in enumerate(cells)
            for right in cells[left_index + 1 :]
        )
        if expected_overlap:
            with pytest.raises(ValidationError, match="extents must not overlap"):
                CanonicalTable(row_count=6, column_count=6, cells=tuple(cells))
        else:
            assert CanonicalTable(
                row_count=6,
                column_count=6,
                cells=tuple(cells),
            ).cells == tuple(cells)


def test_canonical_table_contract_enforces_fixed_resource_limits() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1000000"):
        CanonicalTable(row_count=MAX_CANONICAL_TABLE_AXIS + 1)
    assert any(
        getattr(constraint, "max_length", None) == MAX_CANONICAL_TABLE_CELLS
        for constraint in CanonicalTable.model_fields["cells"].metadata
    )

    document = _document()
    document["tables"][0]["data"] = {
        "table_cells": [{}] * (MAX_CANONICAL_TABLE_CELLS + 1)
    }
    with pytest.raises(CanonicalDocumentError, match="cell limit"):
        build_canonical_document_view(**_fixture_inputs("pdf", document))


@pytest.mark.parametrize(
    "cell",
    [
        3,
        {"text": "missing coordinates"},
        {
            "text": "bad coordinate",
            "start_row_offset_idx": "0",
            "start_col_offset_idx": 0,
        },
        {
            "text": "bad flag",
            "start_row_offset_idx": 0,
            "start_col_offset_idx": 0,
            "column_header": 1,
        },
        {
            "text": "bad span",
            "start_row_offset_idx": 0,
            "start_col_offset_idx": 0,
            "end_col_offset_idx": 2,
            "column_span": 1,
        },
        {
            "start_row_offset_idx": 0,
            "start_col_offset_idx": 0,
        },
        {
            "text": "bad ref",
            "start_row_offset_idx": 0,
            "start_col_offset_idx": 0,
            "ref": {"missing": "#/groups/0"},
        },
        {
            "text": "disagreeing aliases",
            "start_row_offset_idx": 0,
            "start_col_offset_idx": 0,
            "column_span": 1,
            "col_span": 2,
        },
        {
            "text": "zero span",
            "start_row_offset_idx": 0,
            "start_col_offset_idx": 0,
            "row_span": 0,
        },
        {
            "text": "bad end",
            "start_row_offset_idx": 0,
            "start_col_offset_idx": 0,
            "end_row_offset_idx": 0,
        },
    ],
)
def test_malformed_authoritative_table_cells_are_explicit_errors(cell: Any) -> None:
    document = _document()
    document["tables"][0]["data"] = {
        "table_cells": [cell],
        "grid": [["must not be used"]],
    }

    view = build_canonical_document_view(**_fixture_inputs("pdf", document))
    table = next(
        block.table for block in view.blocks if block.kind is CanonicalBlockKind.TABLE
    )
    assert table == CanonicalTable()
    diagnostics = [
        diagnostic
        for diagnostic in view.diagnostics
        if diagnostic.code in {"INVALID_TABLE_CELL", "EMPTY_TABLE_DATA"}
    ]
    assert {diagnostic.code for diagnostic in diagnostics} == {
        "INVALID_TABLE_CELL",
        "EMPTY_TABLE_DATA",
    }
    assert all(
        diagnostic.severity is CanonicalDiagnosticSeverity.ERROR
        for diagnostic in diagnostics
    )


@pytest.mark.parametrize("cell", [True, float("inf")])
def test_invalid_grid_scalars_are_not_stringified(cell: Any) -> None:
    document = _document()
    document["tables"][0]["data"] = {"grid": [[cell]]}

    view = build_canonical_document_view(**_fixture_inputs("pdf", document))

    assert {diagnostic.code for diagnostic in view.diagnostics} >= {
        "INVALID_TABLE_CELL",
        "EMPTY_TABLE_DATA",
    }


def test_malformed_authoritative_cell_collection_never_falls_back_to_grid() -> None:
    document = _document()
    document["tables"][0]["data"] = {
        "table_cells": "not-a-list",
        "grid": [["must not be used"]],
    }

    view = build_canonical_document_view(**_fixture_inputs("pdf", document))
    table = next(
        block.table for block in view.blocks if block.kind is CanonicalBlockKind.TABLE
    )
    assert table == CanonicalTable()
    assert "INVALID_TABLE_CELLS" in {diagnostic.code for diagnostic in view.diagnostics}


def test_valid_grid_fallback_has_explicit_scalar_and_reference_semantics() -> None:
    document = _document()
    document["tables"][0]["data"] = {
        "grid": [
            [
                None,
                1.5,
                {"text": "linked", "ref": "#/groups/0"},
            ]
        ]
    }

    view = build_canonical_document_view(**_fixture_inputs("pdf", document))
    table = next(
        block.table for block in view.blocks if block.kind is CanonicalBlockKind.TABLE
    )
    assert table is not None
    assert [cell.text for cell in table.cells] == ["", "1.5", "linked"]
    assert table.cells[-1].native_ref == "#/groups/0"


def test_table_parser_hard_limits_have_stable_failure_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    too_many_rows = _document()
    too_many_rows["tables"][0]["data"] = {"grid": [[], []]}
    row_inputs = _fixture_inputs("pdf", too_many_rows)
    too_many_columns = _document()
    too_many_columns["tables"][0]["data"] = {"grid": [["a", "b"]]}
    column_inputs = _fixture_inputs("pdf", too_many_columns)
    invalid_dimension = _document()
    invalid_dimension["tables"][0]["data"] = {
        "num_rows": "1",
        "grid": [["a"]],
    }
    dimension_inputs = _fixture_inputs("pdf", invalid_dimension)

    monkeypatch.setattr(canonical_module, "MAX_CANONICAL_TABLE_AXIS", 1)
    with pytest.raises(CanonicalDocumentError, match="TABLE_LIMIT_EXCEEDED"):
        build_canonical_document_view(**row_inputs)

    with pytest.raises(CanonicalDocumentError, match="TABLE_LIMIT_EXCEEDED"):
        build_canonical_document_view(**column_inputs)

    with pytest.raises(CanonicalDocumentError, match="INVALID_TABLE_DIMENSIONS"):
        build_canonical_document_view(**dimension_inputs)


@pytest.mark.parametrize(
    "data",
    [
        {
            "num_rows": MAX_CANONICAL_TABLE_AXIS + 1,
            "grid": [["value"]],
        },
        {
            "table_cells": [
                {
                    "text": "coordinate over limit",
                    "start_row_offset_idx": MAX_CANONICAL_TABLE_AXIS,
                    "start_col_offset_idx": 0,
                }
            ],
        },
        {
            "table_cells": [
                {
                    "text": "span over limit",
                    "start_row_offset_idx": 0,
                    "start_col_offset_idx": 0,
                    "row_span": MAX_CANONICAL_TABLE_AXIS + 1,
                }
            ],
        },
        {
            "table_cells": [
                {
                    "text": "extent over limit",
                    "start_row_offset_idx": MAX_CANONICAL_TABLE_AXIS - 1,
                    "start_col_offset_idx": 0,
                    "row_span": 2,
                }
            ],
        },
        {
            "table_cells": [
                {
                    "text": "end over limit",
                    "start_row_offset_idx": 0,
                    "start_col_offset_idx": 0,
                    "end_row_offset_idx": MAX_CANONICAL_TABLE_AXIS + 1,
                }
            ],
        },
    ],
)
def test_all_table_resource_bound_overages_are_hard_failures(
    data: dict[str, Any],
) -> None:
    document = _document()
    document["tables"][0]["data"] = data

    with pytest.raises(CanonicalDocumentError, match="TABLE_LIMIT_EXCEEDED"):
        build_canonical_document_view(**_fixture_inputs("pdf", document))


def test_builder_does_not_infer_over_invalid_declared_table_dimensions() -> None:
    document = _document()
    document["tables"][0]["data"] = {
        "num_rows": 1,
        "num_cols": 1,
        "table_cells": [
            {
                "text": "wide",
                "start_row_offset_idx": 0,
                "start_col_offset_idx": 0,
                "column_span": 2,
            }
        ],
    }
    with pytest.raises(CanonicalDocumentError, match="declared column dimension"):
        build_canonical_document_view(**_fixture_inputs("pdf", document))

    document["tables"][0]["data"] = {
        "num_rows": 1,
        "num_cols": 1,
        "table_cells": [
            {
                "text": "tall",
                "start_row_offset_idx": 0,
                "start_col_offset_idx": 0,
                "row_span": 2,
            }
        ],
    }
    with pytest.raises(CanonicalDocumentError, match="declared row dimension"):
        build_canonical_document_view(**_fixture_inputs("pdf", document))


def test_canonical_schema_dispatch_rejects_unknown_and_invalid_payloads() -> None:
    view = _fixture("jats")
    payload = json.loads(canonical_document_bytes(view))

    payload["schema_version"] = "deepcritical-canonical-document-view-v99"
    with pytest.raises(UnsupportedCanonicalDocumentVersionError, match="v99"):
        load_canonical_document(json.dumps(payload).encode())
    payload.pop("schema_version")
    with pytest.raises(UnsupportedCanonicalDocumentVersionError, match="None"):
        load_canonical_document(json.dumps(payload).encode())
    with pytest.raises(InvalidCanonicalDocumentError, match="valid JSON"):
        load_canonical_document(b"{not-json")
    with pytest.raises(InvalidCanonicalDocumentError, match="JSON object"):
        load_canonical_document(b"[]")

    tampered = json.loads(canonical_document_bytes(view))
    tampered["blocks"][0]["content_sha256"] = "0" * 64
    with pytest.raises(InvalidCanonicalDocumentError, match="contract"):
        load_canonical_document(json.dumps(tampered).encode())


def test_metadata_is_deeply_immutable_and_serialization_revalidates() -> None:
    view = _fixture("pdf")

    with pytest.raises(TypeError, match="does not support item assignment"):
        view.metadata.identifiers["doi"] = "mutable"  # type: ignore[invalid-assignment]
    assert view.metadata.model_dump(mode="json")["identifiers"] == {
        "pmc": "PMC-CANONICAL"
    }
    assert deepcopy(view) == view
    assert view.model_copy(deep=True) == view
    identifiers = CanonicalDocumentMetadata(
        media_type="application/pdf",
        identifiers={"first": "1", "second": "2"},
    ).identifiers
    reversed_identifiers = CanonicalDocumentMetadata(
        media_type="application/pdf",
        identifiers={"second": "2", "first": "1"},
    ).identifiers
    assert identifiers["second"] == "2"
    with pytest.raises(KeyError, match="missing"):
        identifiers["missing"]
    assert len(identifiers) == 2
    assert repr(identifiers) == "{'first': '1', 'second': '2'}"
    assert identifiers == reversed_identifiers
    assert hash(identifiers) == hash(reversed_identifiers)
    assert identifiers != {"first": "different", "second": "2"}
    assert identifiers != object()
    assert copy(identifiers) is identifiers

    stale = view.model_copy(update={"view_id": "canonical-view-stale"})
    with pytest.raises(InvalidCanonicalDocumentError, match="at serialization"):
        canonical_document_bytes(stale)


def test_recomputed_views_reject_noncanonical_text_during_load_and_save() -> None:
    view = _fixture("pdf")
    base = json.loads(canonical_document_bytes(view))

    invalid_payloads: list[dict[str, Any]] = []
    invalid_title = deepcopy(base)
    invalid_title["metadata"]["title"] = "  invalid\ttitle  "
    invalid_payloads.append(invalid_title)

    invalid_block = deepcopy(base)
    text_block = next(block for block in invalid_block["blocks"] if block["text"])
    text_block["text"] = "  invalid\tblock text  "
    _identify_block(text_block)
    invalid_payloads.append(invalid_block)

    invalid_cell = deepcopy(base)
    table_block = next(
        block for block in invalid_cell["blocks"] if block["table"] is not None
    )
    table_block["table"]["cells"][0]["text"] = "  invalid\tcell text  "
    invalid_payloads.append(invalid_cell)

    for payload in invalid_payloads:
        payload["view_id"] = canonical_view_id(payload)
        with pytest.raises(ValidationError, match="canonical normalization policy"):
            CanonicalDocumentView.model_validate(payload)
        with pytest.raises(InvalidCanonicalDocumentError, match="contract"):
            load_canonical_document(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
            )

    unsafe_metadata = view.metadata.model_copy(update={"title": "  invalid\ttitle  "})
    unsafe_view = view.model_copy(update={"metadata": unsafe_metadata})
    unsafe_view = unsafe_view.model_copy(
        update={"view_id": canonical_view_id(unsafe_view.model_dump(mode="json"))}
    )
    with pytest.raises(InvalidCanonicalDocumentError, match="at serialization"):
        canonical_document_bytes(unsafe_view)


def test_view_contract_binds_product_lineage_and_anchor_roles() -> None:
    base = json.loads(canonical_document_bytes(_fixture("pdf")))

    wrong_lineage = deepcopy(base)
    wrong_lineage["source_products"][0]["source_artifact_ids"] = ["other"]
    _validate_tampered_view(wrong_lineage, "source-product lineage")

    missing_docling = deepcopy(base)
    missing_docling["source_products"] = [
        product
        for product in missing_docling["source_products"]
        if product["name"] != "docling_document"
    ]
    _validate_tampered_view(missing_docling, "exactly one Docling")

    missing_spans = deepcopy(base)
    missing_spans["source_products"] = [
        product
        for product in missing_spans["source_products"]
        if product["name"] != "content_spans"
    ]
    _validate_tampered_view(missing_spans, "exactly one content-span")

    wrong_native_product = deepcopy(base)
    content_product = next(
        product
        for product in wrong_native_product["source_products"]
        if product["name"] == "content_spans"
    )
    wrong_native_product["blocks"][0]["source_anchors"][0]["product_id"] = (
        content_product["product_id"]
    )
    _validate_tampered_view(wrong_native_product, "native anchors")

    missing_primary = deepcopy(base)
    anchored_block = next(
        block for block in missing_primary["blocks"] if len(block["source_anchors"]) > 1
    )
    anchored_block["source_anchors"] = [
        anchor
        for anchor in anchored_block["source_anchors"]
        if anchor["role"] != "primary"
    ]
    _validate_tampered_view(missing_primary, "exactly one primary")

    wrong_primary_node = deepcopy(base)
    primary_anchor = next(
        anchor
        for anchor in wrong_primary_node["blocks"][0]["source_anchors"]
        if anchor["role"] == "primary"
    )
    primary_anchor["node_id"] = "#/texts/999"
    _validate_tampered_view(wrong_primary_node, "block's native node")

    duplicate_native_node = deepcopy(base)
    first = deepcopy(duplicate_native_node["blocks"][0])
    second = deepcopy(duplicate_native_node["blocks"][1])
    for block in (first, second):
        block["parent_block_id"] = None
        block["child_block_ids"] = []
    second["native_node_id"] = first["native_node_id"]
    next(anchor for anchor in second["source_anchors"] if anchor["role"] == "primary")[
        "node_id"
    ] = first["native_node_id"]
    _identify_block(second)
    first["ordinal"] = 0
    second["ordinal"] = 1
    duplicate_native_node["blocks"] = [first, second]
    duplicate_native_node["root_block_ids"] = [
        first["block_id"],
        second["block_id"],
    ]
    duplicate_native_node["relationships"] = []
    duplicate_native_node["diagnostics"] = []
    _validate_tampered_view(duplicate_native_node, "native node IDs")

    wrong_scholarly_product = deepcopy(base)
    scholarly_block = next(
        block
        for block in wrong_scholarly_product["blocks"]
        if any(anchor["role"] == "scholarly" for anchor in block["source_anchors"])
    )
    scholarly_anchor = next(
        anchor
        for anchor in scholarly_block["source_anchors"]
        if anchor["role"] == "scholarly"
    )
    scholarly_anchor["product_id"] = content_product["product_id"]
    _validate_tampered_view(wrong_scholarly_product, "scholarly anchors")

    wrong_relationship_anchor = deepcopy(base)
    relationship = next(
        relation
        for relation in wrong_relationship_anchor["relationships"]
        if relation["source_anchor"] is not None
    )
    relationship["source_anchor"]["role"] = "primary"
    _identify_relationship(relationship)
    _validate_tampered_view(wrong_relationship_anchor, "scholarly evidence")


def test_loaded_view_rejects_unused_and_unpaired_source_products() -> None:
    base = json.loads(canonical_document_bytes(_fixture("pdf")))
    extra_product = _product(
        "native_locator_overlay",
        "f",
        artifact_id=base["artifact_id"],
        producer_run_id="native-locator-run",
    ).model_dump(mode="json")

    extra = deepcopy(base)
    extra["source_products"].append(extra_product)
    missing_grobid = deepcopy(base)
    missing_grobid["source_products"] = [
        product
        for product in missing_grobid["source_products"]
        if product["name"] != "grobid_tei"
    ]
    missing_alignment = deepcopy(base)
    missing_alignment["source_products"] = [
        product
        for product in missing_alignment["source_products"]
        if product["name"] != "alignment_overlay"
    ]
    duplicate_integrity = deepcopy(base)
    duplicate_integrity["source_products"].append(
        _product(
            "content_integrity_overlay",
            "f",
            artifact_id=base["artifact_id"],
            producer_run_id="second-integrity-run",
        ).model_dump(mode="json")
    )

    for payload, match in (
        (extra, "names not used by schema v1"),
        (missing_grobid, "pair exactly one GROBID"),
        (missing_alignment, "pair exactly one GROBID"),
        (duplicate_integrity, "at most one content-integrity"),
    ):
        payload["view_id"] = canonical_view_id(payload)
        with pytest.raises(ValidationError, match=match):
            CanonicalDocumentView.model_validate(payload)
        with pytest.raises(InvalidCanonicalDocumentError, match="contract"):
            load_canonical_document(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )


def test_canonical_configuration_is_closed_and_typed() -> None:
    with pytest.raises(ValidationError):
        CanonicalizationConfig(text_normalization="lowercase-v1")
    with pytest.raises(ValidationError):
        CanonicalizationConfig(extra_policy="not-allowed")


def test_store_round_trip_dispatches_canonical_product_schema(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    source = store.put_blob(b"%PDF-1.7\ncanonical-source")
    artifact = DocumentArtifact(
        artifact_id="canonical-persisted",
        source_sha256=source.sha256,
        acquisition_uri="https://example.test/canonical-persisted",
        media_type="application/pdf",
        raw_location=source.as_location(
            media_type="application/pdf", role=ArtifactLocationRole.RAW
        ),
    )
    store.save_artifact(artifact)
    document = _document()
    docling_blob = store.put_blob(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    native_run_id = "canonical-native-run"
    docling_product = store.data_product_ref(
        name="docling_document",
        blob_sha256=docling_blob.sha256,
        producer_run_id=native_run_id,
        source_artifact_ids=(artifact.artifact_id,),
    )
    span_set = ContentSpanSet(
        artifact_id=artifact.artifact_id,
        processing_run_id=native_run_id,
        representation_product_id=docling_product.product_id,
        spans=build_pdf_content_spans(
            document,
            artifact_id=artifact.artifact_id,
            processing_run_id=native_run_id,
            representation_product_id=docling_product.product_id,
        ),
    )
    spans_blob = store.put_blob(
        json.dumps(
            span_set.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    spans_product = store.data_product_ref(
        name="content_spans",
        blob_sha256=spans_blob.sha256,
        producer_run_id=native_run_id,
        source_artifact_ids=(artifact.artifact_id,),
    )
    configuration: dict[str, Any] = {
        "serve_version": "1.21.0",
        "expected_docling_version": "2.113.0",
        "container_image": "quay.io/docling-project/docling-serve-cpu:v1.21.0",
        "container_digest": None,
        "model_versions": {},
        "model_hashes": {},
        "input_sha256": artifact.source_sha256,
        "input_format": "pdf",
        "options": {
            "to_formats": ["json"],
            "image_export_mode": "embedded",
            "do_ocr": True,
            "table_mode": "accurate",
        },
        "minimum_pdf_locator_coverage": 0.95,
        "quality_validator_version": "docling-quality-v2",
        "content_span_schema_version": "1",
        "pdf_span_algorithm": "provenance-charspan-v2",
    }
    started_at = datetime(2026, 8, 10, 12, tzinfo=UTC)
    store.save_processing_run(
        ProcessingRun(
            run_id=native_run_id,
            artifact_id=artifact.artifact_id,
            stage_id="docling",
            component=ComponentDescriptor(
                component_id="docling",
                component_version="2.113.0",
                capability="document.parse",
            ),
            configuration=configuration,
            configuration_sha256=configuration_sha256(configuration),
            started_at=started_at,
            finished_at=started_at + timedelta(seconds=1),
            status=ProcessingRunStatus.PARTIAL,
            outputs=(docling_product, spans_product),
        )
    )
    persisted_view = build_canonical_document_view(
        artifact=artifact,
        docling_document=document,
        docling_product=docling_product,
        content_span_set=span_set,
        source_products=(docling_product, spans_product),
        configuration=CanonicalizationConfig(),
    )
    blob = store.put_canonical_document(persisted_view)
    product = build_data_product_ref(
        name="canonical_document_view",
        blob_sha256=blob.sha256,
        uri=blob.uri,
        byte_size=blob.byte_size,
        producer_run_id="canonical-run",
        source_artifact_ids=(artifact.artifact_id,),
    )
    canonical_configuration = canonical_invocation_configuration(
        CanonicalizationConfig(),
        persisted_view.source_products,
    )
    store.save_processing_run(
        ProcessingRun(
            run_id="canonical-run",
            artifact_id=artifact.artifact_id,
            stage_id="canonical-document-view",
            component=ComponentDescriptor(
                component_id=CANONICAL_COMPONENT_ID,
                component_version=CANONICAL_COMPONENT_VERSION,
                capability=CANONICAL_COMPONENT_CAPABILITY,
            ),
            configuration=canonical_configuration,
            configuration_sha256=configuration_sha256(canonical_configuration),
            started_at=started_at,
            finished_at=started_at + timedelta(seconds=1),
            status=ProcessingRunStatus.COMPLETE,
            inputs=persisted_view.source_products,
            outputs=(product,),
        )
    )

    assert store.read_canonical_document(product) == persisted_view
    with pytest.raises(ValueError, match="not a canonical"):
        store.read_canonical_document(
            build_data_product_ref(
                name="docling_document",
                blob_sha256=blob.sha256,
                uri=blob.uri,
                byte_size=blob.byte_size,
                producer_run_id="canonical-run",
                source_artifact_ids=(artifact.artifact_id,),
            )
        )


def test_explicit_diagnostics_preserve_unresolved_native_mapping() -> None:
    view = _fixture("bioc")
    document = _document()
    document["body"]["children"].append({"$ref": "#/texts/404"})
    broken_integrity = _integrity_payload(
        document,
        [
            {
                "kind": "figure",
                "source_ref": "#/pictures/0",
                "source_docling_item_ref": "#/pictures/0",
                "declared_target_refs": ["#/texts/404"],
                "resolved_docling_item_refs": [],
                "unresolved_target_refs": ["#/texts/404"],
                "reason_codes": ["caption_target_not_found"],
            }
        ],
    )
    base = _fixture_inputs("bioc", document)
    _add_integrity_product(base)
    broken = build_canonical_document_view(
        **base,
        integrity_report=broken_integrity,
    )

    assert {diagnostic.code for diagnostic in view.diagnostics} == {
        "MISSING_SOURCE_SPAN"
    }
    assert {diagnostic.code for diagnostic in broken.diagnostics} == {
        "UNRESOLVED_NATIVE_REFERENCE",
        "UNRESOLVED_CANONICAL_RELATIONSHIP",
    }
    assert broken.relationships[0].status is CanonicalRelationshipStatus.UNRESOLVED

    falsely_resolved_integrity = _integrity_payload(
        _document(),
        [
            {
                "kind": "figure",
                "source_ref": "#/pictures/0",
                "source_docling_item_ref": "#/pictures/0",
                "declared_target_refs": ["#/texts/404"],
                "resolved_docling_item_refs": ["#/texts/404"],
                "unresolved_target_refs": [],
                "reason_codes": [],
            }
        ],
    )
    failed_inputs = _fixture_inputs("bioc", _document())
    _add_integrity_product(failed_inputs)
    failed_resolution = build_canonical_document_view(
        **failed_inputs,
        integrity_report=falsely_resolved_integrity,
    )
    relationship = failed_resolution.relationships[0]
    assert relationship.status is CanonicalRelationshipStatus.UNRESOLVED
    assert relationship.unresolved_target_refs == ("#/texts/404",)

    silently_omitted_integrity = _integrity_payload(
        _document(),
        [
            {
                "kind": "figure",
                "source_ref": "#/pictures/0",
                "source_docling_item_ref": "#/pictures/0",
                "declared_target_refs": ["#/texts/4", "#/texts/404"],
                "resolved_docling_item_refs": ["#/texts/4"],
                "unresolved_target_refs": [],
                "reason_codes": [],
            }
        ],
    )
    omitted_inputs = _fixture_inputs("bioc", _document())
    _add_integrity_product(omitted_inputs)
    omitted = build_canonical_document_view(
        **omitted_inputs,
        integrity_report=silently_omitted_integrity,
    ).relationships[0]
    assert omitted.status is CanonicalRelationshipStatus.PARTIAL
    assert omitted.unresolved_target_refs == ("#/texts/404",)

    citation = next(
        relationship
        for relationship in _fixture("pdf").relationships
        if relationship.kind is CanonicalRelationshipKind.CITES
    )
    assert citation.status is CanonicalRelationshipStatus.RESOLVED
    assert citation.declared_target_refs == ("#b1",)
    assert citation.target_block_ids


@pytest.mark.parametrize(
    "damage",
    [
        "non_object_record",
        "unknown_kind",
        "malformed_targets",
        "wrong_record_id",
        "wrong_status",
        "wrong_count",
        "extra_field",
    ],
)
def test_builder_rejects_non_replayable_integrity_reports(damage: str) -> None:
    document = _document()
    report = validate_content_integrity(document).to_dict()
    if damage == "non_object_record":
        report["records"][0] = 42
    elif damage == "unknown_kind":
        report["records"][0]["kind"] = "unknown"
    elif damage == "malformed_targets":
        report["records"][0]["resolved_docling_item_refs"] = "not-a-list"
    elif damage == "wrong_record_id":
        report["records"][0]["record_id"] = "0" * 64
    elif damage == "wrong_status":
        report["records"][0]["status"] = "unaligned"
    elif damage == "wrong_count":
        report["resolved_count"] += 1
    else:
        report["unexpected"] = True
    inputs = _fixture_inputs("pdf", document)
    _add_integrity_product(inputs)

    with pytest.raises(CanonicalDocumentError, match="contract is invalid"):
        build_canonical_document_view(**inputs, integrity_report=report)


def test_valid_integrity_error_is_preserved_as_canonical_error() -> None:
    document = _document()
    report = validate_content_integrity(document).to_dict()
    report["issues"].append(
        {
            "code": "UPSTREAM_INTEGRITY_ERROR",
            "message": "A verified source relationship is structurally damaged",
            "severity": "error",
            "item_ref": "#/tables/0",
            "page_number": None,
        }
    )
    inputs = _fixture_inputs("pdf", document)
    _add_integrity_product(inputs)
    integrity_product = inputs["source_products"][-1]

    view = build_canonical_document_view(**inputs, integrity_report=report)

    diagnostic = next(
        item for item in view.diagnostics if item.code == "UPSTREAM_INTEGRITY_ERROR"
    )
    assert diagnostic.severity is CanonicalDiagnosticSeverity.ERROR
    assert diagnostic.product_id == integrity_product.product_id
    assert diagnostic.native_node_ids == ("#/tables/0",)


def test_hierarchy_reconciliation_preserves_all_children_and_diagnoses_damage() -> None:
    complete = _fixture("pdf")
    blocks = {block.native_node_id: block for block in complete.blocks}
    assert blocks["#/texts/3"].parent_block_id == blocks["#/tables/0"].block_id
    assert blocks["#/texts/4"].parent_block_id == blocks["#/pictures/0"].block_id

    reordered_document = _document()
    reordered_document["groups"][0]["children"][:3] = [
        {"$ref": "#/texts/2"},
        {"$ref": "#/texts/0"},
        {"$ref": "#/texts/1"},
    ]
    reordered = build_canonical_document_view(
        **_fixture_inputs("pdf", reordered_document)
    )
    reordered_blocks = {
        block.block_id: block.native_node_id for block in reordered.blocks
    }
    reordered_group = next(
        block for block in reordered.blocks if block.kind is CanonicalBlockKind.GROUP
    )
    assert [
        reordered_blocks[child_id] for child_id in reordered_group.child_block_ids[:3]
    ] == ["#/texts/2", "#/texts/0", "#/texts/1"]

    redundant_body_parent = _document()
    redundant_body_parent["tables"][0]["parent"] = {"$ref": "#/body"}
    assert build_canonical_document_view(
        **_fixture_inputs("pdf", redundant_body_parent)
    ).blocks

    missing_parent = _document()
    missing_parent["texts"][0]["parent"] = {"$ref": "#/groups/404"}
    missing_parent_view = build_canonical_document_view(
        **_fixture_inputs("pdf", missing_parent)
    )
    assert "UNRESOLVED_NATIVE_REFERENCE" in {
        diagnostic.code for diagnostic in missing_parent_view.diagnostics
    }

    body_child_with_parent = _document()
    body_child_with_parent["tables"][0]["parent"] = {"$ref": "#/groups/0"}
    conflicting_root = build_canonical_document_view(
        **_fixture_inputs("pdf", body_child_with_parent)
    )
    assert "MULTIPLE_NATIVE_PARENTS" in {
        diagnostic.code for diagnostic in conflicting_root.diagnostics
    }

    rich_table_document = _document()
    rich_table_document["texts"].append(
        {
            "self_ref": "#/texts/7",
            "label": "paragraph",
            "text": "Rich cell detail",
        }
    )
    rich_table_document["groups"].append(
        {
            "self_ref": "#/groups/1",
            "label": "inline",
            "children": [{"$ref": "#/texts/7"}],
        }
    )
    rich_table_document["tables"][0]["data"] = {
        "num_rows": 1,
        "num_cols": 1,
        "table_cells": [
            {
                "text": "Rich cell",
                "start_row_offset_idx": 0,
                "start_col_offset_idx": 0,
                "ref": {"$ref": "#/groups/1"},
            }
        ],
        "grid": [["ignored"]],
    }
    rich_table = build_canonical_document_view(
        **_fixture_inputs("pdf", rich_table_document)
    )
    rich_blocks = {block.native_node_id: block for block in rich_table.blocks}
    table_block = rich_blocks["#/tables/0"]
    cell_group = rich_blocks["#/groups/1"]
    cell_text = rich_blocks["#/texts/7"]
    assert cell_group.parent_block_id == table_block.block_id
    assert cell_text.parent_block_id == cell_group.block_id
    assert table_block.table is not None
    assert table_block.table.cells[0].native_ref == "#/groups/1"

    all_collections = _document()
    all_collections["key_value_items"] = [
        {
            "self_ref": "#/key_value_items/0",
            "label": "key_value",
            "text": "Key value",
        }
    ]
    all_collections["form_items"] = [
        {
            "self_ref": "#/form_items/0",
            "label": "form",
            "text": "Form value",
        }
    ]
    all_collections["field_regions"] = [
        {
            "self_ref": "#/field_regions/0",
            "label": "field_region",
            "children": [{"$ref": "#/field_items/0"}],
        }
    ]
    all_collections["field_items"] = [
        {
            "self_ref": "#/field_items/0",
            "label": "field",
            "text": "Field value",
        }
    ]
    all_collections["body"]["children"].extend(
        [
            {"$ref": "#/key_value_items/0"},
            {"$ref": "#/form_items/0"},
            {"$ref": "#/field_regions/0"},
        ]
    )
    all_collections["texts"].extend(
        [
            {
                "self_ref": "#/texts/7",
                "label": "paragraph",
                "text": "Furniture child",
            },
            {
                "self_ref": "#/texts/8",
                "label": "paragraph",
                "text": "Furniture parent declaration",
                "parent": {"$ref": "#/furniture"},
            },
        ]
    )
    all_collections["furniture"] = {
        "self_ref": "#/furniture",
        "children": [{"$ref": "#/texts/7"}],
    }
    complete_collections = build_canonical_document_view(
        **_fixture_inputs("pdf", all_collections)
    )
    collection_blocks = {
        block.native_node_id: block for block in complete_collections.blocks
    }
    assert {
        "#/key_value_items/0",
        "#/form_items/0",
        "#/field_regions/0",
        "#/field_items/0",
        "#/texts/7",
        "#/texts/8",
    }.issubset(collection_blocks)
    assert (
        collection_blocks["#/field_items/0"].parent_block_id
        == collection_blocks["#/field_regions/0"].block_id
    )
    assert collection_blocks["#/texts/7"].parent_block_id is None
    assert collection_blocks["#/texts/8"].parent_block_id is None
    assert "UNRESOLVED_NATIVE_REFERENCE" not in {
        diagnostic.code for diagnostic in complete_collections.diagnostics
    }

    document = _document()
    document["body"]["children"] = [
        reference
        for reference in document["body"]["children"]
        if reference["$ref"] not in {"#/tables/0", "#/pictures/0"}
    ]
    document["body"]["children"].append({"$ref": "#/groups/0"})
    document["texts"][0]["parent"] = {"$ref": "#/tables/0"}
    document["tables"][0]["parent"] = {"$ref": "#/texts/0"}
    document["pictures"][0]["parent"] = {"$ref": "#/body"}

    damaged = build_canonical_document_view(**_fixture_inputs("pdf", document))
    codes = {diagnostic.code for diagnostic in damaged.diagnostics}

    assert {
        "MULTIPLE_NATIVE_PARENTS",
        "CYCLIC_NATIVE_HIERARCHY",
        "ORPHANED_NATIVE_NODES",
    }.issubset(codes)
    assert tuple(block.ordinal for block in damaged.blocks) == tuple(
        range(len(damaged.blocks))
    )
    damaged_blocks = {block.block_id: block for block in damaged.blocks}
    assert all(
        block.parent_block_id is None
        or damaged_blocks[block.parent_block_id].ordinal < block.ordinal
        for block in damaged.blocks
    )


def _deep_hierarchy_document(depth: int) -> dict[str, Any]:
    document = _document()
    leaf_children = deepcopy(document["groups"][0]["children"])
    document["groups"] = [
        {
            "self_ref": f"#/groups/{index}",
            "label": "section_group",
            "children": (
                [{"$ref": f"#/groups/{index + 1}"}]
                if index + 1 < depth
                else leaf_children
            ),
        }
        for index in range(depth)
    ]
    return document


def test_deep_acyclic_hierarchy_is_iterative_and_ordered() -> None:
    depth = 1_200
    view = build_canonical_document_view(
        **_fixture_inputs("pdf", _deep_hierarchy_document(depth))
    )
    groups = [block for block in view.blocks if block.kind is CanonicalBlockKind.GROUP]

    assert [block.native_node_id for block in groups] == [
        f"#/groups/{index}" for index in range(depth)
    ]
    assert groups[0].parent_block_id is None
    assert groups[-1].parent_block_id == groups[-2].block_id
    assert "CYCLIC_NATIVE_HIERARCHY" not in {
        diagnostic.code for diagnostic in view.diagnostics
    }


def test_deep_hierarchy_cycle_is_broken_iteratively_and_deterministically() -> None:
    depth = 1_200
    document = _deep_hierarchy_document(depth)
    document["groups"][0]["parent"] = {"$ref": f"#/groups/{depth - 1}"}

    view = build_canonical_document_view(**_fixture_inputs("pdf", document))
    cycle = next(
        diagnostic
        for diagnostic in view.diagnostics
        if diagnostic.code == "CYCLIC_NATIVE_HIERARCHY"
    )
    groups = {
        block.native_node_id: block
        for block in view.blocks
        if block.kind is CanonicalBlockKind.GROUP
    }

    assert len(cycle.native_node_ids) == depth
    assert set(cycle.native_node_ids) == {f"#/groups/{index}" for index in range(depth)}
    assert groups["#/groups/0"].parent_block_id is None
    assert (
        groups[f"#/groups/{depth - 1}"].parent_block_id
        == groups[f"#/groups/{depth - 2}"].block_id
    )


def _fixture_inputs(format_name: str, document: dict[str, Any]) -> dict[str, Any]:
    complete = _fixture(format_name)
    docling_product = next(
        product
        for product in complete.source_products
        if product.name == "docling_document"
    )
    content_product = next(
        product
        for product in complete.source_products
        if product.name == "content_spans"
    )
    artifact_source = f"source-{format_name}".encode()
    artifact = DocumentArtifact(
        artifact_id=complete.artifact_id,
        source_sha256=sha256_bytes(artifact_source),
        acquisition_uri=f"https://example.test/{complete.artifact_id}",
        identifiers={"pmc": "PMC-CANONICAL"},
        media_type=complete.metadata.media_type,
        raw_location=ArtifactLocation(
            uri=f"cas://sha256/{sha256_bytes(artifact_source)}",
            sha256=sha256_bytes(artifact_source),
            byte_size=len(artifact_source),
            media_type=complete.metadata.media_type,
            role=ArtifactLocationRole.RAW,
        ),
    )
    text_spans = tuple(
        ContentSpan(
            span_id=f"broken-{index}",
            artifact_id=complete.artifact_id,
            processing_run_id=docling_product.producer_run_id,
            representation_anchor=RepresentationAnchor(
                product_id=docling_product.product_id,
                node_id=f"#/texts/{index}",
                char_start=0,
                char_end=len(item["text"]),
            ),
            content_sha256=sha256_bytes(item["text"].encode()),
            source_locator=_locator(format_name, index, len(item["text"])),
        )
        for index, item in enumerate(document["texts"])
    )
    formula_text = document["formulas"][0]["text"]
    formula_span = ContentSpan(
        span_id="broken-formula",
        artifact_id=complete.artifact_id,
        processing_run_id=docling_product.producer_run_id,
        representation_anchor=RepresentationAnchor(
            product_id=docling_product.product_id,
            node_id="#/formulas/0",
            char_start=0,
            char_end=len(formula_text),
        ),
        content_sha256=sha256_bytes(formula_text.encode()),
        source_locator=_locator(format_name, len(document["texts"]), len(formula_text)),
    )
    return {
        "artifact": artifact,
        "docling_document": document,
        "docling_product": docling_product,
        "content_span_set": ContentSpanSet(
            artifact_id=complete.artifact_id,
            processing_run_id=docling_product.producer_run_id,
            representation_product_id=docling_product.product_id,
            spans=(*text_spans, formula_span),
        ),
        "source_products": (docling_product, content_product),
        "configuration": CanonicalizationConfig(),
    }


def _add_integrity_product(inputs: dict[str, Any]) -> None:
    inputs["source_products"] = (
        *inputs["source_products"],
        _product(
            "content_integrity_overlay",
            "e",
            artifact_id=inputs["artifact"].artifact_id,
            producer_run_id="integrity-test",
        ),
    )


def _add_scholarly_products(inputs: dict[str, Any]) -> None:
    inputs["source_products"] = (
        *inputs["source_products"],
        _product(
            "grobid_tei",
            "c",
            artifact_id=inputs["artifact"].artifact_id,
            producer_run_id="grobid-test",
        ),
        _product(
            "alignment_overlay",
            "d",
            artifact_id=inputs["artifact"].artifact_id,
            producer_run_id="alignment-test",
        ),
    )


def _integrity_payload(
    document: dict[str, Any],
    records: Any,
    *,
    scholarly_overlay_present: bool = False,
    issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_records: Any = records
    if isinstance(records, list):
        normalized_records = []
        for raw_record in records:
            if not isinstance(raw_record, dict):
                normalized_records.append(raw_record)
                continue
            record = {
                "kind": raw_record.get("kind"),
                "status": raw_record.get(
                    "status",
                    "unaligned" if raw_record.get("reason_codes") else "resolved",
                ),
                "source_ref": raw_record.get("source_ref"),
                "source_docling_item_ref": raw_record.get("source_docling_item_ref"),
                "declared_target_refs": raw_record.get("declared_target_refs", []),
                "resolved_docling_item_refs": raw_record.get(
                    "resolved_docling_item_refs", []
                ),
                "unresolved_target_refs": raw_record.get("unresolved_target_refs", []),
                "reason_codes": raw_record.get("reason_codes", []),
            }
            identity = {
                key: record[key]
                for key in (
                    "kind",
                    "status",
                    "source_ref",
                    "source_docling_item_ref",
                    "declared_target_refs",
                    "resolved_docling_item_refs",
                    "unresolved_target_refs",
                    "reason_codes",
                )
            }
            record["record_id"] = raw_record.get(
                "record_id",
                sha256_bytes(
                    json.dumps(
                        identity,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ),
            )
            normalized_records.append(record)
    resolved_count = (
        sum(record.get("status") == "resolved" for record in normalized_records)
        if isinstance(normalized_records, list)
        and all(isinstance(record, dict) for record in normalized_records)
        else 0
    )
    return {
        "document_sha256": docling_document_sha256(document),
        "scholarly_overlay_present": scholarly_overlay_present,
        "resolved_count": resolved_count,
        "unaligned_count": (
            len(normalized_records) - resolved_count
            if isinstance(normalized_records, list)
            else 0
        ),
        "records": normalized_records,
        "issues": issues or [],
    }


def test_view_contract_rejects_tampered_graph_and_identity() -> None:
    view = _fixture("pdf")
    payload = json.loads(canonical_document_bytes(view))
    payload["root_block_ids"] = payload["root_block_ids"][1:]
    payload["view_id"] = "pending"
    payload["view_id"] = canonical_view_id(payload)

    with pytest.raises(ValidationError, match="roots"):
        CanonicalDocumentView.model_validate(payload)

    duplicate = deepcopy(json.loads(canonical_document_bytes(view)))
    duplicate["source_products"].append(duplicate["source_products"][0])
    duplicate["view_id"] = canonical_view_id(duplicate)
    with pytest.raises(ValidationError, match="source products"):
        CanonicalDocumentView.model_validate(duplicate)

    empty = deepcopy(json.loads(canonical_document_bytes(view)))
    empty["blocks"] = []
    empty["root_block_ids"] = []
    empty["view_id"] = canonical_view_id(empty)
    with pytest.raises(ValidationError, match="at least 1 item"):
        CanonicalDocumentView.model_validate(empty)


def _identify_block(payload: dict[str, Any]) -> None:
    kind = CanonicalBlockKind(payload["kind"])
    table = (
        CanonicalTable.model_validate(payload["table"])
        if payload.get("table") is not None
        else None
    )
    payload["content_sha256"] = canonical_block_content_sha256(
        kind=kind,
        text=payload.get("text"),
        table=table,
    )
    payload["block_id"] = canonical_block_id(
        native_node_id=payload["native_node_id"],
        kind=kind,
        content_sha256=payload["content_sha256"],
    )


def _identify_relationship(payload: dict[str, Any]) -> None:
    anchor = (
        CanonicalSourceAnchor.model_validate(payload["source_anchor"])
        if payload.get("source_anchor") is not None
        else None
    )
    payload["relationship_id"] = canonical_relationship_id(
        kind=CanonicalRelationshipKind(payload["kind"]),
        status=CanonicalRelationshipStatus(payload["status"]),
        source_block_id=payload.get("source_block_id"),
        target_block_ids=tuple(payload.get("target_block_ids", ())),
        declared_target_refs=tuple(payload.get("declared_target_refs", ())),
        unresolved_target_refs=tuple(payload.get("unresolved_target_refs", ())),
        reason_codes=tuple(payload.get("reason_codes", ())),
        source_anchor=anchor,
    )


def _identify_diagnostic(payload: dict[str, Any]) -> None:
    payload["diagnostic_id"] = canonical_diagnostic_id(
        severity=CanonicalDiagnosticSeverity(payload["severity"]),
        code=payload["code"],
        message=payload["message"],
        product_id=payload.get("product_id"),
        native_node_ids=tuple(payload.get("native_node_ids", ())),
    )


def _validate_tampered_view(payload: dict[str, Any], match: str) -> None:
    payload["view_id"] = canonical_view_id(payload)
    with pytest.raises(ValidationError, match=match):
        CanonicalDocumentView.model_validate(payload)


def test_relationship_status_is_derived_and_identity_bound() -> None:
    view = _fixture("pdf")
    resolved = view.relationships[0]
    assert (
        canonical_relationship_status(
            source_block_id=resolved.source_block_id,
            target_block_ids=resolved.target_block_ids,
            declared_target_refs=resolved.declared_target_refs,
            unresolved_target_refs=(),
            reason_codes=(),
        )
        is CanonicalRelationshipStatus.RESOLVED
    )
    assert (
        canonical_relationship_status(
            source_block_id=resolved.source_block_id,
            target_block_ids=resolved.target_block_ids,
            declared_target_refs=(*resolved.declared_target_refs, "missing"),
            unresolved_target_refs=("missing",),
            reason_codes=(),
        )
        is CanonicalRelationshipStatus.PARTIAL
    )
    assert (
        canonical_relationship_status(
            source_block_id=resolved.source_block_id,
            target_block_ids=resolved.target_block_ids,
            declared_target_refs=(*resolved.declared_target_refs, "silently-missing"),
            unresolved_target_refs=(),
            reason_codes=(),
        )
        is CanonicalRelationshipStatus.PARTIAL
    )
    assert (
        canonical_relationship_status(
            source_block_id=None,
            target_block_ids=resolved.target_block_ids,
            declared_target_refs=resolved.declared_target_refs,
            unresolved_target_refs=(),
            reason_codes=(),
        )
        is CanonicalRelationshipStatus.UNRESOLVED
    )

    values = {
        "kind": resolved.kind,
        "source_block_id": resolved.source_block_id,
        "target_block_ids": resolved.target_block_ids,
        "declared_target_refs": resolved.declared_target_refs,
        "unresolved_target_refs": resolved.unresolved_target_refs,
        "reason_codes": resolved.reason_codes,
        "source_anchor": resolved.source_anchor,
    }
    resolved_id = canonical_relationship_id(
        status=CanonicalRelationshipStatus.RESOLVED,
        **values,
    )
    partial_id = canonical_relationship_id(
        status=CanonicalRelationshipStatus.PARTIAL,
        **values,
    )
    assert resolved_id != partial_id


def test_recomputed_relationship_cannot_hide_unaccounted_declared_target() -> None:
    view = _fixture("pdf")
    payload = json.loads(canonical_document_bytes(view))
    relationship = next(
        item for item in payload["relationships"] if item["status"] == "resolved"
    )
    relationship["declared_target_refs"].append("silently-omitted-target")
    _identify_relationship(relationship)
    payload["view_id"] = canonical_view_id(payload)

    with pytest.raises(ValidationError, match="status does not match"):
        CanonicalDocumentView.model_validate(payload)
    with pytest.raises(InvalidCanonicalDocumentError, match="contract"):
        load_canonical_document(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        )

    relationship_index = next(
        index
        for index, item in enumerate(view.relationships)
        if item.status is CanonicalRelationshipStatus.RESOLVED
    )
    original = view.relationships[relationship_index]
    declared = (*original.declared_target_refs, "silently-omitted-target")
    unsafe_relationship = original.model_copy(
        update={
            "declared_target_refs": declared,
            "relationship_id": canonical_relationship_id(
                kind=original.kind,
                status=original.status,
                source_block_id=original.source_block_id,
                target_block_ids=original.target_block_ids,
                declared_target_refs=declared,
                unresolved_target_refs=original.unresolved_target_refs,
                reason_codes=original.reason_codes,
                source_anchor=original.source_anchor,
            ),
        }
    )
    relationships = list(view.relationships)
    relationships[relationship_index] = unsafe_relationship
    unsafe_view = view.model_copy(update={"relationships": tuple(relationships)})
    unsafe_payload = unsafe_view.model_dump(mode="json")
    unsafe_view = unsafe_view.model_copy(
        update={"view_id": canonical_view_id(unsafe_payload)}
    )
    with pytest.raises(InvalidCanonicalDocumentError, match="at serialization"):
        canonical_document_bytes(unsafe_view)


def test_nested_contracts_reject_inconsistent_identifiers_and_content() -> None:
    view = _fixture("pdf")

    for anchor, match in (
        (
            {
                "role": "primary",
                "product_id": "product",
                "node_id": "node",
                "char_start": 0,
            },
            "bounds must be paired",
        ),
        (
            {
                "role": "primary",
                "product_id": "product",
                "node_id": "node",
                "char_start": 1,
                "char_end": 1,
            },
            "char_end must exceed",
        ),
        (
            {
                "role": "primary",
                "product_id": " ",
                "node_id": "node",
            },
            "identifiers must not be empty",
        ),
    ):
        with pytest.raises(ValidationError, match=match):
            CanonicalSourceAnchor.model_validate(anchor)

    with pytest.raises(ValidationError, match="native_ref must not be empty"):
        CanonicalTableCell(
            text="cell",
            row_index=0,
            column_index=0,
            native_ref=" ",
        )
    with pytest.raises(ValidationError, match="canonical normalization policy"):
        CanonicalTableCell(text="  cell\tvalue  ", row_index=0, column_index=0)
    with pytest.raises(ValidationError, match="canonical normalization policy"):
        CanonicalDocumentMetadata(
            title="  document\ttitle  ",
            media_type="application/pdf",
        )
    with pytest.raises(ValidationError, match="canonical normalization policy"):
        CanonicalDocumentMetadata(title="", media_type="application/pdf")

    title = next(
        block for block in view.blocks if block.kind is CanonicalBlockKind.TITLE
    )
    invalid_block_id = title.model_dump(mode="json")
    invalid_block_id["block_id"] = "block-wrong"
    with pytest.raises(ValidationError, match="block ID"):
        CanonicalBlock.model_validate(invalid_block_id)

    table_as_paragraph = next(
        block for block in view.blocks if block.kind is CanonicalBlockKind.TABLE
    ).model_dump(mode="json")
    table_as_paragraph["kind"] = "paragraph"
    _identify_block(table_as_paragraph)
    with pytest.raises(ValidationError, match="only table"):
        CanonicalBlock.model_validate(table_as_paragraph)

    empty_text = title.model_dump(mode="json")
    empty_text["text"] = ""
    _identify_block(empty_text)
    with pytest.raises(ValidationError, match="text must be non-empty"):
        CanonicalBlock.model_validate(empty_text)

    unnormalized_text = title.model_dump(mode="json")
    unnormalized_text["text"] = "  recomputed\tcanonical text  "
    _identify_block(unnormalized_text)
    with pytest.raises(ValidationError, match="canonical normalization policy"):
        CanonicalBlock.model_validate(unnormalized_text)

    group = next(
        block for block in view.blocks if block.kind is CanonicalBlockKind.GROUP
    )
    duplicate_child = group.model_dump(mode="json")
    duplicate_child["child_block_ids"].append(duplicate_child["child_block_ids"][0])
    with pytest.raises(ValidationError, match="child block IDs"):
        CanonicalBlock.model_validate(duplicate_child)

    duplicate_anchor = title.model_dump(mode="json")
    duplicate_anchor["source_anchors"].append(duplicate_anchor["source_anchors"][0])
    with pytest.raises(ValidationError, match="source anchors"):
        CanonicalBlock.model_validate(duplicate_anchor)

    resolved = view.relationships[0].model_dump(mode="json")
    wrong_relationship_id = deepcopy(resolved)
    wrong_relationship_id["relationship_id"] = "relationship-wrong"
    with pytest.raises(ValidationError, match="relationship ID"):
        CanonicalRelationship.model_validate(wrong_relationship_id)

    duplicate_target = deepcopy(resolved)
    duplicate_target["target_block_ids"].append(duplicate_target["target_block_ids"][0])
    _identify_relationship(duplicate_target)
    with pytest.raises(ValidationError, match="targets must be unique"):
        CanonicalRelationship.model_validate(duplicate_target)

    incomplete_resolved = deepcopy(resolved)
    incomplete_resolved["reason_codes"] = ["not-complete"]
    _identify_relationship(incomplete_resolved)
    with pytest.raises(ValidationError, match="status does not match"):
        CanonicalRelationship.model_validate(incomplete_resolved)

    unresolved_with_targets = deepcopy(resolved)
    unresolved_with_targets["status"] = "unresolved"
    _identify_relationship(unresolved_with_targets)
    with pytest.raises(ValidationError, match="status does not match"):
        CanonicalRelationship.model_validate(unresolved_with_targets)

    diagnostic = {
        "diagnostic_id": "canonical-diagnostic-wrong",
        "severity": "warning",
        "code": "TEST",
        "message": "test diagnostic",
    }
    with pytest.raises(ValidationError, match="diagnostic ID"):
        CanonicalMappingDiagnostic.model_validate(diagnostic)
    diagnostic["code"] = " "
    _identify_diagnostic(diagnostic)
    with pytest.raises(ValidationError, match="must not be empty"):
        CanonicalMappingDiagnostic.model_validate(diagnostic)


def test_view_contract_rejects_every_inconsistent_graph_edge() -> None:
    view = _fixture("pdf")
    base = json.loads(canonical_document_bytes(view))

    duplicate_block = deepcopy(base)
    duplicate_block["blocks"].append(deepcopy(duplicate_block["blocks"][0]))
    _validate_tampered_view(duplicate_block, "block IDs must be unique")

    noncontiguous = deepcopy(base)
    noncontiguous["blocks"][1]["ordinal"] = 99
    _validate_tampered_view(noncontiguous, "ordinals must be contiguous")

    duplicate_root = deepcopy(base)
    duplicate_root["root_block_ids"].append(duplicate_root["root_block_ids"][0])
    _validate_tampered_view(duplicate_root, "root block IDs must be unique")

    missing_parent_link = deepcopy(base)
    missing_parent_link["blocks"][0]["child_block_ids"].remove(
        missing_parent_link["blocks"][1]["block_id"]
    )
    _validate_tampered_view(missing_parent_link, "parent and child links")

    parent_after_child = deepcopy(base)
    parent_after_child["blocks"][0], parent_after_child["blocks"][1] = (
        parent_after_child["blocks"][1],
        parent_after_child["blocks"][0],
    )
    for ordinal, block in enumerate(parent_after_child["blocks"]):
        block["ordinal"] = ordinal
    parent_after_child["root_block_ids"] = [
        block["block_id"]
        for block in parent_after_child["blocks"]
        if block["parent_block_id"] is None
    ]
    _validate_tampered_view(parent_after_child, "parents must precede")

    wrong_child_parent = deepcopy(base)
    root_without_parent = next(
        block
        for block in wrong_child_parent["blocks"]
        if block["parent_block_id"] is None and block["kind"] == "table"
    )
    wrong_child_parent["blocks"][0]["child_block_ids"].append(
        root_without_parent["block_id"]
    )
    _validate_tampered_view(wrong_child_parent, "child and parent links")

    unknown_anchor_product = deepcopy(base)
    unknown_anchor_product["blocks"][0]["source_anchors"][0]["product_id"] = (
        "product-missing"
    )
    _validate_tampered_view(unknown_anchor_product, "unknown product")

    duplicate_relationship = deepcopy(base)
    duplicate_relationship["relationships"].append(
        deepcopy(duplicate_relationship["relationships"][0])
    )
    _validate_tampered_view(duplicate_relationship, "relationship IDs must be unique")

    unknown_relationship_source = deepcopy(base)
    relation = unknown_relationship_source["relationships"][0]
    relation["source_block_id"] = "block-missing"
    _identify_relationship(relation)
    _validate_tampered_view(unknown_relationship_source, "source does not exist")

    unknown_relationship_target = deepcopy(base)
    relation = unknown_relationship_target["relationships"][0]
    relation["target_block_ids"] = ["block-missing"]
    _identify_relationship(relation)
    _validate_tampered_view(unknown_relationship_target, "target does not exist")

    unknown_relationship_anchor = deepcopy(base)
    relation = unknown_relationship_anchor["relationships"][-1]
    assert relation["source_anchor"] is not None
    relation["source_anchor"]["product_id"] = "product-missing"
    _identify_relationship(relation)
    _validate_tampered_view(unknown_relationship_anchor, "relationship anchor product")

    diagnostic = {
        "severity": "warning",
        "code": "GRAPH_TEST",
        "message": "graph diagnostic",
        "product_id": base["source_products"][0]["product_id"],
        "native_node_ids": [],
    }
    _identify_diagnostic(diagnostic)
    duplicate_diagnostic = deepcopy(base)
    duplicate_diagnostic["diagnostics"] = [diagnostic, deepcopy(diagnostic)]
    _validate_tampered_view(duplicate_diagnostic, "diagnostic IDs must be unique")

    unknown_diagnostic_product = deepcopy(base)
    diagnostic["product_id"] = "product-missing"
    _identify_diagnostic(diagnostic)
    unknown_diagnostic_product["diagnostics"] = [diagnostic]
    _validate_tampered_view(unknown_diagnostic_product, "diagnostic product")

    wrong_view_id = deepcopy(base)
    wrong_view_id["view_id"] = "canonical-view-wrong"
    with pytest.raises(ValidationError, match="view ID"):
        CanonicalDocumentView.model_validate(wrong_view_id)


def test_builder_rejects_products_that_do_not_match_native_inputs() -> None:
    inputs = _fixture_inputs("pdf", _document())
    docling_product = inputs["docling_product"]

    inputs["source_products"] = tuple(
        product
        for product in inputs["source_products"]
        if product.product_id != docling_product.product_id
    )
    with pytest.raises(CanonicalDocumentError, match="must be a canonical source"):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", _document())
    inputs["content_span_set"] = _retarget_span_set(
        inputs["content_span_set"],
        representation_product_id="product-missing",
    )
    with pytest.raises(CanonicalDocumentError, match="different Docling product"):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", _document())
    span_set = inputs["content_span_set"]
    inputs["content_span_set"] = span_set.model_copy(
        update={"spans": (*span_set.spans, span_set.spans[0])}
    )
    with pytest.raises(CanonicalDocumentError, match="span set contract is invalid"):
        build_canonical_document_view(**inputs)


def test_builder_requires_exact_semantically_consumed_overlay_products() -> None:
    document = _document()
    inputs = _fixture_inputs("pdf", document)
    unused = inputs["docling_product"].model_copy(
        update={
            "name": "native_locator_overlay",
            "product_id": "product-unused-native-overlay",
        }
    )
    inputs["source_products"] = (*inputs["source_products"], unused)
    with pytest.raises(CanonicalDocumentError, match="not used by schema v1"):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", document)
    _add_scholarly_products(inputs)
    with pytest.raises(CanonicalDocumentError, match="require an alignment overlay"):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", document)
    _add_integrity_product(inputs)
    with pytest.raises(CanonicalDocumentError, match="requires its persisted report"):
        build_canonical_document_view(**inputs)

    report = _integrity_payload(document, [])
    with pytest.raises(CanonicalDocumentError, match="requires exactly one integrity"):
        build_canonical_document_view(
            **_fixture_inputs("pdf", document),
            integrity_report=report,
        )

    inputs = _fixture_inputs("pdf", document)
    _add_integrity_product(inputs)
    wrong_document = dict(report, document_sha256="0" * 64)
    with pytest.raises(CanonicalDocumentError, match="different Docling document"):
        build_canonical_document_view(**inputs, integrity_report=wrong_document)

    wrong_overlay_presence = dict(report, scholarly_overlay_present=True)
    with pytest.raises(CanonicalDocumentError, match="provenance does not match"):
        build_canonical_document_view(
            **inputs,
            integrity_report=wrong_overlay_presence,
        )


def test_builder_rejects_cross_artifact_and_cross_run_product_lineage() -> None:
    def replace_product(inputs: dict[str, Any], old: Any, replacement: Any) -> None:
        inputs["source_products"] = tuple(
            replacement if product.product_id == old.product_id else product
            for product in inputs["source_products"]
        )

    inputs = _fixture_inputs("pdf", _document())
    docling = inputs["docling_product"]
    renamed = docling.model_copy(update={"name": "other"})
    inputs["docling_product"] = renamed
    replace_product(inputs, docling, renamed)
    with pytest.raises(CanonicalDocumentError, match="must be Docling"):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", _document())
    docling = inputs["docling_product"]
    replace_product(
        inputs,
        docling,
        docling.model_copy(update={"producer_run_id": "different-run"}),
    )
    with pytest.raises(CanonicalDocumentError, match="must exactly match"):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", _document())
    docling = inputs["docling_product"]
    inputs["source_products"] = (
        *inputs["source_products"],
        docling.model_copy(update={"product_id": "duplicate-docling-product"}),
    )
    with pytest.raises(CanonicalDocumentError, match="exactly one Docling"):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", _document())
    docling = inputs["docling_product"]
    other_artifact = docling.model_copy(update={"source_artifact_ids": ("other",)})
    inputs["docling_product"] = other_artifact
    replace_product(inputs, docling, other_artifact)
    with pytest.raises(CanonicalDocumentError, match="different artifact"):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", _document())
    inputs["content_span_set"] = _retarget_span_set(
        inputs["content_span_set"],
        artifact_id="other",
    )
    with pytest.raises(
        CanonicalDocumentError, match="spans target a different artifact"
    ):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", _document())
    inputs["content_span_set"] = _retarget_span_set(
        inputs["content_span_set"],
        processing_run_id="other",
    )
    with pytest.raises(CanonicalDocumentError, match="different Docling run"):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", _document())
    content_product = next(
        product
        for product in inputs["source_products"]
        if product.name == "content_spans"
    )
    replace_product(
        inputs,
        content_product,
        content_product.model_copy(update={"source_artifact_ids": ("other",)}),
    )
    with pytest.raises(CanonicalDocumentError, match="source-product lineage"):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", _document())
    inputs["source_products"] = (inputs["docling_product"],)
    with pytest.raises(CanonicalDocumentError, match="exactly one content-span"):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", _document())
    content_product = next(
        product
        for product in inputs["source_products"]
        if product.name == "content_spans"
    )
    inputs["source_products"] = (
        *inputs["source_products"],
        content_product.model_copy(update={"product_id": "duplicate-content-product"}),
    )
    with pytest.raises(CanonicalDocumentError, match="exactly one content-span"):
        build_canonical_document_view(**inputs)

    inputs = _fixture_inputs("pdf", _document())
    content_product = next(
        product
        for product in inputs["source_products"]
        if product.name == "content_spans"
    )
    replace_product(
        inputs,
        content_product,
        content_product.model_copy(update={"producer_run_id": "other"}),
    )
    with pytest.raises(CanonicalDocumentError, match="different processing run"):
        build_canonical_document_view(**inputs)


def test_builder_verifies_every_content_span_against_one_exact_native_node() -> None:
    def replace_first_span(inputs: dict[str, Any], span: ContentSpan) -> None:
        original = inputs["content_span_set"]
        inputs["content_span_set"] = original.model_copy(
            update={"spans": (span, *original.spans[1:])}
        )

    document = _document()
    document["texts"][0]["self_ref"] = "title-alias"
    aliased = _fixture_inputs("pdf", document)
    first = aliased["content_span_set"].spans[0]
    replace_first_span(
        aliased,
        first.model_copy(
            update={
                "representation_anchor": first.representation_anchor.model_copy(
                    update={"node_id": "title-alias"}
                )
            }
        ),
    )
    assert build_canonical_document_view(**aliased).blocks

    missing = _fixture_inputs("pdf", _document())
    first = missing["content_span_set"].spans[0]
    replace_first_span(
        missing,
        first.model_copy(
            update={
                "representation_anchor": first.representation_anchor.model_copy(
                    update={"node_id": "missing-node"}
                )
            }
        ),
    )
    with pytest.raises(CanonicalDocumentError, match="does not exist"):
        build_canonical_document_view(**missing)

    ambiguous_document = _document()
    ambiguous_document["texts"][0]["self_ref"] = "duplicate-node"
    ambiguous_document["texts"][1]["self_ref"] = "duplicate-node"
    ambiguous = _fixture_inputs("pdf", ambiguous_document)
    first = ambiguous["content_span_set"].spans[0]
    replace_first_span(
        ambiguous,
        first.model_copy(
            update={
                "representation_anchor": first.representation_anchor.model_copy(
                    update={"node_id": "duplicate-node"}
                )
            }
        ),
    )
    with pytest.raises(CanonicalDocumentError, match="is ambiguous"):
        build_canonical_document_view(**ambiguous)

    out_of_bounds = _fixture_inputs("pdf", _document())
    first = out_of_bounds["content_span_set"].spans[0]
    replace_first_span(
        out_of_bounds,
        first.model_copy(
            update={
                "representation_anchor": first.representation_anchor.model_copy(
                    update={"char_end": 10000}
                )
            }
        ),
    )
    with pytest.raises(CanonicalDocumentError, match="native text bounds"):
        build_canonical_document_view(**out_of_bounds)

    wrong_hash = _fixture_inputs("pdf", _document())
    first = wrong_hash["content_span_set"].spans[0]
    replace_first_span(
        wrong_hash,
        first.model_copy(update={"content_sha256": "0" * 64}),
    )
    with pytest.raises(CanonicalDocumentError, match="does not match native text"):
        build_canonical_document_view(**wrong_hash)

    for update in (
        {"artifact_id": "other"},
        {"processing_run_id": "other"},
        {
            "representation_anchor": first.representation_anchor.model_copy(
                update={"product_id": "product-other"}
            )
        },
    ):
        invalid_lineage = _fixture_inputs("pdf", _document())
        invalid_span = (
            invalid_lineage["content_span_set"].spans[0].model_copy(update=update)
        )
        replace_first_span(invalid_lineage, invalid_span)
        with pytest.raises(
            CanonicalDocumentError, match="span set contract is invalid"
        ):
            build_canonical_document_view(**invalid_lineage)


def test_builder_reports_malformed_native_structures_and_missing_anchors() -> None:
    document = _document()
    inputs = _fixture_inputs("pdf", document)
    malformed = {
        "name": " Fallback\tTitle ",
        "body": "not-an-object",
        "furniture": "not-an-object",
        "texts": [
            42,
            {
                "self_ref": "declared-text",
                "label": "unknown-native-label",
                "orig": "Native text",
            },
        ],
        "tables": [
            {"self_ref": "#/tables/0", "label": "table"},
            {
                "self_ref": "#/tables/1",
                "label": "table",
                "data": {"grid": "not-a-list"},
            },
            {
                "self_ref": "#/tables/2",
                "label": "table",
                "data": {"table_cells": [{"text": " Cell\nvalue "}, 3]},
            },
            {
                "self_ref": "#/tables/3",
                "label": "table",
                "data": {"grid": [3]},
            },
            {
                "self_ref": "#/tables/4",
                "label": "table",
                "data": {"table_cells": [3]},
            },
        ],
        "pictures": "not-a-list",
        "formulas": [],
        "groups": [{"self_ref": "#/groups/0", "children": "not-a-list"}],
    }
    inputs["docling_document"] = malformed
    inputs["content_span_set"] = inputs["content_span_set"].model_copy(
        update={"spans": ()}
    )
    _add_integrity_product(inputs)
    view = build_canonical_document_view(
        **inputs,
        integrity_report=_integrity_payload(malformed, []),
    )

    assert view.metadata.title == "Fallback Title"
    assert {diagnostic.code for diagnostic in view.diagnostics} == {
        "INVALID_DOCUMENT_FURNITURE",
        "INVALID_NATIVE_COLLECTION",
        "INVALID_NATIVE_NODE",
        "INVALID_TABLE_CELL",
        "INVALID_TABLE_DATA",
        "INVALID_TABLE_GRID",
        "INVALID_TABLE_GRID_ROW",
        "EMPTY_TABLE_DATA",
        "MISSING_DOCUMENT_BODY",
        "MISSING_SOURCE_SPAN",
    }
    tables = [block for block in view.blocks if block.kind is CanonicalBlockKind.TABLE]
    assert all(table.table == CanonicalTable() for table in tables)
    assert any(block.kind is CanonicalBlockKind.OTHER for block in view.blocks)

    absent_furniture = _document()
    absent_furniture["furniture"] = None
    absent_furniture_view = build_canonical_document_view(
        **_fixture_inputs("pdf", absent_furniture)
    )
    assert "INVALID_DOCUMENT_FURNITURE" not in {
        diagnostic.code for diagnostic in absent_furniture_view.diagnostics
    }

    invalid_children = _document()
    invalid_children["groups"][0]["children"] = "not-a-list"
    for text_item in invalid_children["texts"]:
        if text_item.get("parent") == {"$ref": "#/groups/0"}:
            text_item.pop("parent")
    child_inputs = _fixture_inputs("pdf", _document())
    child_inputs["docling_document"] = invalid_children
    child_view = build_canonical_document_view(**child_inputs)
    group = next(
        block for block in child_view.blocks if block.kind is CanonicalBlockKind.GROUP
    )
    assert group.child_block_ids == ()

    mixed_children = _document()
    mixed_children["groups"][0]["children"] = [
        42,
        {"$ref": ""},
        {"$ref": "#/texts/0"},
    ]
    mixed_view = build_canonical_document_view(**_fixture_inputs("pdf", mixed_children))
    mixed_blocks = {block.native_node_id: block for block in mixed_view.blocks}
    assert (
        mixed_blocks["#/texts/0"].parent_block_id == mixed_blocks["#/groups/0"].block_id
    )

    missing_furniture_child = _document()
    missing_furniture_child["furniture"] = {
        "self_ref": "#/furniture",
        "children": [{"$ref": "#/texts/404"}],
    }
    furniture_view = build_canonical_document_view(
        **_fixture_inputs("pdf", missing_furniture_child)
    )
    assert "UNRESOLVED_NATIVE_REFERENCE" in {
        diagnostic.code for diagnostic in furniture_view.diagnostics
    }


def test_builder_preserves_ambiguous_partial_and_scholarly_diagnostics() -> None:
    document = _document()
    document["groups"][0]["children"][0] = {"$ref": "declared-title"}
    document["texts"][0]["self_ref"] = "declared-title"
    document["texts"][1]["self_ref"] = "ambiguous"
    document["texts"][2]["self_ref"] = "ambiguous"
    document["groups"][0]["children"].append({"$ref": "ambiguous"})
    inputs = _fixture_inputs("pdf", _document())
    inputs["docling_document"] = document
    _add_integrity_product(inputs)
    inputs["integrity_report"] = _integrity_payload(
        document,
        [
            {
                "kind": "figure",
                "source_ref": "#/pictures/0",
                "source_docling_item_ref": "#/pictures/0",
                "declared_target_refs": ["#/texts/4"],
                "resolved_docling_item_refs": ["#/texts/4"],
                "unresolved_target_refs": ["#/texts/404"],
                "reason_codes": ["one-target-missing"],
            },
        ],
    )
    view = build_canonical_document_view(**inputs)

    assert CanonicalRelationshipStatus.PARTIAL in {
        relationship.status for relationship in view.relationships
    }
    assert "AMBIGUOUS_NATIVE_REFERENCE" in {
        diagnostic.code for diagnostic in view.diagnostics
    }

    overlay = DoclingGrobidAligner(minimum_score=0.7).align(_document(), _tei())
    missing_product_inputs = _fixture_inputs("pdf", _document())
    with pytest.raises(CanonicalDocumentError, match="requires exactly one GROBID"):
        build_canonical_document_view(
            **missing_product_inputs,
            scholarly_overlay=overlay,
        )

    invalid_records = list(overlay.records)
    aligned_index = next(
        index
        for index, record in enumerate(invalid_records)
        if record.status.value == "aligned"
    )
    invalid_records[aligned_index] = replace(
        invalid_records[aligned_index],
        docling_item_ref="missing-aligned-node",
    )
    invalid_overlay = replace(overlay, records=tuple(invalid_records))
    invalid_overlay_inputs = _fixture_inputs("pdf", _document())
    _add_scholarly_products(invalid_overlay_inputs)
    invalid_alignment = build_canonical_document_view(
        **invalid_overlay_inputs,
        scholarly_overlay=invalid_overlay,
    )
    assert "UNRESOLVED_NATIVE_REFERENCE" in {
        diagnostic.code for diagnostic in invalid_alignment.diagnostics
    }

    unrelated_overlay = DoclingGrobidAligner(minimum_score=1.0).align(
        _document(),
        b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><p>'
        b"No native block contains this scholarly sentence."
        b"</p></body></text></TEI>",
    )
    assert unrelated_overlay.unaligned_count == 1
    with_product_inputs = _fixture_inputs("pdf", _document())
    _add_scholarly_products(with_product_inputs)
    unresolved = build_canonical_document_view(
        **with_product_inputs,
        scholarly_overlay=unrelated_overlay,
    )
    assert "UNRESOLVED_SCHOLARLY_ANCHOR" in {
        diagnostic.code for diagnostic in unresolved.diagnostics
    }


def test_schema_version_constant_is_descriptive() -> None:
    assert CANONICAL_DOCUMENT_SCHEMA_VERSION == (
        "deepcritical-canonical-document-view-v1"
    )
