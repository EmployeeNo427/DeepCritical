from __future__ import annotations

from copy import deepcopy

import pytest

from DeepResearch.src.document_processing.alignment import DoclingGrobidAligner
from DeepResearch.src.document_processing.validation import (
    ContentIntegrityKind,
    ContentIntegrityStatus,
    QualitySeverity,
    parse_content_integrity_report,
    probably_image_only,
    validate_content_integrity,
)


def _docling_document() -> dict:
    return {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "paper",
        "body": {
            "self_ref": "#/body",
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/tables/0"},
                {"$ref": "#/pictures/0"},
                {"$ref": "#/texts/3"},
            ],
        },
        "furniture": {"self_ref": "#/furniture", "children": []},
        "groups": [],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "paragraph",
                "text": "[1]",
            },
            {
                "self_ref": "#/texts/1",
                "label": "caption",
                "text": "Table 1. Participant characteristics.",
            },
            {
                "self_ref": "#/texts/2",
                "label": "caption",
                "text": "Figure 1. Biomarker concentrations.",
            },
            {
                "self_ref": "#/texts/3",
                "label": "paragraph",
                "text": "Smith 2024 Alzheimer cohort study.",
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "captions": [{"$ref": "#/texts/1"}],
                "data": {"grid": [["Group", "Count"], ["Control", "12"]]},
            }
        ],
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "label": "picture",
                "captions": [{"$ref": "#/texts/2"}],
            }
        ],
        "key_value_items": [],
        "pages": {"1": {"page_no": 1}},
    }


def _aligned_overlay(document: dict):
    tei = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body><p><ref type="bibr" target="#b1">[1]</ref></p></body>
        <back><listBibl><biblStruct xml:id="b1">
          Smith 2024 Alzheimer cohort study.
        </biblStruct></listBibl></back>
      </text>
    </TEI>"""
    return DoclingGrobidAligner(minimum_score=0.7).align(document, tei)


def test_integrity_overlay_resolves_tables_figures_and_citations() -> None:
    document = _docling_document()
    overlay = _aligned_overlay(document)

    report = validate_content_integrity(document, scholarly_overlay=overlay)

    assert report.scholarly_overlay_present
    assert report.resolved_count == 3
    assert report.unaligned_count == 0
    assert report.issues == ()
    assert [record.kind for record in report.records] == [
        ContentIntegrityKind.TABLE,
        ContentIntegrityKind.FIGURE,
        ContentIntegrityKind.CITATION,
    ]
    assert report.records[0].resolved_docling_item_refs == ("#/texts/1",)
    assert report.records[1].resolved_docling_item_refs == ("#/texts/2",)
    citation = report.records[2]
    assert citation.source_docling_item_ref == "#/texts/0"
    assert citation.declared_target_refs == ("#b1",)
    assert citation.resolved_docling_item_refs == ("#/texts/3",)

    repeated = validate_content_integrity(document, scholarly_overlay=overlay)
    assert repeated.to_dict() == report.to_dict()
    assert parse_content_integrity_report(report.to_dict()) == report


def test_persisted_integrity_report_requires_exact_identity_replay() -> None:
    report = validate_content_integrity(_docling_document()).to_dict()
    wrong_id = deepcopy(report)
    wrong_id["records"][0]["record_id"] = "0" * 64
    with pytest.raises(ValueError, match="record_id"):
        parse_content_integrity_report(wrong_id)

    unknown = deepcopy(report)
    unknown["records"][0]["kind"] = "unknown"
    with pytest.raises(ValueError, match="kind is unknown"):
        parse_content_integrity_report(unknown)

    duplicate_refs = deepcopy(report)
    duplicate_refs["records"][0]["declared_target_refs"] = [
        "#/texts/1",
        "#/texts/1",
    ]
    with pytest.raises(ValueError, match="values must be unique"):
        parse_content_integrity_report(duplicate_refs)

    extra = deepcopy(report)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="fields do not match v1"):
        parse_content_integrity_report(extra)

    non_string_field = deepcopy(report)
    non_string_field[1] = "not a JSON object key"
    with pytest.raises(ValueError, match="fields must be strings"):
        parse_content_integrity_report(non_string_field)

    resolved_with_unresolved_target = deepcopy(report)
    resolved_with_unresolved_target["records"][0]["unresolved_target_refs"] = [
        "#/texts/404"
    ]
    with pytest.raises(ValueError, match="cannot retain unresolved targets"):
        parse_content_integrity_report(resolved_with_unresolved_target)


def test_persisted_integrity_report_rejects_every_untyped_container() -> None:
    base = validate_content_integrity(_docling_document()).to_dict()

    with pytest.raises(ValueError, match="must be an object"):
        parse_content_integrity_report(None)

    malformed = deepcopy(base)
    malformed["scholarly_overlay_present"] = 1
    with pytest.raises(ValueError, match="must be a boolean"):
        parse_content_integrity_report(malformed)

    malformed = deepcopy(base)
    malformed["records"] = {}
    with pytest.raises(ValueError, match="records must be a list"):
        parse_content_integrity_report(malformed)

    malformed = deepcopy(base)
    malformed["records"].append(deepcopy(malformed["records"][0]))
    with pytest.raises(ValueError, match="record IDs must be unique"):
        parse_content_integrity_report(malformed)

    malformed = deepcopy(base)
    malformed["issues"] = {}
    with pytest.raises(ValueError, match="issues must be a list"):
        parse_content_integrity_report(malformed)

    malformed = deepcopy(base)
    malformed["resolved_count"] = -1
    with pytest.raises(ValueError, match="non-negative integer"):
        parse_content_integrity_report(malformed)

    malformed = deepcopy(base)
    malformed["unaligned_count"] += 1
    with pytest.raises(ValueError, match="unaligned_count does not match"):
        parse_content_integrity_report(malformed)


def test_persisted_integrity_record_and_issue_scalars_are_strict() -> None:
    base = validate_content_integrity(_docling_document()).to_dict()

    malformed = deepcopy(base)
    malformed["records"][0]["source_ref"] = ""
    with pytest.raises(ValueError, match="non-empty string"):
        parse_content_integrity_report(malformed)

    malformed = deepcopy(base)
    malformed["records"][0]["record_id"] = "not-a-sha"
    with pytest.raises(ValueError, match="SHA-256"):
        parse_content_integrity_report(malformed)

    malformed = deepcopy(base)
    malformed["records"][0]["status"] = 1
    with pytest.raises(ValueError, match="status must be a string"):
        parse_content_integrity_report(malformed)

    malformed = deepcopy(base)
    malformed["records"][0]["reason_codes"] = "not-a-list"
    with pytest.raises(ValueError, match="reason_codes must be a list"):
        parse_content_integrity_report(malformed)

    valid_issue = deepcopy(base)
    valid_issue["issues"] = [
        {
            "code": "PINNED_ISSUE",
            "message": "Pinned issue",
            "severity": "warning",
            "item_ref": None,
            "page_number": 1,
        }
    ]
    assert parse_content_integrity_report(valid_issue).issues[0].page_number == 1

    for field, value, match in (
        ("item_ref", "", "non-empty string"),
        ("page_number", 0, "greater than zero"),
        ("severity", 1, "severity must be a string"),
    ):
        malformed = deepcopy(valid_issue)
        malformed["issues"][0][field] = value
        with pytest.raises(ValueError, match=match):
            parse_content_integrity_report(malformed)

    invalid_document = _docling_document()
    invalid_document["tables"] = [42]
    source_missing = validate_content_integrity(invalid_document).to_dict()
    assert source_missing["records"][0]["source_docling_item_ref"] is None
    assert parse_content_integrity_report(source_missing).records[0].source_ref


def test_integrity_overlay_explicitly_records_every_unaligned_relationship() -> None:
    document = _docling_document()
    document["tables"][0]["captions"] = [{"$ref": "#/texts/404"}]
    document["pictures"][0]["captions"] = [{"missing_ref": "#/texts/2"}]
    tei = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body><ref type="bibr" target="#missing">[404]</ref></body></text>
    </TEI>"""
    overlay = DoclingGrobidAligner(minimum_score=0.7).align(document, tei)

    report = validate_content_integrity(document, scholarly_overlay=overlay)

    assert len(report.records) == 3
    assert report.resolved_count == 0
    assert report.unaligned_count == 3
    assert all(
        record.status is ContentIntegrityStatus.UNALIGNED for record in report.records
    )
    table, figure, citation = report.records
    assert table.declared_target_refs == ("#/texts/404",)
    assert table.unresolved_target_refs == ("#/texts/404",)
    assert table.reason_codes == ("caption_target_not_found",)
    assert figure.unresolved_target_refs == ("#/pictures/0/captions/0",)
    assert figure.reason_codes == ("malformed_caption_reference",)
    assert citation.declared_target_refs == ("#missing",)
    assert citation.unresolved_target_refs == ("#missing",)
    assert citation.reason_codes == (
        "citation_source_unaligned",
        "citation_target_not_found",
    )
    assert [issue.code for issue in report.issues] == [
        "UNALIGNED_TABLE",
        "UNALIGNED_FIGURE",
        "UNALIGNED_CITATION",
    ]
    assert report.issues[1].severity is QualitySeverity.ERROR


def test_caption_target_must_be_a_unique_caption_item() -> None:
    document = _docling_document()
    document["tables"][0]["captions"] = [{"$ref": "#/texts/0"}]
    duplicate = deepcopy(document["texts"][2])
    duplicate["self_ref"] = "#/texts/2"
    document["texts"].append(duplicate)

    report = validate_content_integrity(document)

    table, figure = report.records
    assert table.status is ContentIntegrityStatus.UNALIGNED
    assert table.reason_codes == ("caption_target_not_caption",)
    assert figure.status is ContentIntegrityStatus.UNALIGNED
    assert figure.reason_codes == ("caption_target_ambiguous",)
    assert not report.scholarly_overlay_present


def test_duplicate_source_refs_keep_unique_explicit_records() -> None:
    document = _docling_document()
    duplicate = deepcopy(document["tables"][0])
    document["tables"].append(duplicate)

    report = validate_content_integrity(document)
    table_records = [
        record for record in report.records if record.kind is ContentIntegrityKind.TABLE
    ]

    assert len(table_records) == 2
    assert len({record.record_id for record in table_records}) == 2
    assert [record.source_ref for record in table_records] == [
        "#/tables/0",
        "#/tables/1",
    ]
    assert all(
        record.reason_codes == ("duplicate_source_ref",) for record in table_records
    )


def test_malformed_content_collections_are_diagnosed_not_ignored() -> None:
    document = _docling_document()
    document["tables"] = {"unexpected": "mapping"}
    del document["pictures"]

    report = validate_content_integrity(document)

    assert [issue.code for issue in report.issues] == [
        "INVALID_TABLES_COLLECTION",
        "MISSING_PICTURES_COLLECTION",
    ]
    assert all(issue.severity is QualitySeverity.ERROR for issue in report.issues)


def test_multi_target_citation_preserves_resolved_and_unresolved_targets() -> None:
    document = _docling_document()
    document["texts"][0]["text"] = "[1,2]"
    tei = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body><ref type="bibr" target="#b1 #b2">[1,2]</ref></body>
        <back><listBibl><biblStruct xml:id="b1">
          Smith 2024 Alzheimer cohort study.
        </biblStruct></listBibl></back>
      </text>
    </TEI>"""
    overlay = DoclingGrobidAligner(minimum_score=0.7).align(document, tei)

    report = validate_content_integrity(document, scholarly_overlay=overlay)
    citation = next(
        record
        for record in report.records
        if record.kind is ContentIntegrityKind.CITATION
    )

    assert citation.status is ContentIntegrityStatus.UNALIGNED
    assert citation.declared_target_refs == ("#b1", "#b2")
    assert citation.resolved_docling_item_refs == ("#/texts/3",)
    assert citation.unresolved_target_refs == ("#b2",)
    assert citation.reason_codes == ("citation_target_not_found",)


def test_non_bibliographic_tei_refs_are_not_reported_as_citations() -> None:
    document = _docling_document()
    tei = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body><ref type="figure" target="#fig1">Figure 1</ref></body></text>
    </TEI>"""
    overlay = DoclingGrobidAligner(minimum_score=0.7).align(document, tei)

    report = validate_content_integrity(document, scholarly_overlay=overlay)

    assert [record.kind for record in report.records] == [
        ContentIntegrityKind.TABLE,
        ContentIntegrityKind.FIGURE,
    ]


def test_image_only_detection_uses_page_level_provenance_and_ratio() -> None:
    document = {
        "pages": {str(page): {"page_no": page} for page in range(1, 6)},
        "texts": [
            {
                "text": "a" * 20,
                "prov": [{"page_no": 1}],
            },
            {
                "text": "b" * 20,
                "prov": [{"page_no": 2}],
            },
        ],
    }

    assert not probably_image_only(
        document,
        minimum_characters_per_page=20,
        image_only_page_ratio=0.8,
    )

    document["texts"].pop()
    assert probably_image_only(
        document,
        minimum_characters_per_page=20,
        image_only_page_ratio=0.8,
    )


def test_image_only_detection_uses_character_spans_for_multi_page_text() -> None:
    document = {
        "pages": {"1": {"page_no": 1}, "2": {"page_no": 2}},
        "texts": [
            {
                "text": "x" * 40,
                "prov": [
                    {"page_no": 1, "charspan": [0, 25]},
                    {"page_no": 2, "charspan": [25, 40]},
                ],
            }
        ],
    }

    assert probably_image_only(
        document,
        minimum_characters_per_page=20,
        image_only_page_ratio=0.5,
    )
    assert not probably_image_only(
        document,
        minimum_characters_per_page=20,
        image_only_page_ratio=0.6,
    )


def test_image_only_detection_does_not_credit_unlocated_text() -> None:
    document = {
        "pages": {"1": {"page_no": 1}},
        "texts": [{"text": "searchable-looking text" * 10, "prov": []}],
    }

    assert probably_image_only(document)


@pytest.mark.parametrize(
    ("minimum_characters_per_page", "image_only_page_ratio"),
    [(-1, 0.8), (20, -0.1), (20, 1.1)],
)
def test_image_only_detection_rejects_invalid_thresholds(
    minimum_characters_per_page: int,
    image_only_page_ratio: float,
) -> None:
    with pytest.raises(ValueError, match="must be"):
        probably_image_only(
            {},
            minimum_characters_per_page=minimum_characters_per_page,
            image_only_page_ratio=image_only_page_ratio,
        )
