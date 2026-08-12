"""Regenerate the deterministic canonical-document fixture bundle."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

_V1_ROOT = Path(__file__).resolve().parent
_CANONICAL_ROOT = _V1_ROOT.parent
_REPOSITORY_ROOT = _V1_ROOT.parents[4]
_TEST_MODULE = _REPOSITORY_ROOT / "tests/test_document_processing/test_canonical.py"
_FORMATS = ("pdf", "jats", "bioc")


def _assemble_pdf(objects: list[bytes]) -> bytes:
    body = bytearray(b"%PDF-1.4\n%\x00\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, payload in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{object_number} 0 obj\n".encode())
        body.extend(payload)
        body.extend(b"\nendobj\n")
    xref_offset = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    body.extend(b"0000000000 65535 f\r\n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n\r\n".encode())
    body.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(body)


def _pdf_text(text: str, *, x: int, y: int, size: int = 10) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"/F1 {size} Tf\n1 0 0 1 {x} {y} Tm\n({escaped}) Tj\n".encode()


def _article_pdf() -> bytes:
    content = bytearray(b"BT\n")
    for text, x, y, size in (
        ("APOE4 pathway", 10, 760, 16),
        ("Methods", 10, 715, 13),
        ("Endosomal pH was measured.", 10, 675, 10),
        ("Table 1. Cohort.", 10, 605, 10),
        ("Group", 10, 570, 10),
        ("N", 150, 570, 10),
        ("APOE4", 10, 550, 10),
        ("12", 150, 550, 10),
        ("Figure 1. Endosomes.", 10, 455, 10),
        ("[1]", 10, 405, 10),
        ("Smith 2026 endosomal study.", 10, 365, 10),
        ("pH = -log10[H+]", 10, 315, 10),
    ):
        content.extend(_pdf_text(text, x=x, y=y, size=size))
    content.extend(b"ET\n")
    return _assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            ),
            b"<< /Length "
            + str(len(content)).encode()
            + b" >>\nstream\n"
            + bytes(content)
            + b"endstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_record(relative_path: str, *, role: str) -> dict[str, object]:
    payload = (_V1_ROOT / relative_path).resolve().read_bytes()
    return {
        "byte_size": len(payload),
        "role": role,
        "sha256": _sha256(payload),
        "transform": "raw-bytes",
    }


def _write_generated_outputs() -> None:
    (_V1_ROOT / "article.pdf").write_bytes(_article_pdf())
    sys.path.insert(0, str(_REPOSITORY_ROOT))
    namespace = runpy.run_path(str(_TEST_MODULE))
    build_fixture = cast(
        "Callable[[str], tuple[Any, Mapping[str, bytes]]]",
        namespace["_fixture_bundle"],
    )
    canonical_document_bytes = cast(
        "Callable[[Any], bytes]", namespace["canonical_document_bytes"]
    )
    for format_name in _FORMATS:
        view, intermediates = build_fixture(format_name)
        output_root = _V1_ROOT / format_name
        output_root.mkdir(exist_ok=True)
        for name, payload in intermediates.items():
            (output_root / name).write_bytes(payload)
        (_CANONICAL_ROOT / f"{format_name}.json").write_bytes(
            canonical_document_bytes(view)
        )


def _write_manifest() -> None:
    roles = {
        "README.md": "fixture-documentation",
        "generate.py": "deterministic-generator",
        "article.pdf": "raw-pdf-source",
        "article.jats.xml": "raw-jats-source",
        "article.bioc.json": "raw-bioc-source",
        "docling_document.json": "shared-synthetic-docling-input",
        "grobid.tei.xml": "synthetic-scholarly-input",
        "pdf/content_spans.json": "frozen-derived-product",
        "pdf/alignment_overlay.json": "frozen-derived-product",
        "pdf/content_integrity_overlay.json": "frozen-derived-product",
        "jats/content_spans.json": "frozen-derived-product",
        "jats/content_integrity_overlay.json": "frozen-derived-product",
        "bioc/content_spans.json": "frozen-derived-product",
        "bioc/content_integrity_overlay.json": "frozen-derived-product",
        "../pdf.json": "expected-canonical-output",
        "../jats.json": "expected-canonical-output",
        "../bioc.json": "expected-canonical-output",
    }
    files = {
        path: _file_record(path, role=role) for path, role in sorted(roles.items())
    }
    docling_file = (_V1_ROOT / "docling_document.json").read_bytes()
    docling_product = _canonical_json(json.loads(docling_file))
    files["docling_document.json"]["product_bytes"] = {
        "byte_size": len(docling_product),
        "sha256": _sha256(docling_product),
        "transform": "canonical-json-v1",
    }
    manifest = {
        "schema_version": "deepcritical-canonical-fixture-manifest-v1",
        "fixture_version": "v1",
        "encoding": {
            "canonical-json-v1": {
                "allow_nan": False,
                "ensure_ascii": False,
                "separators": [",", ":"],
                "sort_keys": True,
                "text_encoding": "utf-8",
            }
        },
        "configuration": {
            "alignment_minimum_score": 0.7,
            "anchoring_policy": "source-spans-and-native-nodes-v1",
            "text_normalization": "unicode-nfc-collapse-whitespace-v1",
        },
        "components": {
            "bioc-adapter": "1",
            "canonical-document-view": "1",
            "docling-content-integrity": "1",
            "docling-grobid-aligner": "2",
            "jats-locator-adapter": "1",
        },
        "external_services": {
            "captured_from_services": False,
            "container_image_digests": {},
            "service_versions": {},
            "statement": (
                "This is a deterministic synthetic unit-contract fixture, not a "
                "Docling or GROBID service capture. Live tests separately attest "
                "configured external services."
            ),
        },
        "provenance": {
            "article_pdf": "generated byte-for-byte by generate.py",
            "docling_document": (
                "hand-authored serialized Docling v1 shape whose represented text "
                "is present in each raw source"
            ),
            "grobid_tei": (
                "hand-authored TEI P5-shaped scholarly evidence for the PDF bundle"
            ),
            "intermediates": (
                "derived by repository production adapters, span builders, aligner, "
                "and integrity validator, then frozen for byte comparison"
            ),
        },
        "bundles": {
            "bioc": {
                "expected_canonical": "../bioc.json",
                "intermediates": [
                    "bioc/content_spans.json",
                    "bioc/content_integrity_overlay.json",
                ],
                "raw_source": "article.bioc.json",
            },
            "jats": {
                "expected_canonical": "../jats.json",
                "intermediates": [
                    "jats/content_spans.json",
                    "jats/content_integrity_overlay.json",
                ],
                "raw_source": "article.jats.xml",
            },
            "pdf": {
                "expected_canonical": "../pdf.json",
                "intermediates": [
                    "pdf/content_spans.json",
                    "pdf/alignment_overlay.json",
                    "pdf/content_integrity_overlay.json",
                ],
                "raw_source": "article.pdf",
                "scholarly_source": "grobid.tei.xml",
            },
        },
        "files": files,
    }
    (_V1_ROOT / "manifest.json").write_text(
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    _write_generated_outputs()
    _write_manifest()


if __name__ == "__main__":
    main()
