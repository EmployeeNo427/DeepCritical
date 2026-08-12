"""Opt-in compatibility contracts for the real document-processing stack.

The module is inert unless ``DEEPCRITICAL_RUN_LIVE_DOCUMENT_PROCESSING=1`` is
present before pytest starts. Its fixtures generate reusable synthetic PDFs, so
the lane never uploads licensed or corpus material.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from defusedxml import ElementTree
from pypdf import PdfReader

from DeepResearch.src.document_processing.canonical import canonical_document_bytes
from DeepResearch.src.document_processing.clients import (
    ContainerOCRmyPDFRunner,
    DoclingConversionResult,
    DoclingServeClient,
    GrobidClient,
    GrobidResult,
    OCRResult,
)
from DeepResearch.src.document_processing.models import (
    ArtifactRelationship,
    ProcessingRun,
    ProcessingRunStatus,
    RuntimeAttestation,
    RuntimeAttestationSource,
    utc_now,
)
from DeepResearch.src.document_processing.orchestration import (
    CompiledStage,
    LocalStageExecutor,
    StageContext,
    StageResult,
)
from DeepResearch.src.document_processing.pipeline import (
    DocumentProcessingConfig,
    DocumentProcessingResult,
    DocumentProcessor,
)
from DeepResearch.src.document_processing.storage import ContentAddressedStore

_LIVE_ENABLED = os.getenv("DEEPCRITICAL_RUN_LIVE_DOCUMENT_PROCESSING") == "1"
_LIVE_OCR_ENABLED = os.getenv("DEEPCRITICAL_RUN_LIVE_DOCUMENT_PROCESSING_OCR") == "1"
_LIVE_ALL_REAL_ENABLED = (
    os.getenv("DEEPCRITICAL_RUN_LIVE_DOCUMENT_PROCESSING_ALL_REAL") == "1"
)
_DIGEST_ADDRESSED_IMAGE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
_TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"

pytestmark = [
    pytest.mark.document_processing_live,
    pytest.mark.integration,
    pytest.mark.requires_network,
    pytest.mark.skipif(
        not _LIVE_ENABLED,
        reason=(
            "set DEEPCRITICAL_RUN_LIVE_DOCUMENT_PROCESSING=1 to run real "
            "parser-stack contracts"
        ),
    ),
]


@dataclass(frozen=True, slots=True)
class _LiveStackConfig:
    docling_url: str
    docling_api_key: str
    grobid_url: str
    grobid_api_key: str
    connect_timeout_seconds: float
    request_timeout_seconds: float
    task_timeout_seconds: float
    expected_docling_version: str
    expected_docling_serve_version: str
    expected_grobid_version: str


def _required_environment(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required when live document-processing tests run"
        )
    return value


def _positive_environment_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _configured_service_version(component_id: str, field: str) -> str:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "document_processing"
        / "default.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    value = config["services"][component_id][field]
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"configured {component_id}.{field} must be a non-empty string"
        )
    return value.strip()


@pytest.fixture(scope="session")
def live_stack_config() -> _LiveStackConfig:
    """Load only namespaced live credentials, never unit-test fixture keys."""

    return _LiveStackConfig(
        docling_url=os.getenv("DEEPCRITICAL_LIVE_DOCLING_URL", "http://127.0.0.1:5001"),
        docling_api_key=_required_environment("DEEPCRITICAL_LIVE_DOCLING_API_KEY"),
        grobid_url=os.getenv("DEEPCRITICAL_LIVE_GROBID_URL", "http://127.0.0.1:8070"),
        grobid_api_key=_required_environment("DEEPCRITICAL_LIVE_GROBID_API_KEY"),
        connect_timeout_seconds=_positive_environment_float(
            "DEEPCRITICAL_LIVE_CONNECT_TIMEOUT_SECONDS", 10.0
        ),
        request_timeout_seconds=_positive_environment_float(
            "DEEPCRITICAL_LIVE_REQUEST_TIMEOUT_SECONDS", 600.0
        ),
        task_timeout_seconds=_positive_environment_float(
            "DEEPCRITICAL_LIVE_TASK_TIMEOUT_SECONDS", 1800.0
        ),
        expected_docling_version=os.getenv(
            "DEEPCRITICAL_LIVE_EXPECTED_DOCLING_VERSION",
            _configured_service_version("docling", "component_version"),
        ).strip(),
        expected_docling_serve_version=os.getenv(
            "DEEPCRITICAL_LIVE_EXPECTED_DOCLING_SERVE_VERSION",
            _configured_service_version("docling", "serve_version"),
        ).strip(),
        expected_grobid_version=os.getenv(
            "DEEPCRITICAL_LIVE_EXPECTED_GROBID_VERSION",
            _configured_service_version("grobid", "component_version"),
        ).strip(),
    )


@dataclass(frozen=True, slots=True)
class _DockerRuntimeReporter:
    """Bind one parser invocation to the container inspected by the CI host."""

    expected_reporter_id: str
    component_id: str
    component_version: str
    component_versions: Mapping[str, str]
    container_reference: str
    container_id: str
    expected_image_id: str

    async def resolve(
        self,
        *,
        component_id: str,
        remote_task_id: str | None,
    ) -> RuntimeAttestation:
        if component_id != self.component_id:
            raise ValueError("runtime reporter component does not match")
        if remote_task_id is None or not remote_task_id.strip():
            raise ValueError("runtime reporter requires an invocation identity")
        if not _DIGEST_ADDRESSED_IMAGE.fullmatch(self.container_reference):
            raise ValueError("runtime reporter requires an immutable image reference")
        inspected = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.Image}}",
                self.container_id,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if inspected != self.expected_image_id:
            raise ValueError(
                "running parser container does not use the expected image ID"
            )
        return RuntimeAttestation(
            component_id=component_id,
            component_version=self.component_version,
            invocation_id=remote_task_id,
            source=RuntimeAttestationSource.AUTHENTICATED_DEPLOYMENT_REPORTER,
            reporter_id=self.expected_reporter_id,
            observed_at=utc_now(),
            workload_id=self.container_id,
            container_reference=self.container_reference,
            container_digest=self.container_reference.rsplit("@", maxsplit=1)[1],
            container_image_id=inspected,
            component_versions=dict(self.component_versions),
        )


class _RecordingExecutor:
    def __init__(self) -> None:
        self.stage_ids: list[str] = []
        self.local = LocalStageExecutor()

    async def execute(
        self,
        stage: CompiledStage,
        context: StageContext,
    ) -> StageResult:
        self.stage_ids.append(stage.spec.stage_id)
        return await self.local.execute(stage, context)


def _fixture_docling_document(*, image_only: bool = False) -> dict[str, Any]:
    text = (
        "x"
        if image_only
        else "This deterministic upstream document reports reproducible evidence."
    )
    return {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "live-pipeline-fixture",
        "body": {
            "self_ref": "#/body",
            "children": [{"$ref": "#/texts/0"}],
        },
        "furniture": {"self_ref": "#/furniture", "children": []},
        "groups": [],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "paragraph",
                "text": text,
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {
                            "l": 10,
                            "t": 40,
                            "r": 180,
                            "b": 20,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    }
                ],
            }
        ],
        "tables": [],
        "pictures": [],
        "key_value_items": [],
        "pages": {"1": {"page_no": 1}},
    }


class _FixtureDocling:
    def __init__(self, *, image_only: bool = False) -> None:
        self.image_only = image_only
        self.calls = 0

    async def convert(self, *args: Any, **kwargs: Any) -> DoclingConversionResult:
        self.calls += 1
        document = _fixture_docling_document(image_only=self.image_only)
        raw = {
            "document": {"json_content": document},
            "status": "success",
            "processing_time": 0.01,
            "timings": {},
            "errors": [],
        }
        return DoclingConversionResult(
            document=document,
            raw_response=raw,
            status="success",
            processing_time_seconds=0.01,
            remote_task_id=f"fixture-docling-{self.calls}",
        )


class _FixtureGrobid:
    coordinates = GrobidClient.DEFAULT_COORDINATES
    consolidate_header = 0
    consolidate_citations = 0
    segment_sentences = True

    def __init__(self, *, force_ocr: bool = False) -> None:
        self.force_ocr = force_ocr
        self.calls = 0

    async def process_fulltext(
        self,
        content: bytes,
        *,
        filename: str,
    ) -> GrobidResult:
        self.calls += 1
        if self.force_ocr and self.calls == 1:
            tei = b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text/></TEI>'
        else:
            tei = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div>
            <head coords="1,10,10,80,10">Results</head>
            <p coords="1,10,30,80,10">Deterministic fixture evidence remains usable
            after the real OCR boundary and supports compiled-pipeline validation.</p>
            </div></body></text></TEI>"""
        return GrobidResult(
            tei_xml=tei,
            coordinates=self.coordinates,
            status_code=200,
            remote_request_id=f"fixture-grobid-{self.calls}",
        )


class _NeverGrobid(_FixtureGrobid):
    async def process_fulltext(
        self,
        content: bytes,
        *,
        filename: str,
    ) -> GrobidResult:
        raise AssertionError("GROBID must not run in the Docling-only smoke")


class _NeverOCR:
    languages = ("eng",)
    rotate_pages = True
    deskew = True
    jobs = 1
    optimize = 1

    async def convert(self, content: bytes) -> OCRResult:
        raise AssertionError("OCR must not run in this compiled-pipeline smoke")


def _assert_compiled_pipeline_result(
    *,
    processor: DocumentProcessor,
    store: ContentAddressedStore,
    result: DocumentProcessingResult,
    executor: _RecordingExecutor,
    skipped_stages: set[str],
) -> None:
    stage_ids = [stage.spec.stage_id for stage in processor.compiled_pipeline.stages]
    assert executor.stage_ids == [
        stage_id for stage_id in stage_ids if stage_id not in skipped_stages
    ]
    assert set(stage_ids) - set(executor.stage_ids) == skipped_stages
    assert result.status in {
        ProcessingRunStatus.COMPLETE,
        ProcessingRunStatus.PARTIAL,
    }
    assert result.processing_runs
    expected_specification = processor.pipeline_spec.model_dump(
        mode="json",
        by_alias=True,
    )
    expected_registry = processor.component_registry.contract_snapshot()
    for run in result.processing_runs:
        assert store.get_processing_run(run.run_id) == run
        policy = run.output_policy_snapshot
        assert policy["schema"] == "deepcritical-document-output-policy-v2"
        assert policy["local_pipeline"]["specification"] == expected_specification
        assert policy["local_pipeline"]["registry"] == expected_registry
        for product in run.outputs:
            assert product.producer_run_id == run.run_id
            assert run.artifact_id in product.source_artifact_ids
            store.verify_blob(product.blob_sha256)
        for product in run.inputs:
            producer = store.get_processing_run(product.producer_run_id)
            assert product in producer.outputs
    canonical_run = _run_for_component(
        result.processing_runs, "canonical-document-view"
    )
    canonical_product = canonical_run.require_output("canonical_document_view")
    canonical_view = store.read_canonical_document(canonical_product)
    assert result.canonical_document_sha256 == canonical_product.blob_sha256
    assert canonical_view.artifact_id == result.artifact.artifact_id
    assert canonical_view.blocks
    assert all(block.source_anchors for block in canonical_view.blocks)
    assert {product.name for product in canonical_view.source_products} >= {
        "docling_document",
        "content_spans",
        "content_integrity_overlay",
    }


def _run_for_component(
    runs: tuple[ProcessingRun, ...],
    component_id: str,
) -> ProcessingRun:
    selected = tuple(run for run in runs if run.component_id == component_id)
    assert len(selected) == 1
    return selected[0]


_CAPTURE_SUFFIXES = {
    "application/json": ".json",
    "application/pdf": ".pdf",
    "application/xml": ".xml",
    "text/plain": ".txt",
}
_CAPTURE_PRODUCT_SUFFIXES = {
    "alignment_overlay": ".json",
    "canonical_document_view": ".json",
    "content_integrity_overlay": ".json",
    "content_spans": ".json",
    "diagnostics_manifest": ".json",
    "docling_document": ".json",
    "docling_response": ".json",
    "grobid_tei": ".tei.xml",
    "ocr_log": ".json",
    "ocr_sidecar": ".txt",
    "runtime_attestation": ".json",
    "searchable_pdf": ".pdf",
}


def _capture_live_pipeline_evidence(
    *,
    capture_directory: Path,
    source_pdf: bytes,
    store: ContentAddressedStore,
    result: DocumentProcessingResult,
) -> Path:
    """Capture synthetic native products after verifying their durable chains."""

    capture_directory.mkdir(parents=True, exist_ok=True)
    source_path = capture_directory / "source-rich-raster.pdf"
    source_path.write_bytes(source_pdf)
    captured_products: list[dict[str, Any]] = []
    for run_index, run in enumerate(result.processing_runs, start=1):
        stage_label = re.sub(
            r"[^a-zA-Z0-9_.-]+",
            "-",
            run.stage_id,
        )
        for product in run.outputs:
            payload = store.read_data_product_bytes(product)
            suffix = _CAPTURE_PRODUCT_SUFFIXES.get(
                product.name,
                _CAPTURE_SUFFIXES.get(product.media_type, ".bin"),
            )
            filename = (
                f"{run_index:02d}-{stage_label}-{run.component_id}-"
                f"{product.name}{suffix}"
            )
            (capture_directory / filename).write_bytes(payload)
            captured_products.append(
                {
                    **product.model_dump(mode="json"),
                    "capture_file": filename,
                }
            )

    capture_files_by_product_id = {
        product["product_id"]: product["capture_file"] for product in captured_products
    }
    canonical_run = _run_for_component(
        result.processing_runs,
        "canonical-document-view",
    )
    canonical_view = store.read_canonical_document(
        canonical_run.require_output("canonical_document_view")
    )
    selected_native_products = []
    for product in canonical_view.source_products:
        if product.name not in {"docling_document", "grobid_tei"}:
            continue
        capture_file = capture_files_by_product_id.get(product.product_id)
        if capture_file is None:
            raise AssertionError(
                f"selected native product {product.product_id!r} was not captured"
            )
        selected_native_products.append(
            {
                **product.model_dump(mode="json"),
                "capture_file": capture_file,
            }
        )

    manifest = {
        "schema": "deepcritical-live-all-real-evidence-v1",
        "synthetic_input": True,
        "tested_revision": os.getenv("TESTED_REVISION", "local-unreported"),
        "capture": {
            "captured_at": utc_now().isoformat(),
            "github_run_id": os.getenv("GITHUB_RUN_ID", "local-unreported"),
            "github_run_attempt": os.getenv(
                "GITHUB_RUN_ATTEMPT",
                "local-unreported",
            ),
        },
        "source": {
            "artifact_id": result.artifact.artifact_id,
            "blob_sha256": result.artifact.source_sha256,
            "capture_file": source_path.name,
        },
        "runtime_images": {
            "docling": {
                "reference": _required_environment(
                    "DEEPCRITICAL_LIVE_DOCLING_IMAGE_REF"
                ),
                "image_id": _required_environment("DEEPCRITICAL_LIVE_DOCLING_IMAGE_ID"),
            },
            "grobid": {
                "reference": _required_environment(
                    "DEEPCRITICAL_LIVE_GROBID_IMAGE_REF"
                ),
                "image_id": _required_environment("DEEPCRITICAL_LIVE_GROBID_IMAGE_ID"),
                "dockerfile_blob": _required_environment(
                    "DEEPCRITICAL_LIVE_GROBID_DOCKERFILE_BLOB"
                ),
            },
            "ocrmypdf": {
                "reference": _required_environment("DEEPCRITICAL_LIVE_OCR_IMAGE"),
            },
        },
        "result": {
            "status": result.status.value,
            "route": list(result.route),
            "canonical_document_sha256": result.canonical_document_sha256,
            "derivative_artifact_ids": list(result.derivative_artifact_ids),
        },
        "artifacts": [
            artifact.model_dump(mode="json") for artifact in store.list_artifacts()
        ],
        "processing_runs": [
            run.model_dump(mode="json") for run in result.processing_runs
        ],
        "diagnostics": [
            diagnostic.model_dump(mode="json") for diagnostic in result.diagnostics
        ],
        "captured_products": captured_products,
        "selected_native_products": selected_native_products,
    }
    manifest_path = capture_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _assemble_pdf(objects: list[bytes]) -> bytes:
    body = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, payload in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{object_number} 0 obj\n".encode())
        body.extend(payload)
        body.extend(b"\nendobj\n")
    xref_offset = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode())
    body.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(body)


@pytest.fixture(scope="session")
def synthetic_scholarly_pdf() -> bytes:
    """Return a deterministic born-digital paper-shaped one-page PDF."""

    content = (
        b"BT\n/F1 16 Tf\n72 742 Td\n"
        b"(DeepCritical Parser Contract Study) Tj\n"
        b"0 -28 Td\n/F1 11 Tf\n(Ada Researcher and Ben Scientist) Tj\n"
        b"0 -34 Td\n/F1 13 Tf\n(Abstract) Tj\n"
        b"0 -20 Td\n/F1 10 Tf\n"
        b"(This reusable synthetic paper tests scientific document parsing.) Tj\n"
        b"0 -16 Td\n"
        b"(The experiment reports reproducible evidence and one citation.) Tj\n"
        b"0 -30 Td\n/F1 13 Tf\n(References) Tj\n"
        b"0 -20 Td\n/F1 10 Tf\n"
        b"(Researcher A. Synthetic parser evaluation. 2026.) Tj\nET\n"
    )
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
            + content
            + b"endstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


_BITMAP_FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def _raster_only_pdf(
    lines: tuple[str, ...],
    *,
    width: int,
    height: int,
    scale: int,
) -> bytes:
    if not lines or width <= 0 or height <= 0 or scale <= 0:
        raise ValueError("raster fixture dimensions and lines must be non-empty")
    unsupported = set("".join(lines)) - {" ", *_BITMAP_FONT}
    if unsupported:
        raise ValueError(f"unsupported raster fixture characters: {unsupported!r}")

    pixels = bytearray(b"\xff" * (width * height))
    glyph_width = 6 * scale
    line_height = 12 * scale
    rendered_height = (len(lines) - 1) * line_height + 7 * scale
    start_y = (height - rendered_height) // 2
    if start_y < 0:
        raise ValueError("raster fixture lines exceed the image height")
    for line_index, text in enumerate(lines):
        start_x = (width - len(text) * glyph_width) // 2
        if start_x < 0:
            raise ValueError("raster fixture line exceeds the image width")
        for character_index, character in enumerate(text):
            glyph = _BITMAP_FONT.get(character)
            if glyph is None:
                continue
            glyph_x = start_x + character_index * glyph_width
            for row, pattern in enumerate(glyph):
                for column, pixel in enumerate(pattern):
                    if pixel != "1":
                        continue
                    for y_offset in range(scale):
                        row_start = (
                            start_y + line_index * line_height + row * scale + y_offset
                        ) * width
                        for x_offset in range(scale):
                            pixels[row_start + glyph_x + column * scale + x_offset] = 0

    compressed = zlib.compress(bytes(pixels), level=9)
    display_width = 540
    display_height = round(display_width * height / width)
    display_y = (792 - display_height) // 2
    content = (
        f"q\n{display_width} 0 0 {display_height} 36 {display_y} cm\n/Im0 Do\nQ\n"
    ).encode()
    image = (
        b"<< /Type /XObject /Subtype /Image /Width "
        + str(width).encode()
        + b" /Height "
        + str(height).encode()
        + b" "
        b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
        b"/Length "
        + str(len(compressed)).encode()
        + b" >>\nstream\n"
        + compressed
        + b"\nendstream"
    )
    return _assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
            ),
            b"<< /Length "
            + str(len(content)).encode()
            + b" >>\nstream\n"
            + content
            + b"endstream",
            image,
        ]
    )


@pytest.fixture(scope="session")
def synthetic_image_only_pdf() -> bytes:
    """Return a deterministic raster-only PDF for the opt-in OCR contract."""

    return _raster_only_pdf(
        ("DEEPCRITICAL OCR TEST",),
        width=1800,
        height=600,
        scale=12,
    )


@pytest.fixture(scope="session")
def synthetic_rich_image_only_pdf() -> bytes:
    """Return a rich raster paper that both real OCR boundaries can parse."""

    return _raster_only_pdf(
        (
            "DEEPCRITICAL PARSER CONTRACT STUDY",
            "ABSTRACT",
            "THIS RASTER PAPER REPORTS A CONTROLLED EXPERIMENT",
            "THE STUDY TESTS REPRODUCIBLE DOCUMENT PROCESSING",
            "METHODS",
            "CELLS RECEIVED CONTROL OR TREATMENT CONDITIONS",
            "PARSER OUTPUTS WERE CAPTURED FOR CRITICAL REVIEW",
            "RESULTS",
            "TREATMENT INCREASED RESPONSE RELATIVE TO CONTROL",
            "THE OBSERVED RESULT WAS CONSISTENT ACROSS REPEATS",
            "DISCUSSION",
            "THE EVIDENCE SUPPORTS REPRODUCIBLE ASSESSMENT",
            "REFERENCES",
            "RESEARCHER SYNTHETIC PARSER EVALUATION",
        ),
        width=3200,
        height=2100,
        scale=8,
    )


@pytest.mark.asyncio
async def test_live_docling_async_conversion_contract(
    live_stack_config: _LiveStackConfig,
    synthetic_scholarly_pdf: bytes,
) -> None:
    client = DoclingServeClient(
        live_stack_config.docling_url,
        api_key=live_stack_config.docling_api_key,
        connect_timeout_seconds=live_stack_config.connect_timeout_seconds,
        request_timeout_seconds=live_stack_config.request_timeout_seconds,
        task_timeout_seconds=live_stack_config.task_timeout_seconds,
        poll_interval_seconds=0.5,
        use_async_api=True,
    )

    health = await client.health()
    versions = await client.version()
    result = await client.convert(
        synthetic_scholarly_pdf,
        filename="deepcritical-live-contract.pdf",
        media_type="application/pdf",
        options={"do_ocr": False},
    )

    assert health.ready is True
    assert isinstance(health.readiness, Mapping)
    assert isinstance(health.versions, Mapping)
    assert versions.get("docling") == live_stack_config.expected_docling_version
    assert versions.get("docling-serve") == (
        live_stack_config.expected_docling_serve_version
    )
    assert health.versions == versions
    assert result.status in {"success", "partial_success"}
    assert result.remote_task_id
    assert result.raw_response.get("status") == result.status
    assert result.document.get("schema_name") == "DoclingDocument"
    assert isinstance(result.document.get("texts"), list)
    assert result.document["texts"]


@pytest.mark.asyncio
async def test_live_grobid_tei_contract(
    live_stack_config: _LiveStackConfig,
    synthetic_scholarly_pdf: bytes,
) -> None:
    client = GrobidClient(
        live_stack_config.grobid_url,
        api_key=live_stack_config.grobid_api_key,
        connect_timeout_seconds=live_stack_config.connect_timeout_seconds,
        timeout_seconds=live_stack_config.request_timeout_seconds,
    )

    assert await client.health() is True
    version = await client.version()
    result = await client.process_fulltext(
        synthetic_scholarly_pdf, filename="deepcritical-live-contract.pdf"
    )

    root = ElementTree.fromstring(result.tei_xml)
    assert version == live_stack_config.expected_grobid_version
    assert result.status_code == 200
    assert root.tag == f"{{{_TEI_NAMESPACE}}}TEI"
    assert root.find(f"{{{_TEI_NAMESPACE}}}teiHeader") is not None
    assert root.find(f"{{{_TEI_NAMESPACE}}}text") is not None
    assert set(result.coordinates) == set(GrobidClient.DEFAULT_COORDINATES)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _LIVE_OCR_ENABLED,
    reason=(
        "set DEEPCRITICAL_RUN_LIVE_DOCUMENT_PROCESSING_OCR=1 to run the "
        "digest-addressed OCR container contract"
    ),
)
async def test_live_digest_addressed_ocr_derivative_contract(
    synthetic_image_only_pdf: bytes,
) -> None:
    runtime = os.getenv("DEEPCRITICAL_LIVE_CONTAINER_RUNTIME", "docker").strip()
    if not runtime or shutil.which(runtime) is None:
        raise RuntimeError(
            "DEEPCRITICAL_LIVE_CONTAINER_RUNTIME must name an installed OCI runtime"
        )
    image = _required_environment("DEEPCRITICAL_LIVE_OCR_IMAGE")
    if not _DIGEST_ADDRESSED_IMAGE.fullmatch(image):
        raise RuntimeError(
            "DEEPCRITICAL_LIVE_OCR_IMAGE must be an exact image@sha256:<64 hex> "
            "reference that is already present locally"
        )

    runner = ContainerOCRmyPDFRunner(
        image,
        runtime_executable=runtime,
        timeout_seconds=_positive_environment_float(
            "DEEPCRITICAL_LIVE_OCR_TIMEOUT_SECONDS", 600.0
        ),
        probe_timeout_seconds=_positive_environment_float(
            "DEEPCRITICAL_LIVE_OCR_PROBE_TIMEOUT_SECONDS", 60.0
        ),
        jobs=1,
        require_digest_addressed=True,
    )
    result = await runner.convert(synthetic_image_only_pdf)

    derivative = PdfReader(io.BytesIO(result.pdf_bytes))
    assert result.exit_code == 0
    assert result.pdf_bytes.startswith(b"%PDF-")
    assert len(derivative.pages) == 1
    assert result.sidecar_text.strip()
    assert derivative.pages[0].extract_text().strip()
    assert result.runtime_attestation is not None
    assert result.runtime_attestation.container_reference == image
    assert result.runtime_attestation.container_digest == image.rsplit("@", 1)[1]
    assert result.runtime_attestation.component_versions["ocrmypdf"].startswith(
        "17.4.1"
    )
    assert result.runtime_attestation.component_versions["tesseract"]


@pytest.mark.asyncio
async def test_live_compiled_docling_pipeline(
    live_stack_config: _LiveStackConfig,
    tmp_path: Path,
) -> None:
    base_client = DoclingServeClient(
        live_stack_config.docling_url,
        api_key=live_stack_config.docling_api_key,
        connect_timeout_seconds=live_stack_config.connect_timeout_seconds,
        request_timeout_seconds=live_stack_config.request_timeout_seconds,
        task_timeout_seconds=live_stack_config.task_timeout_seconds,
        poll_interval_seconds=0.5,
        use_async_api=True,
    )
    versions = await base_client.version()
    attested_versions = {
        "docling": versions["docling"],
        "docling_serve": versions["docling-serve"],
    }
    image_reference = _required_environment("DEEPCRITICAL_LIVE_DOCLING_IMAGE_REF")
    reporter = _DockerRuntimeReporter(
        expected_reporter_id="github-actions-docker-inspector",
        component_id="docling",
        component_version=versions["docling"],
        component_versions=attested_versions,
        container_reference=image_reference,
        container_id=_required_environment("DEEPCRITICAL_LIVE_DOCLING_CONTAINER_ID"),
        expected_image_id=_required_environment("DEEPCRITICAL_LIVE_DOCLING_IMAGE_ID"),
    )
    client = DoclingServeClient(
        live_stack_config.docling_url,
        api_key=live_stack_config.docling_api_key,
        connect_timeout_seconds=live_stack_config.connect_timeout_seconds,
        request_timeout_seconds=live_stack_config.request_timeout_seconds,
        task_timeout_seconds=live_stack_config.task_timeout_seconds,
        poll_interval_seconds=0.5,
        use_async_api=True,
        runtime_reporter=reporter,
    )
    executor = _RecordingExecutor()
    store = ContentAddressedStore(tmp_path / "docling-store")
    processor = DocumentProcessor(
        store,
        docling=client,
        grobid=_NeverGrobid(),
        ocrmypdf=_NeverOCR(),
        config=DocumentProcessingConfig(
            docling_version=live_stack_config.expected_docling_version,
            docling_serve_version=live_stack_config.expected_docling_serve_version,
            docling_container_image=image_reference.split("@", maxsplit=1)[0],
            docling_container_digest=image_reference.rsplit("@", maxsplit=1)[1],
            docling_options={"do_ocr": False},
            grobid_enabled=False,
            ocr_enabled=False,
            preflight_enabled=False,
            require_runtime_identity=False,
        ),
        stage_executor=executor,
    )
    artifact = processor.ingest_bytes(
        b"<html><body><h1>Methods</h1><p>Compiled Docling smoke.</p></body></html>",
        acquisition_uri="https://example.test/live-docling.html",
        media_type="text/html",
        identifiers={"filename": "live-docling.html"},
    )

    result = await processor.process_artifact(artifact.artifact_id)

    _assert_compiled_pipeline_result(
        processor=processor,
        store=store,
        result=result,
        executor=executor,
        skipped_stages={
            "primary-grobid",
            "ocr",
            "fallback-grobid",
            "alignment",
        },
    )
    run = _run_for_component(result.processing_runs, "docling")
    assert run.runtime_attestation is not None
    assert run.runtime_attestation.container_reference == image_reference
    assert run.runtime_attestation.container_image_id == reporter.expected_image_id
    assert run.component_versions == attested_versions


@pytest.mark.asyncio
async def test_live_compiled_grobid_pipeline(
    live_stack_config: _LiveStackConfig,
    synthetic_scholarly_pdf: bytes,
    tmp_path: Path,
) -> None:
    base_client = GrobidClient(
        live_stack_config.grobid_url,
        api_key=live_stack_config.grobid_api_key,
        connect_timeout_seconds=live_stack_config.connect_timeout_seconds,
        timeout_seconds=live_stack_config.request_timeout_seconds,
    )
    version = await base_client.version()
    image_reference = _required_environment("DEEPCRITICAL_LIVE_GROBID_IMAGE_REF")
    reporter = _DockerRuntimeReporter(
        expected_reporter_id="github-actions-docker-inspector",
        component_id="grobid",
        component_version=version,
        component_versions={"grobid": version},
        container_reference=image_reference,
        container_id=_required_environment("DEEPCRITICAL_LIVE_GROBID_CONTAINER_ID"),
        expected_image_id=_required_environment("DEEPCRITICAL_LIVE_GROBID_IMAGE_ID"),
    )
    client = GrobidClient(
        live_stack_config.grobid_url,
        api_key=live_stack_config.grobid_api_key,
        connect_timeout_seconds=live_stack_config.connect_timeout_seconds,
        timeout_seconds=live_stack_config.request_timeout_seconds,
        runtime_reporter=reporter,
    )
    executor = _RecordingExecutor()
    store = ContentAddressedStore(tmp_path / "grobid-store")
    processor = DocumentProcessor(
        store,
        docling=_FixtureDocling(),
        grobid=client,
        ocrmypdf=_NeverOCR(),
        # Only GROBID is live in this isolated lane. The deterministic Docling
        # fixture still records the exact native contract required for replay.
        config=DocumentProcessingConfig(
            grobid_version=version,
            grobid_container_image=image_reference.split("@", maxsplit=1)[0],
            grobid_container_digest=image_reference.rsplit("@", maxsplit=1)[1],
            grobid_minimum_text_characters=0,
            ocr_enabled=False,
            preflight_enabled=False,
            require_runtime_identity=False,
        ),
        stage_executor=executor,
    )
    artifact = processor.ingest_bytes(
        synthetic_scholarly_pdf,
        acquisition_uri="https://example.test/live-grobid.pdf",
        media_type="application/pdf",
        identifiers={"filename": "live-grobid.pdf"},
    )

    result = await processor.process_artifact(artifact.artifact_id)

    _assert_compiled_pipeline_result(
        processor=processor,
        store=store,
        result=result,
        executor=executor,
        skipped_stages={"ocr", "fallback-grobid"},
    )
    run = _run_for_component(result.processing_runs, "grobid")
    assert run.runtime_attestation is not None
    assert run.runtime_attestation.container_reference == image_reference
    assert run.runtime_attestation.container_image_id == reporter.expected_image_id
    assert run.component_versions == {"grobid": version}


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _LIVE_OCR_ENABLED,
    reason=(
        "set DEEPCRITICAL_RUN_LIVE_DOCUMENT_PROCESSING_OCR=1 to run the "
        "compiled digest-addressed OCR pipeline"
    ),
)
async def test_live_compiled_ocr_pipeline(
    synthetic_image_only_pdf: bytes,
    tmp_path: Path,
) -> None:
    image_reference = _required_environment("DEEPCRITICAL_LIVE_OCR_IMAGE")
    digest = image_reference.rsplit("@", maxsplit=1)[1]
    runner = ContainerOCRmyPDFRunner(
        image_reference,
        runtime_executable=os.getenv(
            "DEEPCRITICAL_LIVE_CONTAINER_RUNTIME",
            "docker",
        ).strip(),
        timeout_seconds=_positive_environment_float(
            "DEEPCRITICAL_LIVE_OCR_TIMEOUT_SECONDS",
            600.0,
        ),
        probe_timeout_seconds=_positive_environment_float(
            "DEEPCRITICAL_LIVE_OCR_PROBE_TIMEOUT_SECONDS",
            60.0,
        ),
        jobs=1,
        require_digest_addressed=True,
    )
    grobid = _FixtureGrobid(force_ocr=True)
    executor = _RecordingExecutor()
    store = ContentAddressedStore(tmp_path / "ocr-store")
    processor = DocumentProcessor(
        store,
        docling=_FixtureDocling(image_only=True),
        grobid=grobid,
        ocrmypdf=runner,
        # Only OCRmyPDF is live in this isolated lane. The deterministic parser
        # fixtures still record the exact native contracts required for replay.
        config=DocumentProcessingConfig(
            grobid_minimum_text_characters=80,
            ocrmypdf_version="17.4.1",
            ocr_mode="container_cli",
            ocr_container_image=image_reference.split("@", maxsplit=1)[0],
            ocr_container_digest=digest,
            preflight_enabled=False,
            require_runtime_identity=True,
        ),
        stage_executor=executor,
    )
    artifact = processor.ingest_bytes(
        synthetic_image_only_pdf,
        acquisition_uri="https://example.test/live-ocr.pdf",
        media_type="application/pdf",
        identifiers={"filename": "live-ocr.pdf"},
    )

    result = await processor.process_artifact(artifact.artifact_id)

    _assert_compiled_pipeline_result(
        processor=processor,
        store=store,
        result=result,
        executor=executor,
        skipped_stages=set(),
    )
    assert grobid.calls == 2
    run = _run_for_component(result.processing_runs, "ocrmypdf")
    assert run.runtime_attestation is not None
    assert run.runtime_attestation.container_reference == image_reference
    assert run.runtime_attestation.container_digest == digest
    assert run.component_versions["ocrmypdf"] == "17.4.1"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _LIVE_ALL_REAL_ENABLED,
    reason=(
        "set DEEPCRITICAL_RUN_LIVE_DOCUMENT_PROCESSING_ALL_REAL=1 to run one "
        "capture-capable pipeline across every real parser boundary"
    ),
)
async def test_live_all_real_pipeline_captures_native_outputs(
    live_stack_config: _LiveStackConfig,
    synthetic_rich_image_only_pdf: bytes,
    tmp_path: Path,
) -> None:
    tested_revision = _required_environment("TESTED_REVISION")
    if re.fullmatch(r"[0-9a-f]{40}", tested_revision) is None:
        raise RuntimeError("TESTED_REVISION must be an exact 40-character Git SHA")
    capture_directory = Path(
        _required_environment("DEEPCRITICAL_LIVE_CAPTURE_DIR")
    ).resolve()

    docling_probe = DoclingServeClient(
        live_stack_config.docling_url,
        api_key=live_stack_config.docling_api_key,
        connect_timeout_seconds=live_stack_config.connect_timeout_seconds,
        request_timeout_seconds=live_stack_config.request_timeout_seconds,
        task_timeout_seconds=live_stack_config.task_timeout_seconds,
        poll_interval_seconds=0.5,
        use_async_api=True,
    )
    docling_versions = await docling_probe.version()
    assert docling_versions.get("docling") == (
        live_stack_config.expected_docling_version
    )
    assert docling_versions.get("docling-serve") == (
        live_stack_config.expected_docling_serve_version
    )
    docling_image = _required_environment("DEEPCRITICAL_LIVE_DOCLING_IMAGE_REF")
    docling_reporter = _DockerRuntimeReporter(
        expected_reporter_id="github-actions-docker-inspector",
        component_id="docling",
        component_version=live_stack_config.expected_docling_version,
        component_versions={
            "docling": live_stack_config.expected_docling_version,
            "docling_serve": live_stack_config.expected_docling_serve_version,
        },
        container_reference=docling_image,
        container_id=_required_environment("DEEPCRITICAL_LIVE_DOCLING_CONTAINER_ID"),
        expected_image_id=_required_environment("DEEPCRITICAL_LIVE_DOCLING_IMAGE_ID"),
    )
    docling = DoclingServeClient(
        live_stack_config.docling_url,
        api_key=live_stack_config.docling_api_key,
        connect_timeout_seconds=live_stack_config.connect_timeout_seconds,
        request_timeout_seconds=live_stack_config.request_timeout_seconds,
        task_timeout_seconds=live_stack_config.task_timeout_seconds,
        poll_interval_seconds=0.5,
        use_async_api=True,
        runtime_reporter=docling_reporter,
    )

    grobid_probe = GrobidClient(
        live_stack_config.grobid_url,
        api_key=live_stack_config.grobid_api_key,
        connect_timeout_seconds=live_stack_config.connect_timeout_seconds,
        timeout_seconds=live_stack_config.request_timeout_seconds,
    )
    grobid_version = await grobid_probe.version()
    assert grobid_version == live_stack_config.expected_grobid_version
    grobid_image = _required_environment("DEEPCRITICAL_LIVE_GROBID_IMAGE_REF")
    grobid_reporter = _DockerRuntimeReporter(
        expected_reporter_id="github-actions-docker-inspector",
        component_id="grobid",
        component_version=grobid_version,
        component_versions={"grobid": grobid_version},
        container_reference=grobid_image,
        container_id=_required_environment("DEEPCRITICAL_LIVE_GROBID_CONTAINER_ID"),
        expected_image_id=_required_environment("DEEPCRITICAL_LIVE_GROBID_IMAGE_ID"),
    )
    grobid = GrobidClient(
        live_stack_config.grobid_url,
        api_key=live_stack_config.grobid_api_key,
        connect_timeout_seconds=live_stack_config.connect_timeout_seconds,
        timeout_seconds=live_stack_config.request_timeout_seconds,
        runtime_reporter=grobid_reporter,
    )

    runtime = os.getenv("DEEPCRITICAL_LIVE_CONTAINER_RUNTIME", "docker").strip()
    if not runtime or shutil.which(runtime) is None:
        raise RuntimeError(
            "DEEPCRITICAL_LIVE_CONTAINER_RUNTIME must name an installed OCI runtime"
        )
    ocr_image = _required_environment("DEEPCRITICAL_LIVE_OCR_IMAGE")
    if not _DIGEST_ADDRESSED_IMAGE.fullmatch(ocr_image):
        raise RuntimeError(
            "DEEPCRITICAL_LIVE_OCR_IMAGE must be an exact image@sha256 reference"
        )
    ocr = ContainerOCRmyPDFRunner(
        ocr_image,
        runtime_executable=runtime,
        timeout_seconds=_positive_environment_float(
            "DEEPCRITICAL_LIVE_OCR_TIMEOUT_SECONDS",
            900.0,
        ),
        probe_timeout_seconds=_positive_environment_float(
            "DEEPCRITICAL_LIVE_OCR_PROBE_TIMEOUT_SECONDS",
            60.0,
        ),
        jobs=1,
        require_digest_addressed=True,
    )

    executor = _RecordingExecutor()
    store = ContentAddressedStore(tmp_path / "all-real-store")
    processor = DocumentProcessor(
        store,
        docling=docling,
        grobid=grobid,
        ocrmypdf=ocr,
        config=DocumentProcessingConfig(
            docling_version=live_stack_config.expected_docling_version,
            docling_serve_version=live_stack_config.expected_docling_serve_version,
            docling_container_image=docling_image.split("@", maxsplit=1)[0],
            docling_container_digest=docling_image.rsplit("@", maxsplit=1)[1],
            docling_options={"do_ocr": True},
            grobid_version=grobid_version,
            grobid_container_image=grobid_image.split("@", maxsplit=1)[0],
            grobid_container_digest=grobid_image.rsplit("@", maxsplit=1)[1],
            grobid_minimum_text_characters=100,
            ocrmypdf_version="17.4.1",
            ocr_mode="container_cli",
            ocr_container_image=ocr_image.split("@", maxsplit=1)[0],
            ocr_container_digest=ocr_image.rsplit("@", maxsplit=1)[1],
            ocr_languages=("eng",),
            detect_image_only_pdfs=True,
            # Docling still performs real OCR. This deliberately unreachable
            # threshold proves that the external OCR fallback executes as well.
            minimum_text_characters_per_page=1_000_000,
            image_only_page_ratio=1.0,
            preflight_enabled=True,
            require_runtime_identity=True,
        ),
        stage_executor=executor,
    )
    artifact = processor.ingest_bytes(
        synthetic_rich_image_only_pdf,
        acquisition_uri="https://example.test/live-all-real-raster.pdf",
        media_type="application/pdf",
        identifiers={"filename": "live-all-real-raster.pdf"},
    )

    result = await processor.process_artifact(artifact.artifact_id)

    _assert_compiled_pipeline_result(
        processor=processor,
        store=store,
        result=result,
        executor=executor,
        skipped_stages=set(),
    )
    docling_run = _run_for_component(result.processing_runs, "docling")
    assert docling_run.stage_id == "docling"
    assert docling_run.configuration["options"]["do_ocr"] is True
    assert docling_run.runtime_attestation is not None
    assert docling_run.runtime_attestation.container_reference == docling_image
    assert docling_run.runtime_attestation.container_image_id == (
        docling_reporter.expected_image_id
    )

    ocr_run = _run_for_component(result.processing_runs, "ocrmypdf")
    assert ocr_run.stage_id == "ocr"
    assert "image_only_pages" in ocr_run.configuration["fallback_reason"].split("+")
    assert ocr_run.runtime_attestation is not None
    assert ocr_run.runtime_attestation.container_reference == ocr_image
    assert ocr_run.component_versions["ocrmypdf"] == "17.4.1"
    searchable_pdf = store.read_data_product_bytes(
        ocr_run.require_output("searchable_pdf")
    )
    searchable_reader = PdfReader(io.BytesIO(searchable_pdf))
    assert searchable_reader.pages[0].extract_text().strip()

    assert len(result.derivative_artifact_ids) == 1
    derivative_id = result.derivative_artifact_ids[0]
    derivative = store.get_artifact(derivative_id)
    assert derivative.relationship is ArtifactRelationship.DERIVATIVE
    assert derivative.parent_artifact_id == artifact.artifact_id
    assert derivative.raw_location.created_by_run_id == ocr_run.run_id
    grobid_runs = tuple(
        run for run in result.processing_runs if run.component_id == "grobid"
    )
    assert len(grobid_runs) == 2
    assert {run.stage_id for run in grobid_runs} == {
        "primary-grobid",
        "fallback-grobid",
    }
    primary_grobid = next(
        run for run in grobid_runs if run.artifact_id == artifact.artifact_id
    )
    fallback_grobid = next(
        run for run in grobid_runs if run.artifact_id == derivative_id
    )
    assert primary_grobid.stage_id == "primary-grobid"
    if primary_grobid.runtime_attestation is not None:
        assert primary_grobid.runtime_attestation.container_reference == grobid_image
        assert primary_grobid.runtime_attestation.container_image_id == (
            grobid_reporter.expected_image_id
        )
    assert fallback_grobid.require_output("grobid_tei")
    assert fallback_grobid.runtime_attestation is not None
    assert fallback_grobid.runtime_attestation.container_reference == grobid_image
    assert fallback_grobid.runtime_attestation.container_image_id == (
        grobid_reporter.expected_image_id
    )

    canonical_run = _run_for_component(
        result.processing_runs,
        "canonical-document-view",
    )
    canonical_product = canonical_run.require_output("canonical_document_view")
    canonical_view = store.read_canonical_document(canonical_product)
    assert canonical_run.stage_id == "canonicalize"
    unowned_runs = tuple(
        run for run in result.processing_runs if run.pipeline_run_id is None
    )
    assert len(unowned_runs) == 1
    intake_preflight = unowned_runs[0]
    assert intake_preflight.component_id == "document-preflight"
    assert intake_preflight.stage_id == "document-preflight"
    assert intake_preflight.stage_invocation_id is None

    pipeline_runs = tuple(
        run for run in result.processing_runs if run.pipeline_run_id is not None
    )
    assert pipeline_runs
    stage_invocations: dict[str, str] = {}
    for run in pipeline_runs:
        invocation_id = run.stage_invocation_id
        assert invocation_id is not None
        assert invocation_id.startswith("stage-invocation-")
        assert stage_invocations.setdefault(run.stage_id, invocation_id) == (
            invocation_id
        )
    assert len(set(stage_invocations.values())) == len(stage_invocations)
    pipeline_run_ids = {run.pipeline_run_id for run in pipeline_runs}
    assert len(pipeline_run_ids) == 1
    pipeline_run_id = next(iter(pipeline_run_ids))
    assert pipeline_run_id is not None
    assert pipeline_run_id.startswith("workflow-")
    assert canonical_view.artifact_id == artifact.artifact_id
    assert canonical_view.blocks
    assert fallback_grobid.require_output("grobid_tei") in (
        canonical_view.source_products
    )
    assert store.read_blob(canonical_product.blob_sha256) == canonical_document_bytes(
        canonical_view
    )

    manifest_path = _capture_live_pipeline_evidence(
        capture_directory=capture_directory,
        source_pdf=synthetic_rich_image_only_pdf,
        store=store,
        result=result,
    )
    capture_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert capture_manifest["tested_revision"] == tested_revision
    assert capture_manifest["source"]["blob_sha256"] == artifact.source_sha256
    assert {
        "canonical_document_view",
        "content_spans",
        "docling_document",
        "docling_response",
        "grobid_tei",
        "ocr_sidecar",
        "runtime_attestation",
        "searchable_pdf",
    } <= {product["name"] for product in capture_manifest["captured_products"]}
    assert {
        product["name"] for product in capture_manifest["selected_native_products"]
    } == {"docling_document", "grobid_tei"}
    assert all(
        "stage-invocation-" not in product["capture_file"]
        for product in capture_manifest["captured_products"]
    )
