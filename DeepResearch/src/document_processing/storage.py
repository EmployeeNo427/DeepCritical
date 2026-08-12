"""Filesystem-backed, content-addressed persistence for processing records."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, TypeVar

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from pydantic import BaseModel, ValidationError

from .alignment import (
    ScholarlyAlignmentOverlay,
    verify_scholarly_alignment_overlay,
)
from .canonical import (
    CANONICAL_COMPONENT_CAPABILITY,
    CANONICAL_COMPONENT_ID,
    CANONICAL_COMPONENT_VERSION,
    CanonicalDiagnosticSeverity,
    CanonicalDocumentView,
    CanonicalizationConfig,
    build_canonical_document_view,
    canonical_document_bytes,
    canonical_invocation_configuration,
    load_canonical_document,
)
from .models import (
    PROCESSING_RUN_SCHEMA_V1,
    PROCESSING_RUN_SCHEMA_V2,
    ArtifactLocation,
    ArtifactLocationRole,
    ArtifactRelationship,
    ContentSpanSet,
    DataProductRef,
    DocumentArtifact,
    ExecutionCheckpoint,
    IntakeQuarantineRecord,
    ProcessingDiagnostic,
    ProcessingRun,
    ProcessingRunDiagnosticManifest,
    ProcessingRunStatus,
    RuntimeAttestationSource,
)
from .native_contracts import (
    DOCLING_COMPONENT_VERSION,
    DOCLING_CONTAINER_IMAGE,
    DOCLING_SERVE_VERSION,
    GROBID_COMPONENT_VERSION,
    GROBID_CONTAINER_IMAGE,
    OCR_COMPONENT_VERSION,
    OCR_CONTAINER_IMAGE,
    OCR_DIGEST_RUNNER_VERSION,
    OUTPUT_POLICY_SCHEMA_VERSION,
    REMOTE_ATTESTATION_CONTRACT_VERSION,
    RUNTIME_ATTESTATION_SCHEMA_VERSION,
)
from .products import build_data_product_ref, validate_product_contract
from .routing import InputFormat
from .span_replay import ContentSpanReplayError, replay_content_span_set
from .validation import (
    DoclingQualityValidator,
    QualitySeverity,
    validate_content_integrity,
)

_SHA256_PATTERN = frozenset("0123456789abcdef")
_RECORD_MODEL = TypeVar("_RECORD_MODEL", bound=BaseModel)

_REMOTE_ATTESTATION_SOURCE = RuntimeAttestationSource.AUTHENTICATED_DEPLOYMENT_REPORTER
_REMOTE_TRUST_POLICY_KEYS = frozenset(
    {
        "reporter_configured",
        "expected_reporter_id",
        "expected_source",
        "attestation_contract_version",
        "attestation_schema_version",
    }
)
_OCR_TRUST_POLICY_KEYS = _REMOTE_TRUST_POLICY_KEYS | frozenset(
    {"local_digest_runner_version"}
)

_DOCLING_CONFIGURATION_KEYS = frozenset(
    {
        "serve_version",
        "expected_docling_version",
        "container_image",
        "container_digest",
        "model_versions",
        "model_hashes",
        "input_sha256",
        "input_format",
        "options",
        "minimum_pdf_locator_coverage",
        "quality_validator_version",
        "content_span_schema_version",
    }
)
_DOCLING_LOCATOR_CONFIGURATION_KEYS = {
    InputFormat.PDF: frozenset({"pdf_span_algorithm"}),
    InputFormat.JATS: frozenset(
        {"jats_locator_alignment_algorithm", "native_locator_overlay_sha256"}
    ),
    InputFormat.BIOC_JSON: frozenset(
        {"bioc_locator_alignment_algorithm", "native_locator_overlay_sha256"}
    ),
    InputFormat.BIOC_XML: frozenset(
        {"bioc_locator_alignment_algorithm", "native_locator_overlay_sha256"}
    ),
}
_GROBID_CONFIGURATION_KEYS = frozenset(
    {
        "input_sha256",
        "expected_grobid_version",
        "container_image",
        "container_digest",
        "model_versions",
        "model_hashes",
        "coordinates",
        "consolidate_header",
        "consolidate_citations",
        "segment_sentences",
        "minimum_text_characters",
    }
)
_OCR_CONFIGURATION_KEYS = frozenset(
    {
        "input_sha256",
        "fallback_reason",
        "expected_ocrmypdf_version",
        "mode",
        "container_image",
        "container_digest",
        "languages",
        "rotate_pages",
        "deskew",
        "jobs",
        "optimize",
        "skip_text",
        "output_type",
    }
)


class StorageError(RuntimeError):
    """Base exception for document-processing persistence failures."""


class BlobNotFoundError(StorageError):
    """Raised when a requested content hash is not stored."""


class BlobTooLargeError(StorageError):
    """Raised when a streaming blob crosses an explicit byte limit."""

    def __init__(self, *, max_bytes: int, observed_bytes: int) -> None:
        super().__init__(
            f"blob exceeded {max_bytes} bytes while streaming "
            f"(observed at least {observed_bytes})"
        )
        self.max_bytes = max_bytes
        self.observed_bytes = observed_bytes


class HashMismatchError(StorageError):
    """Raised when stored bytes do not match their claimed digest."""


class RecordNotFoundError(StorageError):
    """Raised when a requested immutable record does not exist."""


class RecordConflictError(StorageError):
    """Raised when an ID is reused for different immutable record content."""


class CorruptRecordError(StorageError):
    """Raised when a stored record cannot be decoded or validated."""


class UnsupportedSchemaVersionError(StorageError):
    """Raised when durable data uses an unknown or missing contract version."""


@dataclass(frozen=True, slots=True)
class StoredBlob:
    """Result of an idempotent blob write."""

    sha256: str
    byte_size: int
    uri: str
    path: Path

    def as_location(
        self,
        *,
        media_type: str,
        role: ArtifactLocationRole,
        created_by_run_id: str | None = None,
    ) -> ArtifactLocation:
        """Create a provenance location for this verified blob."""

        return ArtifactLocation(
            uri=self.uri,
            sha256=self.sha256,
            byte_size=self.byte_size,
            media_type=media_type,
            role=role,
            created_by_run_id=created_by_run_id,
        )


class ContentAddressedStore:
    """Immutable local storage for source bytes and processing state.

    Blobs are addressed by SHA-256 and records by a hash of their stable ID.
    Complete temporary files are linked into place, making publication atomic
    without allowing an existing value to be overwritten.  Repeating the same
    write is idempotent; repeating an ID with different content is an error.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._blob_root = self.root / "blobs" / "sha256"
        self._record_root = self.root / "records"
        self._temporary_root = self.root / ".tmp"
        legacy_runs = self._record_root / "parser_runs"
        if legacy_runs.is_dir() and any(legacy_runs.glob("*.json")):
            raise UnsupportedSchemaVersionError(
                "unversioned records/parser_runs data is unsupported; "
                "migrate or remove the prototype store before opening it"
            )
        for directory in (
            self._blob_root,
            self._record_root / "artifacts",
            self._record_root / "processing_runs",
            self._record_root / "diagnostics",
            self._record_root / "execution_checkpoints",
            self._record_root / "intake_quarantines",
            self._temporary_root,
        ):
            _ensure_directory_durable(directory)

    def put_blob(
        self,
        data: bytes | bytearray | memoryview | BinaryIO,
        *,
        expected_sha256: str | None = None,
        max_bytes: int | None = None,
    ) -> StoredBlob:
        """Persist bytes once and return their verified content identity."""

        if expected_sha256 is not None:
            _validate_sha256(expected_sha256)
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes cannot be negative")

        temporary_path = self._temporary_path("blob")
        digest = hashlib.sha256()
        byte_size = 0
        try:
            with temporary_path.open("xb") as destination:
                if isinstance(data, (bytes, bytearray, memoryview)):
                    chunks = (bytes(data),)
                elif hasattr(data, "read"):
                    chunks = iter(lambda: data.read(1024 * 1024), b"")
                else:
                    raise TypeError("data must be bytes-like or a binary stream")

                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise TypeError("binary streams must return bytes")
                    next_size = byte_size + len(chunk)
                    if max_bytes is not None and next_size > max_bytes:
                        raise BlobTooLargeError(
                            max_bytes=max_bytes,
                            observed_bytes=next_size,
                        )
                    destination.write(chunk)
                    digest.update(chunk)
                    byte_size = next_size
                destination.flush()
                os.fsync(destination.fileno())

            actual_sha256 = digest.hexdigest()
            if expected_sha256 is not None and actual_sha256 != expected_sha256:
                raise HashMismatchError(
                    f"expected blob {expected_sha256}, got {actual_sha256}"
                )

            destination_path = self.blob_path(actual_sha256)
            _ensure_directory_durable(destination_path.parent)
            try:
                self._publish_no_replace(temporary_path, destination_path)
            except FileExistsError:
                # Another writer (or an earlier identical write) published the
                # same content address first. Verification below distinguishes
                # valid idempotency from corruption.
                pass
            self.verify_blob(actual_sha256)
            return StoredBlob(
                sha256=actual_sha256,
                byte_size=byte_size,
                uri=self.blob_uri(actual_sha256),
                path=destination_path,
            )
        finally:
            temporary_path.unlink(missing_ok=True)

    def put_blob_file(
        self,
        source: str | Path,
        *,
        expected_sha256: str | None = None,
        max_bytes: int | None = None,
    ) -> StoredBlob:
        """Stream a file into the store without modifying the source."""

        source_path = Path(source)
        with source_path.open("rb") as source_stream:
            return self.put_blob(
                source_stream,
                expected_sha256=expected_sha256,
                max_bytes=max_bytes,
            )

    def blob_uri(self, sha256: str) -> str:
        """Return the canonical URI for a digest in this store."""

        _validate_sha256(sha256)
        return f"cas://sha256/{sha256}"

    def blob_path(self, sha256: str) -> Path:
        """Return the deterministic local path for a digest."""

        _validate_sha256(sha256)
        return self._blob_root / sha256[:2] / sha256[2:]

    def read_blob(self, sha256: str) -> bytes:
        """Read bytes after verifying their content hash."""

        path = self.verify_blob(sha256)
        return path.read_bytes()

    def put_canonical_document(self, view: CanonicalDocumentView) -> StoredBlob:
        """Stage schema-valid canonical bytes pending producer-run admission.

        CAS blob presence alone does not make a trusted data product.  The
        producer identity and exact native-source semantics are verified when
        the corresponding processing run is admitted.
        """

        # ``FrozenModel`` only provides shallow immutability and
        # ``model_copy(update=...)`` deliberately skips validation.  Rebuild the
        # complete view immediately before serialization so neither a mutated
        # nested mapping nor a stale deterministic identity can become durable.
        validated_view = CanonicalDocumentView.model_validate(
            view.model_dump(mode="python")
        )
        return self.put_blob(canonical_document_bytes(validated_view))

    def read_data_product_bytes(self, product: DataProductRef) -> bytes:
        """Read a durable producer-declared product after full verification.

        A syntactically valid reference is not sufficient provenance.  The
        reference must also describe bytes in this store, name existing source
        artifacts, and exactly match an output on its durable producer run.
        """

        validated_product = self._revalidate_product(product)
        self._verify_durable_product_chain((validated_product,))
        return self.read_blob(validated_product.blob_sha256)

    def read_canonical_document(self, product: DataProductRef) -> CanonicalDocumentView:
        """Load and deterministically reproduce a durable canonical product."""

        validated_product = self._revalidate_product(product)
        validate_product_contract(validated_product)
        if validated_product.name != "canonical_document_view":
            raise ValueError("data product is not a canonical document view")
        self._verify_durable_product_chain((validated_product,))
        producer = self.get_processing_run(validated_product.producer_run_id)
        return self._verify_canonical_product(validated_product, producer)

    def verify_blob(self, sha256: str) -> Path:
        """Verify that a blob exists and matches its content address."""

        path = self.blob_path(sha256)
        if not path.is_file():
            raise BlobNotFoundError(f"blob not found: {sha256}")

        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != sha256:
            raise HashMismatchError(f"stored blob {sha256} hashes to {actual}")
        return path

    def save_artifact(self, artifact: DocumentArtifact) -> Path:
        """Persist an artifact after validating blobs and parent lineage."""

        self._verify_location(artifact.raw_location)
        for location in artifact.derived_locations:
            self._verify_location(location)
        if artifact.parent_artifact_id is not None:
            self.get_artifact(artifact.parent_artifact_id)
        raw_creator_owner = artifact.parent_artifact_id
        if artifact.raw_location.created_by_run_id is not None:
            if raw_creator_owner is None:
                raise RecordConflictError(
                    "a source artifact raw location cannot claim a creator processing run"
                )
            self._validate_location_creator(
                artifact.raw_location,
                expected_artifact_id=raw_creator_owner,
            )
        for location in artifact.derived_locations:
            if location.created_by_run_id is not None:
                self._validate_location_creator(
                    location,
                    expected_artifact_id=artifact.artifact_id,
                )
        return self._save_record("artifacts", artifact.artifact_id, artifact)

    def _validate_location_creator(
        self,
        location: ArtifactLocation,
        *,
        expected_artifact_id: str,
    ) -> None:
        creator_run_id = location.created_by_run_id
        if creator_run_id is None:
            return
        creator = self.get_processing_run(creator_run_id)
        if creator.artifact_id != expected_artifact_id:
            raise RecordConflictError(
                f"creator run {creator_run_id} belongs to a different artifact"
            )
        if not any(
            product.blob_sha256 == location.sha256 for product in creator.outputs
        ):
            raise RecordConflictError(
                f"creator run {creator_run_id} does not declare the location hash"
            )

    def get_artifact(self, artifact_id: str) -> DocumentArtifact:
        """Load and validate an immutable artifact record."""

        return self._get_record(
            "artifacts", artifact_id, DocumentArtifact, id_field="artifact_id"
        )

    def list_artifacts(self) -> tuple[DocumentArtifact, ...]:
        """Return all artifacts in stable ID order."""

        return tuple(
            sorted(
                self._read_record_directory("artifacts", DocumentArtifact),
                key=lambda record: record.artifact_id,
            )
        )

    def save_intake_quarantine(self, record: IntakeQuarantineRecord) -> Path:
        """Persist a rejected path intake without storing unsafe source bytes."""

        self.verify_blob(record.preflight_result_sha256)
        return self._save_record("intake_quarantines", record.intake_id, record)

    def get_intake_quarantine(self, intake_id: str) -> IntakeQuarantineRecord:
        return self._get_record(
            "intake_quarantines",
            intake_id,
            IntakeQuarantineRecord,
            id_field="intake_id",
        )

    def list_intake_quarantines(self) -> tuple[IntakeQuarantineRecord, ...]:
        return tuple(
            sorted(
                self._read_record_directory(
                    "intake_quarantines",
                    IntakeQuarantineRecord,
                ),
                key=lambda record: (record.created_at, record.intake_id),
            )
        )

    def save_processing_run(self, processing_run: ProcessingRun) -> Path:
        """Persist a processing run after validating every typed product."""

        processing_run = ProcessingRun.model_validate(
            processing_run.model_dump(mode="python")
        )
        self.get_artifact(processing_run.artifact_id)
        self._verify_durable_product_chain(processing_run.inputs)
        for product in processing_run.outputs:
            self._verify_product(product)
            self._verify_output_lineage(product, processing_run)
            if product.name == "canonical_document_view":
                self._verify_canonical_product(product, processing_run)
        return self._save_record(
            "processing_runs", processing_run.run_id, processing_run
        )

    def data_product_ref(
        self,
        *,
        name: str,
        blob_sha256: str,
        producer_run_id: str,
        source_artifact_ids: tuple[str, ...],
    ) -> DataProductRef:
        """Build a registered product reference for an already stored blob."""

        path = self.verify_blob(blob_sha256)
        return build_data_product_ref(
            name=name,
            blob_sha256=blob_sha256,
            uri=self.blob_uri(blob_sha256),
            byte_size=path.stat().st_size,
            producer_run_id=producer_run_id,
            source_artifact_ids=source_artifact_ids,
        )

    def data_product_refs(
        self,
        outputs: Mapping[str, str],
        *,
        producer_run_id: str,
        source_artifact_ids: tuple[str, ...],
    ) -> tuple[DataProductRef, ...]:
        """Build ordered registered references for a mapping of named digests."""

        return tuple(
            self.data_product_ref(
                name=name,
                blob_sha256=digest,
                producer_run_id=producer_run_id,
                source_artifact_ids=source_artifact_ids,
            )
            for name, digest in outputs.items()
        )

    def get_processing_run(self, run_id: str) -> ProcessingRun:
        """Load and validate one processing-run record."""

        return self._get_record(
            "processing_runs", run_id, ProcessingRun, id_field="run_id"
        )

    def list_processing_runs(
        self,
        *,
        artifact_id: str | None = None,
        component_id: str | None = None,
        status: ProcessingRunStatus | None = None,
        configuration_sha256: str | None = None,
        output_policy_sha256: str | None = None,
    ) -> tuple[ProcessingRun, ...]:
        """Query persisted run state for deterministic resume decisions."""

        if configuration_sha256 is not None:
            _validate_sha256(configuration_sha256)
        if output_policy_sha256 is not None:
            _validate_sha256(output_policy_sha256)
        records = self._read_record_directory("processing_runs", ProcessingRun)
        selected = (
            record
            for record in records
            if (artifact_id is None or record.artifact_id == artifact_id)
            and (component_id is None or record.component_id == component_id)
            and (status is None or record.status is status)
            and (
                configuration_sha256 is None
                or record.configuration_sha256 == configuration_sha256
            )
            and (
                output_policy_sha256 is None
                or record.output_policy_sha256 == output_policy_sha256
            )
        )
        return tuple(
            sorted(selected, key=lambda record: (record.started_at, record.run_id))
        )

    def latest_processing_run(
        self,
        artifact_id: str,
        component_id: str,
        *,
        configuration_sha256: str | None = None,
        output_policy_sha256: str | None = None,
    ) -> ProcessingRun | None:
        """Return the most recent matching persisted attempt, if any."""

        runs = self.list_processing_runs(
            artifact_id=artifact_id,
            component_id=component_id,
            configuration_sha256=configuration_sha256,
            output_policy_sha256=output_policy_sha256,
        )
        return runs[-1] if runs else None

    def has_complete_run(
        self,
        artifact_id: str,
        component_id: str,
        *,
        configuration_sha256: str | None = None,
        output_policy_sha256: str | None = None,
    ) -> bool:
        """Whether a complete run also has its required diagnostics indexed."""

        return any(
            self.processing_run_diagnostics_reconciled(run)
            for run in self.list_processing_runs(
                artifact_id=artifact_id,
                component_id=component_id,
                status=ProcessingRunStatus.COMPLETE,
                configuration_sha256=configuration_sha256,
                output_policy_sha256=output_policy_sha256,
            )
        )

    def processing_run_diagnostics_reconciled(
        self, processing_run: ProcessingRun
    ) -> bool:
        """Whether every diagnostic in a run's durable manifest is indexed."""

        manifest_product = processing_run.output("diagnostics_manifest")
        if manifest_product is None:
            return True
        manifest = ProcessingRunDiagnosticManifest.model_validate_json(
            self.read_blob(manifest_product.blob_sha256)
        )
        if (
            manifest.artifact_id != processing_run.artifact_id
            or manifest.processing_run_id != processing_run.run_id
        ):
            raise RecordConflictError(
                "diagnostics manifest does not match processing run"
            )
        for expected in manifest.diagnostics:
            try:
                actual = self.get_diagnostic(expected.diagnostic_id)
            except RecordNotFoundError:
                return False
            if actual != expected:
                raise RecordConflictError(
                    "indexed diagnostic does not match diagnostics manifest"
                )
        return True

    def list_resume_candidates(
        self,
        component_id: str,
        *,
        configuration_sha256: str | None = None,
        output_policy_sha256: str | None = None,
    ) -> tuple[DocumentArtifact, ...]:
        """Return artifacts with no run or a latest partial/failed attempt.

        Explicitly quarantined artifacts are not automatically retried.  A
        caller must create a deliberate remediation run after resolving the
        quarantine condition.
        """

        if output_policy_sha256 is not None:
            _validate_sha256(output_policy_sha256)
        candidates: list[DocumentArtifact] = []
        for artifact in self.list_artifacts():
            latest = self.latest_processing_run(
                artifact.artifact_id,
                component_id,
                configuration_sha256=configuration_sha256,
                output_policy_sha256=output_policy_sha256,
            )
            if (
                latest is None
                or latest.status
                in {
                    ProcessingRunStatus.PARTIAL,
                    ProcessingRunStatus.FAILED,
                }
                or (
                    latest.status is ProcessingRunStatus.COMPLETE
                    and not self.processing_run_diagnostics_reconciled(latest)
                )
            ):
                candidates.append(artifact)
        return tuple(candidates)

    def save_diagnostic(self, diagnostic: ProcessingDiagnostic) -> Path:
        """Persist a diagnostic with validated artifact/run references."""

        self.get_artifact(diagnostic.artifact_id)
        if diagnostic.processing_run_id is not None:
            processing_run = self.get_processing_run(diagnostic.processing_run_id)
            if processing_run.artifact_id != diagnostic.artifact_id:
                raise RecordConflictError(
                    "diagnostic artifact does not match its processing run"
                )
        return self._save_record("diagnostics", diagnostic.diagnostic_id, diagnostic)

    def save_execution_checkpoint(self, checkpoint: ExecutionCheckpoint) -> Path:
        """Persist a remote task ID before polling so restarts can resume it."""

        self.get_artifact(checkpoint.artifact_id)
        return self._save_record(
            "execution_checkpoints", checkpoint.checkpoint_id, checkpoint
        )

    def get_execution_checkpoint(self, checkpoint_id: str) -> ExecutionCheckpoint:
        return self._get_record(
            "execution_checkpoints",
            checkpoint_id,
            ExecutionCheckpoint,
            id_field="checkpoint_id",
        )

    def delete_execution_checkpoint(self, checkpoint_id: str) -> None:
        """Clear staging state only after a terminal processing run is durable."""

        path = self._record_path("execution_checkpoints", checkpoint_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return
        _fsync_directory(path.parent)

    def get_diagnostic(self, diagnostic_id: str) -> ProcessingDiagnostic:
        """Load and validate one diagnostic record."""

        return self._get_record(
            "diagnostics",
            diagnostic_id,
            ProcessingDiagnostic,
            id_field="diagnostic_id",
        )

    def list_diagnostics(
        self,
        *,
        artifact_id: str | None = None,
        processing_run_id: str | None = None,
    ) -> tuple[ProcessingDiagnostic, ...]:
        """Return diagnostics filtered by evidence lineage."""

        records = self._read_record_directory("diagnostics", ProcessingDiagnostic)
        selected = (
            record
            for record in records
            if (artifact_id is None or record.artifact_id == artifact_id)
            and (
                processing_run_id is None
                or record.processing_run_id == processing_run_id
            )
        )
        return tuple(
            sorted(
                selected, key=lambda record: (record.created_at, record.diagnostic_id)
            )
        )

    def _verify_location(self, location: ArtifactLocation) -> None:
        expected_uri = self.blob_uri(location.sha256)
        if location.uri != expected_uri:
            raise HashMismatchError(
                f"location URI {location.uri!r} does not match {location.sha256}"
            )
        path = self.verify_blob(location.sha256)
        if path.stat().st_size != location.byte_size:
            raise HashMismatchError(
                f"location size for {location.sha256} does not match stored blob"
            )

    def _verify_docling_native_sources(
        self,
        *,
        canonical_artifact: DocumentArtifact,
        docling_product: DataProductRef,
        spans_product: DataProductRef,
        docling_payload: dict[str, Any],
        span_set: ContentSpanSet,
    ) -> ProcessingRun:
        """Verify the native Docling pair and the invocation that owns it."""

        if docling_product.producer_run_id != spans_product.producer_run_id:
            raise RecordConflictError(
                "canonical Docling document and content spans must share one producer"
            )
        producer = self.get_processing_run(docling_product.producer_run_id)
        self._require_component_contract(
            producer,
            component_id="docling",
            component_version=DOCLING_COMPONENT_VERSION,
            capability="document.parse",
            statuses=frozenset(
                {ProcessingRunStatus.COMPLETE, ProcessingRunStatus.PARTIAL}
            ),
            purpose="Docling native source",
        )
        if producer.artifact_id != canonical_artifact.artifact_id:
            raise RecordConflictError(
                "Docling native-source producer does not own the canonical artifact"
            )
        if (
            docling_product not in producer.outputs
            or spans_product not in producer.outputs
        ):
            raise RecordConflictError(
                "Docling native-source producer does not declare the exact document/span pair"
            )

        configuration = producer.configuration
        try:
            input_format = InputFormat(configuration.get("input_format"))
        except (TypeError, ValueError) as exc:
            raise RecordConflictError(
                "Docling producer input format is not supported"
            ) from exc
        if input_format is InputFormat.UNKNOWN:
            raise RecordConflictError("Docling producer input format cannot be unknown")
        expected_keys = _DOCLING_CONFIGURATION_KEYS | (
            _DOCLING_LOCATOR_CONFIGURATION_KEYS.get(input_format, frozenset())
        )
        if configuration.keys() != expected_keys:
            raise RecordConflictError(
                "Docling producer configuration does not match the production schema"
            )
        if (
            configuration["serve_version"] != DOCLING_SERVE_VERSION
            or configuration["expected_docling_version"] != DOCLING_COMPONENT_VERSION
            or configuration["quality_validator_version"] != "docling-quality-v2"
            or configuration["content_span_schema_version"] != "1"
        ):
            raise RecordConflictError(
                "Docling producer version configuration is not approved"
            )
        if (
            input_format is InputFormat.PDF
            and configuration["pdf_span_algorithm"] != "provenance-charspan-v2"
        ):
            raise RecordConflictError("Docling PDF span algorithm is not approved")
        self._verify_container_configuration(
            producer,
            purpose="Docling",
            component_id="docling",
            expected_image=DOCLING_CONTAINER_IMAGE,
            expected_component_versions={
                "docling": DOCLING_COMPONENT_VERSION,
                "docling_serve": DOCLING_SERVE_VERSION,
            },
        )

        options = configuration["options"]
        if not isinstance(options, dict):
            raise RecordConflictError("Docling producer options must be an object")
        to_formats = options.get("to_formats")
        if (
            not isinstance(to_formats, list)
            or any(not isinstance(item, str) for item in to_formats)
            or "json" not in to_formats
        ):
            raise RecordConflictError(
                "Docling producer options must request serialized JSON"
            )
        minimum_coverage = configuration["minimum_pdf_locator_coverage"]
        if (
            isinstance(minimum_coverage, bool)
            or not isinstance(minimum_coverage, (int, float))
            or not 0.95 <= minimum_coverage <= 1
        ):
            raise RecordConflictError(
                "Docling producer PDF locator coverage threshold is invalid"
            )

        expected_direct_inputs = self._native_artifact_inputs(canonical_artifact)
        if input_format in {InputFormat.BIOC_JSON, InputFormat.BIOC_XML}:
            html_product, locator_product = self._verify_adapter_inputs(
                artifact=canonical_artifact,
                input_format=input_format,
                products=producer.inputs,
                expected_direct_inputs=expected_direct_inputs,
            )
            expected_input_sha256 = html_product.blob_sha256
        elif input_format is InputFormat.JATS:
            if len(producer.inputs) != len(expected_direct_inputs) + 1:
                raise RecordConflictError(
                    "JATS Docling inputs do not match source and locator provenance"
                )
            if producer.inputs[: len(expected_direct_inputs)] != expected_direct_inputs:
                raise RecordConflictError(
                    "JATS Docling source inputs do not match artifact provenance"
                )
            locator_product = producer.inputs[-1]
            self._verify_adapter_product(
                artifact=canonical_artifact,
                input_format=input_format,
                product=locator_product,
                expected_direct_inputs=expected_direct_inputs,
            )
            expected_input_sha256 = canonical_artifact.source_sha256
        else:
            if producer.inputs != expected_direct_inputs:
                raise RecordConflictError(
                    "Docling producer inputs do not match artifact provenance"
                )
            locator_product = None
            expected_input_sha256 = canonical_artifact.source_sha256

        if configuration["input_sha256"] != expected_input_sha256:
            raise RecordConflictError(
                "Docling producer input hash does not match its exact input bytes"
            )
        if input_format in {
            InputFormat.JATS,
            InputFormat.BIOC_JSON,
            InputFormat.BIOC_XML,
        }:
            if locator_product is None:  # pragma: no cover - branch construction guard
                raise RecordConflictError(
                    "structured Docling source is missing its native locator input"
                )
            alignment_key = (
                "jats_locator_alignment_algorithm"
                if input_format is InputFormat.JATS
                else "bioc_locator_alignment_algorithm"
            )
            expected_locator_sha256 = self._docling_locator_configuration_sha256(
                locator_product
            )
            if (
                configuration[alignment_key] != "normalized-exact-v1"
                or configuration["native_locator_overlay_sha256"]
                != expected_locator_sha256
            ):
                raise RecordConflictError(
                    "Docling native-locator configuration does not match its adapter input"
                )

        if (
            span_set.artifact_id != producer.artifact_id
            or span_set.processing_run_id != producer.run_id
            or span_set.representation_product_id != docling_product.product_id
        ):
            raise RecordConflictError(
                "content spans are not bound to their Docling producer and representation"
            )

        quality = DoclingQualityValidator(float(minimum_coverage)).validate(
            docling_payload,
            require_pdf_geometry=input_format is InputFormat.PDF,
        )
        fatal_empty = any(
            issue.code in {"NO_TEXT_ITEMS", "NO_NONEMPTY_TEXT_ITEMS"}
            for issue in quality.issues
        )
        if fatal_empty:
            raise RecordConflictError(
                "Docling native source is not semantically usable"
            )
        has_errors = any(
            issue.severity is QualitySeverity.ERROR for issue in quality.issues
        )
        if has_errors:
            raise RecordConflictError(
                "Docling producer contains semantic validation errors"
            )
        return producer

    def _docling_locator_configuration_sha256(self, product: DataProductRef) -> str:
        payload_bytes = self.read_blob(product.blob_sha256)
        try:
            payload = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecordConflictError(
                "native-locator adapter product is not valid JSON"
            ) from exc
        if not isinstance(payload, list) or any(
            not isinstance(item, dict) for item in payload
        ):
            raise RecordConflictError(
                "native-locator adapter product must be a list of objects"
            )
        if payload_bytes != _canonical_json_bytes(payload):
            raise RecordConflictError(
                "native-locator adapter product is not deterministically serialized"
            )
        return product.blob_sha256

    def _verify_adapter_inputs(
        self,
        *,
        artifact: DocumentArtifact,
        input_format: InputFormat,
        products: tuple[DataProductRef, ...],
        expected_direct_inputs: tuple[DataProductRef, ...],
    ) -> tuple[DataProductRef, DataProductRef]:
        if len(products) != 2:
            raise RecordConflictError(
                "BioC Docling inputs must be the adapter HTML and locator products"
            )
        html_product, locator_product = products
        if html_product.name != "html_projection":
            raise RecordConflictError("BioC Docling input is not an HTML projection")
        adapter = self._verify_adapter_product(
            artifact=artifact,
            input_format=input_format,
            product=locator_product,
            expected_direct_inputs=expected_direct_inputs,
        )
        if (
            html_product.producer_run_id != adapter.run_id
            or html_product not in adapter.outputs
        ):
            raise RecordConflictError(
                "BioC HTML and locator inputs must share the approved adapter producer"
            )
        return html_product, locator_product

    def _verify_adapter_product(
        self,
        *,
        artifact: DocumentArtifact,
        input_format: InputFormat,
        product: DataProductRef,
        expected_direct_inputs: tuple[DataProductRef, ...],
    ) -> ProcessingRun:
        if product.name != "native_locator_overlay":
            raise RecordConflictError(
                "Docling native-locator input has the wrong product contract"
            )
        adapter = self.get_processing_run(product.producer_run_id)
        component_id = (
            "jats-locator-adapter"
            if input_format is InputFormat.JATS
            else "bioc-adapter"
        )
        self._require_component_contract(
            adapter,
            component_id=component_id,
            component_version="1",
            capability="document.adapt",
            statuses=frozenset(
                {ProcessingRunStatus.COMPLETE, ProcessingRunStatus.PARTIAL}
            ),
            purpose=f"{input_format.value} locator adapter",
        )
        if adapter.artifact_id != artifact.artifact_id:
            raise RecordConflictError(
                "native-locator adapter does not own the Docling artifact"
            )
        expected_configuration = {
            "adapter_version": "1",
            "input_format": input_format.value,
            "input_sha256": artifact.source_sha256,
        }
        if adapter.configuration != expected_configuration:
            raise RecordConflictError(
                "native-locator adapter configuration does not match its artifact"
            )
        if adapter.inputs != expected_direct_inputs or product not in adapter.outputs:
            raise RecordConflictError(
                "native-locator adapter lineage does not match its artifact"
            )
        return adapter

    def _verify_grobid_native_source(
        self,
        *,
        canonical_artifact: DocumentArtifact,
        product: DataProductRef,
        tei_xml: bytes,
    ) -> ProcessingRun:
        producer = self.get_processing_run(product.producer_run_id)
        self._require_component_contract(
            producer,
            component_id="grobid",
            component_version=GROBID_COMPONENT_VERSION,
            capability="document.parse.scholarly",
            statuses=frozenset(
                {ProcessingRunStatus.COMPLETE, ProcessingRunStatus.PARTIAL}
            ),
            purpose="GROBID native source",
        )
        if product not in producer.outputs:
            raise RecordConflictError(
                "GROBID native-source producer does not declare the exact TEI product"
            )
        grobid_artifact = self.get_artifact(producer.artifact_id)
        expected_grobid_inputs = self._verify_grobid_artifact_lineage(
            canonical_artifact=canonical_artifact,
            grobid_artifact=grobid_artifact,
        )

        configuration = producer.configuration
        if configuration.keys() != _GROBID_CONFIGURATION_KEYS:
            raise RecordConflictError(
                "GROBID producer configuration does not match the production schema"
            )
        if configuration["expected_grobid_version"] != GROBID_COMPONENT_VERSION:
            raise RecordConflictError(
                "GROBID producer version configuration is not approved"
            )
        self._verify_container_configuration(
            producer,
            purpose="GROBID",
            component_id="grobid",
            expected_image=GROBID_CONTAINER_IMAGE,
            expected_component_versions={"grobid": GROBID_COMPONENT_VERSION},
        )
        if producer.inputs != expected_grobid_inputs:
            raise RecordConflictError(
                "GROBID producer inputs do not match artifact provenance"
            )
        if configuration["input_sha256"] != grobid_artifact.source_sha256:
            raise RecordConflictError(
                "GROBID producer input hash does not match its artifact bytes"
            )
        for option in ("consolidate_header", "consolidate_citations"):
            if type(configuration[option]) is not int or configuration[option] not in {
                0,
                1,
            }:
                raise RecordConflictError(f"GROBID {option} must be 0 or 1")
        if not isinstance(configuration["segment_sentences"], bool):
            raise RecordConflictError("GROBID segment_sentences must be a boolean")
        coordinates = configuration["coordinates"]
        if not isinstance(coordinates, list) or any(
            not isinstance(value, str) or not value.strip() for value in coordinates
        ):
            raise RecordConflictError("GROBID coordinates must be a string list")
        minimum_characters = configuration["minimum_text_characters"]
        if (
            isinstance(minimum_characters, bool)
            or not isinstance(minimum_characters, int)
            or minimum_characters < 0
        ):
            raise RecordConflictError(
                "GROBID minimum text characters must be a non-negative integer"
            )
        try:
            root = ElementTree.fromstring(tei_xml)
        except (DefusedXmlException, ElementTree.ParseError) as exc:
            raise RecordConflictError("GROBID source is not usable TEI XML") from exc
        if root.tag != "{http://www.tei-c.org/ns/1.0}TEI":
            raise RecordConflictError(
                "GROBID source root is not TEI in the canonical namespace"
            )
        text_length = len(" ".join("".join(root.itertext()).split()))
        if text_length == 0 or text_length < minimum_characters:
            raise RecordConflictError(
                "GROBID source does not satisfy its recorded usability threshold"
            )
        return producer

    def _verify_grobid_artifact_lineage(
        self,
        *,
        canonical_artifact: DocumentArtifact,
        grobid_artifact: DocumentArtifact,
    ) -> tuple[DataProductRef, ...]:
        """Bind GROBID to the canonical source or one exact OCR derivative."""

        if grobid_artifact.artifact_id == canonical_artifact.artifact_id:
            return self._native_artifact_inputs(grobid_artifact)
        if (
            grobid_artifact.relationship is not ArtifactRelationship.DERIVATIVE
            or grobid_artifact.parent_artifact_id != canonical_artifact.artifact_id
            or grobid_artifact.media_type != "application/pdf"
        ):
            raise RecordConflictError(
                "GROBID native source is not a direct PDF derivative of the "
                "canonical artifact"
            )

        creator_run_id = grobid_artifact.raw_location.created_by_run_id
        if creator_run_id is None:  # guarded by durable artifact admission
            raise RecordConflictError("GROBID OCR derivative has no creator run")
        creator = self.get_processing_run(creator_run_id)
        self._require_component_contract(
            creator,
            component_id="ocrmypdf",
            component_version=OCR_COMPONENT_VERSION,
            capability="document.ocr",
            statuses=frozenset(
                {ProcessingRunStatus.COMPLETE, ProcessingRunStatus.PARTIAL}
            ),
            purpose="GROBID OCR derivative",
        )
        if creator.artifact_id != canonical_artifact.artifact_id:
            raise RecordConflictError(
                "GROBID OCR derivative creator does not own the canonical artifact"
            )
        creator_products = tuple(
            output
            for output in creator.outputs
            if output.name == "searchable_pdf"
            and output.blob_sha256 == grobid_artifact.source_sha256
        )
        if len(creator_products) != 1:
            raise RecordConflictError(
                "GROBID OCR derivative does not resolve to one searchable-PDF product"
            )
        creator_product = creator_products[0]
        if creator.inputs != self._native_artifact_inputs(canonical_artifact):
            raise RecordConflictError(
                "GROBID OCR derivative creator inputs do not match source provenance"
            )
        self._verify_ocr_configuration(creator, canonical_artifact)
        return (creator_product,)

    def _verify_ocr_configuration(
        self,
        producer: ProcessingRun,
        source_artifact: DocumentArtifact,
    ) -> None:
        configuration = producer.configuration
        if configuration.keys() != _OCR_CONFIGURATION_KEYS:
            raise RecordConflictError(
                "GROBID OCR derivative configuration does not match the production schema"
            )
        if (
            configuration["input_sha256"] != source_artifact.source_sha256
            or configuration["expected_ocrmypdf_version"] != OCR_COMPONENT_VERSION
        ):
            raise RecordConflictError(
                "GROBID OCR derivative configuration does not match its exact source"
            )
        fallback_reason = configuration["fallback_reason"]
        if not isinstance(fallback_reason, str) or not fallback_reason.strip():
            raise RecordConflictError("GROBID OCR fallback reason is invalid")

        mode = configuration["mode"]
        image = configuration["container_image"]
        digest = configuration["container_digest"]
        if mode == "container_cli":
            if (
                not isinstance(image, str)
                or not image.strip()
                or image != OCR_CONTAINER_IMAGE
                or (digest is not None and not _is_oci_sha256(digest))
            ):
                raise RecordConflictError(
                    "GROBID OCR derivative container identity is not approved"
                )
        elif mode == "local_cli":
            if image is not None or digest is not None:
                raise RecordConflictError(
                    "local GROBID OCR derivative cannot claim a container identity"
                )
        else:
            raise RecordConflictError("GROBID OCR derivative mode is not approved")

        languages = configuration["languages"]
        if (
            not isinstance(languages, list)
            or not languages
            or any(
                not isinstance(value, str) or not value.strip() for value in languages
            )
        ):
            raise RecordConflictError("GROBID OCR derivative languages are invalid")
        for option in ("rotate_pages", "deskew", "skip_text"):
            if not isinstance(configuration[option], bool):
                raise RecordConflictError(
                    f"GROBID OCR derivative {option} must be a boolean"
                )
        if configuration["skip_text"] is not True:
            raise RecordConflictError(
                "GROBID OCR derivative must preserve existing text with skip_text"
            )
        jobs = configuration["jobs"]
        optimize = configuration["optimize"]
        if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
            raise RecordConflictError("GROBID OCR derivative jobs are invalid")
        if (
            isinstance(optimize, bool)
            or not isinstance(optimize, int)
            or not 0 <= optimize <= 3
        ):
            raise RecordConflictError("GROBID OCR derivative optimize is invalid")
        if configuration["output_type"] != "pdf":
            raise RecordConflictError("GROBID OCR derivative output type is invalid")

        attestation = producer.runtime_attestation
        if attestation is None:
            trust_policy, document_config = self._runtime_trust_policy(
                producer,
                component_id="ocrmypdf",
                purpose="GROBID OCR derivative",
            )
            self._verify_persisted_parser_policy(
                producer,
                component_id="ocrmypdf",
                purpose="GROBID OCR derivative",
                trust_policy=trust_policy,
                document_config=document_config,
                expected_image=OCR_CONTAINER_IMAGE,
            )
            return
        trust_policy, document_config = self._runtime_trust_policy(
            producer,
            component_id="ocrmypdf",
            purpose="GROBID OCR derivative",
        )
        self._verify_persisted_parser_policy(
            producer,
            component_id="ocrmypdf",
            purpose="GROBID OCR derivative",
            trust_policy=trust_policy,
            document_config=document_config,
            expected_image=OCR_CONTAINER_IMAGE,
        )
        component_versions = attestation.component_versions
        if (
            attestation.source
            is not RuntimeAttestationSource.DIGEST_ADDRESSED_OCI_INVOCATION
            or component_versions.keys() != {"ocrmypdf", "tesseract"}
            or component_versions["ocrmypdf"] != OCR_COMPONENT_VERSION
            or not isinstance(component_versions["tesseract"], str)
            or not component_versions["tesseract"].strip()
            or digest != attestation.container_digest
            or not isinstance(image, str)
            or image != OCR_CONTAINER_IMAGE
            or attestation.container_reference
            != f"{OCR_CONTAINER_IMAGE}@{attestation.container_digest}"
        ):
            raise RecordConflictError(
                "GROBID OCR derivative runtime attestation conflicts with configuration"
            )
        self._verify_runtime_trust_evidence(
            producer,
            component_id="ocrmypdf",
            purpose="GROBID OCR derivative",
            trust_policy=trust_policy,
            document_config=document_config,
            expected_image=OCR_CONTAINER_IMAGE,
            expected_component_versions={"ocrmypdf": OCR_COMPONENT_VERSION},
        )
        attestation_bytes = _canonical_json_bytes(attestation.model_dump(mode="json"))
        if (
            self.read_blob(producer.runtime_attestation_sha256 or "")
            != attestation_bytes
        ):  # pragma: no cover - canonical SHA-256 output makes inequality unreachable
            raise RecordConflictError(
                "GROBID OCR derivative runtime attestation bytes are not canonical"
            )

    def _verify_container_configuration(
        self,
        run: ProcessingRun,
        *,
        purpose: str,
        component_id: str,
        expected_image: str,
        expected_component_versions: dict[str, str],
    ) -> None:
        configuration = run.configuration
        image = configuration.get("container_image")
        digest = configuration.get("container_digest")
        model_versions = configuration.get("model_versions")
        model_hashes = configuration.get("model_hashes")
        if not isinstance(image, str) or not image.strip() or image != expected_image:
            raise RecordConflictError(f"{purpose} container image is not approved")
        if digest is not None and not _is_oci_sha256(digest):
            raise RecordConflictError(f"{purpose} container digest is invalid")
        if (
            not isinstance(model_versions, dict)
            or not isinstance(model_hashes, dict)
            or not _valid_version_mapping(model_versions)
            or not _valid_hash_mapping(model_hashes)
        ):
            raise RecordConflictError(f"{purpose} model inventory is invalid")
        if model_versions.keys() != model_hashes.keys():
            raise RecordConflictError(
                f"{purpose} model version/hash inventories do not match"
            )

        attestation = run.runtime_attestation
        trust_policy, document_config = self._runtime_trust_policy(
            run,
            component_id=component_id,
            purpose=purpose,
        )
        self._verify_persisted_parser_policy(
            run,
            component_id=component_id,
            purpose=purpose,
            trust_policy=trust_policy,
            document_config=document_config,
            expected_image=expected_image,
        )
        if attestation is None:
            if (
                run.status is ProcessingRunStatus.COMPLETE
                and run.runtime_identity_required
            ):
                raise RecordConflictError(
                    f"complete {purpose} producer lacks required runtime attestation"
                )
            return
        if (
            digest != attestation.container_digest
            or model_versions != attestation.model_versions
            or model_hashes != attestation.model_hashes
            or attestation.container_reference
            != f"{expected_image}@{attestation.container_digest}"
        ):
            raise RecordConflictError(
                f"{purpose} runtime attestation conflicts with its production configuration"
            )
        self._verify_runtime_trust_evidence(
            run,
            component_id=component_id,
            purpose=purpose,
            trust_policy=trust_policy,
            document_config=document_config,
            expected_image=expected_image,
            expected_component_versions=expected_component_versions,
        )
        attestation_bytes = _canonical_json_bytes(attestation.model_dump(mode="json"))
        if (  # pragma: no cover - canonical SHA-256 output makes inequality unreachable
            self.read_blob(run.runtime_attestation_sha256 or "") != attestation_bytes
        ):
            raise RecordConflictError(
                f"{purpose} runtime attestation bytes are not canonical"
            )

    def _runtime_trust_policy(
        self,
        run: ProcessingRun,
        *,
        component_id: str,
        purpose: str,
    ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
        snapshot = run.output_policy_snapshot
        if not snapshot:
            if run.runtime_attestation is not None:
                raise RecordConflictError(
                    f"{purpose} runtime attestation has no persisted trust policy"
                )
            return None, None
        if snapshot.get("schema") != OUTPUT_POLICY_SCHEMA_VERSION:
            raise RecordConflictError(f"{purpose} output policy schema is not approved")
        document_config = snapshot.get("document_processing_config")
        policies = snapshot.get("runtime_trust_policy")
        if not isinstance(document_config, Mapping) or not isinstance(
            policies, Mapping
        ):
            raise RecordConflictError(
                f"{purpose} output policy lacks runtime trust configuration"
            )
        policy = policies.get(component_id)
        if not isinstance(policy, Mapping):
            raise RecordConflictError(
                f"{purpose} output policy lacks a component trust policy"
            )
        expected_keys = (
            _OCR_TRUST_POLICY_KEYS
            if component_id == "ocrmypdf"
            else _REMOTE_TRUST_POLICY_KEYS
        )
        if policy.keys() != expected_keys:
            raise RecordConflictError(
                f"{purpose} runtime trust policy shape is not approved"
            )
        required_identity = document_config.get("require_runtime_identity")
        if type(required_identity) is not bool or required_identity is not (
            run.runtime_identity_required
        ):
            raise RecordConflictError(
                f"{purpose} required runtime identity differs from its output policy"
            )
        expected_source = (
            RuntimeAttestationSource.DIGEST_ADDRESSED_OCI_INVOCATION.value
            if component_id == "ocrmypdf"
            else _REMOTE_ATTESTATION_SOURCE.value
        )
        expected_contract = (
            OCR_DIGEST_RUNNER_VERSION
            if component_id == "ocrmypdf"
            else REMOTE_ATTESTATION_CONTRACT_VERSION
        )
        reporter_configured = policy["reporter_configured"]
        reporter_id = policy["expected_reporter_id"]
        if (
            type(reporter_configured) is not bool
            or policy["expected_source"] != expected_source
            or policy["attestation_contract_version"] != expected_contract
            or policy["attestation_schema_version"]
            != RUNTIME_ATTESTATION_SCHEMA_VERSION
        ):
            raise RecordConflictError(
                f"{purpose} runtime trust policy values are not approved"
            )
        if component_id == "ocrmypdf":
            if (
                reporter_id != OCR_DIGEST_RUNNER_VERSION
                or policy["local_digest_runner_version"] != OCR_DIGEST_RUNNER_VERSION
                or reporter_configured
                is not (document_config.get("ocr_mode") == "container_cli")
            ):
                raise RecordConflictError(
                    f"{purpose} OCR runtime trust policy values are not approved"
                )
        elif reporter_configured:
            if not isinstance(reporter_id, str) or not reporter_id.strip():
                raise RecordConflictError(
                    f"{purpose} runtime trust policy reporter is not approved"
                )
        elif reporter_id is not None:
            raise RecordConflictError(
                f"{purpose} unconfigured runtime trust policy names a reporter"
            )
        return policy, document_config

    @staticmethod
    def _verify_persisted_parser_policy(
        run: ProcessingRun,
        *,
        component_id: str,
        purpose: str,
        trust_policy: Mapping[str, Any] | None,
        document_config: Mapping[str, Any] | None,
        expected_image: str,
    ) -> None:
        if trust_policy is None or document_config is None:
            return
        configuration = run.configuration
        if component_id == "docling":
            policy_matches = (
                document_config.get("docling_version") == DOCLING_COMPONENT_VERSION
                and document_config.get("docling_serve_version")
                == DOCLING_SERVE_VERSION
                and document_config.get("docling_container_image") == expected_image
                and document_config.get("docling_container_digest")
                == configuration.get("container_digest")
                and document_config.get("docling_model_versions")
                == configuration.get("model_versions")
                and document_config.get("docling_model_hashes")
                == configuration.get("model_hashes")
            )
        elif component_id == "grobid":
            policy_matches = (
                document_config.get("grobid_version") == GROBID_COMPONENT_VERSION
                and document_config.get("grobid_container_image") == expected_image
                and document_config.get("grobid_container_digest")
                == configuration.get("container_digest")
                and document_config.get("grobid_model_versions")
                == configuration.get("model_versions")
                and document_config.get("grobid_model_hashes")
                == configuration.get("model_hashes")
            )
        elif component_id == "ocrmypdf":
            policy_matches = (
                document_config.get("ocrmypdf_version") == OCR_COMPONENT_VERSION
                and document_config.get("ocr_mode") == configuration.get("mode")
                and document_config.get("ocr_container_image") == expected_image
                and document_config.get("ocr_container_digest")
                == configuration.get("container_digest")
            )
        else:  # pragma: no cover - all callers use a registered native component
            raise RecordConflictError(
                f"{purpose} runtime trust policy names an unsupported component"
            )
        if not policy_matches:
            raise RecordConflictError(
                f"{purpose} invocation conflicts with persisted parser policy"
            )

    def _verify_runtime_trust_evidence(
        self,
        run: ProcessingRun,
        *,
        component_id: str,
        purpose: str,
        trust_policy: Mapping[str, Any] | None,
        document_config: Mapping[str, Any] | None,
        expected_image: str,
        expected_component_versions: Mapping[str, str],
    ) -> None:
        attestation = run.runtime_attestation
        if attestation is None:
            return
        if trust_policy is None or document_config is None:
            raise RecordConflictError(
                f"{purpose} runtime attestation has no persisted trust policy"
            )
        expected_source = (
            RuntimeAttestationSource.DIGEST_ADDRESSED_OCI_INVOCATION
            if component_id == "ocrmypdf"
            else _REMOTE_ATTESTATION_SOURCE
        )
        expected_contract = (
            OCR_DIGEST_RUNNER_VERSION
            if component_id == "ocrmypdf"
            else REMOTE_ATTESTATION_CONTRACT_VERSION
        )
        if (
            trust_policy["reporter_configured"] is not True
            or not isinstance(trust_policy["expected_reporter_id"], str)
            or not trust_policy["expected_reporter_id"].strip()
            or attestation.reporter_id != trust_policy["expected_reporter_id"]
            or trust_policy["expected_source"] != expected_source.value
            or attestation.source is not expected_source
            or trust_policy["attestation_contract_version"] != expected_contract
            or trust_policy["attestation_schema_version"]
            != RUNTIME_ATTESTATION_SCHEMA_VERSION
            or attestation.schema_version != RUNTIME_ATTESTATION_SCHEMA_VERSION
            or attestation.component_id != component_id
            or attestation.invocation_id != run.component_invocation_id
            or attestation.container_reference
            != f"{expected_image}@{attestation.container_digest}"
        ):
            raise RecordConflictError(
                f"{purpose} runtime attestation conflicts with persisted trust policy"
            )
        if component_id == "ocrmypdf":
            if (
                trust_policy["local_digest_runner_version"] != OCR_DIGEST_RUNNER_VERSION
                or attestation.component_versions.keys() != {"ocrmypdf", "tesseract"}
                or attestation.component_versions.get("ocrmypdf")
                != OCR_COMPONENT_VERSION
                or not attestation.component_versions.get("tesseract", "").strip()
            ):
                raise RecordConflictError(
                    f"{purpose} runtime-attested component versions are not approved"
                )
        elif attestation.component_versions != expected_component_versions:
            raise RecordConflictError(
                f"{purpose} runtime-attested component versions are not approved"
            )

    def _native_artifact_inputs(
        self, artifact: DocumentArtifact
    ) -> tuple[DataProductRef, ...]:
        creator_run_id = artifact.raw_location.created_by_run_id
        if creator_run_id is None:
            return ()
        creator = self.get_processing_run(creator_run_id)
        products = tuple(
            product
            for product in creator.outputs
            if product.blob_sha256 == artifact.source_sha256
        )
        if len(products) != 1:
            raise RecordConflictError(
                "derived native artifact does not resolve to exactly one creator product"
            )
        return products

    def _verify_canonical_product(
        self,
        product: DataProductRef,
        producer: ProcessingRun,
    ) -> CanonicalDocumentView:
        """Reproduce one canonical product from its exact durable inputs."""

        expected_component = (
            CANONICAL_COMPONENT_ID,
            CANONICAL_COMPONENT_VERSION,
            CANONICAL_COMPONENT_CAPABILITY,
        )
        actual_component = (
            producer.component_id,
            producer.component_version,
            producer.component.capability,
        )
        if actual_component != expected_component:
            raise RecordConflictError(
                "canonical product producer identity does not match the "
                "canonicalization contract"
            )
        if producer.status not in {
            ProcessingRunStatus.COMPLETE,
            ProcessingRunStatus.PARTIAL,
        }:
            raise RecordConflictError(
                "canonical product producer must have a complete or partial status"
            )
        if product not in producer.outputs:
            raise RecordConflictError(
                f"canonical product {product.product_id!r} is not declared by "
                f"producer run {producer.run_id!r}"
            )

        try:
            configuration = CanonicalizationConfig.model_validate(
                producer.configuration.get("policy")
            )
        except ValidationError as exc:
            raise RecordConflictError("canonical producer policy is invalid") from exc
        expected_configuration = canonical_invocation_configuration(
            configuration,
            producer.inputs,
        )
        if producer.configuration != expected_configuration:
            raise RecordConflictError(
                "canonical producer configuration does not match its exact inputs"
            )

        artifact = self.get_artifact(producer.artifact_id)
        stored_bytes = self.read_blob(product.blob_sha256)
        stored_view = load_canonical_document(stored_bytes)
        if stored_view.artifact_id != artifact.artifact_id:
            raise RecordConflictError(
                f"canonical product {product.product_id!r} artifact does not match "
                f"producer artifact {artifact.artifact_id!r}"
            )
        if stored_view.source_sha256 != artifact.source_sha256:
            raise RecordConflictError(
                f"canonical product {product.product_id!r} source hash does not "
                f"match producer artifact {artifact.artifact_id!r}"
            )
        if stored_view.source_products != producer.inputs:
            raise RecordConflictError(
                f"canonical product {product.product_id!r} source products do not "
                f"exactly match producer run {producer.run_id!r} inputs"
            )

        products_by_name: dict[str, list[DataProductRef]] = {}
        for source_product in producer.inputs:
            products_by_name.setdefault(source_product.name, []).append(source_product)

        docling_product = _require_single_product(
            products_by_name,
            "docling_document",
        )
        spans_product = _require_single_product(products_by_name, "content_spans")
        docling_bytes = self.read_blob(docling_product.blob_sha256)
        docling_payload = _strict_canonical_json_object(
            docling_bytes,
            purpose="canonical Docling source",
        )
        span_bytes = self.read_blob(spans_product.blob_sha256)
        try:
            span_payload = _strict_canonical_json_object(
                span_bytes,
                purpose="canonical content-span source",
            )
            validated_span_set = ContentSpanSet.model_validate(span_payload)
        except ValidationError as exc:
            raise RecordConflictError(
                "canonical content-span source is invalid"
            ) from exc
        if span_bytes != _canonical_json_bytes(
            validated_span_set.model_dump(mode="json")
        ):
            raise RecordConflictError(
                "canonical content-span source is not deterministically serialized"
            )
        span_producer = self._verify_docling_native_sources(
            canonical_artifact=artifact,
            docling_product=docling_product,
            spans_product=spans_product,
            docling_payload=docling_payload,
            span_set=validated_span_set,
        )
        native_outputs = tuple(
            output
            for output in span_producer.outputs
            if output.name == "native_locator_overlay"
        )
        native_product = native_outputs[0] if native_outputs else None
        native_bytes: bytes | None = None
        if native_product is not None:
            self._verify_product(native_product)
            self._verify_output_lineage(native_product, span_producer)
            native_bytes = self.read_blob(native_product.blob_sha256)

        adapter_inputs = tuple(
            input_product
            for input_product in span_producer.inputs
            if input_product.name == "native_locator_overlay"
        )
        adapter_producer = (
            self.get_processing_run(adapter_inputs[0].producer_run_id)
            if adapter_inputs
            else None
        )
        html_inputs = tuple(
            input_product
            for input_product in span_producer.inputs
            if input_product.name == "html_projection"
        )
        html_projection_bytes = (
            self.read_blob(html_inputs[0].blob_sha256) if html_inputs else None
        )
        native_alignment_outputs = tuple(
            output
            for output in span_producer.outputs
            if output.name in {"jats_locator_alignment", "bioc_locator_alignment"}
        )
        native_alignment_product = (
            native_alignment_outputs[0] if native_alignment_outputs else None
        )
        native_alignment_bytes: bytes | None = None
        if native_alignment_product is not None:
            self._verify_product(native_alignment_product)
            self._verify_output_lineage(native_alignment_product, span_producer)
            native_alignment_bytes = self.read_blob(
                native_alignment_product.blob_sha256
            )
        try:
            span_set = replay_content_span_set(
                artifact=artifact,
                source_bytes=self.read_blob(artifact.raw_location.sha256),
                producer=span_producer,
                docling_product=docling_product,
                docling_bytes=docling_bytes,
                docling_document=docling_payload,
                content_spans_product=spans_product,
                content_spans_bytes=span_bytes,
                native_locator_product=native_product,
                native_locator_bytes=native_bytes,
                adapter_producer=adapter_producer,
                html_projection_bytes=html_projection_bytes,
                native_alignment_product=native_alignment_product,
                native_alignment_bytes=native_alignment_bytes,
            )
        except ContentSpanReplayError as exc:
            raise RecordConflictError(
                "canonical content-span source does not reproduce from its "
                "exact Docling invocation"
            ) from exc

        grobid_products = tuple(products_by_name.get("grobid_tei", ()))
        alignment_products = tuple(products_by_name.get("alignment_overlay", ()))
        if bool(grobid_products) != bool(alignment_products):  # pragma: no cover
            # CanonicalDocumentView rejects unpaired scholarly sources before
            # an admitted product can reach this defense-in-depth boundary.
            raise RecordConflictError(
                "canonical scholarly sources require paired GROBID and alignment products"
            )
        if (  # pragma: no cover
            len(grobid_products) > 1 or len(alignment_products) > 1
        ):
            # CanonicalDocumentView also enforces exact scholarly uniqueness.
            raise RecordConflictError("canonical scholarly sources must be unique")

        scholarly_overlay: ScholarlyAlignmentOverlay | None = None
        alignment_product: DataProductRef | None = None
        if grobid_products:
            grobid_product = grobid_products[0]
            alignment_product = alignment_products[0]
            grobid_bytes = self.read_blob(grobid_product.blob_sha256)
            self._verify_grobid_native_source(
                canonical_artifact=artifact,
                product=grobid_product,
                tei_xml=grobid_bytes,
            )
            alignment_run = self.get_processing_run(alignment_product.producer_run_id)
            self._require_component_contract(
                alignment_run,
                component_id="docling-grobid-aligner",
                component_version="2",
                capability="document.align",
                statuses=frozenset({ProcessingRunStatus.COMPLETE}),
                purpose="scholarly alignment",
            )
            if alignment_run.inputs != (docling_product, grobid_product):
                raise RecordConflictError(
                    "scholarly alignment inputs do not match the canonical sources"
                )
            if alignment_run.artifact_id != artifact.artifact_id:
                raise RecordConflictError(
                    "scholarly alignment producer does not own the canonical artifact"
                )
            minimum_score = alignment_run.configuration.get("minimum_score")
            if (
                isinstance(minimum_score, bool)
                or not isinstance(minimum_score, (int, float))
                or not 0 <= minimum_score <= 1
            ):
                raise RecordConflictError(
                    "scholarly alignment minimum score is invalid"
                )
            expected_alignment_configuration = {
                "algorithm": "token-sequence-v2",
                "minimum_score": minimum_score,
                "docling_document_sha256": docling_product.blob_sha256,
                "grobid_tei_sha256": grobid_product.blob_sha256,
            }
            if alignment_run.configuration != expected_alignment_configuration:
                raise RecordConflictError(
                    "scholarly alignment configuration does not match its inputs"
                )
            alignment_bytes = self.read_blob(alignment_product.blob_sha256)
            try:
                alignment_payload = json.loads(alignment_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RecordConflictError(
                    "scholarly alignment product is not valid JSON"
                ) from exc
            if not isinstance(alignment_payload, dict):
                raise RecordConflictError(
                    "scholarly alignment product must be a JSON object"
                )
            try:
                scholarly_overlay = ScholarlyAlignmentOverlay.from_dict(
                    alignment_payload
                )
                verify_scholarly_alignment_overlay(
                    docling_payload,
                    grobid_bytes,
                    scholarly_overlay,
                    minimum_score=float(minimum_score),
                )
            except ValueError as exc:
                raise RecordConflictError(
                    "scholarly alignment product does not reproduce from its inputs"
                ) from exc
            if alignment_bytes != _canonical_json_bytes(scholarly_overlay.to_dict()):
                raise RecordConflictError(
                    "scholarly alignment product is not deterministically serialized"
                )

        integrity_products = tuple(
            products_by_name.get("content_integrity_overlay", ())
        )
        if len(integrity_products) > 1:  # pragma: no cover
            # CanonicalDocumentView permits at most one integrity product.
            raise RecordConflictError(
                "canonical content-integrity source must be unique"
            )
        integrity_payload: dict[str, object] | None = None
        if integrity_products:
            integrity_product = integrity_products[0]
            integrity_run = self.get_processing_run(integrity_product.producer_run_id)
            self._require_component_contract(
                integrity_run,
                component_id="docling-content-integrity",
                component_version="1",
                capability="document.validate",
                statuses=frozenset(
                    {
                        ProcessingRunStatus.COMPLETE,
                        ProcessingRunStatus.PARTIAL,
                    }
                ),
                purpose="content integrity",
            )
            expected_integrity_inputs = (docling_product,) + (
                (alignment_product,) if alignment_product is not None else ()
            )
            if integrity_run.inputs != expected_integrity_inputs:
                raise RecordConflictError(
                    "content-integrity inputs do not match the canonical sources"
                )
            if integrity_run.artifact_id != artifact.artifact_id:
                raise RecordConflictError(
                    "content-integrity producer does not own the canonical artifact"
                )
            expected_integrity_configuration = {
                "algorithm": "explicit-content-integrity-v1",
                "docling_document_sha256": docling_product.blob_sha256,
                "scholarly_alignment_sha256": (
                    alignment_product.blob_sha256
                    if alignment_product is not None
                    else None
                ),
            }
            if integrity_run.configuration != expected_integrity_configuration:
                raise RecordConflictError(
                    "content-integrity configuration does not match its inputs"
                )
            expected_integrity_payload = validate_content_integrity(
                docling_payload,
                scholarly_overlay=scholarly_overlay,
            ).to_dict()
            if (
                expected_integrity_payload["issues"]
                and integrity_run.status is ProcessingRunStatus.COMPLETE
            ):
                raise RecordConflictError(
                    "complete content-integrity producer masks replayed issues"
                )
            integrity_bytes = self.read_blob(integrity_product.blob_sha256)
            if integrity_bytes != _canonical_json_bytes(expected_integrity_payload):
                raise RecordConflictError(
                    "content-integrity product does not reproduce from its inputs"
                )
            integrity_payload = expected_integrity_payload

        try:
            rebuilt_view = build_canonical_document_view(
                artifact=artifact,
                docling_document=docling_payload,
                docling_product=docling_product,
                content_span_set=span_set,
                source_products=producer.inputs,
                configuration=configuration,
                scholarly_overlay=scholarly_overlay,
                integrity_report=integrity_payload,
            )
        except ValueError as exc:
            raise RecordConflictError(
                "canonical product cannot be rebuilt from its exact sources"
            ) from exc
        rebuilt_bytes = canonical_document_bytes(rebuilt_view)
        if (
            any(
                diagnostic.severity is CanonicalDiagnosticSeverity.ERROR
                for diagnostic in rebuilt_view.diagnostics
            )
            and producer.status is ProcessingRunStatus.COMPLETE
        ):
            raise RecordConflictError(
                "complete canonical producer masks replayed semantic errors"
            )
        if stored_view.view_id != rebuilt_view.view_id:
            raise RecordConflictError(
                "canonical view identity does not match its exact sources"
            )
        if stored_bytes != rebuilt_bytes:
            raise RecordConflictError(
                "canonical document bytes do not match their deterministic rebuild"
            )
        return stored_view

    @staticmethod
    def _require_component_contract(
        run: ProcessingRun,
        *,
        component_id: str,
        component_version: str,
        capability: str,
        statuses: frozenset[ProcessingRunStatus],
        purpose: str,
    ) -> None:
        if (
            run.component_id,
            run.component_version,
            run.component.capability,
        ) != (component_id, component_version, capability):
            raise RecordConflictError(
                f"{purpose} producer identity does not match its contract"
            )
        if run.status not in statuses:
            raise RecordConflictError(
                f"{purpose} producer status does not permit durable reuse"
            )

    def _verify_product(self, product: DataProductRef) -> None:
        try:
            validate_product_contract(product)
        except ValueError as exc:
            raise RecordConflictError(str(exc)) from exc
        expected_uri = self.blob_uri(product.blob_sha256)
        if product.uri != expected_uri:
            raise HashMismatchError(
                f"data product {product.name!r} URI does not match its hash"
            )
        path = self.verify_blob(product.blob_sha256)
        if path.stat().st_size != product.byte_size:
            raise HashMismatchError(
                f"data product {product.name!r} size does not match stored blob"
            )
        for artifact_id in product.source_artifact_ids:
            self.get_artifact(artifact_id)

    @staticmethod
    def _revalidate_product(product: DataProductRef) -> DataProductRef:
        try:
            return DataProductRef.model_validate(product.model_dump(mode="python"))
        except ValidationError as exc:
            raise RecordConflictError("data product reference is invalid") from exc

    def _verify_durable_product_chain(
        self,
        products: tuple[DataProductRef, ...],
    ) -> None:
        pending = [(product, False) for product in reversed(products)]
        visiting_run_ids: set[str] = set()
        verified_run_ids: set[str] = set()
        while pending:
            product, exiting = pending.pop()
            if exiting:
                producer_run_id = product.producer_run_id
                visiting_run_ids.remove(producer_run_id)
                verified_run_ids.add(producer_run_id)
                continue

            validated_product = self._revalidate_product(product)
            producer_run_id = validated_product.producer_run_id
            self._verify_product(validated_product)
            producer = self.get_processing_run(producer_run_id)
            self._verify_output_lineage(validated_product, producer)
            if validated_product not in producer.outputs:
                raise RecordConflictError(
                    f"data product {validated_product.product_id!r} is not declared "
                    f"by producer run {producer.run_id!r}"
                )
            if producer_run_id in visiting_run_ids:
                raise RecordConflictError(
                    f"processing-run input provenance contains a cycle at "
                    f"{producer_run_id!r}"
                )
            if producer_run_id in verified_run_ids:
                continue
            visiting_run_ids.add(producer_run_id)
            pending.append((validated_product, True))
            pending.extend(
                (input_product, False) for input_product in reversed(producer.inputs)
            )

    @staticmethod
    def _verify_output_lineage(
        product: DataProductRef,
        producer: ProcessingRun,
    ) -> None:
        expected_lineage = tuple(
            dict.fromkeys(
                (
                    producer.artifact_id,
                    *(
                        artifact_id
                        for input_product in producer.inputs
                        for artifact_id in input_product.source_artifact_ids
                    ),
                )
            )
        )
        if product.source_artifact_ids != expected_lineage:
            raise RecordConflictError(
                f"data product {product.product_id!r} lineage must exactly equal "
                f"producer artifact and inherited input lineage {expected_lineage!r}"
            )

    def _save_record(
        self, category: str, record_id: str, record: _RECORD_MODEL
    ) -> Path:
        path = self._record_path(category, record_id)
        # Pydantic's frozen models are only shallowly immutable: nested dicts can
        # still be changed, and model_copy(update=...) does not re-run validation.
        # Reconstruct the record from plain Python data immediately before
        # serialization so invalid mutations can never become durable state.
        validated_record = type(record).model_validate(record.model_dump(mode="python"))
        serialized = _canonical_record_bytes(validated_record)

        if path.exists():
            if self._stored_record_matches(path, serialized, validated_record):
                return path
            raise RecordConflictError(
                f"immutable {category} ID already has different content: {record_id}"
            )

        temporary_path = self._temporary_path(category)
        try:
            with temporary_path.open("xb") as destination:
                destination.write(serialized)
                destination.flush()
                os.fsync(destination.fileno())
            try:
                self._publish_no_replace(temporary_path, path)
            except FileExistsError:
                if not self._stored_record_matches(path, serialized, validated_record):
                    raise RecordConflictError(
                        f"concurrent conflicting {category} write: {record_id}"
                    ) from None
            return path
        finally:
            temporary_path.unlink(missing_ok=True)

    def _get_record(
        self,
        category: str,
        record_id: str,
        model: type[_RECORD_MODEL],
        *,
        id_field: str,
    ) -> _RECORD_MODEL:
        path = self._record_path(category, record_id)
        if not path.is_file():
            raise RecordNotFoundError(f"{category} record not found: {record_id}")
        record = self._decode_record(path, model)
        if getattr(record, id_field) != record_id:
            raise CorruptRecordError(
                f"{category} record key does not match its stored ID: {record_id}"
            )
        return record

    def _read_record_directory(
        self, category: str, model: type[_RECORD_MODEL]
    ) -> tuple[_RECORD_MODEL, ...]:
        directory = self._record_root / category
        return tuple(
            self._decode_record(path, model) for path in directory.glob("*.json")
        )

    def _decode_record(self, path: Path, model: type[_RECORD_MODEL]) -> _RECORD_MODEL:
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CorruptRecordError(f"invalid record at {path}") from exc
        if not isinstance(payload, dict):
            raise CorruptRecordError(f"record at {path} must be a JSON object")
        schema_field = model.model_fields.get("schema_version")
        expected_version = schema_field.default if schema_field is not None else None
        actual_version = payload.get("schema_version")
        if model is ProcessingRun and actual_version == PROCESSING_RUN_SCHEMA_V1:
            # v1 was originally written without stage-invocation identity.  A
            # short-lived fork revision also wrote paired invocation IDs while
            # retaining the v1 tag.  Both shapes are validated against the v2
            # invariant after this explicit, in-memory migration.  Durable v1
            # bytes remain immutable; all new records are written as v2.
            payload = dict(payload)
            payload["schema_version"] = PROCESSING_RUN_SCHEMA_V2
            actual_version = PROCESSING_RUN_SCHEMA_V2
        if not isinstance(expected_version, str) or actual_version != expected_version:
            raise UnsupportedSchemaVersionError(
                f"unsupported schema_version {actual_version!r} at {path}; "
                f"expected {expected_version!r}"
            )
        try:
            return model.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            raise CorruptRecordError(f"invalid record at {path}") from exc

    def _stored_record_matches(
        self,
        path: Path,
        serialized: bytes,
        record: BaseModel,
    ) -> bool:
        """Compare immutable records, including an in-place legacy run record."""

        if path.read_bytes() == serialized:
            return True
        if isinstance(record, ProcessingRun):
            return self._decode_record(path, ProcessingRun) == record
        return False

    def _record_path(self, category: str, record_id: str) -> Path:
        if not record_id.strip():
            raise ValueError("record ID must not be empty")
        key = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
        return self._record_root / category / f"{key}.json"

    def _temporary_path(self, prefix: str) -> Path:
        descriptor, name = tempfile.mkstemp(
            dir=self._temporary_root,
            prefix=f"{prefix}-",
            suffix=".tmp",
        )
        os.close(descriptor)
        path = Path(name)
        path.unlink()
        return path

    @staticmethod
    def _publish_no_replace(source: Path, destination: Path) -> None:
        """Atomically publish a complete file without replacing a peer."""

        try:
            os.link(source, destination)
        except FileExistsError:
            # A concurrent publisher may not have synced the shared directory
            # entry yet. Sync it here before accepting the existing record.
            _fsync_directory(destination.parent)
            raise
        _fsync_directory(destination.parent)


def _ensure_directory_durable(directory: Path) -> None:
    """Create missing directories and durably publish each new directory entry."""

    missing: list[Path] = []
    candidate = directory
    while not candidate.exists():
        missing.append(candidate)
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    for candidate in reversed(missing):
        try:
            candidate.mkdir()
        except FileExistsError:
            if not candidate.is_dir():
                raise
        # Sync both the new directory's contents and the parent entry that
        # names it. This also closes the race where a peer created it first.
        _fsync_directory(candidate)
        _fsync_directory(candidate.parent)


def _fsync_directory(directory: Path) -> None:
    """Flush directory entries on POSIX; Windows has no portable equivalent."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_record_bytes(record: BaseModel) -> bytes:
    data = record.model_dump(mode="json")
    return (
        json.dumps(
            data,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_canonical_json_object(
    payload: bytes,
    *,
    purpose: str,
) -> dict[str, Any]:
    """Decode one canonical JSON object without accepting ambiguous syntax."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RecordConflictError(f"{purpose} is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise RecordConflictError(f"{purpose} must be a JSON object")
    try:
        canonical_bytes = _canonical_json_bytes(decoded)
    except (TypeError, ValueError) as exc:  # pragma: no cover - parser defense
        raise RecordConflictError(f"{purpose} is not canonical JSON") from exc
    if payload != canonical_bytes:
        raise RecordConflictError(f"{purpose} is not deterministically serialized")
    return decoded


def _is_oci_sha256(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        character in _SHA256_PATTERN for character in digest
    )


def _valid_version_mapping(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str)
        and bool(key.strip())
        and isinstance(version, str)
        and bool(version.strip())
        for key, version in value.items()
    )


def _valid_hash_mapping(value: object) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str)
        and bool(key.strip())
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in _SHA256_PATTERN for character in digest)
        for key, digest in value.items()
    )


def _require_single_product(
    products_by_name: Mapping[str, list[DataProductRef]],
    name: str,
) -> DataProductRef:
    products = products_by_name.get(name)
    if products is None or len(products) != 1:
        raise RecordConflictError(
            f"canonical sources require exactly one {name!r} product"
        )
    return products[0]


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in _SHA256_PATTERN for character in value):
        raise ValueError("SHA-256 hashes must be 64 lowercase hexadecimal characters")


__all__ = [
    "BlobNotFoundError",
    "BlobTooLargeError",
    "ContentAddressedStore",
    "CorruptRecordError",
    "HashMismatchError",
    "RecordConflictError",
    "RecordNotFoundError",
    "StorageError",
    "StoredBlob",
    "UnsupportedSchemaVersionError",
]
