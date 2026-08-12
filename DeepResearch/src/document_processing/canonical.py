"""Versioned, project-owned canonical document representation.

The canonical view is derived from immutable parser-native products.  It keeps
ordered structure and stable block identities without absorbing scientific
annotations or replacing the native products used to build it.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from .alignment import AlignmentStatus, ScholarlyAlignmentOverlay
from .models import (
    ContentSpan,
    ContentSpanSet,
    DataProductRef,
    DocumentArtifact,
    FrozenModel,
    Sha256,
    SourceLocator,
    sha256_bytes,
)
from .validation import (
    ContentIntegrityKind,
    ContentIntegrityReport,
    QualitySeverity,
    docling_document_sha256,
    parse_content_integrity_report,
    validate_docling_reference_definitions,
)

CANONICAL_DOCUMENT_SCHEMA_VERSION = "deepcritical-canonical-document-view-v1"
CANONICAL_TEXT_NORMALIZATION = "unicode-nfc-collapse-whitespace-v1"
CANONICAL_ANCHORING_POLICY = "source-spans-and-native-nodes-v1"
CANONICAL_COMPONENT_ID = "canonical-document-view"
CANONICAL_COMPONENT_VERSION = "1"
CANONICAL_COMPONENT_CAPABILITY = "document.canonicalize"
CANONICAL_ADAPTER_VERSION = "1"
MAX_CANONICAL_TABLE_CELLS = 100_000
MAX_CANONICAL_TABLE_AXIS = 1_000_000
MAX_CANONICAL_NATIVE_NODES = 100_000
MAX_CANONICAL_HIERARCHY_EDGES = 500_000
_NATIVE_NODE_COLLECTIONS = (
    "texts",
    "tables",
    "pictures",
    "formulas",
    "groups",
    "key_value_items",
    "form_items",
    "field_regions",
    "field_items",
)
_CANONICAL_SOURCE_PRODUCT_NAMES = frozenset(
    {
        "docling_document",
        "content_spans",
        "grobid_tei",
        "alignment_overlay",
        "content_integrity_overlay",
    }
)


class _FrozenStringMapping(Mapping[str, str]):
    """Small immutable mapping that remains safe under deep-copy operations."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, str]) -> None:
        self._items = tuple(values.items())

    def __getitem__(self, key: str) -> str:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self._items) == dict(other)

    def __hash__(self) -> int:
        # Mapping equality is independent of insertion order, so its hash must
        # be as well.  Values are contractually strings and therefore hashable.
        return hash(frozenset(self._items))

    def __copy__(self) -> _FrozenStringMapping:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenStringMapping:
        return self


class CanonicalDocumentError(ValueError):
    """Base error for canonical document construction and loading."""


class UnsupportedCanonicalDocumentVersionError(CanonicalDocumentError):
    """A canonical document blob uses an unknown or missing schema version."""


class InvalidCanonicalDocumentError(CanonicalDocumentError):
    """A canonical document blob is malformed or violates its contract."""


class CanonicalBlockKind(StrEnum):
    """Closed structural vocabulary for processor-independent document blocks."""

    TITLE = "title"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FORMULA = "formula"
    CITATION = "citation"
    REFERENCE = "reference"
    GROUP = "group"
    OTHER = "other"


class CanonicalAnchorRole(StrEnum):
    """Why a native representation anchor is attached to a canonical block."""

    PRIMARY = "primary"
    SOURCE = "source"
    SCHOLARLY = "scholarly"


class CanonicalRelationshipKind(StrEnum):
    """Closed relationships retained from parser-native structure."""

    HAS_CAPTION = "has_caption"
    CITES = "cites"


class CanonicalRelationshipStatus(StrEnum):
    """Resolution state for one native structural relationship."""

    RESOLVED = "resolved"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"


class CanonicalDiagnosticSeverity(StrEnum):
    """Severity of a canonicalization mapping diagnostic."""

    WARNING = "warning"
    ERROR = "error"


class CanonicalizationConfig(BaseModel):
    """Closed, persisted policy for one canonicalization component instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    text_normalization: Literal["unicode-nfc-collapse-whitespace-v1"] = (
        CANONICAL_TEXT_NORMALIZATION
    )
    anchoring_policy: Literal["source-spans-and-native-nodes-v1"] = (
        CANONICAL_ANCHORING_POLICY
    )


def canonical_invocation_configuration(
    configuration: CanonicalizationConfig,
    source_products: tuple[DataProductRef, ...],
) -> dict[str, Any]:
    """Return the complete persisted invocation contract for canonicalization."""

    return {
        "adapter_version": CANONICAL_ADAPTER_VERSION,
        "policy": configuration.model_dump(mode="json"),
        "input_products": [
            {
                "name": product.name,
                "product_id": product.product_id,
                "blob_sha256": product.blob_sha256,
                "payload_schema_version": product.payload_schema_version,
            }
            for product in source_products
        ],
    }


class CanonicalSourceAnchor(FrozenModel):
    """Exact node or character range in one immutable native product."""

    role: CanonicalAnchorRole
    product_id: str
    node_id: str
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, gt=0)
    source_locator: SourceLocator | None = None

    @model_validator(mode="after")
    def _validate_range(self) -> CanonicalSourceAnchor:
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("canonical anchor character bounds must be paired")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end <= self.char_start
        ):
            raise ValueError("canonical anchor char_end must exceed char_start")
        if not self.product_id.strip() or not self.node_id.strip():
            raise ValueError("canonical anchor identifiers must not be empty")
        return self


class CanonicalTableCell(FrozenModel):
    """One normalized table cell with its source grid extent."""

    text: str
    row_index: int = Field(ge=0, lt=MAX_CANONICAL_TABLE_AXIS, strict=True)
    column_index: int = Field(ge=0, lt=MAX_CANONICAL_TABLE_AXIS, strict=True)
    row_span: int = Field(
        default=1,
        ge=1,
        le=MAX_CANONICAL_TABLE_AXIS,
        strict=True,
    )
    column_span: int = Field(
        default=1,
        ge=1,
        le=MAX_CANONICAL_TABLE_AXIS,
        strict=True,
    )
    column_header: bool = Field(default=False, strict=True)
    row_header: bool = Field(default=False, strict=True)
    row_section: bool = Field(default=False, strict=True)
    fillable: bool = Field(default=False, strict=True)
    native_ref: str | None = None

    @field_validator("text")
    @classmethod
    def _validate_text_normalization(cls, value: str) -> str:
        if normalize_canonical_text(value) != value:
            raise ValueError(
                "canonical table cell text must use the canonical normalization policy"
            )
        return value

    @field_validator("native_ref")
    @classmethod
    def _validate_native_ref(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("canonical table cell native_ref must not be empty")
        return value


class CanonicalTable(FrozenModel):
    """Normalized table cells with exact row, column, and span structure."""

    row_count: int = Field(
        default=0,
        ge=0,
        le=MAX_CANONICAL_TABLE_AXIS,
        strict=True,
    )
    column_count: int = Field(
        default=0,
        ge=0,
        le=MAX_CANONICAL_TABLE_AXIS,
        strict=True,
    )
    cells: tuple[CanonicalTableCell, ...] = Field(
        default=(),
        max_length=MAX_CANONICAL_TABLE_CELLS,
    )

    @model_validator(mode="after")
    def _validate_grid(self) -> CanonicalTable:
        coordinates: set[tuple[int, int]] = set()
        for cell in self.cells:
            coordinate = (cell.row_index, cell.column_index)
            if coordinate in coordinates:
                raise ValueError("canonical table cell coordinates must be unique")
            coordinates.add(coordinate)
            if cell.row_index + cell.row_span > self.row_count:
                raise ValueError("canonical table cell exceeds row_count")
            if cell.column_index + cell.column_span > self.column_count:
                raise ValueError("canonical table cell exceeds column_count")
        if _table_cells_overlap(self.cells):
            raise ValueError("canonical table cell extents must not overlap")
        if not self.cells and (self.row_count or self.column_count):
            raise ValueError("empty canonical tables must have zero dimensions")
        return self


class CanonicalBlock(FrozenModel):
    """One immutable block in canonical document order."""

    block_id: str
    native_node_id: str
    native_label: str
    kind: CanonicalBlockKind
    ordinal: int = Field(ge=0)
    parent_block_id: str | None = None
    child_block_ids: tuple[str, ...] = ()
    text: str | None = None
    table: CanonicalTable | None = None
    content_sha256: Sha256
    source_anchors: tuple[CanonicalSourceAnchor, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_content(self) -> CanonicalBlock:
        if self.text is not None and normalize_canonical_text(self.text) != self.text:
            raise ValueError(
                "canonical block text must use the canonical normalization policy"
            )
        expected_hash = canonical_block_content_sha256(
            kind=self.kind,
            text=self.text,
            table=self.table,
        )
        if self.content_sha256 != expected_hash:
            raise ValueError("canonical block content hash does not match its content")
        expected_id = canonical_block_id(
            native_node_id=self.native_node_id,
            kind=self.kind,
            content_sha256=self.content_sha256,
        )
        if self.block_id != expected_id:
            raise ValueError("canonical block ID does not match its stable identity")
        if (self.kind is CanonicalBlockKind.TABLE) != (self.table is not None):
            raise ValueError("only table blocks may contain canonical table data")
        if self.text is not None and not self.text:
            raise ValueError("canonical block text must be non-empty when present")
        if len(set(self.child_block_ids)) != len(self.child_block_ids):
            raise ValueError("canonical child block IDs must be unique")
        anchor_keys = {
            json.dumps(
                anchor.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for anchor in self.source_anchors
        }
        if len(anchor_keys) != len(self.source_anchors):
            raise ValueError("canonical source anchors must be unique")
        return self


class CanonicalRelationship(FrozenModel):
    """Resolved and unresolved native relationship evidence."""

    relationship_id: str
    kind: CanonicalRelationshipKind
    status: CanonicalRelationshipStatus
    source_block_id: str | None = None
    target_block_ids: tuple[str, ...] = ()
    declared_target_refs: tuple[str, ...] = ()
    unresolved_target_refs: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    source_anchor: CanonicalSourceAnchor | None = None

    @model_validator(mode="after")
    def _validate_identity(self) -> CanonicalRelationship:
        if len(set(self.target_block_ids)) != len(self.target_block_ids):
            raise ValueError("canonical relationship targets must be unique")
        expected_status = canonical_relationship_status(
            source_block_id=self.source_block_id,
            target_block_ids=self.target_block_ids,
            declared_target_refs=self.declared_target_refs,
            unresolved_target_refs=self.unresolved_target_refs,
            reason_codes=self.reason_codes,
        )
        if self.status is not expected_status:
            raise ValueError(
                "canonical relationship status does not match its resolution"
            )
        expected_id = canonical_relationship_id(
            kind=self.kind,
            status=self.status,
            source_block_id=self.source_block_id,
            target_block_ids=self.target_block_ids,
            declared_target_refs=self.declared_target_refs,
            unresolved_target_refs=self.unresolved_target_refs,
            reason_codes=self.reason_codes,
            source_anchor=self.source_anchor,
        )
        if self.relationship_id != expected_id:
            raise ValueError("canonical relationship ID does not match its content")
        return self


class CanonicalMappingDiagnostic(FrozenModel):
    """Explicit evidence that a native node or relationship did not map cleanly."""

    diagnostic_id: str
    severity: CanonicalDiagnosticSeverity
    code: str
    message: str
    product_id: str | None = None
    native_node_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_identity(self) -> CanonicalMappingDiagnostic:
        expected_id = canonical_diagnostic_id(
            severity=self.severity,
            code=self.code,
            message=self.message,
            product_id=self.product_id,
            native_node_ids=self.native_node_ids,
        )
        if self.diagnostic_id != expected_id:
            raise ValueError("canonical diagnostic ID does not match its content")
        if not self.code.strip() or not self.message.strip():
            raise ValueError("canonical diagnostic values must not be empty")
        return self


class CanonicalDocumentMetadata(FrozenModel):
    """Source-level metadata retained without scientific interpretation."""

    title: str | None = None
    media_type: str
    identifiers: Mapping[str, str] = Field(default_factory=dict)

    @field_validator("title")
    @classmethod
    def _validate_title_normalization(cls, value: str | None) -> str | None:
        if value is not None and (
            not value or normalize_canonical_text(value) != value
        ):
            raise ValueError(
                "canonical document title must use the canonical normalization policy"
            )
        return value

    @field_validator("identifiers")
    @classmethod
    def _freeze_identifiers(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return _FrozenStringMapping(value)

    @field_serializer("identifiers")
    def _serialize_identifiers(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)


class CanonicalDocumentView(FrozenModel):
    """Versioned canonical structure targeting exact immutable native products."""

    schema_version: Literal["deepcritical-canonical-document-view-v1"] = (
        CANONICAL_DOCUMENT_SCHEMA_VERSION
    )
    view_id: str
    artifact_id: str
    source_sha256: Sha256
    normalization_policy: Literal["unicode-nfc-collapse-whitespace-v1"]
    anchoring_policy: Literal["source-spans-and-native-nodes-v1"]
    metadata: CanonicalDocumentMetadata
    source_products: tuple[DataProductRef, ...] = Field(min_length=2)
    root_block_ids: tuple[str, ...] = Field(min_length=1)
    blocks: tuple[CanonicalBlock, ...] = Field(min_length=1)
    relationships: tuple[CanonicalRelationship, ...] = ()
    diagnostics: tuple[CanonicalMappingDiagnostic, ...] = ()

    @model_validator(mode="after")
    def _validate_graph_and_identity(self) -> CanonicalDocumentView:
        if len(self.blocks) > MAX_CANONICAL_NATIVE_NODES:
            raise ValueError(
                "NATIVE_NODE_LIMIT_EXCEEDED: canonical document exceeds the "
                "native node limit"
            )
        hierarchy_edge_count = 0
        for block in self.blocks:
            hierarchy_edge_count += len(block.child_block_ids)
            if hierarchy_edge_count > MAX_CANONICAL_HIERARCHY_EDGES:
                raise ValueError(
                    "NATIVE_HIERARCHY_EDGE_LIMIT_EXCEEDED: canonical document "
                    "exceeds the hierarchy edge limit"
                )
        products = {product.product_id: product for product in self.source_products}
        if len(products) != len(self.source_products):
            raise ValueError("canonical source products must be unique")
        if any(
            self.artifact_id not in product.source_artifact_ids
            for product in self.source_products
        ):
            raise ValueError(
                "canonical source-product lineage must include the source artifact"
            )
        products_by_name: dict[str, list[DataProductRef]] = {}
        for product in self.source_products:
            products_by_name.setdefault(product.name, []).append(product)
        if any(
            name not in _CANONICAL_SOURCE_PRODUCT_NAMES for name in products_by_name
        ):
            raise ValueError(
                "canonical source products contain names not used by schema v1"
            )
        if len(products_by_name.get("docling_document", ())) != 1:
            raise ValueError("canonical view requires exactly one Docling product")
        if len(products_by_name.get("content_spans", ())) != 1:
            raise ValueError("canonical view requires exactly one content-span product")
        scholarly_product_counts = (
            len(products_by_name.get("grobid_tei", ())),
            len(products_by_name.get("alignment_overlay", ())),
        )
        if scholarly_product_counts not in {(0, 0), (1, 1)}:
            raise ValueError(
                "canonical scholarly products must pair exactly one GROBID and "
                "alignment product"
            )
        if len(products_by_name.get("content_integrity_overlay", ())) > 1:
            raise ValueError(
                "canonical view permits at most one content-integrity product"
            )
        docling_product_id = products_by_name["docling_document"][0].product_id
        blocks = {block.block_id: block for block in self.blocks}
        if len(blocks) != len(self.blocks):
            raise ValueError("canonical block IDs must be unique")
        native_node_ids = [block.native_node_id for block in self.blocks]
        if len(set(native_node_ids)) != len(native_node_ids):
            raise ValueError("canonical native node IDs must be unique")
        if tuple(block.ordinal for block in self.blocks) != tuple(
            range(len(self.blocks))
        ):
            raise ValueError("canonical block ordinals must be contiguous")
        if len(set(self.root_block_ids)) != len(self.root_block_ids):
            raise ValueError("canonical root block IDs must be unique")
        expected_roots = tuple(
            block.block_id for block in self.blocks if block.parent_block_id is None
        )
        if self.root_block_ids != expected_roots:
            raise ValueError("canonical roots must match blocks without parents")
        child_ids_by_parent = {
            block.block_id: frozenset(block.child_block_ids) for block in self.blocks
        }
        for block in self.blocks:
            if block.parent_block_id is not None:
                parent = blocks.get(block.parent_block_id)
                if (
                    parent is None
                    or block.block_id not in child_ids_by_parent[parent.block_id]
                ):
                    raise ValueError("canonical parent and child links must agree")
                if parent.ordinal >= block.ordinal:
                    raise ValueError("canonical parents must precede their children")
            for child_id in block.child_block_ids:
                child = blocks.get(child_id)
                if child is None or child.parent_block_id != block.block_id:
                    raise ValueError("canonical child and parent links must agree")
            for anchor in block.source_anchors:
                if anchor.product_id not in products:
                    raise ValueError("canonical anchor references an unknown product")
                product_name = products[anchor.product_id].name
                if (
                    anchor.role
                    in {CanonicalAnchorRole.PRIMARY, CanonicalAnchorRole.SOURCE}
                    and anchor.product_id != docling_product_id
                ):
                    raise ValueError(
                        "canonical native anchors must target the Docling product"
                    )
                if (
                    anchor.role is CanonicalAnchorRole.SCHOLARLY
                    and product_name != "grobid_tei"
                ):
                    raise ValueError(
                        "canonical scholarly anchors must target a GROBID product"
                    )
            primary_anchors = tuple(
                anchor
                for anchor in block.source_anchors
                if anchor.role is CanonicalAnchorRole.PRIMARY
            )
            if len(primary_anchors) != 1:
                raise ValueError(
                    "canonical blocks require exactly one primary native anchor"
                )
            if primary_anchors[0].node_id != block.native_node_id:
                raise ValueError(
                    "canonical primary anchor must target its block's native node"
                )
        relationship_ids: set[str] = set()
        for relationship in self.relationships:
            if relationship.relationship_id in relationship_ids:
                raise ValueError("canonical relationship IDs must be unique")
            relationship_ids.add(relationship.relationship_id)
            if (
                relationship.source_block_id is not None
                and relationship.source_block_id not in blocks
            ):
                raise ValueError("canonical relationship source does not exist")
            if any(target not in blocks for target in relationship.target_block_ids):
                raise ValueError("canonical relationship target does not exist")
            if (
                relationship.source_anchor is not None
                and relationship.source_anchor.product_id not in products
            ):
                raise ValueError("canonical relationship anchor product does not exist")
            if relationship.source_anchor is not None and (
                relationship.source_anchor.role is not CanonicalAnchorRole.SCHOLARLY
                or products[relationship.source_anchor.product_id].name != "grobid_tei"
            ):
                raise ValueError(
                    "canonical relationship anchors must target scholarly evidence"
                )
        diagnostic_ids = [diagnostic.diagnostic_id for diagnostic in self.diagnostics]
        if len(set(diagnostic_ids)) != len(diagnostic_ids):
            raise ValueError("canonical diagnostic IDs must be unique")
        if any(
            diagnostic.product_id is not None and diagnostic.product_id not in products
            for diagnostic in self.diagnostics
        ):
            raise ValueError("canonical diagnostic product does not exist")
        expected_view_id = canonical_view_id(self.model_dump(mode="json"))
        if self.view_id != expected_view_id:
            raise ValueError("canonical view ID does not match its content")
        return self


@dataclass(frozen=True, slots=True)
class _NativeNode:
    canonical_ref: str
    declared_ref: str
    collection: str
    item: dict[str, Any]
    native_label: str
    kind: CanonicalBlockKind
    text: str | None
    table: CanonicalTable | None


def normalize_canonical_text(value: str) -> str:
    """Normalize text using the only policy accepted by schema version 1."""

    normalized = unicodedata.normalize("NFC", value)
    return re.sub(r"\s+", " ", normalized, flags=re.UNICODE).strip()


def canonical_block_content_sha256(
    *,
    kind: CanonicalBlockKind,
    text: str | None,
    table: CanonicalTable | None,
) -> str:
    """Hash normalized block content independently of parser-native metadata."""

    return _canonical_hash(
        {
            "kind": kind.value,
            "text": text,
            "table": table.model_dump(mode="json") if table is not None else None,
        }
    )


def canonical_block_id(
    *, native_node_id: str, kind: CanonicalBlockKind, content_sha256: str
) -> str:
    """Return a deterministic local block identity."""

    return "block-" + _canonical_hash(
        {
            "schema": "deepcritical-canonical-block-id-v1",
            "native_node_id": native_node_id,
            "kind": kind.value,
            "content_sha256": content_sha256,
        }
    )


def canonical_relationship_id(
    *,
    kind: CanonicalRelationshipKind,
    status: CanonicalRelationshipStatus,
    source_block_id: str | None,
    target_block_ids: tuple[str, ...],
    declared_target_refs: tuple[str, ...],
    unresolved_target_refs: tuple[str, ...],
    reason_codes: tuple[str, ...],
    source_anchor: CanonicalSourceAnchor | None,
) -> str:
    """Return a deterministic relationship identity."""

    return "relationship-" + _canonical_hash(
        {
            "schema": "deepcritical-canonical-relationship-id-v1",
            "kind": kind.value,
            "status": status.value,
            "source_block_id": source_block_id,
            "target_block_ids": target_block_ids,
            "declared_target_refs": declared_target_refs,
            "unresolved_target_refs": unresolved_target_refs,
            "reason_codes": reason_codes,
            "source_anchor": (
                source_anchor.model_dump(mode="json")
                if source_anchor is not None
                else None
            ),
        }
    )


def _declared_target_accounting(
    *,
    declared_target_refs: tuple[str, ...],
    resolved_target_count: int,
    unresolved_target_refs: tuple[str, ...],
) -> tuple[tuple[str, ...], bool]:
    """Reconcile target counts without comparing TEI and Docling references."""

    remaining_unresolved = Counter(unresolved_target_refs)
    pending_declared: list[str] = []
    for declared_ref in declared_target_refs:
        if remaining_unresolved[declared_ref] == 0:
            pending_declared.append(declared_ref)
        else:
            remaining_unresolved[declared_ref] -= 1
    missing = tuple(pending_declared[resolved_target_count:])
    return missing, resolved_target_count == len(pending_declared)


def canonical_relationship_status(
    *,
    source_block_id: str | None,
    target_block_ids: tuple[str, ...],
    declared_target_refs: tuple[str, ...],
    unresolved_target_refs: tuple[str, ...],
    reason_codes: tuple[str, ...],
) -> CanonicalRelationshipStatus:
    """Derive relationship resolution from its immutable mapping evidence."""

    _, targets_fully_accounted = _declared_target_accounting(
        declared_target_refs=declared_target_refs,
        resolved_target_count=len(target_block_ids),
        unresolved_target_refs=unresolved_target_refs,
    )
    complete = (
        source_block_id is not None
        and bool(target_block_ids)
        and targets_fully_accounted
        and not unresolved_target_refs
        and not reason_codes
    )
    if complete:
        return CanonicalRelationshipStatus.RESOLVED
    if source_block_id is not None and target_block_ids:
        return CanonicalRelationshipStatus.PARTIAL
    return CanonicalRelationshipStatus.UNRESOLVED


def canonical_diagnostic_id(
    *,
    severity: CanonicalDiagnosticSeverity,
    code: str,
    message: str,
    product_id: str | None,
    native_node_ids: tuple[str, ...],
) -> str:
    """Return a deterministic mapping-diagnostic identity."""

    return "canonical-diagnostic-" + _canonical_hash(
        {
            "schema": "deepcritical-canonical-diagnostic-id-v1",
            "severity": severity.value,
            "code": code,
            "message": message,
            "product_id": product_id,
            "native_node_ids": native_node_ids,
        }
    )


def canonical_view_id(payload: Mapping[str, Any]) -> str:
    """Return the deterministic identity of a complete canonical view payload."""

    identity = dict(payload)
    identity.pop("view_id", None)
    return "canonical-view-" + _canonical_hash(identity)


def canonical_document_bytes(view: CanonicalDocumentView) -> bytes:
    """Serialize a validated canonical view with deterministic JSON bytes."""

    payload = view.model_dump(mode="json")
    try:
        validated = CanonicalDocumentView.model_validate(payload)
    except ValidationError as exc:
        raise InvalidCanonicalDocumentError(
            "canonical document contract is invalid at serialization"
        ) from exc
    return json.dumps(
        validated.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def load_canonical_document(data: bytes) -> CanonicalDocumentView:
    """Dispatch on schema version before validating a canonical document blob."""

    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCanonicalDocumentError(
            "canonical document is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise InvalidCanonicalDocumentError("canonical document must be a JSON object")
    version = payload.get("schema_version")
    if version != CANONICAL_DOCUMENT_SCHEMA_VERSION:
        raise UnsupportedCanonicalDocumentVersionError(
            f"unsupported canonical document schema_version {version!r}"
        )
    try:
        return CanonicalDocumentView.model_validate(payload)
    except ValidationError as exc:
        raise InvalidCanonicalDocumentError(
            "canonical document contract is invalid"
        ) from exc


def build_canonical_document_view(
    *,
    artifact: DocumentArtifact,
    docling_document: dict[str, Any],
    docling_product: DataProductRef,
    content_span_set: ContentSpanSet,
    source_products: tuple[DataProductRef, ...],
    configuration: CanonicalizationConfig,
    scholarly_overlay: ScholarlyAlignmentOverlay | None = None,
    integrity_report: Mapping[str, Any] | None = None,
) -> CanonicalDocumentView:
    """Build a deterministic canonical view from immutable native products."""

    _enforce_native_resource_limits(docling_document)
    try:
        validate_docling_reference_definitions(docling_document)
    except ValueError as exc:
        raise CanonicalDocumentError(str(exc)) from exc
    try:
        content_span_set = ContentSpanSet.model_validate(
            content_span_set.model_dump(mode="python")
        )
    except ValidationError as exc:
        raise CanonicalDocumentError("content span set contract is invalid") from exc
    products = tuple(dict.fromkeys(source_products))
    product_ids = {product.product_id for product in products}
    if docling_product.product_id not in product_ids:
        raise CanonicalDocumentError(
            "Docling product must be a canonical source product"
        )
    matching_docling_products = tuple(
        product
        for product in products
        if product.product_id == docling_product.product_id
    )
    if matching_docling_products != (docling_product,):
        raise CanonicalDocumentError(
            "Docling input must exactly match its canonical source product"
        )
    if content_span_set.representation_product_id != docling_product.product_id:
        raise CanonicalDocumentError("content spans target a different Docling product")
    if docling_product.name != "docling_document":
        raise CanonicalDocumentError("canonical primary product must be Docling output")
    if tuple(product for product in products if product.name == "docling_document") != (
        docling_product,
    ):
        raise CanonicalDocumentError(
            "canonical inputs require exactly one Docling product"
        )
    if artifact.artifact_id not in docling_product.source_artifact_ids:
        raise CanonicalDocumentError("Docling product targets a different artifact")
    if content_span_set.artifact_id != artifact.artifact_id:
        raise CanonicalDocumentError("content spans target a different artifact")
    if content_span_set.processing_run_id != docling_product.producer_run_id:
        raise CanonicalDocumentError("content spans target a different Docling run")
    if any(
        artifact.artifact_id not in product.source_artifact_ids for product in products
    ):
        raise CanonicalDocumentError(
            "canonical source-product lineage targets a different artifact"
        )
    content_span_products = tuple(
        product for product in products if product.name == "content_spans"
    )
    if len(content_span_products) != 1:
        raise CanonicalDocumentError(
            "canonical inputs require exactly one content-span product"
        )
    if content_span_products[0].producer_run_id != content_span_set.processing_run_id:
        raise CanonicalDocumentError(
            "content-span product targets a different processing run"
        )
    unsupported_source_names = tuple(
        product.name
        for product in products
        if product.name not in _CANONICAL_SOURCE_PRODUCT_NAMES
    )
    if unsupported_source_names:
        raise CanonicalDocumentError(
            "canonical inputs contain products that are not used by schema v1"
        )
    grobid_products = tuple(
        product for product in products if product.name == "grobid_tei"
    )
    alignment_products = tuple(
        product for product in products if product.name == "alignment_overlay"
    )
    if scholarly_overlay is None:
        if grobid_products or alignment_products:
            raise CanonicalDocumentError(
                "scholarly source products require an alignment overlay"
            )
    elif len(grobid_products) != 1 or len(alignment_products) != 1:
        raise CanonicalDocumentError(
            "a scholarly overlay requires exactly one GROBID and alignment product"
        )
    integrity_products = tuple(
        product for product in products if product.name == "content_integrity_overlay"
    )
    parsed_integrity_report: ContentIntegrityReport | None = None
    if integrity_report is None:
        if integrity_products:
            raise CanonicalDocumentError(
                "an integrity source product requires its persisted report"
            )
    else:
        if len(integrity_products) != 1:
            raise CanonicalDocumentError(
                "an integrity report requires exactly one integrity source product"
            )
        try:
            parsed_integrity_report = parse_content_integrity_report(integrity_report)
        except ValueError as exc:
            raise CanonicalDocumentError(
                "content-integrity report contract is invalid"
            ) from exc
        if parsed_integrity_report.document_sha256 != docling_document_sha256(
            docling_document
        ):
            raise CanonicalDocumentError(
                "integrity report targets a different Docling document"
            )
        if parsed_integrity_report.scholarly_overlay_present is not (
            scholarly_overlay is not None
        ):
            raise CanonicalDocumentError(
                "integrity report scholarly-overlay provenance does not match inputs"
            )

    diagnostics: list[CanonicalMappingDiagnostic] = []
    if parsed_integrity_report is not None:
        integrity_product = integrity_products[0]
        diagnostics.extend(
            _diagnostic(
                severity=(
                    CanonicalDiagnosticSeverity.ERROR
                    if issue.severity is QualitySeverity.ERROR
                    else CanonicalDiagnosticSeverity.WARNING
                ),
                code=issue.code,
                message=issue.message,
                product_id=integrity_product.product_id,
                native_node_ids=(issue.item_ref,) if issue.item_ref is not None else (),
            )
            for issue in parsed_integrity_report.issues
        )
    nodes = _collect_native_nodes(docling_document, diagnostics, docling_product)
    canonical_index = {node.canonical_ref: node for node in nodes}
    declared_index = {node.declared_ref: node for node in nodes}

    span_node_by_id: dict[str, _NativeNode] = {}
    for span in content_span_set.spans:
        node_id = span.representation_anchor.node_id
        node = canonical_index.get(node_id)
        if node is None:
            node = declared_index.get(node_id)
            if node is None:
                raise CanonicalDocumentError(
                    f"content span native node {node_id!r} does not exist"
                )
        native_text = _native_text(node.item)
        start = span.representation_anchor.char_start
        end = span.representation_anchor.char_end
        if end > len(native_text):
            raise CanonicalDocumentError(
                f"content span for {node_id!r} exceeds native text bounds"
            )
        if sha256_bytes(native_text[start:end].encode("utf-8")) != span.content_sha256:
            raise CanonicalDocumentError(
                f"content span for {node_id!r} does not match native text"
            )
        span_node_by_id[span.span_id] = node

    def resolve(reference: str, *, context: str) -> _NativeNode | None:
        direct = canonical_index.get(reference)
        if direct is not None:
            return direct
        declared = declared_index.get(reference)
        if declared is not None:
            return declared
        diagnostics.append(
            _diagnostic(
                severity=CanonicalDiagnosticSeverity.ERROR,
                code="UNRESOLVED_NATIVE_REFERENCE",
                message=f"{context} reference {reference!r} did not resolve uniquely",
                product_id=docling_product.product_id,
                native_node_ids=(reference,),
            )
        )
        return None

    body_roots: list[str] = []
    body_root_set: set[str] = set()
    declared_parents: dict[str, list[str | None]] = {
        node.canonical_ref: [] for node in nodes
    }
    body = docling_document.get("body")
    if isinstance(body, Mapping):
        for reference in _reference_values(body.get("children")):
            node = resolve(reference, context="document body")
            if node is not None:
                if node.canonical_ref not in body_root_set:
                    body_root_set.add(node.canonical_ref)
                    body_roots.append(node.canonical_ref)
                declared_parents[node.canonical_ref].append(None)
    else:
        diagnostics.append(
            _diagnostic(
                severity=CanonicalDiagnosticSeverity.ERROR,
                code="MISSING_DOCUMENT_BODY",
                message="Docling document body is missing or malformed",
                product_id=docling_product.product_id,
            )
        )
    furniture_roots: list[str] = []
    furniture_root_set: set[str] = set()
    furniture = docling_document.get("furniture")
    if isinstance(furniture, Mapping):
        for reference in _reference_values(furniture.get("children")):
            node = resolve(reference, context="document furniture")
            if node is not None:
                if node.canonical_ref not in furniture_root_set:
                    furniture_root_set.add(node.canonical_ref)
                    furniture_roots.append(node.canonical_ref)
                declared_parents[node.canonical_ref].append(None)
    elif furniture is not None:
        diagnostics.append(
            _diagnostic(
                severity=CanonicalDiagnosticSeverity.ERROR,
                code="INVALID_DOCUMENT_FURNITURE",
                message="Docling document furniture is malformed",
                product_id=docling_product.product_id,
            )
        )

    explicit_parent_by_ref: dict[str, str | None] = {}
    declared_child_order: dict[str, list[str]] = {}
    for parent in nodes:
        for child_reference in _child_references(parent.item, table=parent.table):
            child = resolve(child_reference, context="document hierarchy child")
            if child is not None:
                declared_parents[child.canonical_ref].append(parent.canonical_ref)
                declared_child_order.setdefault(parent.canonical_ref, []).append(
                    child.canonical_ref
                )
        parent_reference = _parent_reference(parent.item)
        if parent_reference is None:
            continue
        if parent_reference in {"#/body", "#/furniture"}:
            top_level_roots = (
                body_roots if parent_reference == "#/body" else furniture_roots
            )
            top_level_root_set = (
                body_root_set if parent_reference == "#/body" else furniture_root_set
            )
            explicit_parent_by_ref[parent.canonical_ref] = None
            declared_parents[parent.canonical_ref].append(None)
            if parent.canonical_ref not in top_level_root_set:
                top_level_root_set.add(parent.canonical_ref)
                top_level_roots.append(parent.canonical_ref)
            continue
        resolved_parent = resolve(
            parent_reference,
            context="document hierarchy parent",
        )
        if resolved_parent is not None:
            explicit_parent_by_ref[parent.canonical_ref] = resolved_parent.canonical_ref
            declared_parents[parent.canonical_ref].append(resolved_parent.canonical_ref)

    parent_by_ref: dict[str, str | None] = {}
    for node in nodes:
        candidates = tuple(dict.fromkeys(declared_parents[node.canonical_ref]))
        if len(candidates) > 1:
            diagnostics.append(
                _diagnostic(
                    severity=CanonicalDiagnosticSeverity.ERROR,
                    code="MULTIPLE_NATIVE_PARENTS",
                    message="Native node has conflicting parent declarations",
                    product_id=docling_product.product_id,
                    native_node_ids=(node.canonical_ref,),
                )
            )
        if node.canonical_ref in explicit_parent_by_ref:
            parent_by_ref[node.canonical_ref] = explicit_parent_by_ref[
                node.canonical_ref
            ]
        elif candidates:
            parent_by_ref[node.canonical_ref] = candidates[0]
        else:
            parent_by_ref[node.canonical_ref] = None

    order_index = {node.canonical_ref: index for index, node in enumerate(nodes)}
    visit_state: dict[str, int] = {}

    for node in nodes:
        reference = node.canonical_ref
        if visit_state.get(reference, 0) == 2:
            continue
        path: list[str] = []
        path_index: dict[str, int] = {}
        while visit_state.get(reference, 0) != 2:
            if visit_state.get(reference, 0) == 1:
                cycle = path[path_index[reference] :]
                breaker = min(cycle, key=order_index.__getitem__)
                parent_by_ref[breaker] = None
                diagnostics.append(
                    _diagnostic(
                        severity=CanonicalDiagnosticSeverity.ERROR,
                        code="CYCLIC_NATIVE_HIERARCHY",
                        message="Native hierarchy cycle was broken deterministically",
                        product_id=docling_product.product_id,
                        native_node_ids=tuple(cycle),
                    )
                )
                break
            visit_state[reference] = 1
            path_index[reference] = len(path)
            path.append(reference)
            parent = parent_by_ref[reference]
            if parent is None:
                break
            reference = parent
        for path_reference in reversed(path):
            visit_state[path_reference] = 2

    children_by_ref: dict[str, list[str]] = {node.canonical_ref: [] for node in nodes}
    child_sets_by_ref: dict[str, set[str]] = {
        node.canonical_ref: set() for node in nodes
    }
    for parent in nodes:
        children = children_by_ref[parent.canonical_ref]
        child_set = child_sets_by_ref[parent.canonical_ref]
        for child_ref in declared_child_order.get(parent.canonical_ref, []):
            if (
                parent_by_ref[child_ref] == parent.canonical_ref
                and child_ref not in child_set
            ):
                child_set.add(child_ref)
                children.append(child_ref)
    for child in nodes:
        parent_ref = parent_by_ref[child.canonical_ref]
        if parent_ref is None:
            continue
        child_set = child_sets_by_ref[parent_ref]
        if child.canonical_ref not in child_set:
            child_set.add(child.canonical_ref)
            children_by_ref[parent_ref].append(child.canonical_ref)

    reachable: set[str] = set()

    for reference in (*body_roots, *furniture_roots):
        if parent_by_ref.get(reference) is None:
            pending = [reference]
            while pending:
                reachable_reference = pending.pop()
                if reachable_reference in reachable:
                    continue
                reachable.add(reachable_reference)
                pending.extend(reversed(children_by_ref.get(reachable_reference, ())))
    orphaned = tuple(
        node.canonical_ref for node in nodes if node.canonical_ref not in reachable
    )
    if (isinstance(body, Mapping) or isinstance(furniture, Mapping)) and orphaned:
        diagnostics.append(
            _diagnostic(
                severity=CanonicalDiagnosticSeverity.WARNING,
                code="ORPHANED_NATIVE_NODES",
                message="Native nodes are unreachable from document body or furniture",
                product_id=docling_product.product_id,
                native_node_ids=orphaned,
            )
        )

    ordered_nodes: list[_NativeNode] = []
    visited: set[str] = set()

    for reference in (
        *body_roots,
        *furniture_roots,
        *(node.canonical_ref for node in nodes),
    ):
        if parent_by_ref.get(reference) is None:
            pending = [reference]
            while pending:
                ordered_reference = pending.pop()
                if ordered_reference in visited:
                    continue
                visited.add(ordered_reference)
                ordered_nodes.append(canonical_index[ordered_reference])
                pending.extend(reversed(children_by_ref.get(ordered_reference, ())))

    spans_by_node: dict[str, list[ContentSpan]] = {}
    for span in content_span_set.spans:
        span_node = span_node_by_id[span.span_id]
        spans_by_node.setdefault(span_node.canonical_ref, []).append(span)

    grobid_product = grobid_products[0] if grobid_products else None
    scholarly_by_node: dict[str, list[CanonicalSourceAnchor]] = {}
    if scholarly_overlay is not None:
        scholarly_product = cast("DataProductRef", grobid_product)
        for record in scholarly_overlay.records:
            annotation = record.annotation
            if record.status is AlignmentStatus.ALIGNED and record.docling_item_ref:
                aligned_node = resolve(
                    record.docling_item_ref,
                    context="scholarly alignment",
                )
                if aligned_node is None:
                    continue
                scholarly_by_node.setdefault(aligned_node.canonical_ref, []).append(
                    CanonicalSourceAnchor(
                        role=CanonicalAnchorRole.SCHOLARLY,
                        product_id=scholarly_product.product_id,
                        node_id=annotation.tei_path,
                    )
                )
            else:
                diagnostics.append(
                    _diagnostic(
                        severity=CanonicalDiagnosticSeverity.WARNING,
                        code="UNRESOLVED_SCHOLARLY_ANCHOR",
                        message="GROBID annotation did not align to a canonical block",
                        product_id=scholarly_product.product_id,
                        native_node_ids=(annotation.tei_path,),
                    )
                )

    provisional: list[dict[str, Any]] = []
    block_id_by_ref: dict[str, str] = {}
    for ordinal, node in enumerate(ordered_nodes):
        content_hash = canonical_block_content_sha256(
            kind=node.kind,
            text=node.text,
            table=node.table,
        )
        block_id = canonical_block_id(
            native_node_id=node.canonical_ref,
            kind=node.kind,
            content_sha256=content_hash,
        )
        block_id_by_ref[node.canonical_ref] = block_id
        anchors: list[CanonicalSourceAnchor] = [
            CanonicalSourceAnchor(
                role=CanonicalAnchorRole.PRIMARY,
                product_id=docling_product.product_id,
                node_id=node.canonical_ref,
                char_start=0 if node.text else None,
                char_end=len(_native_text(node.item)) if node.text else None,
            )
        ]
        matched_spans = spans_by_node.get(node.canonical_ref, [])
        for span in matched_spans:
            anchors.append(
                CanonicalSourceAnchor(
                    role=CanonicalAnchorRole.SOURCE,
                    product_id=span.representation_anchor.product_id,
                    node_id=span.representation_anchor.node_id,
                    char_start=span.representation_anchor.char_start,
                    char_end=span.representation_anchor.char_end,
                    source_locator=span.source_locator,
                )
            )
        anchors.extend(scholarly_by_node.get(node.canonical_ref, []))
        if node.text and not matched_spans:
            diagnostics.append(
                _diagnostic(
                    severity=CanonicalDiagnosticSeverity.WARNING,
                    code="MISSING_SOURCE_SPAN",
                    message="Canonical text block has no source-locator span",
                    product_id=docling_product.product_id,
                    native_node_ids=(node.declared_ref,),
                )
            )
        provisional.append(
            {
                "block_id": block_id,
                "native_node_id": node.canonical_ref,
                "native_label": node.native_label,
                "kind": node.kind,
                "ordinal": ordinal,
                "text": node.text,
                "table": node.table,
                "content_sha256": content_hash,
                "source_anchors": tuple(_unique_anchors(anchors)),
            }
        )

    def parent_block_id(native_node_id: str) -> str | None:
        parent_ref = parent_by_ref[native_node_id]
        return block_id_by_ref[parent_ref] if parent_ref is not None else None

    blocks = tuple(
        CanonicalBlock(
            **values,
            parent_block_id=parent_block_id(values["native_node_id"]),
            child_block_ids=tuple(
                block_id_by_ref[child]
                for child in children_by_ref.get(values["native_node_id"], [])
            ),
        )
        for values in provisional
    )

    relationships = _build_relationships(
        parsed_integrity_report,
        resolve=resolve,
        block_id_by_ref=block_id_by_ref,
        scholarly_overlay=scholarly_overlay,
        grobid_product=grobid_product,
        diagnostics=diagnostics,
    )
    title = next(
        (block.text for block in blocks if block.kind is CanonicalBlockKind.TITLE),
        None,
    )
    if title is None:
        name = docling_document.get("name")
        title = normalize_canonical_text(name) if isinstance(name, str) else None
        title = title or None
    payload: dict[str, Any] = {
        "schema_version": CANONICAL_DOCUMENT_SCHEMA_VERSION,
        "artifact_id": artifact.artifact_id,
        "source_sha256": artifact.source_sha256,
        "normalization_policy": configuration.text_normalization,
        "anchoring_policy": configuration.anchoring_policy,
        "metadata": CanonicalDocumentMetadata(
            title=title,
            media_type=artifact.media_type,
            identifiers=artifact.identifiers,
        ),
        "source_products": products,
        "root_block_ids": tuple(
            block.block_id for block in blocks if block.parent_block_id is None
        ),
        "blocks": blocks,
        "relationships": relationships,
        "diagnostics": tuple(_unique_diagnostics(diagnostics)),
    }
    payload["view_id"] = canonical_view_id(_json_payload(payload))
    return CanonicalDocumentView.model_validate(payload)


def _collect_native_nodes(
    document: Mapping[str, Any],
    diagnostics: list[CanonicalMappingDiagnostic],
    docling_product: DataProductRef,
) -> tuple[_NativeNode, ...]:
    nodes: list[_NativeNode] = []
    for collection in _NATIVE_NODE_COLLECTIONS:
        values = document.get(collection, [])
        if not isinstance(values, list):
            diagnostics.append(
                _diagnostic(
                    severity=CanonicalDiagnosticSeverity.ERROR,
                    code="INVALID_NATIVE_COLLECTION",
                    message=f"Docling collection {collection!r} is not a list",
                    product_id=docling_product.product_id,
                    native_node_ids=(f"#/{collection}",),
                )
            )
            continue
        for index, raw_item in enumerate(values):
            canonical_ref = f"#/{collection}/{index}"
            if not isinstance(raw_item, Mapping):
                diagnostics.append(
                    _diagnostic(
                        severity=CanonicalDiagnosticSeverity.ERROR,
                        code="INVALID_NATIVE_NODE",
                        message="Docling collection item is not an object",
                        product_id=docling_product.product_id,
                        native_node_ids=(canonical_ref,),
                    )
                )
                continue
            item = dict(raw_item)
            declared = item.get("self_ref")
            declared_ref = (
                declared if isinstance(declared, str) and declared else canonical_ref
            )
            native_label_value = item.get("label")
            native_label = (
                native_label_value
                if isinstance(native_label_value, str) and native_label_value
                else collection.rstrip("s")
            )
            kind = _block_kind(collection, native_label)
            raw_text = _native_text(item)
            text = normalize_canonical_text(raw_text) or None
            table = (
                _canonical_table(
                    item,
                    native_node_id=canonical_ref,
                    diagnostics=diagnostics,
                    docling_product=docling_product,
                )
                if kind is CanonicalBlockKind.TABLE
                else None
            )
            nodes.append(
                _NativeNode(
                    canonical_ref=canonical_ref,
                    declared_ref=declared_ref,
                    collection=collection,
                    item=item,
                    native_label=native_label,
                    kind=kind,
                    text=text,
                    table=table,
                )
            )
    return tuple(nodes)


def _enforce_native_resource_limits(document: Mapping[str, Any]) -> None:
    """Reject oversized native graphs before decoding nodes or references.

    Counts intentionally use raw declaration slots.  Malformed collection
    entries and malformed references therefore cannot evade admission bounds
    merely because later semantic decoding would ignore them.
    """

    collections: list[tuple[str, list[Any]]] = []
    node_count = 0
    for collection in _NATIVE_NODE_COLLECTIONS:
        values = document.get(collection, [])
        if not isinstance(values, list):
            continue
        node_count += len(values)
        if node_count > MAX_CANONICAL_NATIVE_NODES:
            raise CanonicalDocumentError(
                "NATIVE_NODE_LIMIT_EXCEEDED: Docling document exceeds the native "
                "node limit"
            )
        collections.append((collection, values))

    edge_count = 0

    def add_edges(count: int) -> None:
        nonlocal edge_count
        edge_count += count
        if edge_count > MAX_CANONICAL_HIERARCHY_EDGES:
            raise CanonicalDocumentError(
                "NATIVE_HIERARCHY_EDGE_LIMIT_EXCEEDED: Docling document exceeds "
                "the hierarchy edge limit"
            )

    for root_name in ("body", "furniture"):
        root = document.get(root_name)
        if isinstance(root, Mapping):
            add_edges(_raw_reference_slot_count(root, "children"))

    for collection, values in collections:
        for raw_item in values:
            if not isinstance(raw_item, Mapping):
                continue
            add_edges(_raw_reference_slot_count(raw_item, "children"))
            if "parent" in raw_item:
                add_edges(1)
            if collection == "tables":
                _count_raw_table_reference_slots(raw_item, add_edges=add_edges)


def _raw_reference_slot_count(container: Mapping[str, Any], field: str) -> int:
    if field not in container:
        return 0
    value = container[field]
    return len(value) if isinstance(value, list) else 1


def _count_raw_table_reference_slots(
    item: Mapping[str, Any],
    *,
    add_edges: Callable[[int], None],
) -> None:
    data = item.get("data")
    if not isinstance(data, Mapping):
        return
    if "table_cells" in data:
        table_cells = data["table_cells"]
        if not isinstance(table_cells, list):
            return
        for raw_cell in table_cells:
            if isinstance(raw_cell, Mapping) and "ref" in raw_cell:
                add_edges(1)
        return
    grid = data.get("grid")
    if not isinstance(grid, list):
        return
    for row in grid:
        if not isinstance(row, list):
            continue
        for raw_cell in row:
            if isinstance(raw_cell, Mapping) and "ref" in raw_cell:
                add_edges(1)


def _block_kind(collection: str, label: str) -> CanonicalBlockKind:
    normalized = label.casefold().replace("-", "_").replace(" ", "_")
    if collection == "tables":
        return CanonicalBlockKind.TABLE
    if collection == "pictures":
        return CanonicalBlockKind.FIGURE
    if collection == "formulas":
        return CanonicalBlockKind.FORMULA
    if collection == "groups":
        return CanonicalBlockKind.GROUP
    aliases = {
        "title": CanonicalBlockKind.TITLE,
        "document_title": CanonicalBlockKind.TITLE,
        "section_header": CanonicalBlockKind.SECTION,
        "section": CanonicalBlockKind.SECTION,
        "paragraph": CanonicalBlockKind.PARAGRAPH,
        "text": CanonicalBlockKind.PARAGRAPH,
        "list_item": CanonicalBlockKind.LIST_ITEM,
        "caption": CanonicalBlockKind.CAPTION,
        "formula": CanonicalBlockKind.FORMULA,
        "citation": CanonicalBlockKind.CITATION,
        "reference": CanonicalBlockKind.REFERENCE,
    }
    return aliases.get(normalized, CanonicalBlockKind.OTHER)


def _native_text(item: Mapping[str, Any]) -> str:
    for key in ("text", "orig", "caption_text", "name"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


class _RangeMaximumTree:
    """Range-add/range-maximum tree used by the table rectangle sweep."""

    __slots__ = ("_lazy", "_maximum", "_size")

    def __init__(self, size: int) -> None:
        self._size = size
        self._maximum = [0] * (4 * size)
        self._lazy = [0] * (4 * size)

    def add(self, left: int, right: int, delta: int) -> None:
        self._add(1, 0, self._size - 1, left, right, delta)

    def maximum(self, left: int, right: int) -> int:
        return self._query(1, 0, self._size - 1, left, right)

    def _add(
        self,
        node: int,
        node_left: int,
        node_right: int,
        query_left: int,
        query_right: int,
        delta: int,
    ) -> None:
        if query_left <= node_left and node_right <= query_right:
            self._maximum[node] += delta
            self._lazy[node] += delta
            return
        midpoint = (node_left + node_right) // 2
        if query_left <= midpoint:
            self._add(
                node * 2,
                node_left,
                midpoint,
                query_left,
                query_right,
                delta,
            )
        if query_right > midpoint:
            self._add(
                node * 2 + 1,
                midpoint + 1,
                node_right,
                query_left,
                query_right,
                delta,
            )
        self._maximum[node] = self._lazy[node] + max(
            self._maximum[node * 2],
            self._maximum[node * 2 + 1],
        )

    def _query(
        self,
        node: int,
        node_left: int,
        node_right: int,
        query_left: int,
        query_right: int,
    ) -> int:
        if query_left <= node_left and node_right <= query_right:
            return self._maximum[node]
        midpoint = (node_left + node_right) // 2
        result = 0
        if query_left <= midpoint:
            result = self._query(
                node * 2,
                node_left,
                midpoint,
                query_left,
                query_right,
            )
        if query_right > midpoint:
            result = max(
                result,
                self._query(
                    node * 2 + 1,
                    midpoint + 1,
                    node_right,
                    query_left,
                    query_right,
                ),
            )
        return self._lazy[node] + result


def _table_cells_overlap(cells: tuple[CanonicalTableCell, ...]) -> bool:
    """Detect overlap among half-open cell rectangles without expanding spans."""

    if len(cells) < 2:
        return False
    column_boundaries = sorted(
        {
            boundary
            for cell in cells
            for boundary in (
                cell.column_index,
                cell.column_index + cell.column_span,
            )
        }
    )
    column_positions = {
        boundary: index for index, boundary in enumerate(column_boundaries)
    }
    events: list[tuple[int, int, int, int]] = []
    for cell in cells:
        left = column_positions[cell.column_index]
        right = column_positions[cell.column_index + cell.column_span] - 1
        events.append((cell.row_index, 1, left, right))
        events.append((cell.row_index + cell.row_span, 0, left, right))
    tree = _RangeMaximumTree(len(column_boundaries) - 1)
    for _, event_kind, left, right in sorted(events):
        if event_kind == 0:
            tree.add(left, right, -1)
        else:
            if tree.maximum(left, right) > 0:
                return True
            tree.add(left, right, 1)
    return False


class _TableCellError(ValueError):
    pass


def _canonical_table(
    item: Mapping[str, Any],
    *,
    native_node_id: str,
    diagnostics: list[CanonicalMappingDiagnostic],
    docling_product: DataProductRef,
) -> CanonicalTable:
    data = item.get("data")
    if not isinstance(data, Mapping):
        _append_table_diagnostic(
            diagnostics,
            docling_product=docling_product,
            native_node_id=native_node_id,
            code="INVALID_TABLE_DATA",
            message="Docling table data is missing or is not an object",
        )
        return CanonicalTable()

    row_count = _declared_table_dimension(data, "num_rows")
    column_count = _declared_table_dimension(data, "num_cols")
    entries: list[tuple[Any, int, int, bool, str]] = []
    authoritative_cells = "table_cells" in data
    if authoritative_cells:
        table_cells = data["table_cells"]
        if not isinstance(table_cells, list):
            _append_table_diagnostic(
                diagnostics,
                docling_product=docling_product,
                native_node_id=native_node_id,
                code="INVALID_TABLE_CELLS",
                message="Docling table_cells is not a list",
            )
            return CanonicalTable()
        _enforce_table_cell_limit(len(table_cells))
        entries.extend(
            (raw_cell, 0, 0, True, f"table_cells[{index}]")
            for index, raw_cell in enumerate(table_cells)
        )
    else:
        grid = data.get("grid")
        if not isinstance(grid, list):
            _append_table_diagnostic(
                diagnostics,
                docling_product=docling_product,
                native_node_id=native_node_id,
                code="INVALID_TABLE_GRID",
                message="Docling table grid is missing or is not a list",
            )
            return CanonicalTable()
        if len(grid) > MAX_CANONICAL_TABLE_AXIS:
            raise CanonicalDocumentError(
                "TABLE_LIMIT_EXCEEDED: structured table exceeds the row limit"
            )
        total_cells = 0
        for row_index, raw_row in enumerate(grid):
            if not isinstance(raw_row, list):
                _append_table_diagnostic(
                    diagnostics,
                    docling_product=docling_product,
                    native_node_id=native_node_id,
                    code="INVALID_TABLE_GRID_ROW",
                    message=f"Docling table grid row {row_index} is not a list",
                )
                continue
            if len(raw_row) > MAX_CANONICAL_TABLE_AXIS:
                raise CanonicalDocumentError(
                    "TABLE_LIMIT_EXCEEDED: structured table exceeds the column limit"
                )
            total_cells += len(raw_row)
            _enforce_table_cell_limit(total_cells)
            entries.extend(
                (
                    raw_cell,
                    row_index,
                    column_index,
                    False,
                    f"grid[{row_index}][{column_index}]",
                )
                for column_index, raw_cell in enumerate(raw_row)
            )

    parsed_cells: list[CanonicalTableCell] = []
    for raw_cell, fallback_row, fallback_column, authoritative, context in entries:
        try:
            parsed_cells.append(
                _canonical_table_cell(
                    raw_cell,
                    fallback_row=fallback_row,
                    fallback_column=fallback_column,
                    authoritative=authoritative,
                )
            )
        except _TableCellError as exc:
            _append_table_diagnostic(
                diagnostics,
                docling_product=docling_product,
                native_node_id=native_node_id,
                code="INVALID_TABLE_CELL",
                message=f"Docling {context} is invalid: {exc}",
            )

    if not parsed_cells:
        _append_table_diagnostic(
            diagnostics,
            docling_product=docling_product,
            native_node_id=native_node_id,
            code="EMPTY_TABLE_DATA",
            message="Docling table contains no valid structured cells",
        )
        return CanonicalTable()

    unique_cells: dict[tuple[int, int], CanonicalTableCell] = {}
    for cell in parsed_cells:
        coordinate = (cell.row_index, cell.column_index)
        existing = unique_cells.get(coordinate)
        if existing == cell and not authoritative_cells:
            continue
        if existing is not None:
            raise CanonicalDocumentError(
                "structured table contains conflicting cells at one coordinate"
            )
        unique_cells[coordinate] = cell
    cells = tuple(
        sorted(
            unique_cells.values(),
            key=lambda cell: (cell.row_index, cell.column_index),
        )
    )
    if _table_cells_overlap(cells):
        raise CanonicalDocumentError(
            "structured table contains overlapping cell extents"
        )

    inferred_rows = max(cell.row_index + cell.row_span for cell in cells)
    inferred_columns = max(cell.column_index + cell.column_span for cell in cells)
    if row_count is not None and row_count < inferred_rows:
        raise CanonicalDocumentError(
            "INVALID_TABLE_DIMENSIONS: structured table cells exceed the "
            "declared row dimension"
        )
    if column_count is not None and column_count < inferred_columns:
        raise CanonicalDocumentError(
            "INVALID_TABLE_DIMENSIONS: structured table cells exceed the "
            "declared column dimension"
        )
    return CanonicalTable(
        row_count=inferred_rows if row_count is None else row_count,
        column_count=inferred_columns if column_count is None else column_count,
        cells=cells,
    )


def _append_table_diagnostic(
    diagnostics: list[CanonicalMappingDiagnostic],
    *,
    docling_product: DataProductRef,
    native_node_id: str,
    code: str,
    message: str,
) -> None:
    diagnostics.append(
        _diagnostic(
            severity=CanonicalDiagnosticSeverity.ERROR,
            code=code,
            message=message,
            product_id=docling_product.product_id,
            native_node_ids=(native_node_id,),
        )
    )


def _enforce_table_cell_limit(count: int) -> None:
    if count > MAX_CANONICAL_TABLE_CELLS:
        raise CanonicalDocumentError(
            "TABLE_LIMIT_EXCEEDED: structured table exceeds the cell limit"
        )


def _declared_table_dimension(data: Mapping[str, Any], field_name: str) -> int | None:
    if field_name not in data:
        return None
    value = data[field_name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CanonicalDocumentError(
            "INVALID_TABLE_DIMENSIONS: structured table "
            f"{field_name} is not a non-negative integer"
        )
    if value > MAX_CANONICAL_TABLE_AXIS:
        raise CanonicalDocumentError(
            "TABLE_LIMIT_EXCEEDED: structured table declared dimension exceeds "
            "the axis limit"
        )
    return value


def _canonical_table_cell(
    value: Any,
    *,
    fallback_row: int,
    fallback_column: int,
    authoritative: bool,
) -> CanonicalTableCell:
    if not isinstance(value, Mapping):
        if authoritative:
            raise _TableCellError("authoritative cells must be objects")
        if value is None:
            text = ""
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise _TableCellError("scalar cell values must be finite")
            text = str(value)
        else:
            raise _TableCellError("grid cells must be objects or scalar values")
        return CanonicalTableCell(
            text=normalize_canonical_text(text),
            row_index=fallback_row,
            column_index=fallback_column,
        )

    raw_text = value.get("text")
    if not isinstance(raw_text, str):
        raise _TableCellError("cell text must be a string")
    row_index = _table_coordinate(
        value,
        "start_row_offset_idx",
        fallback=None if authoritative else fallback_row,
    )
    column_index = _table_coordinate(
        value,
        "start_col_offset_idx",
        fallback=None if authoritative else fallback_column,
    )
    row_span = _table_span(
        value,
        start=row_index,
        end_field="end_row_offset_idx",
        span_fields=("row_span",),
    )
    column_span = _table_span(
        value,
        start=column_index,
        end_field="end_col_offset_idx",
        span_fields=("column_span", "col_span"),
    )
    flags: dict[str, bool] = {}
    for flag in ("column_header", "row_header", "row_section", "fillable"):
        flag_value = value.get(flag, False)
        if not isinstance(flag_value, bool):
            raise _TableCellError(f"{flag} must be a boolean")
        flags[flag] = flag_value
    native_ref = None
    if "ref" in value and value["ref"] is not None:
        native_ref = _reference_value(value["ref"])
        if native_ref is None or not native_ref.strip():
            raise _TableCellError("ref must contain a non-empty native reference")
    return CanonicalTableCell(
        text=normalize_canonical_text(raw_text),
        row_index=row_index,
        column_index=column_index,
        row_span=row_span,
        column_span=column_span,
        native_ref=native_ref,
        **flags,
    )


def _table_coordinate(
    value: Mapping[str, Any],
    field_name: str,
    *,
    fallback: int | None,
) -> int:
    if field_name not in value:
        if fallback is None:
            raise _TableCellError(f"{field_name} is required")
        return fallback
    coordinate = value[field_name]
    if (
        not isinstance(coordinate, int)
        or isinstance(coordinate, bool)
        or coordinate < 0
    ):
        raise _TableCellError(f"{field_name} must be a non-negative integer")
    if coordinate >= MAX_CANONICAL_TABLE_AXIS:
        raise CanonicalDocumentError(
            "TABLE_LIMIT_EXCEEDED: structured table cell coordinate exceeds "
            "the axis limit"
        )
    return coordinate


def _table_span(
    value: Mapping[str, Any],
    *,
    start: int,
    end_field: str,
    span_fields: tuple[str, ...],
) -> int:
    raw_spans = [value[field] for field in span_fields if field in value]
    if len(raw_spans) > 1 and any(item != raw_spans[0] for item in raw_spans[1:]):
        raise _TableCellError(f"{span_fields!r} disagree")
    explicit_span = None
    if raw_spans:
        candidate = raw_spans[0]
        if (
            not isinstance(candidate, int)
            or isinstance(candidate, bool)
            or candidate <= 0
        ):
            raise _TableCellError(f"{span_fields[0]} must be a positive integer")
        if candidate > MAX_CANONICAL_TABLE_AXIS:
            raise CanonicalDocumentError(
                "TABLE_LIMIT_EXCEEDED: structured table cell span exceeds "
                "the axis limit"
            )
        explicit_span = candidate
    end_span = None
    if end_field in value:
        end = value[end_field]
        if not isinstance(end, int) or isinstance(end, bool) or end <= start:
            raise _TableCellError(f"{end_field} must be an exclusive integer end")
        if end > MAX_CANONICAL_TABLE_AXIS:
            raise CanonicalDocumentError(
                "TABLE_LIMIT_EXCEEDED: structured table cell end exceeds the axis limit"
            )
        end_span = end - start
    if explicit_span is not None and end_span is not None and explicit_span != end_span:
        raise _TableCellError(f"{end_field} disagrees with {span_fields[0]}")
    span = end_span or explicit_span or 1
    if start + span > MAX_CANONICAL_TABLE_AXIS:
        raise CanonicalDocumentError(
            "TABLE_LIMIT_EXCEEDED: structured table cell extent exceeds the axis limit"
        )
    return span


def _child_references(
    item: Mapping[str, Any],
    *,
    table: CanonicalTable | None,
) -> tuple[str, ...]:
    references = list(_reference_values(item.get("children")))
    if table is not None:
        references.extend(
            cell.native_ref for cell in table.cells if cell.native_ref is not None
        )
    return tuple(dict.fromkeys(references))


def _parent_reference(item: Mapping[str, Any]) -> str | None:
    parent = item.get("parent")
    if not isinstance(parent, Mapping):
        return None
    reference = parent.get("$ref")
    return reference if isinstance(reference, str) and reference else None


def _reference_values(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    references: list[str] = []
    for entry in value:
        if isinstance(entry, Mapping):
            reference = entry.get("$ref")
            if isinstance(reference, str) and reference:
                references.append(reference)
    return tuple(references)


def _reference_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        reference = value.get("$ref")
        return reference if isinstance(reference, str) and reference else None
    return value if isinstance(value, str) and value else None


def _build_relationships(
    integrity_report: ContentIntegrityReport | None,
    *,
    resolve: Any,
    block_id_by_ref: Mapping[str, str],
    scholarly_overlay: ScholarlyAlignmentOverlay | None,
    grobid_product: DataProductRef | None,
    diagnostics: list[CanonicalMappingDiagnostic],
) -> tuple[CanonicalRelationship, ...]:
    records = integrity_report.records if integrity_report is not None else ()
    annotation_by_id = (
        {
            record.annotation.annotation_id: record.annotation
            for record in scholarly_overlay.records
        }
        if scholarly_overlay is not None
        else {}
    )
    relationships: list[CanonicalRelationship] = []
    for record in records:
        relation_kind = (
            CanonicalRelationshipKind.CITES
            if record.kind is ContentIntegrityKind.CITATION
            else CanonicalRelationshipKind.HAS_CAPTION
        )
        source_ref = record.source_docling_item_ref or record.source_ref
        source_node = resolve(source_ref, context="relationship source")
        source_block_id = (
            block_id_by_ref.get(source_node.canonical_ref)
            if source_node is not None
            else None
        )
        target_ids: list[str] = []
        failed_target_refs: list[str] = []
        for target_ref in record.resolved_docling_item_refs:
            target_node = resolve(target_ref, context="relationship target")
            if target_node is not None:
                target_ids.append(block_id_by_ref[target_node.canonical_ref])
                continue
            failed_target_refs.append(target_ref)
        declared = record.declared_target_refs
        unresolved = tuple(
            dict.fromkeys(
                (
                    *record.unresolved_target_refs,
                    *failed_target_refs,
                )
            )
        )
        reasons = record.reason_codes
        unique_target_ids = tuple(dict.fromkeys(target_ids))
        missing_declared_refs, _ = _declared_target_accounting(
            declared_target_refs=declared,
            resolved_target_count=len(unique_target_ids),
            unresolved_target_refs=unresolved,
        )
        unresolved = tuple(dict.fromkeys((*unresolved, *missing_declared_refs)))
        status = canonical_relationship_status(
            source_block_id=source_block_id,
            target_block_ids=unique_target_ids,
            declared_target_refs=declared,
            unresolved_target_refs=unresolved,
            reason_codes=reasons,
        )
        source_anchor = None
        annotation_id = record.source_ref
        annotation = annotation_by_id.get(annotation_id)
        if annotation is not None and grobid_product is not None:
            source_anchor = CanonicalSourceAnchor(
                role=CanonicalAnchorRole.SCHOLARLY,
                product_id=grobid_product.product_id,
                node_id=annotation.tei_path,
            )
        relationship_id = canonical_relationship_id(
            kind=relation_kind,
            status=status,
            source_block_id=source_block_id,
            target_block_ids=unique_target_ids,
            declared_target_refs=declared,
            unresolved_target_refs=unresolved,
            reason_codes=reasons,
            source_anchor=source_anchor,
        )
        relationships.append(
            CanonicalRelationship(
                relationship_id=relationship_id,
                kind=relation_kind,
                status=status,
                source_block_id=source_block_id,
                target_block_ids=unique_target_ids,
                declared_target_refs=declared,
                unresolved_target_refs=unresolved,
                reason_codes=reasons,
                source_anchor=source_anchor,
            )
        )
        if status is not CanonicalRelationshipStatus.RESOLVED:
            diagnostics.append(
                _diagnostic(
                    severity=CanonicalDiagnosticSeverity.WARNING,
                    code="UNRESOLVED_CANONICAL_RELATIONSHIP",
                    message=f"{relation_kind.value} relationship did not resolve completely",
                    product_id=(
                        grobid_product.product_id
                        if annotation is not None and grobid_product
                        else None
                    ),
                    native_node_ids=tuple(
                        dict.fromkeys((str(source_ref), *declared, *unresolved))
                    ),
                )
            )
    return tuple(relationships)


def _diagnostic(
    *,
    severity: CanonicalDiagnosticSeverity,
    code: str,
    message: str,
    product_id: str | None = None,
    native_node_ids: tuple[str, ...] = (),
) -> CanonicalMappingDiagnostic:
    diagnostic_id = canonical_diagnostic_id(
        severity=severity,
        code=code,
        message=message,
        product_id=product_id,
        native_node_ids=native_node_ids,
    )
    return CanonicalMappingDiagnostic(
        diagnostic_id=diagnostic_id,
        severity=severity,
        code=code,
        message=message,
        product_id=product_id,
        native_node_ids=native_node_ids,
    )


def _unique_anchors(
    anchors: list[CanonicalSourceAnchor],
) -> tuple[CanonicalSourceAnchor, ...]:
    unique: dict[str, CanonicalSourceAnchor] = {}
    for anchor in anchors:
        key = json.dumps(
            anchor.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        unique.setdefault(key, anchor)
    return tuple(unique.values())


def _unique_diagnostics(
    diagnostics: list[CanonicalMappingDiagnostic],
) -> tuple[CanonicalMappingDiagnostic, ...]:
    return tuple(
        {diagnostic.diagnostic_id: diagnostic for diagnostic in diagnostics}.values()
    )


def _json_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        json.loads(
            json.dumps(
                payload,
                default=lambda value: value.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    )


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


__all__ = [
    "CANONICAL_ADAPTER_VERSION",
    "CANONICAL_ANCHORING_POLICY",
    "CANONICAL_COMPONENT_CAPABILITY",
    "CANONICAL_COMPONENT_ID",
    "CANONICAL_COMPONENT_VERSION",
    "CANONICAL_DOCUMENT_SCHEMA_VERSION",
    "CANONICAL_TEXT_NORMALIZATION",
    "MAX_CANONICAL_HIERARCHY_EDGES",
    "MAX_CANONICAL_NATIVE_NODES",
    "MAX_CANONICAL_TABLE_AXIS",
    "MAX_CANONICAL_TABLE_CELLS",
    "CanonicalAnchorRole",
    "CanonicalBlock",
    "CanonicalBlockKind",
    "CanonicalDiagnosticSeverity",
    "CanonicalDocumentError",
    "CanonicalDocumentMetadata",
    "CanonicalDocumentView",
    "CanonicalMappingDiagnostic",
    "CanonicalRelationship",
    "CanonicalRelationshipKind",
    "CanonicalRelationshipStatus",
    "CanonicalSourceAnchor",
    "CanonicalTable",
    "CanonicalTableCell",
    "CanonicalizationConfig",
    "InvalidCanonicalDocumentError",
    "UnsupportedCanonicalDocumentVersionError",
    "build_canonical_document_view",
    "canonical_block_content_sha256",
    "canonical_block_id",
    "canonical_document_bytes",
    "canonical_invocation_configuration",
    "canonical_relationship_id",
    "canonical_relationship_status",
    "canonical_view_id",
    "load_canonical_document",
    "normalize_canonical_text",
]
