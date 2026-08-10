# ADR 0001: Project-owned canonical document view

- Status: Accepted
- Date: 2026-07-27
- Implemented: 2026-08-05

## Context

DeepCritical currently preserves Docling, JATS, BioC, GROBID, and OCR outputs.
Calling a serialized `DoclingDocument` canonical would make downstream
annotations and processor-independent contracts depend permanently on one
processor's data model. Rewriting native outputs into a shared structure during
ingestion would also discard useful native fidelity and make upgrades harder to
audit.

## Decision

DeepCritical provides the versioned, project-owned
`CanonicalDocumentView` contract
`deepcritical-canonical-document-view-v1`.

Each processor will preserve its native output unchanged as a typed
`DataProductRef`. The allow-listed `canonical-document-view` component runs
after native alignment and integrity validation. Its deterministic adapter
creates an immutable `canonical_document_view` product containing:

- document-order blocks and an explicit parent/child hierarchy;
- closed block kinds for titles, sections, paragraphs, lists, tables, figures,
  captions, formulas, citations, references, groups, and unclassified content;
- stable block IDs derived from native node identity, block kind, and normalized
  content hash;
- exact anchors to immutable parser-native products, including character ranges
  and PDF, JATS, or BioC source locators where available;
- normalized tables whose authoritative structured cells retain coordinates,
  spans, header/section flags, fillable state, and rich-cell references, plus
  source-level metadata;
- caption and citation relationships whose `resolved`, `partial`, or
  `unresolved` status is derived from their immutable mapping evidence; and
- stable mapping diagnostics whenever native structure cannot be represented
  without ambiguity.

Version 1 accepts only the persisted normalization policy
`unicode-nfc-collapse-whitespace-v1` and anchoring policy
`source-spans-and-native-nodes-v1`. Unknown fields and policy names fail
configuration validation. Loaders dispatch on `schema_version` before model
validation, and every block, relationship, diagnostic, and complete view
revalidates its content-derived identity. Parent and child declarations from
all native node collections are reconciled into one ordered hierarchy;
conflicting declarations, cycles, and missing nodes produce diagnostics.
Immutable tuples and read-only metadata mappings protect nested state, and the
store rebuilds the complete model immediately before deterministic
serialization so unvalidated copies or stale identities cannot become durable.

Native products remain immutable after the canonical view is introduced.
`docling_document` remains a native Docling product rather than being relabelled
as canonical. `ContentSpan.representation_anchor` still identifies the exact
product, native node, and character range it targets, and
`DoclingItemLocator` remains valid where a source format has no stable source
coordinate contract.

Scientific annotations are deliberately excluded from the canonical document.
When introduced, they must be separate immutable `AnnotationSet` products that
identify the exact canonical-view product and block/span IDs they target. This
prevents reprocessing from silently moving labels and prevents scientific
interpretation from changing document identity.

## Consequences

Downstream code must use adapters when it needs a processor-independent view,
while audit and debugging code can retain full access to native representations.
The compiled reference pipeline now includes this explicit conversion stage and
records its non-empty normalization and anchoring configuration in pipeline and
output-policy provenance. The stage reconstructs the view exclusively from
persisted inputs after verifying each product's CAS bytes, source-artifact
lineage, and exact declaration on its durable producer run; its recorded inputs
must exactly equal the view's `source_products`. A mapping gap produces
inspectable diagnostics rather than an invented coordinate. This adds one
durable product and processing run, but avoids a repository-wide migration
whenever Docling or another processor changes its native schema.
