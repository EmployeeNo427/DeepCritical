# Canonical service-capture fixture v1

This fixture freezes the smallest reviewed native-output slice from the
successful all-real document-processing workflow run `31587722084` at exact
revision `a2c91ddcd7f4c38cd011a11487e02296c3c356a8`.

The source is the repository's deterministic, raster-only synthetic paper.
`docling_document.json` is the exact canonical JSON emitted by the pinned
Docling service. `grobid.tei.xml` is the exact TEI emitted by the pinned GROBID
fallback after the real OCRmyPDF derivative made the raster input searchable.
The primary GROBID attempt on the raster source failed and is not selected.

`manifest.json` is a deliberately sanitized allowlist. It records the
successful workflow and evidence artifact, digest-addressed runtime identities,
attestation contracts, accepted stage outcomes, exact file hashes, and replay
expectations. It omits credentials, invocation and workload identifiers,
timestamps, raw service responses, OCR derivatives, runtime attestations, and
the live canonical output.

Offline tests reconstruct content spans, scholarly alignment, content
integrity, and the canonical view from these three immutable inputs. At the
production alignment threshold (`0.72`), the captured OCR readings yield three
explicitly unaligned GROBID annotations. The replay therefore requires no
invented scholarly anchors and preserves three
`UNRESOLVED_SCHOLARLY_ANCHOR` diagnostics.

The intentionally omitted OCR derivative means the fixture cannot truthfully
reconstruct the full durable live producer chain. Tests use the content-addressed
store for canonical serialization and then dispatch through
`load_canonical_document`; they do not claim a verified
`read_canonical_document` replay.
