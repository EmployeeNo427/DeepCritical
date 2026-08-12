"""Pinned native-parser identities admitted by canonical schema v1."""

DOCLING_COMPONENT_VERSION = "2.96.1"
DOCLING_SERVE_VERSION = "1.21.0"
DOCLING_CONTAINER_IMAGE = "quay.io/docling-project/docling-serve-cpu:v1.21.0"
DOCLING_IMAGE_TAG = "docling-serve-cpu:v1.21.0"

OUTPUT_POLICY_SCHEMA_VERSION = "deepcritical-document-output-policy-v2"
RUNTIME_ATTESTATION_SCHEMA_VERSION = "deepcritical-runtime-attestation-v1"
REMOTE_ATTESTATION_CONTRACT_VERSION = (
    "deepcritical-authenticated-runtime-attestation-reporter-v1"
)
OCR_DIGEST_RUNNER_VERSION = "deepcritical-container-ocr-runner-v1"

GROBID_COMPONENT_VERSION = "0.9.0"
GROBID_CONTAINER_IMAGE = "deepcritical/grobid:0.9.0-full-p0-c2"
GROBID_IMAGE_TAG = "0.9.0-full-p0-c2"

OCR_COMPONENT_VERSION = "17.4.1"
OCR_CONTAINER_IMAGE = "jbarlow83/ocrmypdf:v17.4.1"
OCR_IMAGE_TAG = "ocrmypdf:v17.4.1"

__all__ = [
    "DOCLING_COMPONENT_VERSION",
    "DOCLING_CONTAINER_IMAGE",
    "DOCLING_IMAGE_TAG",
    "DOCLING_SERVE_VERSION",
    "GROBID_COMPONENT_VERSION",
    "GROBID_CONTAINER_IMAGE",
    "GROBID_IMAGE_TAG",
    "OCR_COMPONENT_VERSION",
    "OCR_CONTAINER_IMAGE",
    "OCR_DIGEST_RUNNER_VERSION",
    "OCR_IMAGE_TAG",
    "OUTPUT_POLICY_SCHEMA_VERSION",
    "REMOTE_ATTESTATION_CONTRACT_VERSION",
    "RUNTIME_ATTESTATION_SCHEMA_VERSION",
]
