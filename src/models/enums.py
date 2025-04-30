"""Shared enum definitions used by both SQLAlchemy models and Pydantic schemas."""

import enum


class JobStatus(str, enum.Enum):
    pending = "pending"
    extracting = "extracting"
    verifying = "verifying"
    aggregating = "aggregating"
    completed = "completed"
    failed = "failed"


class ClaimType(str, enum.Enum):
    molecular_property = "molecular_property"
    target_interaction = "target_interaction"
    citation = "citation"
    clinical_outcome = "clinical_outcome"
    pharmacokinetic = "pharmacokinetic"
    mechanism_of_action = "mechanism_of_action"
    general_biomedical = "general_biomedical"


class Verdict(str, enum.Enum):
    verified = "verified"
    refuted = "refuted"
    unverifiable = "unverifiable"
    partially_supported = "partially_supported"


class Severity(str, enum.Enum):
    clean = "clean"
    minor = "minor"
    major = "major"
    critical = "critical"
