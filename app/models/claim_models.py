from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================
# PATIENT INFORMATION
# ============================================================

class PatientInfo(BaseModel):
    patient_id: str
    age: int
    gender: Optional[str] = None


# ============================================================
# POLICY INFORMATION
# ============================================================

class PolicyInfo(BaseModel):
    policy_id: str
    policy_name: str
    policy_start_date: date
    continuous_coverage_months: int


# ============================================================
# CLAIM INFORMATION
# ============================================================

class ClaimInfo(BaseModel):
    claim_id: str
    diagnosis: str
    treatment_type: str
    claimed_amount: float
    hospitalization_days: Optional[int] = None
    experimental: bool = False


# ============================================================
# MEDICAL DOCUMENT
# ============================================================

class MedicalDocument(BaseModel):
    document_id: str
    document_type: str
    document_text: str


# ============================================================
# COMPLETE CLAIM CASE
# ============================================================

class ClaimCase(BaseModel):
    patient: PatientInfo
    policy: PolicyInfo
    claim: ClaimInfo
    medical_documents: List[MedicalDocument] = Field(
        default_factory=list
    )


# ============================================================
# POLICY EVIDENCE
# ============================================================

class PolicyEvidence(BaseModel):
    chunk_id: str
    page_number: int
    text: str
    relevance_score: float


# ============================================================
# FINAL CLAIM DECISION
# ============================================================

class ClaimDecision(BaseModel):
    claim_id: str

    decision: Literal[
        "APPROVED",
        "REJECTED",
        "NEEDS_REVIEW",
    ]

    approved_amount: float

    reason: str

    policy_evidence: List[PolicyEvidence] = Field(
        default_factory=list
    )

    confidence: float

    findings: List[str] = Field(
        default_factory=list
    )

    limitations: List[str] = Field(
        default_factory=list
    )

    missing_evidence: List[str] = Field(
        default_factory=list
    )

    # Structured multi-agent execution trace
    trace: List[Dict[str, Any]] = Field(
        default_factory=list
    )