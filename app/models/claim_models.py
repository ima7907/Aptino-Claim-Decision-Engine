from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    patient_id: str
    age: int
    gender: Optional[str] = None


class PolicyInfo(BaseModel):
    policy_id: str
    policy_name: str
    policy_start_date: date
    continuous_coverage_months: int


class ClaimInfo(BaseModel):
    claim_id: str
    diagnosis: str
    treatment_type: str
    claimed_amount: float
    hospitalization_days: Optional[int] = None

    # Indicates whether the treatment is identified
    # as experimental or unproven.
    experimental: bool = False


class MedicalDocument(BaseModel):
    document_id: str
    document_type: str
    document_text: str


class ClaimCase(BaseModel):
    patient: PatientInfo
    policy: PolicyInfo
    claim: ClaimInfo
    medical_documents: List[MedicalDocument] = Field(
        default_factory=list
    )


class PolicyEvidence(BaseModel):
    chunk_id: str
    page_number: int
    text: str
    relevance_score: float


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
        default_factory=list,
        description="Key findings identified during claim analysis",
    )

    limitations: List[str] = Field(
        default_factory=list,
        description="Known limitations of the automated analysis",
    )

    missing_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence that should be verified or provided",
    )

    trace: List[str] = Field(
        default_factory=list,
        description="High-level processing steps performed by the system",
    )