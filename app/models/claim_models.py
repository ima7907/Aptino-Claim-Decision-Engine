from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    patient_id: str = Field(
        ...,
        description="Unique patient identifier",
    )

    age: int = Field(
        ...,
        ge=0,
        le=120,
        description="Patient age",
    )

    gender: Optional[str] = Field(
        default=None,
        description="Patient gender",
    )


class PolicyInfo(BaseModel):
    policy_id: str = Field(
        ...,
        description="Unique policy identifier",
    )

    policy_name: str = Field(
        ...,
        description="Name of the insurance policy",
    )

    policy_start_date: str = Field(
        ...,
        description="Policy start date in YYYY-MM-DD format",
    )

    continuous_coverage_months: int = Field(
        ...,
        ge=0,
        description="Number of months of continuous policy coverage",
    )


class ClaimInfo(BaseModel):
    claim_id: str = Field(
        ...,
        description="Unique claim identifier",
    )

    diagnosis: str = Field(
        ...,
        description="Medical diagnosis",
    )

    treatment_type: str = Field(
        ...,
        description="Type of treatment or hospitalization",
    )

    claimed_amount: float = Field(
        ...,
        ge=0,
        description="Amount claimed by the patient",
    )

    hospitalization_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of hospitalization days",
    )


class MedicalDocument(BaseModel):
    document_id: str = Field(
        ...,
        description="Unique medical document identifier",
    )

    document_type: str = Field(
        ...,
        description="Type of medical document",
    )

    document_text: str = Field(
        ...,
        description="Extracted text from the medical document",
    )


class ClaimCase(BaseModel):
    patient: PatientInfo

    policy: PolicyInfo

    claim: ClaimInfo

    medical_documents: List[MedicalDocument] = Field(
        default_factory=list,
        description="Medical documents related to the claim",
    )


class PolicyEvidence(BaseModel):
    chunk_id: str

    page_number: Optional[int] = None

    text: str

    relevance_score: float


class ClaimDecision(BaseModel):
    claim_id: str

    decision: Literal[
        "APPROVED",
        "REJECTED",
        "NEEDS_REVIEW",
    ]

    approved_amount: float = Field(
        default=0.0,
        ge=0,
    )

    reason: str

    policy_evidence: List[PolicyEvidence] = Field(
        default_factory=list,
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )