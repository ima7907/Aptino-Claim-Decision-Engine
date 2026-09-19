from typing import Any, Dict, List, Optional, TypedDict


class ClaimWorkflowState(TypedDict, total=False):
    """
    Shared structured state passed between the specialized agents.

    This keeps the workflow auditable without exposing hidden
    chain-of-thought.
    """

    # ---------------------------------------------------------
    # INPUT CLAIM DATA
    # ---------------------------------------------------------
    claim_id: str
    diagnosis: str
    treatment_type: str
    claimed_amount: float
    continuous_coverage_months: int
    pre_existing_confirmed: Optional[bool]
    previous_coverage_verified: bool
    experimental: bool

    # ---------------------------------------------------------
    # MEDICAL DOCUMENT AGENT OUTPUT
    # ---------------------------------------------------------
    medical_analysis: List[Dict[str, Any]]
    medical_review_required: bool
    possible_pre_existing_evidence: bool

    # ---------------------------------------------------------
    # POLICY ANALYSIS AGENT OUTPUT
    # ---------------------------------------------------------
    policy_query: str
    policy_evidence: List[Dict[str, Any]]
    retrieval_count: int

    # ---------------------------------------------------------
    # DECISION AGENT OUTPUT
    # ---------------------------------------------------------
    decision: str
    approved_amount: float
    reason: str
    confidence: float
    review_flags: List[str]

    # ---------------------------------------------------------
    # AUDITABLE EXECUTION TRACE
    # ---------------------------------------------------------
    trace: List[Dict[str, Any]]

    # ---------------------------------------------------------
    # VALIDATION / FINAL STATE
    # ---------------------------------------------------------
    validation_status: str
    validation_errors: List[str]
    missing_evidence: List[str]