from typing import Dict
import time

from fastapi import FastAPI, HTTPException

from app.models.claim_models import ClaimCase, ClaimDecision
from app.workflow import ClaimDecisionWorkflow


app = FastAPI(
    title="Aptino Claim Decision Engine",
    description=(
        "AI-assisted health insurance claim decision engine "
        "using hybrid policy retrieval, reranking, medical "
        "document analysis, and a structured multi-agent workflow."
    ),
    version="1.4.0",
)


# ============================================================
# MULTI-AGENT WORKFLOW
# ============================================================

workflow = ClaimDecisionWorkflow()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Aptino Claim Decision Engine is running",
        "status": "success",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check() -> Dict[str, str]:
    return {
        "status": "healthy"
    }


# ============================================================
# CLAIM ANALYSIS
# ============================================================

@app.post("/analyze", response_model=ClaimDecision)
@app.post("/claims/analyze", response_model=ClaimDecision)
def analyze_claim(claim_case: ClaimCase) -> ClaimDecision:

    start_time = time.perf_counter()

    try:

        # ------------------------------------------------------
        # Prepare medical documents for the workflow
        # ------------------------------------------------------

        medical_documents = [
            {
                "document_id": document.document_id,
                "document_type": document.document_type,
                "document_text": document.document_text,
            }
            for document in claim_case.medical_documents
        ]

        # ------------------------------------------------------
        # Run structured multi-agent workflow
        # ------------------------------------------------------

        workflow_state = workflow.run(
            claim_id=claim_case.claim.claim_id,
            diagnosis=claim_case.claim.diagnosis,
            treatment_type=claim_case.claim.treatment_type,
            claimed_amount=claim_case.claim.claimed_amount,
            continuous_coverage_months=(
                claim_case.policy.continuous_coverage_months
            ),
            medical_documents=medical_documents,
            pre_existing_confirmed=None,
            previous_coverage_verified=False,
            experimental=claim_case.claim.experimental,
        )

        # ------------------------------------------------------
        # Medical analysis information
        # ------------------------------------------------------

        medical_review_required = workflow_state.get(
            "medical_review_required",
            False,
        )

        possible_pre_existing = workflow_state.get(
            "possible_pre_existing_evidence",
            False,
        )

        # ------------------------------------------------------
        # Findings
        # ------------------------------------------------------

        findings = [
            f"Claim diagnosis: {claim_case.claim.diagnosis}",
            f"Treatment type: {claim_case.claim.treatment_type}",
            (
                "Continuous coverage: "
                f"{claim_case.policy.continuous_coverage_months} months"
            ),
        ]

        if claim_case.claim.experimental:
            findings.append(
                "Treatment is identified as experimental or unproven."
            )

        if possible_pre_existing:
            findings.append(
                "Medical document contains possible "
                "pre-existing medical history."
            )

        if claim_case.medical_documents:
            findings.append(
                f"{len(claim_case.medical_documents)} "
                "medical document(s) analyzed."
            )
        else:
            findings.append(
                "No medical documents were provided."
            )

        if medical_review_required:
            findings.append(
                "Medical document analysis identified "
                "information requiring human verification."
            )

        # ------------------------------------------------------
        # Missing evidence
        # ------------------------------------------------------

        missing_evidence = []

        if not claim_case.medical_documents:
            missing_evidence.append(
                "Medical documentation"
            )

        missing_evidence.append(
            "Medical verification of pre-existing condition status"
        )

        missing_evidence.append(
            "Previous continuous insurance coverage "
            "or portability information"
        )

        if claim_case.claim.experimental:
            missing_evidence.append(
                "Verification of the applicable policy exclusion "
                "for experimental or unproven treatment"
            )

        # ------------------------------------------------------
        # Limitations
        # ------------------------------------------------------

        limitations = [
            (
                "The system provides a preliminary AI-assisted "
                "claim assessment and does not replace human "
                "claim review."
            ),
            (
                "Medical document analysis is based on the "
                "extracted document text provided to the system."
            ),
            (
                "Previous insurance coverage or portability "
                "information is not currently verified automatically."
            ),
        ]

        # ------------------------------------------------------
        # Policy evidence
        # ------------------------------------------------------

        policy_evidence = workflow_state.get(
            "policy_evidence",
            [],
        )

        # ------------------------------------------------------
        # Execution trace
        # ------------------------------------------------------

        trace = workflow_state.get(
            "trace",
            [],
        )

        elapsed_time = time.perf_counter() - start_time

        trace.append(
            {
                "agent": "System",
                "action": "Completed claim analysis",
                "retrieval_count": workflow_state.get(
                    "retrieval_count",
                    0,
                ),
                "validation_status": workflow_state.get(
                    "validation_status",
                    "NOT_RUN",
                ),
                "elapsed_time_seconds": round(
                    elapsed_time,
                    4,
                ),
            }
        )

        # ------------------------------------------------------
        # Add medical review warning to reason
        # ------------------------------------------------------

        reason = workflow_state.get(
            "reason",
            "",
        )

        if medical_review_required:
            reason += (
                " Medical document analysis also identified "
                "information requiring human verification."
            )

        # ------------------------------------------------------
        # Return structured API response
        # ------------------------------------------------------

        return ClaimDecision(
            claim_id=workflow_state["claim_id"],
            decision=workflow_state["decision"],
            approved_amount=workflow_state["approved_amount"],
            reason=reason,
            policy_evidence=policy_evidence,
            confidence=workflow_state["confidence"],
            findings=findings,
            limitations=limitations,
            missing_evidence=missing_evidence,
            trace=trace,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Claim analysis failed: {str(error)}",
        )