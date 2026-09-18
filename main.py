from typing import Dict

from fastapi import FastAPI, HTTPException

from app.agents.decision_agent import DecisionAgent
from app.agents.medical_document_agent import MedicalDocumentAgent

from app.models.claim_models import (
    ClaimCase,
    ClaimDecision,
)


app = FastAPI(
    title="Aptino Claim Decision Engine",
    description=(
        "AI-assisted health insurance claim decision engine "
        "using hybrid policy retrieval and medical document analysis."
    ),
    version="1.1.0",
)


decision_agent = DecisionAgent()
medical_document_agent = MedicalDocumentAgent()


@app.get("/")
def root() -> Dict[str, str]:
    """
    Health check endpoint.
    """

    return {
        "message": "Aptino Claim Decision Engine is running",
        "status": "success",
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    API health check.
    """

    return {
        "status": "healthy",
    }


@app.post(
    "/claims/analyze",
    response_model=ClaimDecision,
)
def analyze_claim(
    claim_case: ClaimCase,
) -> ClaimDecision:
    """
    Analyze an insurance claim using medical document
    analysis and policy-based decision logic.
    """

    try:

        # ---------------------------------------------------------
        # 1. ANALYZE MEDICAL DOCUMENTS
        # ---------------------------------------------------------

        possible_pre_existing = False
        medical_review_required = False

        medical_analysis_results = []

        for document in claim_case.medical_documents:

            medical_result = (
                medical_document_agent.analyze_document(
                    document_text=document.document_text,
                    claim_diagnosis=claim_case.claim.diagnosis,
                    treatment_type=claim_case.claim.treatment_type,
                )
            )

            medical_analysis_results.append(
                medical_result
            )

            if medical_result.get(
                "possible_pre_existing_evidence",
                False,
            ):
                possible_pre_existing = True

            if medical_result.get(
                "document_status"
            ) == "NEEDS_REVIEW":
                medical_review_required = True

        # ---------------------------------------------------------
        # 2. DETERMINE PRE-EXISTING STATUS
        # ---------------------------------------------------------

        if possible_pre_existing:

            pre_existing_confirmed = True

        else:

            pre_existing_confirmed = None

        # ---------------------------------------------------------
        # 3. VERIFY PREVIOUS COVERAGE
        # ---------------------------------------------------------

        previous_coverage_verified = False

        # ---------------------------------------------------------
        # 4. RUN DECISION AGENT
        # ---------------------------------------------------------

        result = decision_agent.make_decision(
            claim_id=claim_case.claim.claim_id,
            diagnosis=claim_case.claim.diagnosis,
            treatment_type=claim_case.claim.treatment_type,
            claimed_amount=claim_case.claim.claimed_amount,
            continuous_coverage_months=(
                claim_case.policy.continuous_coverage_months
            ),
            pre_existing_confirmed=(
                pre_existing_confirmed
            ),
            previous_coverage_verified=(
                previous_coverage_verified
            ),
        )

        # ---------------------------------------------------------
        # 5. ADD MEDICAL DOCUMENT REVIEW INFORMATION
        # ---------------------------------------------------------

        if medical_review_required:

            medical_flags = []

            for medical_result in medical_analysis_results:

                medical_flags.extend(
                    medical_result.get(
                        "review_flags",
                        [],
                    )
                )

            result["reason"] += (
                " Medical document analysis also identified "
                "information requiring human verification."
            )

        # ---------------------------------------------------------
        # 6. RETURN API RESPONSE
        # ---------------------------------------------------------

        return ClaimDecision(
            claim_id=result["claim_id"],
            decision=result["decision"],
            approved_amount=result["approved_amount"],
            reason=result["reason"],
            policy_evidence=result["policy_evidence"],
            confidence=result["confidence"],
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Claim analysis failed: {str(error)}",
        )