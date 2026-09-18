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
    version="1.3.0",
)


# ============================================================
# AGENT INITIALIZATION
# ============================================================

decision_agent = DecisionAgent()
medical_document_agent = MedicalDocumentAgent()


# ============================================================
# ROOT ENDPOINT
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

@app.post(
    "/claims/analyze",
    response_model=ClaimDecision,
)
def analyze_claim(claim_case: ClaimCase) -> ClaimDecision:

    try:

        # ----------------------------------------------------
        # 1. MEDICAL DOCUMENT ANALYSIS
        # ----------------------------------------------------

        possible_pre_existing = False
        medical_review_required = False

        medical_analysis_results = []

        for document in claim_case.medical_documents:

            medical_result = (
                medical_document_agent.analyze_document(
                    document_text=document.document_text,
                    claim_diagnosis=(
                        claim_case.claim.diagnosis
                    ),
                    treatment_type=(
                        claim_case.claim.treatment_type
                    ),
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

        # ----------------------------------------------------
        # 2. DETERMINE PRE-EXISTING STATUS
        # ----------------------------------------------------

        if possible_pre_existing:
            pre_existing_confirmed = True
        else:
            pre_existing_confirmed = None

        # Previous insurance coverage information is not
        # currently represented in the ClaimCase schema.
        previous_coverage_verified = False

        # ----------------------------------------------------
        # 3. DECISION AGENT
        # ----------------------------------------------------

        result = decision_agent.make_decision(
            claim_id=(
                claim_case.claim.claim_id
            ),

            diagnosis=(
                claim_case.claim.diagnosis
            ),

            treatment_type=(
                claim_case.claim.treatment_type
            ),

            claimed_amount=(
                claim_case.claim.claimed_amount
            ),

            continuous_coverage_months=(
                claim_case.policy.continuous_coverage_months
            ),

            pre_existing_confirmed=(
                pre_existing_confirmed
            ),

            previous_coverage_verified=(
                previous_coverage_verified
            ),

            experimental=(
                claim_case.claim.experimental
            ),
        )

        # ----------------------------------------------------
        # 4. STRUCTURED FINDINGS
        # ----------------------------------------------------

        findings = []

        findings.append(
            f"Claim diagnosis: "
            f"{claim_case.claim.diagnosis}"
        )

        findings.append(
            f"Treatment type: "
            f"{claim_case.claim.treatment_type}"
        )

        findings.append(
            f"Continuous coverage: "
            f"{claim_case.policy.continuous_coverage_months} "
            f"months"
        )

        if claim_case.claim.experimental:
            findings.append(
                "Treatment is identified as experimental "
                "or unproven."
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

        # ----------------------------------------------------
        # 5. MISSING EVIDENCE
        # ----------------------------------------------------

        missing_evidence = []

        if not claim_case.medical_documents:

            missing_evidence.append(
                "Medical documentation"
            )

        if not possible_pre_existing:

            missing_evidence.append(
                "Medical verification of pre-existing "
                "condition status"
            )

        if not previous_coverage_verified:

            missing_evidence.append(
                "Previous continuous insurance coverage "
                "or portability information"
            )

        if claim_case.claim.experimental:

            missing_evidence.append(
                "Verification of the applicable policy "
                "exclusion for experimental or unproven treatment"
            )

        # ----------------------------------------------------
        # 6. LIMITATIONS
        # ----------------------------------------------------

        limitations = [
            (
                "The system provides a preliminary "
                "AI-assisted claim assessment and does "
                "not replace human claim review."
            ),

            (
                "Medical document analysis is based on "
                "the extracted document text provided "
                "to the system."
            ),

            (
                "Previous insurance coverage or portability "
                "information is not currently verified "
                "automatically."
            ),
        ]

        # ----------------------------------------------------
        # 7. PROCESSING TRACE
        # ----------------------------------------------------

        trace = [
            "Claim case received",
            "Medical documents analyzed",
            "Policy evidence retrieved using "
            "hybrid retrieval",
            "Decision Agent evaluated claim "
            "against retrieved policy evidence",
            "Preliminary claim decision generated",
        ]

        # ----------------------------------------------------
        # 8. MEDICAL REVIEW INFORMATION
        # ----------------------------------------------------

        if medical_review_required:

            result["reason"] += (
                " Medical document analysis also "
                "identified information requiring "
                "human verification."
            )

        # ----------------------------------------------------
        # 9. RETURN STRUCTURED RESPONSE
        # ----------------------------------------------------

        return ClaimDecision(
            claim_id=result["claim_id"],

            decision=result["decision"],

            approved_amount=(
                result["approved_amount"]
            ),

            reason=result["reason"],

            policy_evidence=(
                result["policy_evidence"]
            ),

            confidence=result["confidence"],

            findings=findings,

            limitations=limitations,

            missing_evidence=missing_evidence,

            trace=trace,
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Claim analysis failed: "
                f"{str(error)}"
            ),
        )