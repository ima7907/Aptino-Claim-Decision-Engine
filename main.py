from typing import Dict

from fastapi import FastAPI, HTTPException

from app.agents.decision_agent import DecisionAgent
from app.models.claim_models import (
    ClaimCase,
    ClaimDecision,
)


app = FastAPI(
    title="Aptino Claim Decision Engine",
    description=(
        "AI-assisted health insurance claim decision engine "
        "using hybrid policy retrieval."
    ),
    version="1.0.0",
)


decision_agent = DecisionAgent()


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
    Analyze an insurance claim and return a preliminary decision.
    """

    try:
        result = decision_agent.make_decision(
            claim_id=claim_case.claim.claim_id,
            diagnosis=claim_case.claim.diagnosis,
            treatment_type=claim_case.claim.treatment_type,
            claimed_amount=claim_case.claim.claimed_amount,
            continuous_coverage_months=(
                claim_case.policy.continuous_coverage_months
            ),
        )

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