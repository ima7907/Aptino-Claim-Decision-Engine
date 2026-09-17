from typing import Dict, List

from app.agents.policy_agent import PolicyAnalysisAgent


class DecisionAgent:
    """
    Agent responsible for making a preliminary claim decision
    using retrieved policy evidence and basic rule-based logic.
    """

    def __init__(self):
        print("Initializing Decision Agent...")

        self.policy_agent = PolicyAnalysisAgent()

    def make_decision(
        self,
        claim_id: str,
        diagnosis: str,
        treatment_type: str,
        claimed_amount: float,
        continuous_coverage_months: int,
    ) -> Dict:
        """
        Make a preliminary claim decision.

        This implementation uses the policy rule that
        pre-existing diseases require 48 months of
        continuous coverage.
        """

        policy_analysis = self.policy_agent.analyze_claim(
            diagnosis=diagnosis,
            treatment_type=treatment_type,
            continuous_coverage_months=continuous_coverage_months,
            top_k=5,
        )

        evidence: List[Dict] = policy_analysis.get(
            "evidence",
            []
        )

        evidence_text = " ".join(
            item.get("text", "")
            for item in evidence
        ).lower()

        contains_pre_existing_rule = (
            "pre-existing diseases" in evidence_text
            or "pre-existing disease" in evidence_text
            or "pre existing diseases" in evidence_text
        )

        contains_48_month_rule = (
            "48 months" in evidence_text
            or "48 month" in evidence_text
        )

        if (
            contains_pre_existing_rule
            and contains_48_month_rule
            and continuous_coverage_months < 48
        ):
            decision = "NEEDS_REVIEW"

            approved_amount = 0.0

            reason = (
                "The retrieved policy evidence indicates that "
                "pre-existing diseases are subject to a "
                "48-month continuous coverage requirement. "
                "The claim has only "
                f"{continuous_coverage_months} months of coverage. "
                "The claim requires manual review to verify whether "
                "the diagnosis qualifies as a pre-existing disease "
                "and whether any policy exceptions apply."
            )

        else:
            decision = "NEEDS_REVIEW"

            approved_amount = 0.0

            reason = (
                "The available policy evidence is not sufficient "
                "to automatically approve or reject the claim. "
                "Manual review is required."
            )

        formatted_evidence = []

        for item in evidence:
            formatted_evidence.append(
                {
                    "chunk_id": item.get("chunk_id"),
                    "page_number": item.get("page_number"),
                    "text": item.get("text", ""),
                    "relevance_score": item.get(
                        "relevance_score",
                        0.0,
                    ),
                }
            )

        return {
            "claim_id": claim_id,
            "decision": decision,
            "approved_amount": approved_amount,
            "reason": reason,
            "policy_evidence": formatted_evidence,
            "confidence": 0.75 if contains_48_month_rule else 0.40,
        }


if __name__ == "__main__":

    agent = DecisionAgent()

    result = agent.make_decision(
        claim_id="CLM-001",
        diagnosis="Diabetes",
        treatment_type="In-patient hospitalization",
        claimed_amount=50000.0,
        continuous_coverage_months=24,
    )

    print()
    print("Claim Decision Result")
    print("=" * 80)

    print(
        f"Claim ID: {result['claim_id']}"
    )

    print(
        f"Decision: {result['decision']}"
    )

    print(
        f"Approved Amount: "
        f"{result['approved_amount']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']}"
    )

    print()
    print("Reason:")
    print(result["reason"])

    print()
    print(
        "Policy Evidence Count:",
        len(result["policy_evidence"]),
    )