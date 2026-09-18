from typing import Dict, List, Optional

from app.agents.policy_agent import PolicyAnalysisAgent


class DecisionAgent:
    """
    Agent responsible for making a preliminary claim decision
    using retrieved policy evidence and rule-based decision logic.

    The agent does not automatically reject or approve a claim
    when important policy or medical information is missing.
    Such cases are routed to manual review.
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
        pre_existing_confirmed: Optional[bool] = None,
        previous_coverage_verified: bool = False,
        experimental: bool = False,
    ) -> Dict:
        """
        Make a preliminary claim decision.

        Parameters
        ----------
        claim_id:
            Unique claim identifier.

        diagnosis:
            Diagnosis or medical condition associated with the claim.

        treatment_type:
            Type of treatment or hospitalization.

        claimed_amount:
            Amount claimed by the policyholder.

        continuous_coverage_months:
            Number of months of continuous coverage.

        pre_existing_confirmed:
            Whether the available medical documentation has
            confirmed that the condition is pre-existing.

        previous_coverage_verified:
            Whether previous continuous insurance coverage
            has been verified.

        experimental:
            Whether the treatment is identified as experimental
            or unproven in the claim information.

        Returns
        -------
        Dict
            Preliminary claim decision with reason,
            evidence, confidence, and review flags.
        """

        # ---------------------------------------------------------
        # 1. INPUT VALIDATION
        # ---------------------------------------------------------

        if not claim_id:
            raise ValueError("claim_id is required.")

        if not diagnosis:
            raise ValueError("diagnosis is required.")

        if not treatment_type:
            raise ValueError("treatment_type is required.")

        if claimed_amount < 0:
            raise ValueError(
                "claimed_amount cannot be negative."
            )

        if continuous_coverage_months < 0:
            raise ValueError(
                "continuous_coverage_months cannot be negative."
            )

        # ---------------------------------------------------------
        # 2. RETRIEVE POLICY EVIDENCE
        # ---------------------------------------------------------

        policy_analysis = self.policy_agent.analyze_claim(
            diagnosis=diagnosis,
            treatment_type=treatment_type,
            continuous_coverage_months=continuous_coverage_months,
            top_k=5,
        )

        evidence: List[Dict] = policy_analysis.get(
            "evidence",
            [],
        )

        evidence_text = " ".join(
            item.get("text", "")
            for item in evidence
        ).lower()

        # ---------------------------------------------------------
        # 3. DETECT IMPORTANT POLICY RULES
        # ---------------------------------------------------------

        contains_pre_existing_rule = (
            "pre-existing diseases" in evidence_text
            or "pre-existing disease" in evidence_text
            or "pre existing diseases" in evidence_text
            or "pre existing disease" in evidence_text
        )

        contains_48_month_rule = (
            "48 months" in evidence_text
            or "48 month" in evidence_text
        )

        contains_previous_coverage_exception = (
            "previous continuous insurance coverage"
            in evidence_text
            or "continuous insurance coverage"
            in evidence_text
            or "portability"
            in evidence_text
        )

        contains_experimental_rule = (
            "experimental" in evidence_text
            or "unproven" in evidence_text
        )

        # ---------------------------------------------------------
        # 4. REVIEW FLAGS
        # ---------------------------------------------------------

        review_flags: List[str] = []

        if not diagnosis.strip():
            review_flags.append(
                "Diagnosis information is missing."
            )

        if not treatment_type.strip():
            review_flags.append(
                "Treatment information is missing."
            )

        if claimed_amount <= 0:
            review_flags.append(
                "Claimed amount should be verified."
            )

        if pre_existing_confirmed is None:
            review_flags.append(
                "Pre-existing disease status has not been medically verified."
            )

        if (
            contains_previous_coverage_exception
            and not previous_coverage_verified
        ):
            review_flags.append(
                "Previous insurance coverage or portability exception "
                "may require verification."
            )

        if not contains_48_month_rule:
            review_flags.append(
                "The retrieved evidence does not clearly contain "
                "the 48-month waiting-period rule."
            )

        # ---------------------------------------------------------
        # 5. DECISION LOGIC
        # ---------------------------------------------------------

        decision = "NEEDS_REVIEW"
        approved_amount = 0.0

        # ---------------------------------------------------------
        # CASE A:
        # Experimental / unproven treatment
        # ---------------------------------------------------------

        if experimental:
            reason = (
                "The claim identifies the treatment as experimental "
                "or unproven. The retrieved policy evidence should "
                "be reviewed for the applicable exclusion before a "
                "final claim decision is made."
            )

            review_flags.append(
                "Experimental or unproven treatment identified; "
                "policy exclusion requires verification."
            )

            confidence = 0.85

        # ---------------------------------------------------------
        # CASE B:
        # Pre-existing condition confirmed AND coverage < 48 months
        # ---------------------------------------------------------

        elif (
            pre_existing_confirmed is True
            and contains_pre_existing_rule
            and contains_48_month_rule
            and continuous_coverage_months < 48
        ):
            reason = (
                "The available policy evidence indicates that "
                "pre-existing diseases are subject to a 48-month "
                "continuous coverage requirement. The condition "
                "has been marked as pre-existing and the claim has "
                f"{continuous_coverage_months} months of coverage, "
                "which is below the identified waiting period. "
                "Manual review is required to verify the medical "
                "documentation and determine whether any applicable "
                "previous-coverage or portability exception applies."
            )

            confidence = 0.90

        # ---------------------------------------------------------
        # CASE C:
        # Pre-existing condition confirmed AND coverage >= 48 months
        # ---------------------------------------------------------

        elif (
            pre_existing_confirmed is True
            and contains_pre_existing_rule
            and contains_48_month_rule
            and continuous_coverage_months >= 48
        ):
            reason = (
                "The available policy evidence identifies a "
                "48-month continuous coverage requirement for "
                "pre-existing diseases. The claim has "
                f"{continuous_coverage_months} months of coverage, "
                "so the identified waiting-period condition appears "
                "to be satisfied. However, automatic approval is not "
                "made because other policy eligibility conditions "
                "and exclusions still require verification."
            )

            confidence = 0.85

        # ---------------------------------------------------------
        # CASE D:
        # Pre-existing status is unknown
        # ---------------------------------------------------------

        elif (
            pre_existing_confirmed is None
            and contains_pre_existing_rule
            and contains_48_month_rule
        ):
            reason = (
                "The retrieved policy evidence contains a 48-month "
                "continuous coverage requirement for pre-existing "
                "diseases. However, the available claim information "
                "does not establish whether the diagnosis is "
                "pre-existing. Medical documentation and previous "
                "treatment history should be reviewed before making "
                "a final coverage decision."
            )

            confidence = 0.75

        # ---------------------------------------------------------
        # CASE E:
        # Policy evidence is insufficient
        # ---------------------------------------------------------

        else:
            reason = (
                "The available policy evidence is not sufficient "
                "to automatically approve or reject this claim. "
                "Manual review is required to verify the diagnosis, "
                "policy conditions, applicable exclusions, waiting "
                "periods, and any previous-coverage exceptions."
            )

            confidence = 0.40

        # ---------------------------------------------------------
        # 6. FORMAT POLICY EVIDENCE
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # 7. RETURN FINAL RESULT
        # ---------------------------------------------------------

        return {
            "claim_id": claim_id,
            "decision": decision,
            "approved_amount": approved_amount,
            "reason": reason,
            "policy_evidence": formatted_evidence,
            "confidence": confidence,
            "review_flags": review_flags,
            "claim_summary": {
                "diagnosis": diagnosis,
                "treatment_type": treatment_type,
                "claimed_amount": claimed_amount,
                "continuous_coverage_months": (
                    continuous_coverage_months
                ),
                "pre_existing_confirmed": (
                    pre_existing_confirmed
                ),
                "previous_coverage_verified": (
                    previous_coverage_verified
                ),
                "experimental": experimental,
            },
        }


if __name__ == "__main__":

    agent = DecisionAgent()

    result = agent.make_decision(
        claim_id="CLM-001",
        diagnosis="Diabetes",
        treatment_type="In-patient hospitalization",
        claimed_amount=50000.0,
        continuous_coverage_months=24,
        pre_existing_confirmed=True,
        previous_coverage_verified=False,
        experimental=False,
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
    print("Review Flags:")

    for flag in result["review_flags"]:
        print(f"- {flag}")

    print()
    print(
        "Policy Evidence Count:",
        len(result["policy_evidence"]),
    )