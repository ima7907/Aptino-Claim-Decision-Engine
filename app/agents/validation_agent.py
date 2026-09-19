from typing import Any, Dict, List


class ValidationAgent:
    """
    Validates the structured output produced by the claim
    decision workflow before it is returned to the API.
    """

    ALLOWED_DECISIONS = {
        "APPROVED",
        "REJECTED",
        "NEEDS_REVIEW",
    }

    def __init__(self):
        print("Initializing Validation Agent...")

    def validate(
        self,
        claim_id: str,
        decision: str,
        approved_amount: float,
        claimed_amount: float,
        confidence: float,
        reason: str,
        policy_evidence: List[Dict[str, Any]],
        missing_evidence: List[str],
    ) -> Dict[str, Any]:

        errors = []
        warnings = []

        # --------------------------------------------------------
        # CLAIM ID
        # --------------------------------------------------------

        if not claim_id or not claim_id.strip():
            errors.append("Claim ID is missing.")

        # --------------------------------------------------------
        # DECISION
        # --------------------------------------------------------

        if decision not in self.ALLOWED_DECISIONS:
            errors.append(
                f"Invalid decision status: {decision}."
            )

        # --------------------------------------------------------
        # AMOUNT
        # --------------------------------------------------------

        if approved_amount < 0:
            errors.append(
                "Approved amount cannot be negative."
            )

        if approved_amount > claimed_amount:
            errors.append(
                "Approved amount cannot exceed the claimed amount."
            )

        # --------------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------------

        if confidence < 0 or confidence > 1:
            errors.append(
                "Confidence must be between 0 and 1."
            )

        # --------------------------------------------------------
        # REASON
        # --------------------------------------------------------

        if not reason or not reason.strip():
            errors.append(
                "Decision reason is missing."
            )

        # --------------------------------------------------------
        # POLICY EVIDENCE
        # --------------------------------------------------------

        if not policy_evidence:
            errors.append(
                "No policy evidence was retrieved."
            )

        # --------------------------------------------------------
        # NEEDS REVIEW CONSISTENCY
        # --------------------------------------------------------

        if decision == "NEEDS_REVIEW":
            if not missing_evidence:
                warnings.append(
                    "Decision is NEEDS_REVIEW but no missing "
                    "evidence was explicitly identified."
                )

        # --------------------------------------------------------
        # APPROVED CONSISTENCY
        # --------------------------------------------------------

        if decision == "APPROVED":
            if approved_amount <= 0:
                warnings.append(
                    "Decision is APPROVED but approved amount "
                    "is zero or less."
                )

        # --------------------------------------------------------
        # REJECTED CONSISTENCY
        # --------------------------------------------------------

        if decision == "REJECTED":
            if approved_amount != 0:
                errors.append(
                    "Rejected claims should have an approved "
                    "amount of zero."
                )

        # --------------------------------------------------------
        # FINAL STATUS
        # --------------------------------------------------------

        if errors:
            validation_status = "FAILED"
        elif warnings:
            validation_status = "PASSED_WITH_WARNINGS"
        else:
            validation_status = "PASSED"

        return {
            "validation_status": validation_status,
            "validation_errors": errors,
            "validation_warnings": warnings,
        }