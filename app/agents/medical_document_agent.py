from typing import Dict, List


class MedicalDocumentAgent:
    """
    Agent responsible for analyzing medical document text
    and extracting information relevant to claim review.

    This is a lightweight prototype. It does not make a
    medical diagnosis. It identifies relevant evidence that
    may require verification by a human reviewer.
    """

    def __init__(self):
        print("Initializing Medical Document Agent...")

    def analyze_document(
        self,
        document_text: str,
        claim_diagnosis: str,
        treatment_type: str,
    ) -> Dict:
        """
        Analyze medical document text against claim information.
        """

        if not document_text or not document_text.strip():
            return {
                "document_status": "MISSING",
                "diagnosis_match": False,
                "treatment_match": False,
                "possible_pre_existing_evidence": False,
                "evidence": [],
                "review_flags": [
                    "Medical document text was not provided."
                ],
            }

        text = document_text.lower()
        diagnosis = claim_diagnosis.lower().strip()
        treatment = treatment_type.lower().strip()

        evidence: List[str] = []
        review_flags: List[str] = []

        # ---------------------------------------------------------
        # 1. Diagnosis matching
        # ---------------------------------------------------------

        diagnosis_match = (
            diagnosis in text
            if diagnosis
            else False
        )

        if diagnosis_match:
            evidence.append(
                f"Claim diagnosis '{claim_diagnosis}' "
                "was found in the medical document."
            )
        else:
            review_flags.append(
                "Claim diagnosis was not clearly found "
                "in the medical document."
            )

        # ---------------------------------------------------------
        # 2. Treatment matching
        # ---------------------------------------------------------

        treatment_match = (
            treatment in text
            if treatment
            else False
        )

        if treatment_match:
            evidence.append(
                f"Claim treatment '{treatment_type}' "
                "was found in the medical document."
            )
        else:
            review_flags.append(
                "Claim treatment was not clearly found "
                "in the medical document."
            )

        # ---------------------------------------------------------
        # 3. Possible pre-existing evidence
        # ---------------------------------------------------------

        pre_existing_keywords = [
            "pre-existing",
            "pre existing",
            "previous history",
            "past history",
            "history of",
            "known case of",
            "long standing",
            "long-standing",
            "chronic condition",
            "previously diagnosed",
        ]

        matched_pre_existing_terms = [
            keyword
            for keyword in pre_existing_keywords
            if keyword in text
        ]

        possible_pre_existing_evidence = (
            len(matched_pre_existing_terms) > 0
        )

        if possible_pre_existing_evidence:
            evidence.append(
                "The medical document contains terminology "
                "that may indicate previous or pre-existing "
                "medical history."
            )

            review_flags.append(
                "Possible pre-existing medical history detected; "
                "human verification is required."
            )

        # ---------------------------------------------------------
        # 4. Previous treatment / consultation evidence
        # ---------------------------------------------------------

        previous_treatment_keywords = [
            "previous treatment",
            "prior treatment",
            "previous medication",
            "prior medication",
            "follow-up",
            "follow up",
            "previous consultation",
            "prior consultation",
        ]

        matched_previous_treatment_terms = [
            keyword
            for keyword in previous_treatment_keywords
            if keyword in text
        ]

        if matched_previous_treatment_terms:
            evidence.append(
                "The document contains references to "
                "previous treatment or consultation."
            )

            review_flags.append(
                "Previous treatment history should be "
                "verified against the policy terms."
            )

        # ---------------------------------------------------------
        # 5. Document completeness
        # ---------------------------------------------------------

        if len(document_text.strip()) < 50:
            review_flags.append(
                "Medical document text appears too short "
                "for reliable automated analysis."
            )

        # ---------------------------------------------------------
        # 6. Final status
        # ---------------------------------------------------------

        if (
            diagnosis_match
            and treatment_match
            and not possible_pre_existing_evidence
            and len(review_flags) == 0
        ):
            document_status = "SUFFICIENT_FOR_REVIEW"

        else:
            document_status = "NEEDS_REVIEW"

        return {
            "document_status": document_status,
            "diagnosis_match": diagnosis_match,
            "treatment_match": treatment_match,
            "possible_pre_existing_evidence": (
                possible_pre_existing_evidence
            ),
            "matched_pre_existing_terms": (
                matched_pre_existing_terms
            ),
            "evidence": evidence,
            "review_flags": review_flags,
        }


if __name__ == "__main__":

    agent = MedicalDocumentAgent()

    sample_document = """
    Patient: John Doe

    Diagnosis: Diabetes

    Treatment: In-patient hospitalization.

    Medical history:
    Patient has a previous history of diabetes and was
    receiving previous medication before the current admission.
    """

    result = agent.analyze_document(
        document_text=sample_document,
        claim_diagnosis="Diabetes",
        treatment_type="In-patient hospitalization",
    )

    print()
    print("Medical Document Analysis")
    print("=" * 80)

    print(
        "Document Status:",
        result["document_status"],
    )

    print(
        "Diagnosis Match:",
        result["diagnosis_match"],
    )

    print(
        "Treatment Match:",
        result["treatment_match"],
    )

    print(
        "Possible Pre-existing Evidence:",
        result["possible_pre_existing_evidence"],
    )

    print()
    print("Evidence:")

    for item in result["evidence"]:
        print(f"- {item}")

    print()
    print("Review Flags:")

    for flag in result["review_flags"]:
        print(f"- {flag}")