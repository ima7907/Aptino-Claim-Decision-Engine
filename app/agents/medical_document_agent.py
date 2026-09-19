from typing import Dict


class MedicalDocumentAgent:
    def __init__(self):
        print("Initializing Medical Document Agent...")

    def _has_positive_history_evidence(self, text: str, keyword: str) -> bool:
        """
        Detect whether a history keyword is used positively.

        Example:
        "previous history of diabetes" -> True

        But:
        "no previous history of diabetes" -> False
        "without previous history of diabetes" -> False
        "no history of diabetes" -> False
        """
        text_lower = text.lower()

        keyword_position = text_lower.find(keyword)

        if keyword_position == -1:
            return False

        context_start = max(0, keyword_position - 60)
        context = text_lower[context_start:keyword_position]

        negative_phrases = [
            "no ",
            "no previous",
            "no prior",
            "without ",
            "negative for ",
            "denies ",
            "denied ",
            "not ",
            "none ",
            "absent ",
            "never ",
        ]

        return not any(
            phrase in context
            for phrase in negative_phrases
        )

    def analyze_document(
        self,
        document_text: str,
        claim_diagnosis: str,
        treatment_type: str,
    ) -> Dict:

        if not document_text or not document_text.strip():
            return {
                "document_status": "MISSING",
                "diagnosis_match": False,
                "treatment_match": False,
                "possible_pre_existing_evidence": False,
                "matched_pre_existing_terms": [],
                "evidence": [],
                "review_flags": [
                    "Medical document text was not provided."
                ],
            }

        text = document_text.lower()
        diagnosis = claim_diagnosis.lower().strip()
        treatment = treatment_type.lower().strip()

        evidence = []
        review_flags = []

        # --------------------------------------------------------
        # DIAGNOSIS MATCH
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # TREATMENT MATCH
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # PRE-EXISTING / MEDICAL HISTORY DETECTION
        # --------------------------------------------------------

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

        matched_pre_existing_terms = []

        for keyword in pre_existing_keywords:
            if keyword in text:
                if self._has_positive_history_evidence(
                    text,
                    keyword,
                ):
                    matched_pre_existing_terms.append(keyword)

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

        # --------------------------------------------------------
        # PREVIOUS TREATMENT DETECTION
        # --------------------------------------------------------

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
                "The document contains references to previous "
                "treatment or consultation."
            )

            review_flags.append(
                "Previous treatment history should be verified "
                "against the policy terms."
            )

        # --------------------------------------------------------
        # DOCUMENT LENGTH CHECK
        # --------------------------------------------------------

        if len(document_text.strip()) < 50:
            review_flags.append(
                "Medical document text appears too short "
                "for reliable automated analysis."
            )

        # --------------------------------------------------------
        # DOCUMENT STATUS
        # --------------------------------------------------------

        if (
            diagnosis_match
            and treatment_match
            and not possible_pre_existing_evidence
            and not matched_previous_treatment_terms
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