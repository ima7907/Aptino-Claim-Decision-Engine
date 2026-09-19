import json
import requests


API_URL = "http://127.0.0.1:8000/analyze"


# ============================================================
# EXPECTED OUTCOMES
# ============================================================
#
# These expectations are based on the task descriptions and
# treatment information supplied in the official public cases.
#
# NEEDS_REVIEW is used where the supplied case requires
# additional evidence or verification rather than a fully
# automatic final decision.
#
# ============================================================

EXPECTED_DECISIONS = {
    "PUB-001": "NEEDS_REVIEW",
    "PUB-002": "NEEDS_REVIEW",
    "PUB-003": "NEEDS_REVIEW",
    "PUB-004": "NEEDS_REVIEW",
    "PUB-005": "NEEDS_REVIEW",
    "PUB-006": "NEEDS_REVIEW",
    "PUB-007": "NEEDS_REVIEW",
    "PUB-008": "NEEDS_REVIEW",
    "PUB-009": "NEEDS_REVIEW",
    "PUB-010": "NEEDS_REVIEW",
    "PUB-011": "NEEDS_REVIEW",
    "PUB-012": "NEEDS_REVIEW",
}


def build_payload(case):
    """
    Convert the official public test-case format
    into the ClaimCase format expected by the API.
    """

    treatment = case.get(
        "treatment",
        {},
    )

    patient = case.get(
        "patient",
        {},
    )

    expenses = case.get(
        "expenses_inr",
        {},
    )

    medical_text = (
        f"Diagnosis: "
        f"{treatment.get('diagnosis', '')}. "

        f"Treatment: "
        f"{treatment.get('type', '')}. "

        f"Procedure: "
        f"{treatment.get('procedure', '')}. "

        f"Pre-existing condition: "
        f"{treatment.get('pre_existing', False)}. "

        f"Experimental treatment: "
        f"{treatment.get('experimental', False)}."
    )

    return {
        "patient": {
            "patient_id": case["case_id"],
            "age": patient.get(
                "age",
                0,
            ),
            "gender": None,
        },

        "policy": {
            "policy_id": case["policy_id"],

            "policy_name": (
                "USGIC Individual Health Insurance"
            ),

            "policy_start_date": (
                case["policy_start_date"]
            ),

            "continuous_coverage_months": (
                case[
                    "continuous_coverage_months"
                ]
            ),
        },

        "claim": {
            "claim_id": case["case_id"],

            "diagnosis": treatment.get(
                "diagnosis",
                "Unknown",
            ),

            "treatment_type": treatment.get(
                "type",
                "Unknown",
            ),

            "claimed_amount": sum(
                expenses.values()
            ),

            "hospitalization_days": (
                treatment.get(
                    "hospitalization_hours"
                )
            ),

            "experimental": (
                treatment.get(
                    "experimental",
                    False,
                )
            ),
        },

        "medical_documents": [
            {
                "document_id": (
                    f"DOC-{case['case_id']}"
                ),

                "document_type": (
                    "Public Test Case "
                    "Medical Summary"
                ),

                "document_text": medical_text,
            }
        ],
    }


def evaluate_case(
    case,
):
    """
    Send one official case to the API and
    evaluate the returned decision.
    """

    case_id = case["case_id"]

    expected_decision = (
        EXPECTED_DECISIONS.get(
            case_id
        )
    )

    payload = build_payload(
        case
    )

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:

            return {
                "case_id": case_id,
                "status": "FAIL",
                "http_status": (
                    response.status_code
                ),
                "decision": None,
                "expected_decision": (
                    expected_decision
                ),
                "confidence": None,
                "decision_match": False,
                "citation_found": False,
                "retrieval_count": 0,
            }

        result = response.json()

        decision = result.get(
            "decision"
        )

        policy_evidence = result.get(
            "policy_evidence",
            [],
        )

        decision_match = (
            decision
            == expected_decision
        )

        citation_found = (
            len(policy_evidence)
            > 0
        )

        retrieval_count = len(
            policy_evidence
        )

        status = (
            "PASS"
            if decision_match
            and citation_found
            else "FAIL"
        )

        return {
            "case_id": case_id,
            "status": status,
            "http_status": 200,
            "decision": decision,
            "expected_decision": (
                expected_decision
            ),
            "confidence": result.get(
                "confidence"
            ),
            "decision_match": (
                decision_match
            ),
            "citation_found": (
                citation_found
            ),
            "retrieval_count": (
                retrieval_count
            ),
            "policy_evidence": [
                {
                    "chunk_id": item.get(
                        "chunk_id"
                    ),
                    "page_number": item.get(
                        "page_number"
                    ),
                    "relevance_score": item.get(
                        "relevance_score"
                    ),
                }
                for item in policy_evidence
            ],
        }

    except Exception as error:

        return {
            "case_id": case_id,
            "status": "FAIL",
            "http_status": None,
            "decision": None,
            "expected_decision": (
                expected_decision
            ),
            "confidence": None,
            "decision_match": False,
            "citation_found": False,
            "retrieval_count": 0,
            "error": str(error),
        }


def main():

    # ========================================================
    # LOAD OFFICIAL CASES
    # ========================================================

    with open(
        "candidate_data/public_test_cases.json",
        "r",
        encoding="utf-8",
    ) as file:

        cases = json.load(file)

    print(
        "=" * 70
    )

    print(
        "APTINO CLAIM DECISION ENGINE"
    )

    print(
        "OFFICIAL PUBLIC TEST CASE EVALUATION"
    )

    print(
        "=" * 70
    )

    print()

    # ========================================================
    # METRIC COUNTERS
    # ========================================================

    total_cases = len(cases)

    http_passed = 0
    decision_matches = 0
    citation_cases = 0
    abstention_cases = 0

    results = []

    # ========================================================
    # TEST EACH CASE
    # ========================================================

    for case in cases:

        case_id = case["case_id"]

        print(
            f"Testing {case_id}..."
        )

        result = evaluate_case(
            case
        )

        results.append(
            result
        )

        if result.get(
            "http_status"
        ) == 200:

            http_passed += 1

        if result.get(
            "decision_match"
        ):

            decision_matches += 1

        if result.get(
            "citation_found"
        ):

            citation_cases += 1

        if result.get(
            "decision"
        ) == "NEEDS_REVIEW":

            abstention_cases += 1

        print(
            f"Status: "
            f"{result['status']}"
        )

        print(
            f"Expected Decision: "
            f"{result.get('expected_decision')}"
        )

        print(
            f"Actual Decision: "
            f"{result.get('decision')}"
        )

        print(
            f"Confidence: "
            f"{result.get('confidence')}"
        )

        print(
            f"Policy citations: "
            f"{result.get('retrieval_count', 0)}"
        )

        print()

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    decision_accuracy = (
        decision_matches
        / total_cases
        if total_cases
        else 0.0
    )

    citation_hit_rate = (
        citation_cases
        / total_cases
        if total_cases
        else 0.0
    )

    http_success_rate = (
        http_passed
        / total_cases
        if total_cases
        else 0.0
    )

    abstention_rate = (
        abstention_cases
        / total_cases
        if total_cases
        else 0.0
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "OFFICIAL TEST SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total official cases: "
        f"{total_cases}"
    )

    print(
        f"HTTP successful cases: "
        f"{http_passed}/{total_cases}"
    )

    print(
        f"Decision matches: "
        f"{decision_matches}/{total_cases}"
    )

    print(
        f"Decision accuracy: "
        f"{decision_accuracy:.2%}"
    )

    print(
        f"Citation-supported cases: "
        f"{citation_cases}/{total_cases}"
    )

    print(
        f"Citation hit rate: "
        f"{citation_hit_rate:.2%}"
    )

    print(
        f"NEEDS_REVIEW cases: "
        f"{abstention_cases}/{total_cases}"
    )

    print(
        f"Abstention rate: "
        f"{abstention_rate:.2%}"
    )

    print(
        f"HTTP success rate: "
        f"{http_success_rate:.2%}"
    )

    print()

    # ========================================================
    # CASE-LEVEL RESULTS
    # ========================================================

    print(
        "CASE-LEVEL RESULTS"
    )

    print(
        "-" * 70
    )

    for result in results:

        print(
            f"{result['case_id']}: "
            f"expected={result.get('expected_decision')}, "
            f"actual={result.get('decision')}, "
            f"status={result['status']}"
        )

    print()

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = (
        "tests/public_evaluation_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "total_cases": total_cases,
                "http_successful_cases": (
                    http_passed
                ),
                "decision_matches": (
                    decision_matches
                ),
                "decision_accuracy": (
                    decision_accuracy
                ),
                "citation_supported_cases": (
                    citation_cases
                ),
                "citation_hit_rate": (
                    citation_hit_rate
                ),
                "needs_review_cases": (
                    abstention_cases
                ),
                "abstention_rate": (
                    abstention_rate
                ),
                "http_success_rate": (
                    http_success_rate
                ),
                "results": results,
            },
            file,
            indent=2,
        )

    print(
        f"Detailed results saved to: "
        f"{output_file}"
    )

    print()

    if (
        http_passed == total_cases
        and decision_matches == total_cases
        and citation_cases == total_cases
    ):

        print(
            "Overall result: "
            "ALL OFFICIAL EVALUATION CHECKS PASSED"
        )

    else:

        print(
            "Overall result: "
            "REVIEW CASE-LEVEL RESULTS ABOVE"
        )


if __name__ == "__main__":
    main()