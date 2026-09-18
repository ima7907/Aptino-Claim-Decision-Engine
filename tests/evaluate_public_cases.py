import json
import requests


API_URL = "http://127.0.0.1:8000/claims/analyze"


def build_payload(case):
    """
    Convert the official public test-case format
    into the ClaimCase format expected by the API.
    """

    treatment = case.get("treatment", {})
    patient = case.get("patient", {})

    medical_text = (
        f"Diagnosis: {treatment.get('diagnosis', '')}. "
        f"Treatment: {treatment.get('type', '')}. "
        f"Procedure: {treatment.get('procedure', '')}. "
        f"Pre-existing condition: "
        f"{treatment.get('pre_existing', False)}."
    )

    return {
        "patient": {
            "patient_id": case["case_id"],
            "age": patient.get("age", 0),
            "gender": None,
        },

        "policy": {
            "policy_id": case["policy_id"],
            "policy_name": "USGIC Individual Health Insurance",
            "policy_start_date": case["policy_start_date"],
            "continuous_coverage_months": case[
                "continuous_coverage_months"
            ],
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
                case.get("expenses_inr", {}).values()
            ),
            "hospitalization_days": None,
        },

        "medical_documents": [
            {
                "document_id": f"DOC-{case['case_id']}",
                "document_type": "Public Test Case Medical Summary",
                "document_text": medical_text,
            }
        ],
    }


def main():

    with open(
        "candidate_data/public_test_cases.json",
        "r",
        encoding="utf-8",
    ) as file:

        cases = json.load(file)


    print("=" * 70)
    print("APTINO CLAIM DECISION ENGINE")
    print("OFFICIAL PUBLIC TEST CASE EVALUATION")
    print("=" * 70)

    passed = 0
    failed = 0

    results = []


    for case in cases:

        case_id = case["case_id"]

        print()
        print(f"Testing {case_id}...")

        payload = build_payload(case)

        try:

            response = requests.post(
                API_URL,
                json=payload,
                timeout=120,
            )

            if response.status_code == 200:

                result = response.json()

                print("Status: PASS")
                print(
                    f"Decision: "
                    f"{result.get('decision')}"
                )
                print(
                    f"Confidence: "
                    f"{result.get('confidence')}"
                )

                passed += 1

                results.append(
                    {
                        "case_id": case_id,
                        "status": "PASS",
                        "decision": result.get(
                            "decision"
                        ),
                        "confidence": result.get(
                            "confidence"
                        ),
                    }
                )

            else:

                print("Status: FAIL")
                print(
                    f"HTTP Status: "
                    f"{response.status_code}"
                )

                failed += 1

                results.append(
                    {
                        "case_id": case_id,
                        "status": "FAIL",
                        "decision": None,
                        "confidence": None,
                    }
                )

        except Exception as error:

            print("Status: FAIL")
            print(f"Error: {error}")

            failed += 1

            results.append(
                {
                    "case_id": case_id,
                    "status": "FAIL",
                    "decision": None,
                    "confidence": None,
                }
            )


    print()
    print("=" * 70)
    print("OFFICIAL TEST SUMMARY")
    print("=" * 70)

    print(f"Total official cases: {len(cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:

        print("Overall result: ALL OFFICIAL CASES PASSED")

    else:

        print("Overall result: SOME OFFICIAL CASES FAILED")


if __name__ == "__main__":
    main()