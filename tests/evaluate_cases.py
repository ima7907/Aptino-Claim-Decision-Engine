import requests


API_URL = "http://127.0.0.1:8000/claims/analyze"


test_cases = [
    {
        "name": "Pre-existing condition - 24 months",
        "coverage_months": 24,
        "document": (
            "Patient diagnosis: Diabetes. "
            "Treatment: In-patient hospitalization. "
            "Patient has a previous history of diabetes."
        ),
    },
    {
        "name": "Pre-existing condition - 48 months",
        "coverage_months": 48,
        "document": (
            "Patient diagnosis: Diabetes. "
            "Treatment: In-patient hospitalization. "
            "Previous history of diabetes is documented."
        ),
    },
    {
        "name": "Medical history unclear",
        "coverage_months": 24,
        "document": (
            "Patient diagnosis: Diabetes. "
            "Treatment: In-patient hospitalization."
        ),
    },
    {
        "name": "Missing medical document",
        "coverage_months": 24,
        "document": "",
    },
    {
        "name": "Previous treatment mentioned",
        "coverage_months": 36,
        "document": (
            "Patient diagnosis: Hypertension. "
            "Treatment: In-patient hospitalization. "
            "Patient received previous medication and "
            "follow-up treatment."
        ),
    },
    {
        "name": "Different diagnosis",
        "coverage_months": 60,
        "document": (
            "Patient diagnosis: Hypertension. "
            "Treatment: In-patient hospitalization."
        ),
    },
    {
        "name": "Short medical document",
        "coverage_months": 12,
        "document": "Diabetes.",
    },
]


def create_payload(case, index):
    return {
        "patient": {
            "patient_id": f"P{index:03d}",
            "age": 45,
            "gender": "Male",
        },
        "policy": {
            "policy_id": f"POL{index:03d}",
            "policy_name": "USGIC Individual Health Insurance",
            "policy_start_date": "2024-01-01",
            "continuous_coverage_months": case["coverage_months"],
        },
        "claim": {
            "claim_id": f"CLM-EVAL-{index:03d}",
            "diagnosis": "Diabetes",
            "treatment_type": "In-patient hospitalization",
            "claimed_amount": 50000,
            "hospitalization_days": 5,
        },
        "medical_documents": (
            [
                {
                    "document_id": f"DOC{index:03d}",
                    "document_type": "Discharge Summary",
                    "document_text": case["document"],
                }
            ]
            if case["document"]
            else []
        ),
    }


def run_evaluation():
    print("=" * 70)
    print("APTINO CLAIM DECISION ENGINE - EVALUATION")
    print("=" * 70)

    passed = 0
    failed = 0

    for index, case in enumerate(test_cases, start=1):

        print()
        print(f"Test {index}: {case['name']}")

        payload = create_payload(case, index)

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

            else:

                print("Status: FAIL")
                print(
                    f"HTTP Status: "
                    f"{response.status_code}"
                )
                print(
                    f"Response: "
                    f"{response.text}"
                )

                failed += 1

        except Exception as error:

            print("Status: FAIL")
            print(f"Error: {error}")

            failed += 1

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total cases: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("Overall result: ALL TESTS PASSED")
    else:
        print("Overall result: SOME TESTS FAILED")


if __name__ == "__main__":
    run_evaluation()