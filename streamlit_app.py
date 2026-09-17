import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/claims/analyze"


st.set_page_config(
    page_title="Aptino Claim Decision Engine",
    page_icon="🏥",
    layout="centered",
)


st.title("🏥 Aptino Claim Decision Engine")
st.write(
    "Enter the patient, policy, and claim details "
    "to receive a preliminary claim decision."
)


st.header("Patient Information")

patient_id = st.text_input(
    "Patient ID",
    value="PAT-001",
)

age = st.number_input(
    "Patient Age",
    min_value=0,
    max_value=120,
    value=45,
)

gender = st.selectbox(
    "Gender",
    options=[
        "Female",
        "Male",
        "Other",
        "Prefer not to say",
    ],
)


st.header("Policy Information")

policy_id = st.text_input(
    "Policy ID",
    value="POL-001",
)

policy_name = st.text_input(
    "Policy Name",
    value="CSC Individual Health Insurance",
)

policy_start_date = st.date_input(
    "Policy Start Date",
)

continuous_coverage_months = st.number_input(
    "Continuous Coverage Months",
    min_value=0,
    value=24,
)


st.header("Claim Information")

claim_id = st.text_input(
    "Claim ID",
    value="CLM-001",
)

diagnosis = st.text_input(
    "Diagnosis",
    value="Diabetes",
)

treatment_type = st.text_input(
    "Treatment Type",
    value="In-patient hospitalization",
)

claimed_amount = st.number_input(
    "Claimed Amount",
    min_value=0.0,
    value=50000.0,
    step=1000.0,
)

hospitalization_days = st.number_input(
    "Hospitalization Days",
    min_value=0,
    value=5,
)


st.header("Medical Document")

document_id = st.text_input(
    "Document ID",
    value="DOC-001",
)

document_type = st.text_input(
    "Document Type",
    value="Discharge Summary",
)

document_text = st.text_area(
    "Medical Document Text",
    value=(
        "Patient was admitted for diabetes-related "
        "complications."
    ),
    height=150,
)


if st.button(
    "Analyze Claim",
    type="primary",
):

    claim_payload = {
        "patient": {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
        },
        "policy": {
            "policy_id": policy_id,
            "policy_name": policy_name,
            "policy_start_date": str(policy_start_date),
            "continuous_coverage_months": (
                continuous_coverage_months
            ),
        },
        "claim": {
            "claim_id": claim_id,
            "diagnosis": diagnosis,
            "treatment_type": treatment_type,
            "claimed_amount": claimed_amount,
            "hospitalization_days": hospitalization_days,
        },
        "medical_documents": [
            {
                "document_id": document_id,
                "document_type": document_type,
                "document_text": document_text,
            }
        ],
    }

    with st.spinner("Analyzing claim..."):

        try:
            response = requests.post(
                API_URL,
                json=claim_payload,
                timeout=120,
            )

            if response.status_code == 200:

                result = response.json()

                st.success("Claim analysis completed.")

                st.subheader("Claim Decision")

                decision = result.get(
                    "decision",
                    "UNKNOWN",
                )

                if decision == "APPROVED":
                    st.success(
                        f"Decision: {decision}"
                    )

                elif decision == "REJECTED":
                    st.error(
                        f"Decision: {decision}"
                    )

                else:
                    st.warning(
                        f"Decision: {decision}"
                    )

                st.metric(
                    "Approved Amount",
                    f"₹{result.get('approved_amount', 0):,.2f}",
                )

                st.metric(
                    "Confidence",
                    f"{result.get('confidence', 0) * 100:.1f}%",
                )

                st.subheader("Reason")

                st.write(
                    result.get(
                        "reason",
                        "No reason provided.",
                    )
                )

                st.subheader("Policy Evidence")

                evidence = result.get(
                    "policy_evidence",
                    [],
                )

                if evidence:

                    for index, item in enumerate(
                        evidence,
                        start=1,
                    ):

                        with st.expander(
                            f"Evidence {index} - "
                            f"Page {item.get('page_number')}"
                        ):

                            st.write(
                                f"Chunk ID: "
                                f"{item.get('chunk_id')}"
                            )

                            st.write(
                                f"Relevance Score: "
                                f"{item.get('relevance_score', 0):.6f}"
                            )

                            st.write(
                                item.get(
                                    "text",
                                    "",
                                )
                            )

                else:

                    st.info(
                        "No policy evidence was returned."
                    )

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                st.code(
                    response.text
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI server. "
                "Make sure Terminal 1 is running "
                "uvicorn main:app --reload."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The request took too long. "
                "Please try again."
            )

        except Exception as error:

            st.error(
                f"Unexpected error: {str(error)}"
            )