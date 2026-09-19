import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://aptino-claim-decision-engine.onrender.com/analyze"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aptino Claim Decision Engine",
    page_icon="🏥",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("Aptino Claim Decision Engine")
st.caption(
    "AI-assisted health insurance claim analysis using "
    "policy retrieval, reranking, medical document analysis, "
    "and a structured multi-agent workflow."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Claim Information")

patient_id = st.sidebar.text_input(
    "Patient ID",
    value="TEST001",
)

age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=35,
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male", "Other", "Not specified"],
)

policy_id = st.sidebar.text_input(
    "Policy ID",
    value="POL001",
)

policy_name = st.sidebar.text_input(
    "Policy Name",
    value="Test Health Policy",
)

coverage_months = st.sidebar.number_input(
    "Continuous Coverage (months)",
    min_value=0,
    value=14,
)

claim_id = st.sidebar.text_input(
    "Claim ID",
    value="TEST-001",
)

diagnosis = st.sidebar.text_input(
    "Diagnosis",
    value="appendicitis",
)

treatment_type = st.sidebar.text_input(
    "Treatment Type",
    value="inpatient hospitalization",
)

claimed_amount = st.sidebar.number_input(
    "Claimed Amount",
    min_value=0.0,
    value=50000.0,
)

hospitalization_days = st.sidebar.number_input(
    "Hospitalization Days",
    min_value=0,
    value=4,
)

experimental = st.sidebar.checkbox(
    "Experimental Treatment",
    value=False,
)


# ============================================================
# MEDICAL DOCUMENT
# ============================================================

st.subheader("Medical Document")

document_type = st.selectbox(
    "Document Type",
    [
        "discharge_summary",
        "medical_report",
        "hospital_bill",
        "other",
    ],
)

document_text = st.text_area(
    "Paste medical document text",
    value=(
        "Patient was admitted for appendicitis and underwent "
        "inpatient treatment. No previous history of "
        "appendicitis or related chronic condition was "
        "documented."
    ),
    height=180,
)


# ============================================================
# ANALYZE
# ============================================================

if st.button(
    "Analyze Claim",
    type="primary",
    use_container_width=True,
):

    payload = {
        "patient": {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
        },
        "policy": {
            "policy_id": policy_id,
            "policy_name": policy_name,
            "policy_start_date": "2025-01-01",
            "continuous_coverage_months": coverage_months,
        },
        "claim": {
            "claim_id": claim_id,
            "diagnosis": diagnosis,
            "treatment_type": treatment_type,
            "claimed_amount": claimed_amount,
            "hospitalization_days": hospitalization_days,
            "experimental": experimental,
        },
        "medical_documents": [
            {
                "document_id": "DOC001",
                "document_type": document_type,
                "document_text": document_text,
            }
        ],
    }

    with st.spinner("Analyzing claim..."):

        try:

            response = requests.post(
                API_URL,
                json=payload,
                timeout=120,
            )

            if response.status_code != 200:

                st.error(
                    f"API error: {response.status_code}"
                )

                st.code(response.text)

            else:

                result = response.json()

                # ------------------------------------------------
                # DECISION
                # ------------------------------------------------

                st.subheader("Decision")

                decision = result.get(
                    "decision",
                    "NEEDS_REVIEW",
                )

                if decision == "APPROVED":
                    st.success(f"Decision: {decision}")

                elif decision == "REJECTED":
                    st.error(f"Decision: {decision}")

                else:
                    st.warning(f"Decision: {decision}")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Confidence",
                        f"{result.get('confidence', 0) * 100:.1f}%",
                    )

                with col2:
                    st.metric(
                        "Approved Amount",
                        f"{result.get('approved_amount', 0):,.2f}",
                    )

                st.info(
                    result.get(
                        "reason",
                        "No reason provided.",
                    )
                )

                # ------------------------------------------------
                # FINDINGS
                # ------------------------------------------------

                st.subheader("Findings")

                findings = result.get(
                    "findings",
                    [],
                )

                if findings:
                    for item in findings:
                        st.write(f"- {item}")
                else:
                    st.write("No findings reported.")

                # ------------------------------------------------
                # LIMITATIONS
                # ------------------------------------------------

                st.subheader("Limitations")

                limitations = result.get(
                    "limitations",
                    [],
                )

                if limitations:
                    for item in limitations:
                        st.write(f"- {item}")
                else:
                    st.write("No limitations reported.")

                # ------------------------------------------------
                # MISSING EVIDENCE
                # ------------------------------------------------

                st.subheader("Missing Evidence")

                missing = result.get(
                    "missing_evidence",
                    [],
                )

                if missing:
                    for item in missing:
                        st.write(f"- {item}")
                else:
                    st.write("No missing evidence identified.")

                # ------------------------------------------------
                # POLICY EVIDENCE
                # ------------------------------------------------

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
                            f"Evidence {index} — "
                            f"Page {item.get('page_number', 'N/A')} "
                            f"— {item.get('chunk_id', 'N/A')}"
                        ):

                            st.write(
                                item.get(
                                    "text",
                                    "",
                                )
                            )

                            st.caption(
                                "Relevance score: "
                                f"{item.get('relevance_score', 0):.4f}"
                            )

                else:
                    st.warning(
                        "No policy evidence retrieved."
                    )

                # ------------------------------------------------
                # EXECUTION TRACE
                # ------------------------------------------------

                st.subheader("Execution Trace")

                trace = result.get(
                    "trace",
                    [],
                )

                if trace:

                    for step in trace:

                        agent = step.get(
                            "agent",
                            "Unknown",
                        )

                        action = step.get(
                            "action",
                            "",
                        )

                        st.write(
                            f"**{agent}** — {action}"
                        )

                        details = {
                            key: value
                            for key, value in step.items()
                            if key not in {
                                "agent",
                                "action",
                            }
                        }

                        if details:
                            st.json(details)

                else:
                    st.write(
                        "No execution trace available."
                    )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the backend API. "
                "Make sure Uvicorn is running on "
                "http://127.0.0.1:8000."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The API request timed out."
            )

        except Exception as exc:

            st.error(
                f"Unexpected error: {exc}"
            )