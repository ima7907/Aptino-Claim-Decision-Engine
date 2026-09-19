from typing import Any, Dict

from app.agents.medical_document_agent import MedicalDocumentAgent
from app.agents.policy_agent import PolicyAnalysisAgent
from app.agents.decision_agent import DecisionAgent
from app.agents.validation_agent import ValidationAgent
from app.workflow_state import ClaimWorkflowState


class ClaimDecisionWorkflow:
    """
    Structured multi-agent workflow for claim analysis.

    Flow:
        MedicalDocumentAgent
            ↓
        PolicyAnalysisAgent
            ↓
        DecisionAgent
            ↓
        ValidationAgent
    """

    def __init__(self):
        print("Initializing Claim Decision Workflow...")

        self.medical_agent = MedicalDocumentAgent()
        self.policy_agent = PolicyAnalysisAgent()
        self.decision_agent = DecisionAgent()
        self.validation_agent = ValidationAgent()

    # ============================================================
    # INITIAL STATE
    # ============================================================

    def _initialize_state(
        self,
        claim_id: str,
        diagnosis: str,
        treatment_type: str,
        claimed_amount: float,
        continuous_coverage_months: int,
        pre_existing_confirmed,
        previous_coverage_verified: bool,
        experimental: bool,
    ) -> ClaimWorkflowState:

        return {
            "claim_id": claim_id,
            "diagnosis": diagnosis,
            "treatment_type": treatment_type,
            "claimed_amount": claimed_amount,
            "continuous_coverage_months": continuous_coverage_months,
            "pre_existing_confirmed": pre_existing_confirmed,
            "previous_coverage_verified": previous_coverage_verified,
            "experimental": experimental,
            "medical_analysis": [],
            "medical_review_required": False,
            "possible_pre_existing_evidence": False,
            "policy_query": "",
            "policy_evidence": [],
            "retrieval_count": 0,
            "decision": "NEEDS_REVIEW",
            "approved_amount": 0.0,
            "reason": "",
            "confidence": 0.0,
            "review_flags": [],
            "trace": [],
            "validation_status": "NOT_RUN",
            "validation_errors": [],
            "missing_evidence": [],
        }

    # ============================================================
    # MEDICAL DOCUMENT AGENT
    # ============================================================

    def run_medical_agent(
        self,
        state: ClaimWorkflowState,
        medical_documents,
    ) -> ClaimWorkflowState:

        medical_results = []

        review_required = False
        possible_pre_existing = False

        for document in medical_documents:

            # Support both Pydantic objects and dictionaries.
            if isinstance(document, dict):
                document_id = document.get(
                    "document_id",
                    "",
                )
                document_type = document.get(
                    "document_type",
                    "",
                )
                document_text = document.get(
                    "document_text",
                    "",
                )
            else:
                document_id = getattr(
                    document,
                    "document_id",
                    "",
                )
                document_type = getattr(
                    document,
                    "document_type",
                    "",
                )
                document_text = getattr(
                    document,
                    "document_text",
                    "",
                )

            result = self.medical_agent.analyze_document(
                document_text=document_text,
                claim_diagnosis=state["diagnosis"],
                treatment_type=state["treatment_type"],
            )

            medical_results.append(
                {
                    "document_id": document_id,
                    "document_type": document_type,
                    "analysis": result,
                }
            )

            if result.get("document_status") == "NEEDS_REVIEW":
                review_required = True

            if result.get("possible_pre_existing_evidence"):
                possible_pre_existing = True

        state["medical_analysis"] = medical_results
        state["medical_review_required"] = review_required
        state["possible_pre_existing_evidence"] = (
            possible_pre_existing
        )

        state["trace"].append(
            {
                "agent": "MedicalDocumentAgent",
                "action": "Analyzed medical documents",
                "documents_processed": len(medical_documents),
                "review_required": review_required,
            }
        )

        return state

    # ============================================================
    # POLICY ANALYSIS AGENT
    # ============================================================

    def run_policy_agent(
        self,
        state: ClaimWorkflowState,
    ) -> ClaimWorkflowState:

        result = self.policy_agent.analyze_claim(
            diagnosis=state["diagnosis"],
            treatment_type=state["treatment_type"],
            continuous_coverage_months=state[
                "continuous_coverage_months"
            ],
            top_k=5,
        )

        state["policy_query"] = result.get(
            "query",
            "",
        )

        state["policy_evidence"] = result.get(
            "evidence",
            [],
        )

        state["retrieval_count"] = result.get(
            "retrieval_count",
            0,
        )

        state["trace"].append(
            {
                "agent": "PolicyAnalysisAgent",
                "action": (
                    "Retrieved policy evidence using "
                    "hybrid retrieval and reranking"
                ),
                "retrieval_count": state[
                    "retrieval_count"
                ],
                "reranking_applied": any(
                    evidence.get("rerank_score") is not None
                    for evidence in state[
                        "policy_evidence"
                    ]
                ),
            }
        )

        return state

    # ============================================================
    # DECISION AGENT
    # ============================================================

    def run_decision_agent(
        self,
        state: ClaimWorkflowState,
    ) -> ClaimWorkflowState:

        result = self.decision_agent.make_decision(
            claim_id=state["claim_id"],
            diagnosis=state["diagnosis"],
            treatment_type=state["treatment_type"],
            claimed_amount=state["claimed_amount"],
            continuous_coverage_months=state[
                "continuous_coverage_months"
            ],
            pre_existing_confirmed=state[
                "pre_existing_confirmed"
            ],
            previous_coverage_verified=state[
                "previous_coverage_verified"
            ],
            experimental=state["experimental"],
        )

        state["decision"] = result.get(
            "decision",
            "NEEDS_REVIEW",
        )

        state["approved_amount"] = result.get(
            "approved_amount",
            0.0,
        )

        state["reason"] = result.get(
            "reason",
            "",
        )

        state["confidence"] = result.get(
            "confidence",
            0.0,
        )

        state["review_flags"] = result.get(
            "review_flags",
            [],
        )

        state["trace"].append(
            {
                "agent": "DecisionAgent",
                "action": (
                    "Evaluated claim using structured "
                    "claim state and policy evidence"
                ),
                "decision": state["decision"],
            }
        )

        return state

    # ============================================================
    # VALIDATION AGENT
    # ============================================================

    def run_validation_agent(
        self,
        state: ClaimWorkflowState,
    ) -> ClaimWorkflowState:

        validation_result = self.validation_agent.validate(
            claim_id=state["claim_id"],
            decision=state["decision"],
            approved_amount=state["approved_amount"],
            claimed_amount=state["claimed_amount"],
            confidence=state["confidence"],
            reason=state["reason"],
            policy_evidence=state["policy_evidence"],
            missing_evidence=state["missing_evidence"],
        )

        state["validation_status"] = validation_result.get(
            "validation_status",
            "FAILED",
        )

        state["validation_errors"] = validation_result.get(
            "validation_errors",
            [],
        )

        state["trace"].append(
            {
                "agent": "ValidationAgent",
                "action": (
                    "Validated structured claim decision"
                ),
                "validation_status": state[
                    "validation_status"
                ],
                "validation_errors": state[
                    "validation_errors"
                ],
            }
        )

        return state

    # ============================================================
    # MAIN WORKFLOW
    # ============================================================

    def run(
        self,
        claim_id: str,
        diagnosis: str,
        treatment_type: str,
        claimed_amount: float,
        continuous_coverage_months: int,
        medical_documents,
        pre_existing_confirmed=None,
        previous_coverage_verified: bool = False,
        experimental: bool = False,
    ) -> Dict[str, Any]:

        state = self._initialize_state(
            claim_id=claim_id,
            diagnosis=diagnosis,
            treatment_type=treatment_type,
            claimed_amount=claimed_amount,
            continuous_coverage_months=continuous_coverage_months,
            pre_existing_confirmed=pre_existing_confirmed,
            previous_coverage_verified=previous_coverage_verified,
            experimental=experimental,
        )

        # 1. Medical document analysis
        state = self.run_medical_agent(
            state,
            medical_documents,
        )

        # 2. Policy retrieval and reranking
        state = self.run_policy_agent(
            state,
        )

        # 3. Decision
        state = self.run_decision_agent(
            state,
        )

        # 4. Validation
        state = self.run_validation_agent(
            state,
        )

        # Workflow completion trace
        state["trace"].append(
            {
                "agent": "ClaimDecisionWorkflow",
                "action": (
                    "Completed structured multi-agent workflow"
                ),
                "validation_status": state[
                    "validation_status"
                ],
            }
        )

        return state