from pathlib import Path
from typing import List, Dict

from app.retrieval.hybrid_retriever import HybridRetriever


class PolicyAnalysisAgent:
    """
    Agent responsible for finding relevant policy clauses
    related to a claim.
    """

    def __init__(self):
        print("Initializing Policy Analysis Agent...")

        self.project_root = Path(__file__).resolve().parents[2]

        self.chunks_path = (
            self.project_root
            / "data"
            / "processed"
            / "policy_chunks.json"
        )

        self.retriever = HybridRetriever()

    def analyze_claim(
        self,
        diagnosis: str,
        treatment_type: str,
        continuous_coverage_months: int,
        top_k: int = 5,
    ) -> Dict:
        """
        Analyze a claim by retrieving relevant policy clauses.
        """

        query = (
            f"Diagnosis: {diagnosis}. "
            f"Treatment type: {treatment_type}. "
            f"Continuous coverage: "
            f"{continuous_coverage_months} months. "
            f"Find relevant coverage, exclusions, waiting periods, "
            f"and pre-existing disease conditions."
        )

        results = self.retriever.search(
            query=query,
            top_k=top_k,
            retrieval_k=10,
        )

        evidence: List[Dict] = []

        for result in results:
            evidence.append(
                {
                    "chunk_id": result.get("chunk_id"),
                    "page_number": result.get("page_number"),
                    "text": result.get("text", ""),
                    "relevance_score": result.get(
                        "hybrid_score",
                        0.0,
                    ),
                }
            )

        return {
            "query": query,
            "continuous_coverage_months": (
                continuous_coverage_months
            ),
            "evidence": evidence,
        }


if __name__ == "__main__":

    agent = PolicyAnalysisAgent()

    result = agent.analyze_claim(
        diagnosis="Diabetes",
        treatment_type="In-patient hospitalization",
        continuous_coverage_months=24,
        top_k=5,
    )

    print()
    print("Policy Analysis Result")
    print("=" * 80)

    print("Query:")
    print(result["query"])

    print()
    print("Evidence found:", len(result["evidence"]))
    print()

    for evidence in result["evidence"]:

        print("--- Evidence ---")

        print(
            f"Chunk ID: {evidence['chunk_id']}"
        )

        print(
            f"Page: {evidence['page_number']}"
        )

        print(
            f"Relevance score: "
            f"{evidence['relevance_score']:.6f}"
        )

        print(
            evidence["text"][:500]
        )

        print("-" * 80)