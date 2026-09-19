from typing import List, Dict, Optional
import re

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever


class HybridRetriever:
    """
    Hybrid policy retriever combining:

    1. BM25 keyword retrieval
    2. Dense semantic retrieval
    3. Reciprocal Rank Fusion (RRF)
    4. Lightweight policy-aware reranking

    The reranking stage uses deterministic lexical and
    policy-concept matching so that important policy clauses
    receive higher priority before being passed to the agents.
    """

    def __init__(
        self,
        chunks: Optional[List[Dict]] = None,
    ):
        print("Loading policy chunks...")

        print("Loading BM25 retriever...")
        self.bm25_retriever = BM25Retriever(chunks)

        print("Loading dense retriever...")
        self.dense_retriever = DenseRetriever()

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Convert text into normalized word tokens.
        """

        if not text:
            return []

        return re.findall(
            r"\b[a-zA-Z0-9][a-zA-Z0-9-]*\b",
            text.lower(),
        )

    # =========================================================
    # RERANKING
    # =========================================================

    def _calculate_rerank_score(
        self,
        query: str,
        result: Dict,
    ) -> float:
        """
        Calculate a deterministic policy-aware reranking score.

        The score combines:

        - query-term coverage
        - policy concept matching
        - exact phrase matching
        - original hybrid retrieval score
        """

        query_tokens = self._tokenize(query)

        document_text = result.get(
            "text",
            "",
        )

        document_tokens = self._tokenize(
            document_text
        )

        if not query_tokens or not document_tokens:
            return float(
                result.get(
                    "hybrid_score",
                    0.0,
                )
            )

        query_terms = set(query_tokens)
        document_terms = set(document_tokens)

        # -----------------------------------------------------
        # 1. QUERY TERM COVERAGE
        # -----------------------------------------------------

        matched_terms = query_terms.intersection(
            document_terms
        )

        term_coverage = (
            len(matched_terms)
            / len(query_terms)
            if query_terms
            else 0.0
        )

        # -----------------------------------------------------
        # 2. NORMALIZED TEXT
        # -----------------------------------------------------

        normalized_query = " ".join(
            query_tokens
        )

        normalized_document = " ".join(
            document_tokens
        )

        # -----------------------------------------------------
        # 3. EXACT PHRASE MATCH
        # -----------------------------------------------------

        exact_phrase_score = (
            1.0
            if normalized_query in normalized_document
            else 0.0
        )

        # -----------------------------------------------------
        # 4. POLICY CONCEPT MATCHING
        # -----------------------------------------------------

        policy_concepts = {

            "pre-existing": [
                "pre-existing",
                "pre existing",
                "48 months",
                "prior to the first policy",
                "previous continuous insurance coverage",
                "portability",
            ],

            "waiting": [
                "waiting period",
                "48 months",
                "waiting",
            ],

            "hospitalization": [
                "hospitalization",
                "in-patient",
                "inpatient",
                "24 consecutive hours",
                "hospital",
            ],

            "day-care": [
                "day care",
                "day-care",
                "day care treatment",
                "less than 24",
            ],

            "domiciliary": [
                "domiciliary",
                "home treatment",
                "residence",
            ],

            "experimental": [
                "experimental",
                "unproven",
                "investigational",
            ],

            "cosmetic": [
                "cosmetic",
                "aesthetic",
                "plastic surgery",
            ],

            "portability": [
                "portability",
                "previous health insurance policy",
                "continuous preceding years",
                "database and claim history",
            ],
        }

        query_lower = query.lower()

        concept_matches = 0
        concept_total = 0

        for keywords in policy_concepts.values():

            concept_in_query = any(
                keyword in query_lower
                for keyword in keywords
            )

            if not concept_in_query:
                continue

            concept_total += 1

            concept_found_in_document = any(
                keyword in normalized_document
                for keyword in keywords
            )

            if concept_found_in_document:
                concept_matches += 1

        policy_concept_score = (
            concept_matches / concept_total
            if concept_total > 0
            else 0.0
        )

        # -----------------------------------------------------
        # 5. ORIGINAL HYBRID SCORE
        # -----------------------------------------------------

        hybrid_score = float(
            result.get(
                "hybrid_score",
                0.0,
            )
        )

        # -----------------------------------------------------
        # 6. FINAL RERANKING SCORE
        # -----------------------------------------------------

        rerank_score = (
            (0.45 * term_coverage)
            + (0.25 * policy_concept_score)
            + (0.15 * exact_phrase_score)
            + (0.15 * hybrid_score)
        )

        return rerank_score

    # =========================================================
    # APPLY RERANKING
    # =========================================================

    def _rerank(
        self,
        query: str,
        results: List[Dict],
    ) -> List[Dict]:
        """
        Rerank the hybrid retrieval results.
        """

        for result in results:

            result["rerank_score"] = (
                self._calculate_rerank_score(
                    query=query,
                    result=result,
                )
            )

        reranked_results = sorted(
            results,
            key=lambda result: (
                result.get(
                    "rerank_score",
                    0.0,
                ),
                result.get(
                    "hybrid_score",
                    0.0,
                ),
            ),
            reverse=True,
        )

        # Assign final reranking rank.

        for rank, result in enumerate(
            reranked_results,
            start=1,
        ):
            result["rerank_rank"] = rank

        return reranked_results

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 10,
    ) -> List[Dict]:
        """
        Retrieve policy evidence using:

        BM25
            ↓
        Dense retrieval
            ↓
        Reciprocal Rank Fusion
            ↓
        Policy-aware reranking
            ↓
        Final top-k evidence
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        if retrieval_k <= 0:
            retrieval_k = 10

        # -----------------------------------------------------
        # 1. BM25 RETRIEVAL
        # -----------------------------------------------------

        bm25_results = (
            self.bm25_retriever.search(
                query=query,
                top_k=retrieval_k,
            )
        )

        # -----------------------------------------------------
        # 2. DENSE RETRIEVAL
        # -----------------------------------------------------

        dense_results = (
            self.dense_retriever.search(
                query=query,
                top_k=retrieval_k,
            )
        )

        # -----------------------------------------------------
        # 3. COMBINE RESULTS
        # -----------------------------------------------------

        combined_results: Dict[str, Dict] = {}

        # -----------------------------------------------------
        # Add BM25 results
        # -----------------------------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            chunk_id = (
                result.get("chunk_id")
                or result.get("id")
            )

            if not chunk_id:
                continue

            if chunk_id not in combined_results:

                combined_results[chunk_id] = {
                    "chunk_id": chunk_id,
                    "page_number": (
                        result.get("page_number")
                        or result.get("page")
                        or result.get("page_num")
                    ),
                    "text": result.get(
                        "text",
                        "",
                    ),
                    "bm25_rank": None,
                    "dense_rank": None,
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                    "hybrid_score": 0.0,
                    "rerank_score": 0.0,
                    "rerank_rank": None,
                }

            combined_results[
                chunk_id
            ]["bm25_rank"] = rank

            combined_results[
                chunk_id
            ]["bm25_score"] = float(
                result.get(
                    "bm25_score",
                    0.0,
                )
            )

        # -----------------------------------------------------
        # Add dense results
        # -----------------------------------------------------

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):

            chunk_id = (
                result.get("chunk_id")
                or result.get("id")
            )

            if not chunk_id:
                continue

            if chunk_id not in combined_results:

                combined_results[chunk_id] = {
                    "chunk_id": chunk_id,
                    "page_number": (
                        result.get("page_number")
                        or result.get("page")
                        or result.get("page_num")
                    ),
                    "text": result.get(
                        "text",
                        "",
                    ),
                    "bm25_rank": None,
                    "dense_rank": None,
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                    "hybrid_score": 0.0,
                    "rerank_score": 0.0,
                    "rerank_rank": None,
                }

            combined_results[
                chunk_id
            ]["dense_rank"] = rank

            dense_score = (
                result.get(
                    "dense_score"
                )
                if result.get(
                    "dense_score"
                ) is not None
                else result.get(
                    "score",
                    0.0,
                )
            )

            combined_results[
                chunk_id
            ]["dense_score"] = float(
                dense_score
            )

            if not combined_results[
                chunk_id
            ]["text"]:

                combined_results[
                    chunk_id
                ]["text"] = result.get(
                    "text",
                    "",
                )

            if not combined_results[
                chunk_id
            ]["page_number"]:

                combined_results[
                    chunk_id
                ]["page_number"] = (
                    result.get("page_number")
                    or result.get("page")
                    or result.get("page_num")
                )

        # -----------------------------------------------------
        # 4. RECIPROCAL RANK FUSION
        # -----------------------------------------------------

        fusion_constant = 60

        for result in combined_results.values():

            bm25_rank = result[
                "bm25_rank"
            ]

            dense_rank = result[
                "dense_rank"
            ]

            hybrid_score = 0.0

            if bm25_rank is not None:

                hybrid_score += 1 / (
                    fusion_constant
                    + bm25_rank
                )

            if dense_rank is not None:

                hybrid_score += 1 / (
                    fusion_constant
                    + dense_rank
                )

            result[
                "hybrid_score"
            ] = hybrid_score

        # -----------------------------------------------------
        # 5. HYBRID RANKING
        # -----------------------------------------------------

        hybrid_results = sorted(
            combined_results.values(),
            key=lambda result: result[
                "hybrid_score"
            ],
            reverse=True,
        )

        # -----------------------------------------------------
        # 6. POLICY-AWARE RERANKING
        # -----------------------------------------------------

        reranked_results = self._rerank(
            query=query,
            results=hybrid_results,
        )

        # -----------------------------------------------------
        # 7. RETURN FINAL RESULTS
        # -----------------------------------------------------

        return reranked_results[:top_k]


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    query = (
        "Does the policy cover a disease "
        "that existed before the policy started?"
    )

    retriever = HybridRetriever()

    results = retriever.search(
        query=query,
        top_k=5,
        retrieval_k=10,
    )

    print()
    print(
        f"Query: {query}"
    )

    print(
        f"Results found: {len(results)}"
    )

    print()

    for result in results:

        print(
            "--- Final Reranked Result ---"
        )

        print(
            f"Rank: "
            f"{result.get('rerank_rank')}"
        )

        print(
            f"Chunk ID: "
            f"{result.get('chunk_id')}"
        )

        print(
            f"Page: "
            f"{result.get('page_number')}"
        )

        print(
            f"BM25 rank: "
            f"{result.get('bm25_rank')}"
        )

        print(
            f"Dense rank: "
            f"{result.get('dense_rank')}"
        )

        print(
            f"BM25 score: "
            f"{result.get('bm25_score', 0.0):.4f}"
        )

        print(
            f"Dense score: "
            f"{result.get('dense_score', 0.0):.4f}"
        )

        print(
            f"Hybrid score: "
            f"{result.get('hybrid_score', 0.0):.6f}"
        )

        print(
            f"Rerank score: "
            f"{result.get('rerank_score', 0.0):.6f}"
        )

        print()

        print(
            result.get(
                "text",
                "",
            )[:500]
        )

        print(
            "-" * 80
        )