import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever


class HybridRetriever:
    """
    Hybrid policy retriever using:

    1. BM25 sparse retrieval
    2. Dense semantic retrieval
    3. Reciprocal Rank Fusion (RRF)
    4. Lightweight policy-aware reranking

    Dense retrieval can be disabled for low-memory deployment by setting:

        DISABLE_DENSE_RETRIEVAL=true

    When disabled, BM25 retrieval + policy-aware reranking still operate.
    """

    def __init__(
        self,
        chunks_path: str = "data/processed/policy_chunks.json",
    ):
        print("Initializing Hybrid Retriever...")

        self.chunks_path = Path(chunks_path)

        # ------------------------------------------------------------
        # Load policy chunks
        # ------------------------------------------------------------

        import json

        with open(
            self.chunks_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.chunks = json.load(file)

        # ------------------------------------------------------------
        # Load BM25 retriever
        # ------------------------------------------------------------

        print("Loading BM25 retriever...")

        self.bm25_retriever = BM25Retriever(
            chunks=self.chunks
        )

        # ------------------------------------------------------------
        # Dense retrieval switch
        # ------------------------------------------------------------

        self.use_dense = (
            os.getenv(
                "DISABLE_DENSE_RETRIEVAL",
                "false",
            ).lower()
            != "true"
        )

        if self.use_dense:

            print("Loading dense retriever...")

            self.dense_retriever = DenseRetriever(
                chunks_path=str(
                    self.chunks_path
                )
            )

        else:

            print(
                "Dense retrieval disabled for low-memory deployment."
            )

            self.dense_retriever = None

    # ================================================================
    # TEXT HELPERS
    # ================================================================

    @staticmethod
    def _normalize_text(text: str) -> str:

        if not text:
            return ""

        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:

        text = HybridRetriever._normalize_text(
            text
        )

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text,
        )

    # ================================================================
    # POLICY CONCEPTS
    # ================================================================

    @staticmethod
    def _policy_concepts() -> Dict[str, List[str]]:

        return {

            "coverage": [
                "coverage",
                "covered",
                "cover",
                "admissible",
            ],

            "pre_existing": [
                "pre-existing",
                "pre existing",
                "preexisting",
            ],

            "waiting_period": [
                "waiting period",
                "waiting",
                "continuous coverage",
                "48 months",
                "forty eight months",
            ],

            "hospitalization": [
                "hospitalization",
                "hospitalisation",
                "hospital",
                "inpatient",
                "admission",
            ],

            "exclusion": [
                "exclusion",
                "excluded",
                "not covered",
            ],

            "treatment": [
                "treatment",
                "procedure",
                "surgery",
                "medical treatment",
            ],

            "portability": [
                "portability",
                "previous insurance",
                "previous coverage",
                "continuous insurance",
            ],

            "experimental": [
                "experimental",
                "investigational",
                "unproven",
            ],
        }

    # ================================================================
    # TERM COVERAGE
    # ================================================================

    def _calculate_term_coverage(
        self,
        query: str,
        text: str,
    ) -> float:

        query_tokens = set(
            self._tokenize(query)
        )

        if not query_tokens:
            return 0.0

        text_tokens = set(
            self._tokenize(text)
        )

        matched = query_tokens.intersection(
            text_tokens
        )

        return len(matched) / len(
            query_tokens
        )

    # ================================================================
    # POLICY CONCEPT SCORE
    # ================================================================

    def _calculate_policy_concept_score(
        self,
        query: str,
        text: str,
    ) -> float:

        query_normalized = (
            self._normalize_text(query)
        )

        text_normalized = (
            self._normalize_text(text)
        )

        concepts = self._policy_concepts()

        matched_concepts = 0
        query_concepts = 0

        for phrases in concepts.values():

            query_has_concept = any(
                phrase in query_normalized
                for phrase in phrases
            )

            if query_has_concept:

                query_concepts += 1

                chunk_has_concept = any(
                    phrase in text_normalized
                    for phrase in phrases
                )

                if chunk_has_concept:
                    matched_concepts += 1

        if query_concepts == 0:
            return 0.0

        return (
            matched_concepts
            / query_concepts
        )

    # ================================================================
    # EXACT PHRASE SCORE
    # ================================================================

    def _calculate_exact_phrase_score(
        self,
        query: str,
        text: str,
    ) -> float:

        query_normalized = (
            self._normalize_text(query)
        )

        text_normalized = (
            self._normalize_text(text)
        )

        if (
            not query_normalized
            or not text_normalized
        ):
            return 0.0

        query_tokens = self._tokenize(
            query_normalized
        )

        if len(query_tokens) < 2:
            return 0.0

        phrases = []

        for index in range(
            len(query_tokens) - 1
        ):

            phrases.append(
                f"{query_tokens[index]} "
                f"{query_tokens[index + 1]}"
            )

        matched = sum(
            1
            for phrase in phrases
            if phrase in text_normalized
        )

        if not phrases:
            return 0.0

        return matched / len(
            phrases
        )

    # ================================================================
    # RERANKING
    # ================================================================

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if not results:
            return []

        hybrid_scores = [
            float(
                result.get(
                    "hybrid_score",
                    0.0,
                )
            )
            for result in results
        ]

        max_hybrid = (
            max(hybrid_scores)
            if hybrid_scores
            else 0.0
        )

        min_hybrid = (
            min(hybrid_scores)
            if hybrid_scores
            else 0.0
        )

        hybrid_range = (
            max_hybrid
            - min_hybrid
        )

        for result in results:

            text = result.get(
                "text",
                "",
            )

            term_coverage = (
                self._calculate_term_coverage(
                    query,
                    text,
                )
            )

            policy_concept_score = (
                self._calculate_policy_concept_score(
                    query,
                    text,
                )
            )

            exact_phrase_score = (
                self._calculate_exact_phrase_score(
                    query,
                    text,
                )
            )

            raw_hybrid = float(
                result.get(
                    "hybrid_score",
                    0.0,
                )
            )

            if hybrid_range > 0:

                normalized_hybrid = (
                    (
                        raw_hybrid
                        - min_hybrid
                    )
                    / hybrid_range
                )

            else:

                normalized_hybrid = 1.0

            rerank_score = (
                0.45 * term_coverage
                + 0.25 * policy_concept_score
                + 0.15 * exact_phrase_score
                + 0.15 * normalized_hybrid
            )

            result[
                "term_coverage_score"
            ] = round(
                term_coverage,
                6,
            )

            result[
                "policy_concept_score"
            ] = round(
                policy_concept_score,
                6,
            )

            result[
                "exact_phrase_score"
            ] = round(
                exact_phrase_score,
                6,
            )

            result[
                "normalized_hybrid_score"
            ] = round(
                normalized_hybrid,
                6,
            )

            result[
                "rerank_score"
            ] = round(
                rerank_score,
                6,
            )

        results.sort(
            key=lambda item: item.get(
                "rerank_score",
                0.0,
            ),
            reverse=True,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            result[
                "rerank_rank"
            ] = rank

        return results

    # ================================================================
    # RECIPROCAL RANK FUSION
    # ================================================================

    @staticmethod
    def _reciprocal_rank_fusion(
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:

        fused_scores = defaultdict(
            float
        )

        documents = {}

        # ------------------------------------------------------------
        # BM25
        # ------------------------------------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            chunk_id = result.get(
                "chunk_id"
            )

            if chunk_id is None:
                continue

            fused_scores[
                chunk_id
            ] += 1.0 / (
                k + rank
            )

            documents[
                chunk_id
            ] = {
                **documents.get(
                    chunk_id,
                    {},
                ),
                **result,
            }

            documents[
                chunk_id
            ][
                "bm25_rank"
            ] = rank

        # ------------------------------------------------------------
        # Dense
        # ------------------------------------------------------------

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):

            chunk_id = result.get(
                "chunk_id"
            )

            if chunk_id is None:
                continue

            fused_scores[
                chunk_id
            ] += 1.0 / (
                k + rank
            )

            documents[
                chunk_id
            ] = {
                **documents.get(
                    chunk_id,
                    {},
                ),
                **result,
            }

            documents[
                chunk_id
            ][
                "dense_rank"
            ] = rank

        # ------------------------------------------------------------
        # Build fused results
        # ------------------------------------------------------------

        fused_results = []

        for (
            chunk_id,
            score,
        ) in fused_scores.items():

            result = documents[
                chunk_id
            ]

            result[
                "hybrid_score"
            ] = float(score)

            fused_results.append(
                result
            )

        fused_results.sort(
            key=lambda item: item.get(
                "hybrid_score",
                0.0,
            ),
            reverse=True,
        )

        return fused_results

    # ================================================================
    # MAIN SEARCH
    # ================================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 10,
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():
            return []

        # ------------------------------------------------------------
        # BM25
        # ------------------------------------------------------------

        bm25_results = (
            self.bm25_retriever.search(
                query=query,
                top_k=retrieval_k,
            )
        )

        # ------------------------------------------------------------
        # Dense
        # ------------------------------------------------------------

        if self.use_dense:

            dense_results = (
                self.dense_retriever.search(
                    query=query,
                    top_k=retrieval_k,
                )
            )

        else:

            dense_results = []

        # ------------------------------------------------------------
        # RRF
        # ------------------------------------------------------------

        fused_results = (
            self._reciprocal_rank_fusion(
                bm25_results=bm25_results,
                dense_results=dense_results,
            )
        )

        # ------------------------------------------------------------
        # Reranking candidates
        # ------------------------------------------------------------

        rerank_candidates = fused_results[
            : max(
                retrieval_k,
                top_k,
            )
        ]

        # ------------------------------------------------------------
        # Policy-aware reranking
        # ------------------------------------------------------------

        reranked_results = (
            self._rerank_results(
                query=query,
                results=rerank_candidates,
            )
        )

        # ------------------------------------------------------------
        # Final results
        # ------------------------------------------------------------

        final_results = (
            reranked_results[:top_k]
        )

        # ------------------------------------------------------------
        # Retrieval metadata
        # ------------------------------------------------------------

        for result in final_results:

            if self.use_dense:

                result[
                    "retrieval_method"
                ] = "hybrid_bm25_dense"

            else:

                result[
                    "retrieval_method"
                ] = "bm25_low_memory"

            result[
                "dense_retrieval_enabled"
            ] = self.use_dense

            result[
                "reranking_applied"
            ] = True

        return final_results