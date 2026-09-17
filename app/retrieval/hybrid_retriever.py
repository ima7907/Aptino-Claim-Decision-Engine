from typing import List, Dict, Optional

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever


class HybridRetriever:
    """
    Hybrid retriever combining:

    1. BM25 keyword retrieval
    2. Dense semantic retrieval
    3. Reciprocal Rank Fusion
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

    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 10,
    ) -> List[Dict]:
        """
        Retrieve policy chunks using BM25 and dense retrieval,
        then combine the results using Reciprocal Rank Fusion.
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        if retrieval_k <= 0:
            retrieval_k = 10

        # -------------------------------------------------
        # BM25 retrieval
        # -------------------------------------------------

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=retrieval_k,
        )

        # -------------------------------------------------
        # Dense retrieval
        # -------------------------------------------------

        dense_results = self.dense_retriever.search(
            query=query,
            top_k=retrieval_k,
        )

        # -------------------------------------------------
        # Combine results by chunk ID
        # -------------------------------------------------

        combined_results: Dict[str, Dict] = {}

        # Add BM25 results
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
                    "text": result.get("text", ""),
                    "bm25_rank": None,
                    "dense_rank": None,
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                    "hybrid_score": 0.0,
                }

            combined_results[chunk_id]["bm25_rank"] = rank

            combined_results[chunk_id]["bm25_score"] = float(
                result.get("bm25_score", 0.0)
            )

        # Add dense results
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
                    "text": result.get("text", ""),
                    "bm25_rank": None,
                    "dense_rank": None,
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                    "hybrid_score": 0.0,
                }

            combined_results[chunk_id]["dense_rank"] = rank

            dense_score = (
                result.get("dense_score")
                if result.get("dense_score") is not None
                else result.get("score", 0.0)
            )

            combined_results[chunk_id]["dense_score"] = float(
                dense_score
            )

            # Use text from dense result if missing
            if not combined_results[chunk_id]["text"]:
                combined_results[chunk_id]["text"] = result.get(
                    "text",
                    "",
                )

            # Use page number from dense result if missing
            if not combined_results[chunk_id]["page_number"]:
                combined_results[chunk_id]["page_number"] = (
                    result.get("page_number")
                    or result.get("page")
                    or result.get("page_num")
                )

        # -------------------------------------------------
        # Reciprocal Rank Fusion
        # -------------------------------------------------

        fusion_constant = 60

        for result in combined_results.values():

            bm25_rank = result["bm25_rank"]
            dense_rank = result["dense_rank"]

            hybrid_score = 0.0

            if bm25_rank is not None:
                hybrid_score += 1 / (
                    fusion_constant + bm25_rank
                )

            if dense_rank is not None:
                hybrid_score += 1 / (
                    fusion_constant + dense_rank
                )

            result["hybrid_score"] = hybrid_score

        # -------------------------------------------------
        # Sort final results
        # -------------------------------------------------

        ranked_results = sorted(
            combined_results.values(),
            key=lambda result: result["hybrid_score"],
            reverse=True,
        )

        return ranked_results[:top_k]


if __name__ == "__main__":

    query = (
        "Does the policy cover a disease that existed "
        "before the policy started?"
    )

    retriever = HybridRetriever()

    results = retriever.search(
        query=query,
        top_k=5,
        retrieval_k=10,
    )

    print()
    print(f"Query: {query}")
    print(f"Results found: {len(results)}")
    print()

    for result in results:

        print("--- Hybrid Result ---")

        print(
            f"Chunk ID: {result.get('chunk_id')}"
        )

        print(
            f"Page: {result.get('page_number')}"
        )

        print(
            f"BM25 rank: {result.get('bm25_rank')}"
        )

        print(
            f"Dense rank: {result.get('dense_rank')}"
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
            result.get("text", "")[:500]
        )

        print("-" * 80)