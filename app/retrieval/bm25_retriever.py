from pathlib import Path
import json
import re
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "policy_chunks.json"
)


# ---------------------------------------------------------
# Stopwords
# ---------------------------------------------------------

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "by",
    "with",
    "about",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "over",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "there",
    "does",
    "do",
    "did",
    "doing",
    "have",
    "has",
    "had",
    "having",
    "can",
    "could",
    "may",
    "might",
    "must",
    "should",
    "will",
    "would",
    "what",
    "which",
    "who",
    "whom",
    "where",
    "when",
    "why",
    "how",
    "i",
    "you",
    "he",
    "she",
    "we",
    "us",
    "my",
    "your",
    "his",
    "her",
    "our",
}


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """
    Convert text into lowercase tokens.

    Stopwords are removed to improve BM25 retrieval.
    """

    if not text:
        return []

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )

    tokens = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    return tokens


# ---------------------------------------------------------
# Load policy chunks
# ---------------------------------------------------------

def load_chunks() -> List[Dict]:
    """
    Load processed policy chunks from JSON.
    """

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Policy chunks file not found:\n{CHUNKS_FILE}"
        )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError(
            "policy_chunks.json must contain a list."
        )

    return chunks


# ---------------------------------------------------------
# Query expansion
# ---------------------------------------------------------

def expand_query(query: str) -> str:
    """
    Add policy-related terms to improve BM25 retrieval.

    This is a simple rule-based query expansion method.
    """

    expanded_query = query.lower()

    # Pre-existing disease query
    if (
        "existed before" in expanded_query
        or "before the policy started" in expanded_query
        or "existing disease" in expanded_query
        or "prior disease" in expanded_query
        or "disease before" in expanded_query
        or "pre existing" in expanded_query
        or "pre-existing" in expanded_query
    ):
        expanded_query += (
            " pre-existing "
            "preexisting "
            "pre existing "
            "pre-existing disease "
            "existing disease "
            "prior disease "
            "previous disease "
            "48 months "
            "continuous coverage "
            "inception "
            "first policy"
        )

    # Waiting-period queries
    if (
        "waiting period" in expanded_query
        or "wait" in expanded_query
    ):
        expanded_query += (
            " waiting period "
            "initial waiting period "
            "continuous coverage"
        )

    # Hospitalization queries
    if (
        "hospital" in expanded_query
        or "hospitalization" in expanded_query
    ):
        expanded_query += (
            " hospitalization "
            "in-patient "
            "inpatient "
            "nursing home "
            "medical treatment"
        )

    # Day-care treatment queries
    if (
        "day care" in expanded_query
        or "daycare" in expanded_query
    ):
        expanded_query += (
            " day care "
            "day-care "
            "daycare treatment "
            "less than 24 hours"
        )

    # Domiciliary treatment queries
    if (
        "domiciliary" in expanded_query
        or "home treatment" in expanded_query
    ):
        expanded_query += (
            " domiciliary treatment "
            "home treatment "
            "residence "
            "medical treatment"
        )

    # Experimental treatment queries
    if (
        "experimental" in expanded_query
        or "unproven" in expanded_query
    ):
        expanded_query += (
            " experimental treatment "
            "unproven treatment "
            "established medical practice"
        )

    return expanded_query


# ---------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------

class BM25Retriever:
    """
    Sparse keyword-based retriever using BM25.
    """

    def __init__(
        self,
        chunks: Optional[List[Dict]] = None,
    ):
        self.chunks = (
            chunks
            if chunks is not None
            else load_chunks()
        )

        self.tokenized_documents = [
            tokenize(
                chunk.get("text", "")
            )
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Search policy chunks using BM25.
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        expanded_query = expand_query(query)

        query_tokens = tokenize(
            expanded_query
        )

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indexes:
            score = float(
                scores[index]
            )

            # Ignore chunks with no matching terms
            if score <= 0:
                continue

            original_chunk = self.chunks[index]

            chunk = dict(
                original_chunk
            )

            # Support different page field names
            page_number = (
                chunk.get("page_number")
                or chunk.get("page")
                or chunk.get("page_num")
            )

            # Support different chunk ID field names
            chunk_id = (
                chunk.get("chunk_id")
                or chunk.get("id")
                or f"chunk_{index + 1}"
            )

            chunk["chunk_id"] = chunk_id
            chunk["page_number"] = page_number
            chunk["bm25_score"] = score
            chunk["retriever"] = "bm25"

            results.append(chunk)

            if len(results) >= top_k:
                break

        return results


# ---------------------------------------------------------
# Convenience function
# ---------------------------------------------------------

def bm25_search(
    query: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Convenience function for BM25 search.
    """

    retriever = BM25Retriever()

    return retriever.search(
        query=query,
        top_k=top_k,
    )


# ---------------------------------------------------------
# Test the retriever
# ---------------------------------------------------------

if __name__ == "__main__":

    query = (
        "Does the policy cover a disease that existed "
        "before the policy started?"
    )

    print("Loading policy chunks...")

    retriever = BM25Retriever()

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print()
    print(f"Query: {query}")
    print(f"Results found: {len(results)}")
    print()

    for result in results:

        print("--- BM25 Result ---")

        print(
            f"Chunk ID: {result.get('chunk_id')}"
        )

        print(
            f"Page: {result.get('page_number')}"
        )

        print(
            "BM25 score: "
            f"{result.get('bm25_score', 0):.4f}"
        )

        print(
            result.get("text", "")[:500]
        )

        print("-" * 80)