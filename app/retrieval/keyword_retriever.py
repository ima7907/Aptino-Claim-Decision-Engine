import json
import re
from pathlib import Path


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "by",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "with",
}


def load_chunks(
    path: str = "data/processed/policy_chunks.json",
) -> list[dict]:
    """Load policy chunks from JSON."""

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def tokenize(text: str) -> list[str]:
    """
    Convert text into meaningful lowercase words.
    Common stopwords are removed.
    """

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )

    return [
        word
        for word in words
        if word not in STOPWORDS
    ]


def keyword_search(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Search policy chunks using meaningful keyword overlap.
    """

    query_words = set(tokenize(query))
    scored_chunks = []

    for chunk in chunks:
        chunk_words = set(tokenize(chunk["text"]))

        overlap_words = query_words.intersection(
            chunk_words
        )

        overlap_score = len(overlap_words)

        if overlap_score > 0:
            scored_chunks.append(
                {
                    **chunk,
                    "score": overlap_score,
                    "matched_words": sorted(overlap_words),
                }
            )

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_chunks[:top_k]


if __name__ == "__main__":
    chunks = load_chunks()

    query = "waiting period pre-existing disease"

    results = keyword_search(
        query=query,
        chunks=chunks,
        top_k=3,
    )

    print(f"Query: {query}")
    print(f"Results found: {len(results)}")

    for result in results:
        print("\n--- Result ---")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['score']}")
        print(
            "Matched words: "
            f"{result['matched_words']}"
        )
        print(result["text"][:500])