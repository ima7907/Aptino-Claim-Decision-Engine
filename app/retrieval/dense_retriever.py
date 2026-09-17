import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        chunks_path: str = "data/processed/policy_chunks.json",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.chunks_path = Path(chunks_path)

        with open(self.chunks_path, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        self.embeddings = self.model.encode(
            [chunk["text"] for chunk in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        scores = np.dot(self.embeddings, query_embedding)

        ranked_indexes = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in ranked_indexes:
            results.append(
                {
                    **self.chunks[index],
                    "score": float(scores[index]),
                }
            )

        return results


if __name__ == "__main__":
    retriever = DenseRetriever()

    query = "Does the policy cover a disease that existed before the policy started?"

    results = retriever.search(
        query=query,
        top_k=3,
    )

    print(f"\nQuery: {query}")
    print(f"Results found: {len(results)}")

    for result in results:
        print("\n--- Result ---")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['score']:.4f}")
        print(result["text"][:500])