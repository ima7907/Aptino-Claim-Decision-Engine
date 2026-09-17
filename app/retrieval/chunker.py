from app.retrieval.pdf_loader import load_policy_pdf


def create_chunks(
    pages: list[dict],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """
    Split policy pages into overlapping text chunks.
    """

    chunks = []
    chunk_counter = 1

    for page in pages:
        page_number = page["page"]
        text = page["text"]

        if not text:
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_counter}",
                        "page": page_number,
                        "section": f"Page {page_number}",
                        "text": chunk_text,
                    }
                )

                chunk_counter += 1

            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    import json
    from pathlib import Path

    policy_path = (
        "policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    )

    pages = load_policy_pdf(policy_path)
    chunks = create_chunks(pages)

    output_path = Path("data/processed/policy_chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Chunks saved to: {output_path}")