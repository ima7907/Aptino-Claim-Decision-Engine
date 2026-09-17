from pathlib import Path
from pypdf import PdfReader


def load_policy_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of the policy PDF.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"Policy PDF not found: {pdf_file}")

    reader = PdfReader(str(pdf_file))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            {
                "page": page_number,
                "text": text.strip(),
            }
        )

    return pages


if __name__ == "__main__":
    policy_path = (
        "policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    )

    policy_pages = load_policy_pdf(policy_path)

    print(f"Total pages loaded: {len(policy_pages)}")

    if policy_pages:
        print("\nFirst page preview:\n")
        print(policy_pages[0]["text"][:1000])