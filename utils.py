"""
utils.py — PDF text extraction (no LangChain)
===============================================
Uses pypdf directly instead of LangChain's PyPDFLoader —
eliminates any chance of the 'proxies' conflict being triggered here.

Returns simple objects with .page_content and .metadata so rag.py
can iterate over them the same way as before.
"""

import io
from pypdf import PdfReader


class PageDoc:
    """
    Minimal document object — mimics LangChain's Document class
    so rag.py works without any changes to its interface.
    """
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata     = metadata


def extract_text_from_pdfs(uploaded_files) -> list[PageDoc]:
    """
    Extract text from a list of Streamlit UploadedFile objects.

    Reads each PDF page by page using pypdf directly (no LangChain).
    Each page becomes one PageDoc with:
        .page_content  — extracted text
        .metadata      — {"source": "filename.pdf", "page": 0-indexed int}

    Args:
        uploaded_files: List of st.UploadedFile objects from st.file_uploader().

    Returns:
        List of PageDoc objects, one per page across all PDFs.
    """
    all_docs = []

    for uploaded_file in uploaded_files:
        # Read file bytes directly — no temp file needed with pypdf
        file_bytes = uploaded_file.read()
        reader     = PdfReader(io.BytesIO(file_bytes))

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()

            # Skip blank or near-blank pages (image-only pages extract as "")
            if len(text) < 20:
                continue

            all_docs.append(PageDoc(
                page_content=text,
                metadata={
                    "source": uploaded_file.name,
                    "page":   page_num,        # 0-indexed; display as page_num+1
                },
            ))

    if not all_docs:
        raise ValueError(
            "No text could be extracted from the uploaded PDFs. "
            "Make sure they are text-based PDFs and not scanned images. "
            "For scanned PDFs you would need an OCR step (e.g. pytesseract)."
        )

    return all_docs


def format_source_citation(source: dict) -> str:
    """Format a source dict into a readable citation string."""
    return f"[{source['file']}, p.{source['page']}]: {source['snippet']}"
