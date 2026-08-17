from pathlib import Path

import pymupdf

from retrieval.chunker import DocumentChunker


class PDFIngestionPipeline:
    def __init__(self):
        self.chunker = DocumentChunker()

    def process_pdf(self, pdf_path: str) -> list[dict]:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        document_id = pdf_path.stem

        chunks = []

        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text")

                metadata = {
                    "document_id": document_id,
                    "document_name": pdf_path.name,
                    "page_number": page_number,
                    "source_type": "pdf",
                }

                page_chunks = self.chunker.chunk_page(
                    text=text,
                    metadata=metadata,
                )

                for chunk in page_chunks:
                    chunk["metadata"]["chunk_id"] = (
                        f"{document_id}_"
                        f"p{page_number}_"
                        f"c{chunk['metadata']['chunk_number']}"
                    )

                chunks.extend(page_chunks)

        return chunks
