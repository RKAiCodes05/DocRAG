import hashlib
from pathlib import Path

from ingestion.pipeline import PDFIngestionPipeline
from retrieval.embedder import Embedder
from retrieval.vector_store import PineconeVectorStore


def get_file_hash(pdf_path: Path) -> str:
    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()



class DocumentIngestionService:
    def __init__(
        self,
        embedder: Embedder,
    ):
        self.pipeline = PDFIngestionPipeline()

        self.vector_store = (
            PineconeVectorStore()
        )

        # Reuse the same Embedder used by RAG.
        self.embedder = embedder

    def ingest(
        self,
        pdf_path: Path,
        original_filename: str,
    ) -> dict:

        document_hash = get_file_hash(
            pdf_path
        )

        # --------------------------------------------------
        # Duplicate check
        # --------------------------------------------------

        if self.vector_store.document_exists(
            document_hash
        ):
            return {
                "status": "already_exists",
                "document": pdf_path.name,
                "document_hash": document_hash,
                "chunks": 0,
            }

        # --------------------------------------------------
        # PDF processing
        # --------------------------------------------------

        chunks = self.pipeline.process_pdf(
            str(pdf_path)
        )
        for chunk in chunks:
            chunk["metadata"]["document_name"] = original_filename
    

        if not chunks:
            raise ValueError(
                "No text could be extracted from the PDF."
            )

        # --------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = (
            self.embedder.embed_documents(
                texts,
                batch_size=16,
            )
        )

        # --------------------------------------------------
        # Build Pinecone vectors
        # --------------------------------------------------

        vectors = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            metadata = {
                **chunk["metadata"],
                "document_hash": document_hash,
                "text": chunk["text"],
            }

            vector = {
                "id": metadata["chunk_id"],
                "values": embedding.tolist(),
                "metadata": metadata,
            }

            vectors.append(vector)

        # --------------------------------------------------
        # Upload to Pinecone
        # --------------------------------------------------

        self.vector_store.upsert_vectors(
            vectors
        )

        return {
            "status": "indexed",
            "document": pdf_path.name,
            "document_hash": document_hash,
            "chunks": len(chunks),
        }
    def delete(
        self,
        document_hash: str,
    ) -> dict:

        deleted = self.vector_store.delete_document(
        document_hash
        )   

        return {
            "status": "deleted",
            "document_hash": document_hash,
            "result": deleted,
        }
    def list_documents(self) -> list[dict]:
        return self.vector_store.list_documents()