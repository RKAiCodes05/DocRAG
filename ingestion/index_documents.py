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


def main():
    pipeline = PDFIngestionPipeline()
    vector_store = PineconeVectorStore()

    vector_store.create_index()

    pdf_directory = Path("data/pdf")

    pdf_files = sorted(pdf_directory.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    # Check which documents actually need indexing first.
    documents_to_process = []

    for pdf_path in pdf_files:
        document_hash = get_file_hash(pdf_path)

        print(f"\nChecking: {pdf_path.name}")

        if vector_store.document_exists(document_hash):
            print(
                f"Already indexed: {pdf_path.name} "
                f"→ skipping"
            )
            continue

        documents_to_process.append(
            (pdf_path, document_hash)
        )

    # If everything is already indexed,
    # don't even load the embedding model.
    if not documents_to_process:
        print("\nAll documents are already indexed.")
        print("No embedding model was loaded.")
        return

    print(
        f"\nDocuments requiring indexing: "
        f"{len(documents_to_process)}"
    )

    embedder = Embedder()

    all_vectors = []

    for pdf_path, document_hash in documents_to_process:
        print(f"\nProcessing: {pdf_path.name}")

        chunks = pipeline.process_pdf(
            str(pdf_path)
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embedder.embed_documents(
            texts,
            batch_size=16,
        )

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            metadata = chunk["metadata"]

            metadata["document_hash"] = document_hash

            vector = {
                "id": metadata["chunk_id"],
                "values": embedding.tolist(),
                "metadata": {
                    **metadata,
                    "text": chunk["text"],
                },
            }

            all_vectors.append(vector)

        print(f"Chunks: {len(chunks)}")
        print(
            f"Document hash: "
            f"{document_hash[:16]}..."
        )

    print(
        f"\nTotal vectors to upload: "
        f"{len(all_vectors)}"
    )

    vector_store.upsert_vectors(
        all_vectors
    )

    print("\nIndexing complete.")


if __name__ == "__main__":
    main()
