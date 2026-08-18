import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()


class PineconeVectorStore:
    def __init__(self, index_name: str | None = None):
        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY is not set"
            )

        self.index_name = index_name or os.getenv(
            "PINECONE_INDEX_NAME",
            "rag-pdf-qa",
        )

        # Create Pinecone client once
        self.client = Pinecone(
            api_key=api_key
        )

        # Create/reuse index connection once
        self.index = self.client.Index(
            self.index_name
        )

    def create_index(self):
        existing_indexes = (
            self.client.list_indexes().names()
        )

        if self.index_name not in existing_indexes:
            self.client.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )

            # Refresh index connection after creation
            self.index = self.client.Index(
                self.index_name
            )

    def get_index(self):
        return self.index

    def document_exists(
        self,
        document_hash: str,
    ) -> bool:
        result = self.index.query(
            vector=[1.0] * 768,
            top_k=1,
            filter={
                "document_hash": {
                    "$eq": document_hash
                }
            },
            include_metadata=False,
        )

        return len(result.matches) > 0

    def upsert_vectors(
        self,
        vectors: list[dict],
        batch_size: int = 100,
    ):
        for start in range(
            0,
            len(vectors),
            batch_size,
        ):
            batch = vectors[
                start:start + batch_size
            ]

            self.index.upsert(
                vectors=batch
            )

            uploaded = min(
                start + batch_size,
                len(vectors),
            )

            print(
                f"Uploaded "
                f"{uploaded}/{len(vectors)} vectors"
            )

    def delete_document(
        self,
        document_hash: str,
    ):
        self.index.delete(
            filter={
                "document_hash": {
                    "$eq": document_hash
                }
            }
        )

    def list_documents(self) -> list[dict]:
        """
        Return all unique documents currently
        indexed in Pinecone.
        """

        documents = {}

        # Get all vector IDs from the index.
        vector_ids = []

        for page in self.index.list():
            for item in page.vectors:
                vector_ids.append(item.id)
            

        # Fetch metadata in batches.
        batch_size = 100

        for start in range(
            0,
            len(vector_ids),
            batch_size,
        ):
            batch_ids = vector_ids[
                start:start + batch_size
            ]

            result = self.index.fetch(
                ids=batch_ids
            )

            for vector in result.vectors.values():

                metadata = vector.metadata or {}

                document_hash = metadata.get(
                    "document_hash"
                )

                document_name = metadata.get(
                    "document_name"
                )

                if not document_hash:
                    continue

                if document_hash not in documents:
                    documents[document_hash] = {
                        "document": document_name,
                        "document_hash": document_hash,
                        "chunks": 0,
                    }

                documents[
                    document_hash
                ]["chunks"] += 1

        return list(documents.values())

    def search(
        self,
        query_vector,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[dict]:

        if hasattr(
            query_vector,
            "tolist",
        ):
            query_vector = query_vector.tolist()

        result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        matches = []

        for match in result.matches:
            if (
                match.score
                < similarity_threshold
            ):
                continue

            metadata = match.metadata or {}

            matches.append(
                {
                    "id": match.id,
                    "score": match.score,
                    "text": metadata.get(
                        "text",
                        "",
                    ),
                    "document_id": metadata.get(
                        "document_id"
                    ),
                    "document_name": metadata.get(
                        "document_name"
                    ),
                    "page_number": metadata.get(
                        "page_number"
                    ),
                    "chunk_number": metadata.get(
                        "chunk_number"
                    ),
                }
            )

        return matches