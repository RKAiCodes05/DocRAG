from retrieval.embedder import Embedder
from retrieval.vector_store import PineconeVectorStore


class Retriever:
    def __init__(
        self,
        top_k: int = 5,
        similarity_threshold: float = 0.70,
        embedder: Embedder | None = None,
    ):
        # Reuse an existing Embedder when provided.
        # This prevents loading the embedding model multiple times.
        self.embedder = embedder or Embedder()

        self.vector_store = PineconeVectorStore()

        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str) -> list[dict]:
        query = query.strip()

        if not query:
            return []

        query_embedding = self.embedder.embed_query(
            query
        )

        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
        )

        return results

    def has_relevant_context(
        self,
        results: list[dict],
    ) -> bool:
        return len(results) > 0