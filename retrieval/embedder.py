from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 16,
    ):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str):
        return self.model.encode(
            query,
            normalize_embeddings=True,
        )
