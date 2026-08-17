import time

from generation.generator import Generator
from retrieval.retriever import Retriever


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Flow:
        Question
            ↓
        Embedding
            ↓
        Pinecone retrieval
            ↓
        Similarity threshold filtering
            ↓
        Grounded generation
            ↓
        Answer + sources + latency
    """

    def __init__(
        self,
        top_k: int = 5,
        similarity_threshold: float = 0.65,
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        self.retriever = Retriever(
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        self.generator = Generator()

    def answer(self, question: str) -> dict:
        """
        Run the complete RAG pipeline.

        Returns:
            {
                "question": str,
                "answer": str,
                "sources": list,
                "retrieved_chunks": list,
                "latency": {
                    "retrieval_ms": float,
                    "generation_ms": float,
                    "total_ms": float
                }
            }
        """

        # --------------------------------------------------
        # Validate question
        # --------------------------------------------------

        if not isinstance(question, str):
            raise TypeError(
                "Question must be a string."
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        # --------------------------------------------------
        # Retrieval
        # --------------------------------------------------

        retrieval_start = time.perf_counter()

        retrieved_chunks = self.retriever.retrieve(
            question
        )

        retrieval_latency = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        # --------------------------------------------------
        # Generation
        # --------------------------------------------------

        generation_start = time.perf_counter()

        result = self.generator.generate(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        generation_latency = (
            time.perf_counter()
            - generation_start
        ) * 1000

        # --------------------------------------------------
        # Total latency
        # --------------------------------------------------

        total_latency = (
            retrieval_latency
            + generation_latency
        )

        # --------------------------------------------------
        # Return structured result
        # --------------------------------------------------

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": retrieved_chunks,
            "latency": {
                "retrieval_ms": round(
                    retrieval_latency,
                    2,
                ),
                "generation_ms": round(
                    generation_latency,
                    2,
                ),
                "total_ms": round(
                    total_latency,
                    2,
                ),
            },
        }
