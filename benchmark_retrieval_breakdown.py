import statistics
import time

import numpy as np
from dotenv import load_dotenv

from retrieval.embedder import Embedder
from retrieval.vector_store import PineconeVectorStore


load_dotenv()


QUERIES = [
    "What is retrieval augmented generation?",
    "What is the Transformer architecture?",
    "What is FastAPI?",
    "What is Docker?",
    "What are embeddings?",
    "What is a sentence transformer?",
    "What is Hugging Face Transformers?",
    "What is self-attention?",
    "What are the challenges of RAG?",
    "What is retrieval in RAG?",
]


def percentile(values, p):
    values = sorted(values)

    index = int(
        (p / 100) * (len(values) - 1)
    )

    return values[index]


def main():
    print("=" * 80)
    print("RETRIEVAL LATENCY BREAKDOWN")
    print("=" * 80)

    embedder = Embedder()
    vector_store = PineconeVectorStore()

    embedding_times = []
    pinecone_times = []
    total_times = []

    for index, query in enumerate(
        QUERIES,
        start=1,
    ):
        print(
            f"\nQuery {index}: {query}"
        )

        # --------------------------------------------------
        # 1. Query embedding
        # --------------------------------------------------

        start = time.perf_counter()

        vector = embedder.embed_query(
            query
        )

        embedding_time = (
            time.perf_counter()
            - start
        ) * 1000

        # Make sure Pinecone receives a
        # normal Python list.
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        # --------------------------------------------------
        # 2. Direct Pinecone query
        # --------------------------------------------------

        start = time.perf_counter()

        vector_store.get_index().query(
            vector=vector,
            top_k=5,
            include_metadata=True,
        )

        pinecone_time = (
            time.perf_counter()
            - start
        ) * 1000

        # --------------------------------------------------
        # 3. Total retrieval components
        # --------------------------------------------------

        total_time = (
            embedding_time
            + pinecone_time
        )

        embedding_times.append(
            embedding_time
        )

        pinecone_times.append(
            pinecone_time
        )

        total_times.append(
            total_time
        )

        print(
            f"Embedding: {embedding_time:.2f} ms"
        )

        print(
            f"Pinecone:  {pinecone_time:.2f} ms"
        )

        print(
            f"Combined:  {total_time:.2f} ms"
        )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\nEmbedding")
    print(
        f"P50:  "
        f"{percentile(embedding_times, 50):.2f} ms"
    )
    print(
        f"P95:  "
        f"{percentile(embedding_times, 95):.2f} ms"
    )
    print(
        f"Mean: "
        f"{statistics.mean(embedding_times):.2f} ms"
    )

    print("\nPinecone")
    print(
        f"P50:  "
        f"{percentile(pinecone_times, 50):.2f} ms"
    )
    print(
        f"P95:  "
        f"{percentile(pinecone_times, 95):.2f} ms"
    )
    print(
        f"Mean: "
        f"{statistics.mean(pinecone_times):.2f} ms"
    )

    print("\nEmbedding + Pinecone")
    print(
        f"P50:  "
        f"{percentile(total_times, 50):.2f} ms"
    )
    print(
        f"P95:  "
        f"{percentile(total_times, 95):.2f} ms"
    )
    print(
        f"Mean: "
        f"{statistics.mean(total_times):.2f} ms"
    )


if __name__ == "__main__":
    main()
