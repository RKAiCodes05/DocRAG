import statistics
import time

import numpy as np
from dotenv import load_dotenv

from retrieval.embedder import Embedder
from retrieval.vector_store import PineconeVectorStore


load_dotenv()


QUERY = "What is retrieval augmented generation?"
NUM_QUERIES = 20


def percentile(values, p):
    values = sorted(values)
    index = int((p / 100) * (len(values) - 1))
    return values[index]


def main():
    print("=" * 80)
    print("PINECONE WARM-UP BENCHMARK")
    print("=" * 80)

    embedder = Embedder()
    vector_store = PineconeVectorStore()

    print("\nGenerating query vector...")

    vector = embedder.embed_query(QUERY)

    if isinstance(vector, np.ndarray):
        vector = vector.tolist()

    print("Vector ready.")

    # --------------------------------------------------
    # Warm-up query
    # --------------------------------------------------

    print("\nWarm-up query...")

    start = time.perf_counter()

    vector_store.get_index().query(
        vector=vector,
        top_k=5,
        include_metadata=True,
    )

    warmup_time = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"Warm-up: {warmup_time:.2f} ms"
    )

    # --------------------------------------------------
    # Repeated queries
    # --------------------------------------------------

    latencies = []

    print(
        f"\nRunning {NUM_QUERIES} "
        "warm queries..."
    )

    for i in range(NUM_QUERIES):
        start = time.perf_counter()

        result = vector_store.get_index().query(
            vector=vector,
            top_k=5,
            include_metadata=True,
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(latency)

        print(
            f"Query {i + 1:02d}: "
            f"{latency:.2f} ms | "
            f"matches={len(result.matches)}"
        )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"P50:  "
        f"{percentile(latencies, 50):.2f} ms"
    )

    print(
        f"P95:  "
        f"{percentile(latencies, 95):.2f} ms"
    )

    print(
        f"Mean: "
        f"{statistics.mean(latencies):.2f} ms"
    )

    print(
        f"Min:  "
        f"{min(latencies):.2f} ms"
    )

    print(
        f"Max:  "
        f"{max(latencies):.2f} ms"
    )


if __name__ == "__main__":
    main()

