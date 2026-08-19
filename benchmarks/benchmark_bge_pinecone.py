import statistics
import time

from retrieval.embedder import Embedder
from retrieval.vector_store import PineconeVectorStore


def percentile(values, percentile):
    values = sorted(values)
    index = int((percentile / 100) * (len(values) - 1))
    return values[index]


def main():
    embedder = Embedder()
    store = PineconeVectorStore()
    index = store.get_index()

    query = "What is retrieval augmented generation?"

    query_vector = embedder.embed_query(query)

    print("Generated BGE query vector.")
    print(f"Dimensions: {len(query_vector)}")

    latencies = []

    for i in range(20):
        start = time.perf_counter()

        result = index.query(
            vector=query_vector.tolist(),
            top_k=5,
            include_metadata=True,
        )

        latency = (time.perf_counter() - start) * 1000

        latencies.append(latency)

        print(
            f"Query {i + 1:02d}: "
            f"{latency:.2f} ms"
        )

    print("\n" + "=" * 50)
    print("BGE VECTOR → PINECONE")
    print("=" * 50)

    print(
        f"P50: "
        f"{percentile(latencies, 50):.2f} ms"
    )

    print(
        f"P95: "
        f"{percentile(latencies, 95):.2f} ms"
    )

    print(
        f"Mean: "
        f"{statistics.mean(latencies):.2f} ms"
    )

    print(
        f"Min: "
        f"{min(latencies):.2f} ms"
    )

    print(
        f"Max: "
        f"{max(latencies):.2f} ms"
    )


if __name__ == "__main__":
    main()
