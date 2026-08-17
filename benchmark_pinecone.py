import statistics
import time

from retrieval.vector_store import PineconeVectorStore


def percentile(values, percentile):
    values = sorted(values)

    index = int(
        (percentile / 100) * (len(values) - 1)
    )

    return values[index]


def main():
    vector_store = PineconeVectorStore()
    index = vector_store.get_index()

    # Fetch one existing vector from Pinecone.
    response = index.fetch(
        ids=["Attention_is_all_you_need_p1_c0"]
    )

    vector = response.vectors[
        "Attention_is_all_you_need_p1_c0"
    ].values

    print("Vector loaded from Pinecone.")
    print(f"Vector dimensions: {len(vector)}")

    latencies = []

    print("\nRunning 20 raw Pinecone queries...\n")

    for i in range(20):
        start = time.perf_counter()

        result = index.query(
            vector=vector,
            top_k=5,
            include_metadata=False,
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

    print("\n" + "=" * 50)
    print("RAW PINECONE BENCHMARK")
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
