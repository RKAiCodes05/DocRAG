import time

from retrieval.vector_store import PineconeVectorStore


def run_query(index, vector, include_metadata):
    start = time.perf_counter()

    result = index.query(
        vector=vector,
        top_k=5,
        include_metadata=include_metadata,
    )

    latency = (time.perf_counter() - start) * 1000

    return latency, result


def main():
    store = PineconeVectorStore()
    index = store.get_index()

    response = index.fetch(
        ids=["Attention_is_all_you_need_p1_c0"]
    )

    vector = response.vectors[
        "Attention_is_all_you_need_p1_c0"
    ].values

    for include_metadata in [False, True]:
        latencies = []

        print(
            f"\ninclude_metadata={include_metadata}"
        )

        for i in range(10):
            latency, result = run_query(
                index,
                vector,
                include_metadata,
            )

            latencies.append(latency)

            print(
                f"{i + 1:02d}: "
                f"{latency:.2f} ms"
            )

        latencies.sort()

        p50 = latencies[4]
        p95 = latencies[8]

        print(f"P50: {p50:.2f} ms")
        print(f"P95: {p95:.2f} ms")


if __name__ == "__main__":
    main()

