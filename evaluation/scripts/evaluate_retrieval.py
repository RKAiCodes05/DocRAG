import json
import statistics
import time

from retrieval.retriever import Retriever


GROUND_TRUTH_PATH = "eval/ground_truth.json"
RESULTS_PATH = "eval/results.json"

TOP_K_VALUES = [1, 3, 5]
SIMILARITY_THRESHOLD = 0.65


def percentile(values, percentile):
    values = sorted(values)

    index = int(
        (percentile / 100) * (len(values) - 1)
    )

    return values[index]


def load_ground_truth():
    with open(
        GROUND_TRUTH_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def is_hit(results, acceptable_ids, k):
    retrieved_ids = {
        result["id"]
        for result in results[:k]
    }

    return bool(
        retrieved_ids.intersection(
            acceptable_ids
        )
    )


def main():
    questions = load_ground_truth()

    retriever = Retriever(
        top_k=5,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    hits = {
        k: 0
        for k in TOP_K_VALUES
    }

    latencies = []
    failures = []
    question_results = []

    print("=" * 80)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 80)

    print(f"Questions: {len(questions)}")
    print(
        f"Similarity threshold: "
        f"{SIMILARITY_THRESHOLD}"
    )

    for item in questions:
        question_id = item["id"]
        question = item["question"]

        acceptable_ids = set(
            item["acceptable_chunk_ids"]
        )

        start = time.perf_counter()

        results = retriever.retrieve(
            question
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(latency)

        retrieved_ids = [
            result["id"]
            for result in results
        ]

        question_hits = {}

        for k in TOP_K_VALUES:
            hit = is_hit(
                results,
                acceptable_ids,
                k,
            )

            question_hits[f"hit_at_{k}"] = hit

            if hit:
                hits[k] += 1

        if not question_hits["hit_at_5"]:
            failures.append(
                {
                    "id": question_id,
                    "question": question,
                    "acceptable_chunk_ids": list(
                        acceptable_ids
                    ),
                    "retrieved_chunk_ids": retrieved_ids,
                }
            )

        question_results.append(
            {
                "id": question_id,
                "question": question,
                "latency_ms": round(
                    latency,
                    2,
                ),
                "acceptable_chunk_ids": list(
                    acceptable_ids
                ),
                "retrieved_chunk_ids": retrieved_ids,
                **question_hits,
            }
        )

        print(
            f"Q{question_id}: "
            f"{latency:.2f} ms"
        )

    total = len(questions)

    metrics = {}

    for k in TOP_K_VALUES:
        metrics[f"hit_at_{k}"] = round(
            (hits[k] / total) * 100,
            2,
        )

    metrics["latency_ms"] = {
        "p50": round(
            percentile(latencies, 50),
            2,
        ),
        "p95": round(
            percentile(latencies, 95),
            2,
        ),
        "mean": round(
            statistics.mean(latencies),
            2,
        ),
        "min": round(
            min(latencies),
            2,
        ),
        "max": round(
            max(latencies),
            2,
        ),
    }

    evaluation = {
        "evaluation": {
            "total_questions": total,
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "vector_database": "Pinecone",
            "similarity_metric": "cosine",
            "top_k": 5,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        },
        "metrics": metrics,
        "failures": failures,
        "questions": question_results,
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation,
            file,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for k in TOP_K_VALUES:
        print(
            f"Hit@{k}: "
            f"{hits[k]}/{total} "
            f"({metrics[f'hit_at_{k}']:.2f}%)"
        )

    print(
        f"P50: "
        f"{metrics['latency_ms']['p50']:.2f} ms"
    )

    print(
        f"P95: "
        f"{metrics['latency_ms']['p95']:.2f} ms"
    )

    print(
        f"\nResults saved to: "
        f"{RESULTS_PATH}"
    )

    print(
        f"Hit@5 failures: "
        f"{len(failures)}"
    )


if __name__ == "__main__":
    main()
