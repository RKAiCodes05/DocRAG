import statistics
import time

from dotenv import load_dotenv

from rag.pipeline import RAGPipeline


load_dotenv()


QUESTIONS = [
    "What is retrieval augmented generation?",
    "What is the Transformer architecture?",
    "What is FastAPI?",
    "What is Docker?",
    "What is a sentence transformer?",
    "What are embeddings?",
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
    pipeline = RAGPipeline(
        top_k=5,
        similarity_threshold=0.65,
    )

    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []

    print("=" * 80)
    print("END-TO-END RAG BENCHMARK")
    print("=" * 80)

    for index, question in enumerate(
        QUESTIONS,
        start=1,
    ):
        print(
            f"\nQuery {index}: "
            f"{question}"
        )

        start = time.perf_counter()

        result = pipeline.answer(
            question
        )

        total_time = (
            time.perf_counter()
            - start
        ) * 1000

        retrieval = result["latency"][
            "retrieval_ms"
        ]

        generation = result["latency"][
            "generation_ms"
        ]

        total = result["latency"][
            "total_ms"
        ]

        retrieval_latencies.append(
            retrieval
        )

        generation_latencies.append(
            generation
        )

        total_latencies.append(
            total
        )

        print(
            f"Retrieval:  {retrieval:.2f} ms"
        )

        print(
            f"Generation: {generation:.2f} ms"
        )

        print(
            f"Total:      {total:.2f} ms"
        )

        print(
            f"Measured:   {total_time:.2f} ms"
        )

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    print("\nRetrieval")
    print(
        f"P50:  "
        f"{percentile(retrieval_latencies, 50):.2f} ms"
    )
    print(
        f"P95:  "
        f"{percentile(retrieval_latencies, 95):.2f} ms"
    )
    print(
        f"Mean: "
        f"{statistics.mean(retrieval_latencies):.2f} ms"
    )

    print("\nGeneration")
    print(
        f"P50:  "
        f"{percentile(generation_latencies, 50):.2f} ms"
    )
    print(
        f"P95:  "
        f"{percentile(generation_latencies, 95):.2f} ms"
    )
    print(
        f"Mean: "
        f"{statistics.mean(generation_latencies):.2f} ms"
    )

    print("\nTotal RAG Pipeline")
    print(
        f"P50:  "
        f"{percentile(total_latencies, 50):.2f} ms"
    )
    print(
        f"P95:  "
        f"{percentile(total_latencies, 95):.2f} ms"
    )
    print(
        f"Mean: "
        f"{statistics.mean(total_latencies):.2f} ms"
    )


if __name__ == "__main__":
    main()
