from dotenv import load_dotenv

from rag.pipeline import RAGPipeline


load_dotenv()


def main():
    pipeline = RAGPipeline(
        top_k=5,
        similarity_threshold=0.65,
    )

    questions = [
        "What is retrieval augmented generation?",
        "What is FastAPI?",
        "What is the capital of Mars?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        result = pipeline.answer(question)

        # --------------------------------------------------
        # Answer
        # --------------------------------------------------

        print("\nANSWER")
        print("-" * 80)
        print(result["answer"])

        # --------------------------------------------------
        # Latency
        # --------------------------------------------------

        print("\nLATENCY")
        print("-" * 80)

        latency = result["latency"]

        print(
            f"Retrieval:   "
            f"{latency['retrieval_ms']:.2f} ms"
        )

        print(
            f"Generation:  "
            f"{latency['generation_ms']:.2f} ms"
        )

        print(
            f"Total:       "
            f"{latency['total_ms']:.2f} ms"
        )

        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        print("\nSOURCES")
        print("-" * 80)

        if not result["sources"]:
            print("No sources.")
        else:
            for source in result["sources"]:
                print(
                    f"[Source {source['source']}] "
                    f"{source['document']} "
                    f"(page {source['page']}, "
                    f"chunk {source['chunk']}, "
                    f"score={source['score']:.4f})"
                )


if __name__ == "__main__":
    main()
