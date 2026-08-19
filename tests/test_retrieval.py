from retrieval.retriever import Retriever


def main():
    retriever = Retriever(
        top_k=5,
        similarity_threshold=0.70,
    )

    queries = [
        "What is retrieval augmented generation?",
        "What is the Transformer architecture?",
        "What is FastAPI?",
        "What is the capital of Mars?",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = retriever.retrieve(query)

        if not results:
            print("\nNO_RELEVANT_CONTEXT")
            continue

        print(
            f"\nRelevant chunks: {len(results)}"
        )

        for i, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"\n[{i}] "
                f"Score: {result['score']:.4f}"
            )

            print(
                f"Source: "
                f"{result['document_name']}"
            )

            print(
                f"Page: "
                f"{result['page_number']}"
            )

            print(
                f"Text: "
                f"{result['text'][:250]}..."
            )


if __name__ == "__main__":
    main()
