import os

from dotenv import load_dotenv

from generation.generator import Generator


load_dotenv()


def main():
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file."
        )

    generator = Generator()

    # ============================================================
    # TEST 1 — Normal grounded generation
    # ============================================================

    print("\n" + "=" * 80)
    print("TEST 1 — GROUNDED GENERATION")
    print("=" * 80)

    chunks = [
        {
            "document_name": "RAG_for_LLM.pdf",
            "page_number": 1,
            "chunk_number": 1,
            "score": 0.81,
            "text": (
                "Retrieval-augmented generation (RAG) "
                "enhances large language models by "
                "retrieving relevant document chunks "
                "from an external knowledge base."
            ),
        }
    ]

    result = generator.generate(
        question="What is retrieval-augmented generation?",
        retrieved_chunks=chunks,
    )

    print("\nANSWER")
    print("-" * 80)
    print(result["answer"])

    print("\nSOURCES")
    print("-" * 80)

    for source in result["sources"]:
        print(source)

    # ============================================================
    # TEST 2 — No-context / hallucination guard
    # ============================================================

    print("\n" + "=" * 80)
    print("TEST 2 — NO CONTEXT")
    print("=" * 80)

    result = generator.generate(
        question="What is the capital of Mars?",
        retrieved_chunks=[],
    )

    print("\nANSWER")
    print("-" * 80)
    print(result["answer"])

    print("\nSOURCES")
    print("-" * 80)
    print(result["sources"])

    # ============================================================
    # TEST 3 — Multiple sources
    # ============================================================

    print("\n" + "=" * 80)
    print("TEST 3 — MULTIPLE SOURCES")
    print("=" * 80)

    chunks = [
        {
            "document_name": "RAG_for_LLM.pdf",
            "page_number": 1,
            "chunk_number": 1,
            "score": 0.86,
            "text": (
                "Retrieval-augmented generation enhances "
                "large language models by retrieving "
                "relevant document chunks from an "
                "external knowledge base."
            ),
        },
        {
            "document_name": "RAG_eval.pdf",
            "page_number": 1,
            "chunk_number": 1,
            "score": 0.79,
            "text": (
                "RAG evaluation assesses the quality "
                "of retrieval and generation in a "
                "retrieval-augmented generation system."
            ),
        },
    ]

    result = generator.generate(
        question="What is RAG and how can it be evaluated?",
        retrieved_chunks=chunks,
    )

    print("\nANSWER")
    print("-" * 80)
    print(result["answer"])

    print("\nSOURCES")
    print("-" * 80)

    for source in result["sources"]:
        print(source)


if __name__ == "__main__":
    main()
