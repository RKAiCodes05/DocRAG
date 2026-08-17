from dotenv import load_dotenv

from generation.generator import Generator


load_dotenv()


def print_result(test_name, result):
    print("\n" + "=" * 80)
    print(test_name)
    print("=" * 80)

    print("\nANSWER")
    print("-" * 80)
    print(result["answer"])

    print("\nSOURCES")
    print("-" * 80)

    if not result["sources"]:
        print("No sources.")
    else:
        for source in result["sources"]:
            print(
                f"[Source {source['source']}] "
                f"{source['document']} "
                f"page={source['page']} "
                f"chunk={source['chunk']} "
                f"score={source['score']:.4f}"
            )


def main():
    generator = Generator()

    # ==========================================================
    # TEST 1 — Relevant single source
    # ==========================================================

    chunks = [
        {
            "document_name": "RAG_for_LLM.pdf",
            "page_number": 1,
            "chunk_number": 1,
            "score": 0.86,
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

    print_result(
        "TEST 1 — RELEVANT SINGLE SOURCE",
        result,
    )

    # ==========================================================
    # TEST 2 — Empty context
    # ==========================================================

    result = generator.generate(
        question="What is the capital of Mars?",
        retrieved_chunks=[],
    )

    print_result(
        "TEST 2 — EMPTY CONTEXT",
        result,
    )

    # ==========================================================
    # TEST 3 — Multiple relevant sources
    # ==========================================================

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
            "score": 0.81,
            "text": (
                "RAG evaluation assesses the quality "
                "of retrieval and generation in a "
                "retrieval-augmented generation system."
            ),
        },
    ]

    result = generator.generate(
        question="What is RAG and how is it evaluated?",
        retrieved_chunks=chunks,
    )

    print_result(
        "TEST 3 — MULTIPLE RELEVANT SOURCES",
        result,
    )

    # ==========================================================
    # TEST 4 — Irrelevant context
    # ==========================================================

    chunks = [
        {
            "document_name": "Mastering_FastAPI_with_Python.pdf",
            "page_number": 10,
            "chunk_number": 0,
            "score": 0.66,
            "text": (
                "FastAPI is a modern high-performance "
                "web framework for building APIs with Python."
            ),
        }
    ]

    result = generator.generate(
        question="What is the Transformer architecture?",
        retrieved_chunks=chunks,
    )

    print_result(
        "TEST 4 — IRRELEVANT CONTEXT",
        result,
    )


if __name__ == "__main__":
    main()
