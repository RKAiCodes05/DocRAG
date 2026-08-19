import json

from retrieval.retriever import Retriever


def main():
    with open("eval/test_questions.json", "r") as file:
        questions = json.load(file)

    retriever = Retriever(
        top_k=10,
        similarity_threshold=0.0,
    )

    for item in questions:
        question_id = item["id"]
        question = item["question"]

        print("\n" + "=" * 80)
        print(f"QUESTION {question_id}: {question}")
        print("=" * 80)

        results = retriever.retrieve(question)

        for rank, result in enumerate(results, start=1):
            print(
                f"\n[{rank}] "
                f"Score: {result['score']:.4f}"
            )

            print(
                f"Document: {result['document_name']}"
            )

            print(
                f"Page: {result['page_number']}"
            )

            print(
                f"Chunk: {result['chunk_number']}"
            )

            print(
                f"Chunk ID: {result['id']}"
            )

            print(
                f"Text: {result['text'][:500]}"
            )


if __name__ == "__main__":
    main()
