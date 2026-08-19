import json

from retrieval.retriever import Retriever


QUESTIONS_PATH = "eval/negative_questions.json"
RESULTS_PATH = "eval/negative_results.json"

THRESHOLD = 0.65


def main():
    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    retriever = Retriever(
        top_k=5,
        similarity_threshold=THRESHOLD,
    )

    rejected = 0
    question_results = []

    print("=" * 80)
    print("NEGATIVE QUERY EVALUATION")
    print("=" * 80)

    print(
        f"Similarity threshold: {THRESHOLD}"
    )

    for item in questions:
        results = retriever.retrieve(
            item["question"]
        )

        no_context = len(results) == 0

        if no_context:
            rejected += 1

        retrieved = []

        for result in results:
            retrieved.append(
                {
                    "id": result["id"],
                    "score": result["score"],
                    "document": result["document_name"],
                    "page": result["page_number"],
                }
            )

        question_results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "rejected": no_context,
                "retrieved": retrieved,
            }
        )

        print(
            f"\nQ{item['id']}: "
            f"{item['question']}"
        )

        if no_context:
            print("Result: NO_RELEVANT_CONTEXT")
        else:
            print("Result: CONTEXT RETURNED")

    total = len(questions)

    rejection_rate = (
        rejected / total
    ) * 100

    evaluation = {
        "evaluation": {
            "total_negative_questions": total,
            "similarity_threshold": THRESHOLD,
            "embedding_model": (
                "BAAI/bge-base-en-v1.5"
            ),
            "vector_database": "Pinecone",
            "similarity_metric": "cosine",
            "top_k": 5,
        },
        "metrics": {
            "correct_rejections": rejected,
            "rejection_rate": round(
                rejection_rate,
                2,
            ),
        },
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
    print("RESULTS")
    print("=" * 80)

    print(
        f"Rejected: "
        f"{rejected}/{total}"
    )

    print(
        f"Rejection rate: "
        f"{rejection_rate:.2f}%"
    )

    print(
        f"\nResults saved to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
