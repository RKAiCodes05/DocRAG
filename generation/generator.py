import os

from groq import Groq


class Generator:
    def __init__(
        self,
        model: str = "qwen/qwen3.6-27b",
    ):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> dict:

        # Do not allow the LLM to answer without
        # retrieved evidence.
        if not retrieved_chunks:
            return {
                "answer": (
                    "I couldn't find relevant information "
                    "in the provided documents to answer "
                    "this question."
                ),
                "sources": [],
            }

        context_parts = []

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            context_parts.append(
                f"""
SOURCE {index}
Document: {chunk['document_name']}
Page: {chunk['page_number']}
Chunk: {chunk['chunk_number']}

Content:
{chunk['text']}
"""
            )

        context = "\n".join(context_parts)

        system_prompt = """
You are a document question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain enough information
   to answer the question, say that the information
   is not available in the provided documents.
4. Keep the answer concise and directly relevant.
5. Cite the supporting sources using [Source N] notation.
6. Only cite sources that actually support your answer.
"""

        user_prompt = f"""
DOCUMENT CONTEXT
================

{context}

QUESTION
========

{question}

Answer the question using only the document context.
Include source citations such as [Source 1].
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content

        sources = []

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            sources.append(
                {
                    "source": index,
                    "document": chunk["document_name"],
                    "page": chunk["page_number"],
                    "chunk": chunk["chunk_number"],
                    "score": chunk["score"],
                }
            )

        return {
            "answer": answer,
            "sources": sources,
        }
