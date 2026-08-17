from transformers import AutoTokenizer


class DocumentChunker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        chunk_size: int = 500,
        chunk_overlap: int = 75,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=10**6,
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page(self, text: str, metadata: dict) -> list[dict]:
        text = text.strip()

        if not text:
            return []

        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]

        chunks = []
        start = 0
        chunk_number = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))

            chunk_tokens = tokens[start:end]

            chunk_text = self.tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True,
            ).strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_number": chunk_number,
                        },
                    }
                )

            if end >= len(tokens):
                break

            start = end - self.chunk_overlap
            chunk_number += 1

        return chunks
