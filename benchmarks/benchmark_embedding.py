import time
import resource

from ingestion.pipeline import PDFIngestionPipeline
from retrieval.embedder import Embedder


def get_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024


pipeline = PDFIngestionPipeline()

print("Loading documents...")

chunks = []

from pathlib import Path

pdf_directory = Path("data/pdf")

for pdf_path in sorted(pdf_directory.glob("*.pdf")):
    pdf_chunks = pipeline.process_pdf(str(pdf_path))
    chunks.extend(pdf_chunks)

print(f"Total chunks: {len(chunks)}")

texts = [chunk["text"] for chunk in chunks]

print("\nLoading embedding model...")

start_load = time.perf_counter()

embedder = Embedder()

model_load_time = time.perf_counter() - start_load

print(f"Model load time: {model_load_time:.2f} seconds")
print(f"Memory after model load: {get_memory_mb():.2f} MB")

print("\nGenerating embeddings...")

start_embedding = time.perf_counter()

embeddings = embedder.embed_documents(
    texts,
    batch_size=16,
)

embedding_time = time.perf_counter() - start_embedding

print("\n========== BENCHMARK ==========")
print(f"Chunks:              {len(texts)}")
print(f"Embedding shape:     {embeddings.shape}")
print(f"Embedding time:      {embedding_time:.2f} seconds")
print(f"Memory usage:        {get_memory_mb():.2f} MB")
print(
    f"Average per chunk:   "
    f"{embedding_time / len(texts):.4f} seconds"
)
print("================================")
query = "What is retrieval augmented generation?"

start_query = time.perf_counter()

query_embedding = embedder.embed_query(query)

query_time = time.perf_counter() - start_query

print("\n========== QUERY ==========")
print(f"Query:               {query}")
print(f"Query shape:         {query_embedding.shape}")
print(f"Query time:          {query_time:.4f} seconds")
print("===========================")
