from pathlib import Path

from ingestion.pipeline import PDFIngestionPipeline


pipeline = PDFIngestionPipeline()

pdf_directory = Path("data/pdf")

total_chunks = 0

for pdf_path in sorted(pdf_directory.glob("*.pdf")):
    chunks = pipeline.process_pdf(str(pdf_path))

    print(f"{pdf_path.name:<50} {len(chunks):>5} chunks")

    total_chunks += len(chunks)

print("-" * 60)
print(f"Total chunks: {total_chunks}")
