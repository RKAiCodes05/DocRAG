from fastapi import FastAPI

app = FastAPI(
    title="DocRAG: Document Retrieval-Augmented Generation Service",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}
