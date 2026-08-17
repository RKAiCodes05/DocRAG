import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("rag-api")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="RAG PDF QA Service",
    description="""
## Retrieval-Augmented Generation API

A production-oriented RAG service that:

- Generates embeddings using BGE
- Retrieves documents from Pinecone
- Generates grounded answers using an LLM
- Returns source citations
- Provides retrieval and generation latency
- Rejects unsupported questions when relevant context is unavailable

### Pipeline

`Question → Embedding → Pinecone → Context → LLM → Answer + Sources`
""",
    version="1.0.0",
    contact={
        "name": "Rahul Gajera",
    },
    openapi_tags=[
        {
            "name": "system",
            "description": (
                "Service health and readiness endpoints."
            ),
        },
        {
            "name": "rag",
            "description": (
                "Retrieval-Augmented Generation endpoints."
            ),
        },
    ],
)


# ============================================================
# RAG Pipeline
# ============================================================

pipeline = RAGPipeline(
    top_k=5,
    similarity_threshold=0.65,
)


# ============================================================
# Request Schema
# ============================================================

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the RAG system",
        examples=[
            "What is retrieval augmented generation?"
        ],
    )


# ============================================================
# Response Schemas
# ============================================================

class Source(BaseModel):
    source: int
    document: str
    page: int
    chunk: int
    score: float


class Latency(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    latency: Latency


# ============================================================
# Request Logging Middleware
# ============================================================

@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):
    request_id = str(uuid.uuid4())

    start = time.perf_counter()

    response = None

    try:
        response = await call_next(request)

        return response

    finally:
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        if response is not None:
            response.headers["X-Request-ID"] = request_id
            status_code = response.status_code
        else:
            status_code = 500

        logger.info(
            "request_id=%s method=%s path=%s "
            "status=%s latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status_code,
            elapsed_ms,
        )


# ============================================================
# Root Endpoint
# ============================================================

@app.get(
    "/",
    tags=["system"],
)
def root():
    return {
        "service": "RAG PDF QA Service",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get(
    "/health",
    tags=["system"],
)
def health():
    return {
        "status": "healthy",
    }


# ============================================================
# Readiness Endpoint
# ============================================================

@app.get(
    "/ready",
    tags=["system"],
)
def ready():
    try:
        index = (
            pipeline
            .retriever
            .vector_store
            .get_index()
        )

        stats = index.describe_index_stats()

        return {
            "status": "ready",
            "pinecone": "available",
            "vectors": stats.total_vector_count,
        }

    except Exception:
        logger.exception(
            "Readiness check failed"
        )

        raise HTTPException(
            status_code=503,
            detail="Service is not ready",
        )


# ============================================================
# RAG Query Endpoint
# ============================================================

@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["rag"],
)
def query(request: QueryRequest):
    logger.info(
        "RAG query received: %s",
        request.question,
    )

    try:
        result = pipeline.answer(
            request.question
        )

        logger.info(
            "RAG query completed: "
            "retrieval_ms=%.2f "
            "generation_ms=%.2f "
            "total_ms=%.2f "
            "sources=%d",
            result["latency"]["retrieval_ms"],
            result["latency"]["generation_ms"],
            result["latency"]["total_ms"],
            len(result["sources"]),
        )

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            latency=result["latency"],
        )

    except ValueError as exc:
        logger.warning(
            "Invalid query: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        logger.exception(
            "Unexpected error during RAG query"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )