import logging
import time
import uuid
import tempfile
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel, Field

from ingestion.service import DocumentIngestionService
from rag.pipeline import RAGPipeline
from retrieval.embedder import Embedder


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
    title="DocRAG: Document Retrieval-Augmented Generation Service",
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
        {
            "name": "documents",
            "description": (
                "PDF document ingestion and indexing endpoints."
            ),
        },
    ],
)


# ============================================================
# RAG Pipeline
# ============================================================

embedder = Embedder()

pipeline = RAGPipeline(
    top_k=5,
    similarity_threshold=0.65,
    embedder=embedder,
)

document_service = DocumentIngestionService(
    embedder=embedder,
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
        "service": "DocRAG",
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
# Document Upload / Ingestion Endpoint / delete document endpoint
# ============================================================
@app.get(
    "/documents",
    tags=["documents"],
)
async def list_documents():
    try:
        return document_service.list_documents()

    except Exception:
        logger.exception(
            "Failed to list documents"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to list documents.",
        )
        
@app.post(
    "/documents",
    tags=["documents"],
    status_code=201,
)
async def upload_document(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    temp_path = None

    try:
        # --------------------------------------------------
        # Create temporary PDF
        # --------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:

            temp_path = Path(
                temp_file.name
            )

            total_size = 0

            while True:
                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(chunk)

                # 25 MB upload limit
                if total_size > 25 * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "PDF file is too large. "
                            "Maximum size is 25 MB."
                        ),
                    )

                temp_file.write(chunk)

        # --------------------------------------------------
        # Ingest document
        # --------------------------------------------------

        result = document_service.ingest(
            temp_path,
            file.filename,
        )

        # --------------------------------------------------
        # Duplicate document
        # --------------------------------------------------

        if result["status"] == "already_exists":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Document '{file.filename}' "
                    "is already indexed."
                ),
            )

        result["document"] = file.filename

        logger.info(
            "Document indexed: "
            "name=%s chunks=%s hash=%s",
            file.filename,
            result["chunks"],
            result["document_hash"][:16],
        )

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Document ingestion validation failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        logger.exception(
            "Unexpected error during document ingestion"
        )

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        )

    finally:
        # --------------------------------------------------
        # Always delete temporary PDF
        # --------------------------------------------------

        if (
            temp_path is not None
            and temp_path.exists()
        ):
            temp_path.unlink()

@app.delete(
    "/documents/{document_hash}",
    tags=["documents"],
)
async def delete_document(
    document_hash: str,
):
    try:
        result = document_service.delete(
            document_hash
        )

        return result

    except Exception:
        logger.exception(
            "Document deletion failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Document deletion failed.",
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