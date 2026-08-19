# RAG PDF QA Service

> A production-oriented Retrieval-Augmented Generation (RAG) API for question answering over PDF documents.

Build, deploy, and query a scalable RAG backend that combines **document ingestion, semantic retrieval, vector search, and LLM generation** into a single REST API.

## 🚀 Live Demo

**Production API:** https://rag.modelstack.tech  
**Swagger UI:** https://rag.modelstack.tech/docs

## 🎯 What I Built

This project goes beyond a basic PDF chatbot by implementing the complete document-to-answer lifecycle:

```text
PDF
 ↓
Document Ingestion
 ↓
Chunking + Embeddings
 ↓
Pinecone Vector Database
 ↓
Semantic Retrieval
 ↓
Groq LLM
 ↓
Source-Grounded Answer

It also supports document hashing, duplicate detection, document listing, and document-level deletion.
```
## ✨ Highlights
- 📄 PDF ingestion and semantic chunking
- 🔐 SHA-256 document identity and duplicate detection
- 🧠 BGE embeddings
- 🔎 Pinecone semantic retrieval
- 🤖 Groq LLM generation
- 📚 Source-aware responses
- 📋 Document management APIs
- 📊 Retrieval and generation latency tracking
- 🐳 Dockerized application
- 🔄 GitHub Actions CI/CD
- 📦 GitHub Container Registry
- ☁️ Azure production deployment
- 🔒 Caddy reverse proxy with HTTPS

## 🏗️ Architecture
```text
                         Client
                    Swagger / curl / UI
                           │
                           │ HTTPS
                           ▼
                    ┌─────────────┐
                    │    Caddy    │
                    │ TLS / Proxy │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   FastAPI   │
                    │   RAG API   │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       ┌─────────────┐          ┌─────────────┐
       │  Ingestion  │          │    Query     │
       │  Pipeline   │          │   Pipeline   │
       └──────┬──────┘          └──────┬───────┘
              │                        │
              ▼                        ▼
       ┌─────────────┐          ┌─────────────┐
       │ BGE         │          │  Pinecone   │
       │ Embeddings  │◄─────────│  Retrieval  │
       └─────────────┘          └──────┬──────┘
                                       │
                                       ▼
                                ┌─────────────┐
                                │   Groq LLM  │
                                └──────┬──────┘
                                       │
                                       ▼
                                Grounded Answer


```
## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **API** | FastAPI |
| **Language** | Python |
| **PDF Processing** | PyMuPDF |
| **Embeddings** | BGE-base-en-v1.5 |
| **Vector Database** | Pinecone |
| **LLM Inference** | Groq |
| **Containerization** | Docker |
| **Registry** | GitHub Container Registry |
| **CI/CD** | GitHub Actions |
| **Cloud** | Microsoft Azure |
| **Reverse Proxy / TLS** | Caddy |
| **Testing** | pytest |

## 📡 API

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check |
| `POST` | `/query` | Query indexed documents |
| `POST` | `/documents` | Upload and index a PDF |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{document_hash}` | Delete a document |

Interactive API: [https://rag.modelstack.tech/docs](https://rag.modelstack.tech/docs)

## ☁️ Production

The service is deployed on an Azure VM in Korea Central using a containerized production workflow:

```text
GitHub
   ↓
GitHub Actions
   ↓
GHCR
   ↓
Azure VM
   ↓
Docker
   ↓
Caddy + HTTPS
   ↓
FastAPI
   ├── Pinecone
   └── Groq
```
## 📚 Documentation

Detailed technical documentation is available in `docs/`:
- Architecture
- RAG Pipeline
- API Reference
- Deployment
- Development

## 🚀 Quick Start

```bash
git clone https://github.com/RKAiCodes05/rag-pdf-qa-service.git
cd rag-pdf-qa-service

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env

uvicorn api.main:app --reload

```
Local Swagger:  
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 🔮 Future Improvements

- API authentication
- Rate limiting
- Streaming responses
- Hybrid retrieval and reranking
- Advanced RAG evaluation
- Interactive frontend

## 👤 Author

Ram khimani (AI/LLM learner)
