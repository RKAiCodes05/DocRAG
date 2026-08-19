from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


# ============================================================
# TEST 1 — Root endpoint
# ============================================================

def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "DocRAG"
    assert data["status"] == "running"


# ============================================================
# TEST 2 — Health endpoint
# ============================================================

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


# ============================================================
# TEST 3 — Readiness endpoint
# ============================================================

def test_ready():
    response = client.get("/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["pinecone"] == "available"
    assert data["vectors"] > 0


# ============================================================
# TEST 4 — Missing question
# ============================================================

def test_missing_question():
    response = client.post(
        "/query",
        json={},
    )

    assert response.status_code == 422


# ============================================================
# TEST 5 — Empty question
# ============================================================

def test_empty_question():
    response = client.post(
        "/query",
        json={
            "question": ""
        },
    )

    assert response.status_code == 422


# ============================================================
# TEST 6 — Valid RAG question
# ============================================================

def test_valid_query():
    response = client.post(
        "/query",
        json={
            "question": (
                "What is retrieval "
                "augmented generation?"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "question" in data
    assert "answer" in data
    assert "sources" in data
    assert "latency" in data

    assert len(data["sources"]) > 0

    assert "retrieval_ms" in data["latency"]
    assert "generation_ms" in data["latency"]
    assert "total_ms" in data["latency"]


# ============================================================
# TEST 7 — Out-of-domain question
# ============================================================

def test_out_of_domain_query():
    response = client.post(
        "/query",
        json={
            "question": "What is the capital of Mars?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sources"] == []

    assert (
        "couldn't find relevant information"
        in data["answer"].lower()
    )


# ============================================================
# TEST 8 — Request ID
# ============================================================

def test_request_id():
    response = client.get("/health")

    assert response.status_code == 200

    assert "X-Request-ID" in response.headers

    request_id = response.headers["X-Request-ID"]

    assert len(request_id) > 0


# ============================================================
# TEST 9 — Invalid HTTP method
# ============================================================

def test_invalid_method():
    response = client.get("/query")

    assert response.status_code == 405
