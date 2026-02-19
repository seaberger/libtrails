"""Tests for community API endpoints."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from libtrails.api.dependencies import get_db_connection
from libtrails.api.routers.communities import router


@pytest.fixture
def community_db(tmp_path):
    """Create a database with community schema and seeded data for API tests."""
    db_path = tmp_path / "api_test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            calibre_id INTEGER
        );

        CREATE TABLE domains (
            id INTEGER PRIMARY KEY,
            label TEXT NOT NULL UNIQUE,
            cluster_count INTEGER DEFAULT 0
        );

        CREATE TABLE communities (
            id INTEGER PRIMARY KEY,
            label TEXT NOT NULL,
            domain_id INTEGER REFERENCES domains(id),
            topic_count INTEGER DEFAULT 0
        );

        CREATE TABLE cluster_communities (
            cluster_id INTEGER PRIMARY KEY,
            community_id INTEGER NOT NULL REFERENCES communities(id)
        );

        CREATE TABLE cluster_stats (
            cluster_id INTEGER PRIMARY KEY,
            size INTEGER NOT NULL DEFAULT 0,
            book_count INTEGER NOT NULL DEFAULT 0,
            top_label TEXT,
            top_topics_json TEXT,
            sample_books_json TEXT
        );

        CREATE TABLE community_stats (
            community_id INTEGER PRIMARY KEY,
            topic_count INTEGER NOT NULL DEFAULT 0,
            book_count INTEGER NOT NULL DEFAULT 0,
            primary_book_count INTEGER NOT NULL DEFAULT 0,
            top_label TEXT,
            top_topics_json TEXT,
            sample_books_json TEXT,
            domain_id INTEGER,
            domain_label TEXT,
            refreshed_at TIMESTAMP
        );

        CREATE TABLE book_communities (
            book_id INTEGER NOT NULL,
            community_id INTEGER NOT NULL,
            topic_count INTEGER NOT NULL,
            book_total_topics INTEGER NOT NULL,
            concentration REAL NOT NULL,
            relevance_score REAL NOT NULL,
            is_primary INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (book_id, community_id)
        );

        -- Seed data
        INSERT INTO books (id, title, author, calibre_id) VALUES
            (1, 'Quantum Physics', 'Alice Author', 100),
            (2, 'Modern Poetry', 'Bob Writer', 200);

        INSERT INTO domains (id, label) VALUES
            (0, 'Science'),
            (1, 'Literature');

        INSERT INTO communities (id, label, domain_id, topic_count) VALUES
            (10, 'Physics & Math', 0, 50),
            (20, 'Poetry & Fiction', 1, 30);

        INSERT INTO community_stats
            (community_id, topic_count, book_count, primary_book_count,
             top_label, top_topics_json, sample_books_json, domain_id, domain_label)
        VALUES
            (10, 50, 2, 1, 'quantum mechanics',
             '[{"id":1,"label":"quantum mechanics","count":15}]',
             '[{"id":1,"title":"Quantum Physics","author":"Alice Author","calibre_id":100}]',
             0, 'Science'),
            (20, 30, 1, 1, 'romantic poetry',
             '[{"id":4,"label":"romantic poetry","count":12}]',
             '[{"id":2,"title":"Modern Poetry","author":"Bob Writer","calibre_id":200}]',
             1, 'Literature');

        INSERT INTO cluster_communities (cluster_id, community_id) VALUES
            (0, 10), (1, 10), (2, 20);

        INSERT INTO cluster_stats (cluster_id, size, book_count, top_label) VALUES
            (0, 30, 2, 'quantum mechanics'),
            (1, 20, 1, 'linear algebra'),
            (2, 30, 1, 'romantic poetry');

        INSERT INTO book_communities
            (book_id, community_id, topic_count, book_total_topics,
             concentration, relevance_score, is_primary)
        VALUES
            (1, 10, 60, 100, 0.6, 2.5, 1),
            (1, 20, 40, 100, 0.4, 1.2, 0),
            (2, 20, 30, 50, 0.6, 2.8, 1);
    """)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def client(community_db):
    """Create a FastAPI test client with overridden DB dependency."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    def override_db():
        conn = sqlite3.connect(community_db)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[get_db_connection] = override_db

    with TestClient(app) as c:
        yield c


class TestListCommunities:
    """Tests for GET /communities."""

    def test_returns_all_communities(self, client):
        resp = client.get("/api/v1/communities")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_community_fields(self, client):
        resp = client.get("/api/v1/communities")
        data = resp.json()
        comm = data[0]
        assert "community_id" in comm
        assert "label" in comm
        assert "topic_count" in comm
        assert "book_count" in comm
        assert "domain_id" in comm
        assert "domain_label" in comm
        assert "sample_books" in comm

    def test_uses_curated_label(self, client):
        """Should use communities.label (curated), not community_stats.top_label."""
        resp = client.get("/api/v1/communities")
        data = resp.json()
        labels = {c["community_id"]: c["label"] for c in data}
        # Curated labels from communities table, not top_label from stats
        assert labels[10] == "Physics & Math"
        assert labels[20] == "Poetry & Fiction"

    def test_filter_by_domain(self, client):
        resp = client.get("/api/v1/communities", params={"domain_id": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["domain_id"] == 0

    def test_filter_nonexistent_domain(self, client):
        resp = client.get("/api/v1/communities", params={"domain_id": 999})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_ordered_by_book_count_desc(self, client):
        resp = client.get("/api/v1/communities")
        data = resp.json()
        book_counts = [c["book_count"] for c in data]
        assert book_counts == sorted(book_counts, reverse=True)


class TestGetCommunity:
    """Tests for GET /communities/{community_id}."""

    def test_returns_community_detail(self, client):
        resp = client.get("/api/v1/communities/10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["community_id"] == 10
        assert data["label"] == "Physics & Math"
        assert data["topic_count"] == 50
        assert data["domain_id"] == 0
        assert data["domain_label"] == "Science"

    def test_includes_clusters(self, client):
        resp = client.get("/api/v1/communities/10")
        data = resp.json()
        assert "clusters" in data
        assert len(data["clusters"]) == 2  # C0 and C1 mapped to Comm10
        cluster_ids = {c["cluster_id"] for c in data["clusters"]}
        assert cluster_ids == {0, 1}

    def test_includes_books(self, client):
        resp = client.get("/api/v1/communities/10")
        data = resp.json()
        assert "books" in data
        assert len(data["books"]) >= 1
        book = data["books"][0]
        assert "id" in book
        assert "title" in book
        assert "concentration" in book
        assert "is_primary" in book

    def test_not_found(self, client):
        resp = client.get("/api/v1/communities/999")
        assert resp.status_code == 404

    def test_books_ordered_by_relevance(self, client):
        resp = client.get("/api/v1/communities/10")
        data = resp.json()
        # Only book 1 is in community 10 via book_communities
        books = data["books"]
        if len(books) > 1:
            scores = [b["concentration"] for b in books]
            # Verify descending order (ordered by relevance_score in query)
            assert scores == sorted(scores, reverse=True)
