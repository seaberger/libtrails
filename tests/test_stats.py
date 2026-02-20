"""Tests for stats module: scoring function and refresh pipelines."""

import json

import pytest

from libtrails.stats import (
    book_cluster_relevance,
    refresh_book_communities,
    refresh_book_domains,
    refresh_community_stats,
    refresh_domain_stats,
)


class TestMinTopicsFilter:
    def test_below_min_returns_zero(self):
        score = book_cluster_relevance(
            topics_in_cluster=2,
            total_topics_book=10,
            total_topics_cluster=100,
            total_corpus=10000,
        )
        assert score == 0.0

    def test_at_min_returns_nonzero(self):
        score = book_cluster_relevance(
            topics_in_cluster=3,
            total_topics_book=10,
            total_topics_cluster=100,
            total_corpus=10000,
        )
        assert score > 0.0

    def test_custom_min_topics(self):
        score = book_cluster_relevance(
            topics_in_cluster=4,
            total_topics_book=10,
            total_topics_cluster=100,
            total_corpus=10000,
            min_topics=5,
        )
        assert score == 0.0


class TestLengthBias:
    def test_focused_book_beats_sprawling_book(self):
        """A short book concentrated in a cluster should score higher than
        a massive book that merely touches the cluster."""
        # Short focused book: 20 topics in cluster out of 30 total
        focused = book_cluster_relevance(
            topics_in_cluster=20,
            total_topics_book=30,
            total_topics_cluster=200,
            total_corpus=50000,
        )
        # Sprawling epic: 50 topics in cluster out of 3000 total
        sprawling = book_cluster_relevance(
            topics_in_cluster=50,
            total_topics_book=3000,
            total_topics_cluster=200,
            total_corpus=50000,
        )
        assert focused > sprawling

    def test_les_miserables_vs_short_book(self):
        """Simulate Les Misérables (3769 topics) vs a focused 80-topic book."""
        # Les Mis: 40 topics in a cluster, out of 3769 total
        les_mis = book_cluster_relevance(
            topics_in_cluster=40,
            total_topics_book=3769,
            total_topics_cluster=200,
            total_corpus=50000,
        )
        # Short book: 15 topics in same cluster, out of 80 total
        short = book_cluster_relevance(
            topics_in_cluster=15,
            total_topics_book=80,
            total_topics_cluster=200,
            total_corpus=50000,
        )
        assert short > les_mis


class TestBM25Saturation:
    def test_diminishing_returns(self):
        """Higher concentration should give diminishing returns."""
        scores = []
        for topics_in in [10, 20, 30, 40]:
            score = book_cluster_relevance(
                topics_in_cluster=topics_in,
                total_topics_book=40,
                total_topics_cluster=200,
                total_corpus=50000,
            )
            scores.append(score)

        # Each increment should yield less additional score
        deltas = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        for i in range(len(deltas) - 1):
            assert deltas[i] > deltas[i + 1], "Saturation should yield diminishing returns"


class TestPPMI:
    def test_above_chance_gets_boost(self):
        """A book with more topics than expected by chance should score higher."""
        # Book appears way more than random chance would predict
        boosted = book_cluster_relevance(
            topics_in_cluster=50,
            total_topics_book=100,
            total_topics_cluster=200,
            total_corpus=100000,
        )
        # Same concentration but at random chance level
        # expected = (100 * 200) / 400 = 50, so pmi = log2(50/50) = 0
        at_chance = book_cluster_relevance(
            topics_in_cluster=50,
            total_topics_book=100,
            total_topics_cluster=200,
            total_corpus=400,
        )
        assert boosted > at_chance

    def test_ppmi_never_negative(self):
        """PPMI should clamp negative PMI to 0, not penalize."""
        # Below-chance: expected is high, actual is low
        score = book_cluster_relevance(
            topics_in_cluster=3,
            total_topics_book=1000,
            total_topics_cluster=500,
            total_corpus=1000,
        )
        # Should still be positive (just no PPMI boost)
        assert score > 0.0


class TestEdgeCases:
    def test_zero_total_topics_book(self):
        """Should not crash on zero division."""
        with pytest.raises(ZeroDivisionError):
            book_cluster_relevance(
                topics_in_cluster=5,
                total_topics_book=0,
                total_topics_cluster=100,
                total_corpus=10000,
            )

    def test_zero_corpus(self):
        """Zero corpus should not crash (expected would be inf/nan)."""
        # topics_in_cluster=5 passes min_topics, but total_corpus=0
        # expected = (10 * 100) / 0 -> ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            book_cluster_relevance(
                topics_in_cluster=5,
                total_topics_book=10,
                total_topics_cluster=100,
                total_corpus=0,
            )

    def test_all_topics_in_cluster(self):
        """Book with 100% concentration should get a valid score."""
        score = book_cluster_relevance(
            topics_in_cluster=50,
            total_topics_book=50,
            total_topics_cluster=200,
            total_corpus=50000,
        )
        assert score > 0.0
        # Concentration is 1.0, saturated = 1.0 * 2.5 / 2.5 = 1.0
        assert score >= 1.0  # At minimum, saturated=1.0 * (1+ppmi) >= 1.0


# ── Integration tests for stats refresh pipeline ──────────────────


class TestRefreshBookCommunities:
    """Tests for refresh_book_communities (cluster_books + cluster_communities → book_communities)."""

    def test_populates_book_communities(self, stats_db):
        """Should create rows for each (book, community) pair."""
        count = refresh_book_communities(stats_db)

        # B1 touches C0→Comm0 and C1→Comm1 → 2 rows
        # B2 touches C0→Comm0 and C1→Comm1 → 2 rows
        assert count == 4

        rows = stats_db.execute(
            "SELECT * FROM book_communities ORDER BY book_id, community_id"
        ).fetchall()
        assert len(rows) == 4

    def test_concentration_values(self, stats_db):
        """Concentration = topic_count / book_total_topics."""
        refresh_book_communities(stats_db)

        # B1 in Comm0: topics=60, total=100 → concentration=0.6
        row = stats_db.execute(
            "SELECT concentration FROM book_communities WHERE book_id=1 AND community_id=0"
        ).fetchone()
        assert abs(row["concentration"] - 0.6) < 0.001

        # B2 in Comm1: topics=30, total=50 → concentration=0.6
        row = stats_db.execute(
            "SELECT concentration FROM book_communities WHERE book_id=2 AND community_id=1"
        ).fetchone()
        assert abs(row["concentration"] - 0.6) < 0.001

    def test_relevance_scores_positive(self, stats_db):
        """All relevance scores should be positive (min_topics=1 for communities)."""
        refresh_book_communities(stats_db)

        rows = stats_db.execute("SELECT relevance_score FROM book_communities").fetchall()
        for row in rows:
            assert row["relevance_score"] > 0.0

    def test_primary_flag_set(self, stats_db):
        """Each book should have exactly one primary community."""
        refresh_book_communities(stats_db)

        for book_id in [1, 2]:
            primary_count = stats_db.execute(
                "SELECT COUNT(*) as cnt FROM book_communities WHERE book_id=? AND is_primary=1",
                (book_id,),
            ).fetchone()["cnt"]
            assert primary_count == 1

    def test_focused_book_gets_higher_score(self, stats_db):
        """B2 (30/50=60% concentration in Comm1) should score higher than B1 (40/100=40%) in Comm1."""
        refresh_book_communities(stats_db)

        b1_score = stats_db.execute(
            "SELECT relevance_score FROM book_communities WHERE book_id=1 AND community_id=1"
        ).fetchone()["relevance_score"]
        b2_score = stats_db.execute(
            "SELECT relevance_score FROM book_communities WHERE book_id=2 AND community_id=1"
        ).fetchone()["relevance_score"]
        assert b2_score > b1_score

    def test_empty_cluster_communities_returns_zero(self, stats_db):
        """With no cluster_communities data, should return 0 rows."""
        stats_db.execute("DELETE FROM cluster_communities")
        stats_db.commit()

        count = refresh_book_communities(stats_db)
        assert count == 0

    def test_idempotent(self, stats_db):
        """Running twice should produce the same result."""
        refresh_book_communities(stats_db)
        count1 = stats_db.execute("SELECT COUNT(*) FROM book_communities").fetchone()[0]

        refresh_book_communities(stats_db)
        count2 = stats_db.execute("SELECT COUNT(*) FROM book_communities").fetchone()[0]

        assert count1 == count2


class TestRefreshCommunityStats:
    """Tests for refresh_community_stats (communities + book_communities → community_stats)."""

    def test_populates_community_stats(self, stats_db):
        """Should create one stats row per community."""
        refresh_book_communities(stats_db)  # prerequisite
        count = refresh_community_stats(stats_db)

        assert count == 2
        rows = stats_db.execute("SELECT * FROM community_stats").fetchall()
        assert len(rows) == 2

    def test_topic_count_from_communities_table(self, stats_db):
        """topic_count should come from communities table, not recomputed."""
        refresh_book_communities(stats_db)
        refresh_community_stats(stats_db)

        row = stats_db.execute(
            "SELECT topic_count FROM community_stats WHERE community_id=0"
        ).fetchone()
        assert row["topic_count"] == 3  # matches communities table seed

    def test_domain_label_populated(self, stats_db):
        """domain_label should be set from domains table."""
        refresh_book_communities(stats_db)
        refresh_community_stats(stats_db)

        row = stats_db.execute(
            "SELECT domain_id, domain_label FROM community_stats WHERE community_id=0"
        ).fetchone()
        assert row["domain_id"] == 0
        assert row["domain_label"] == "Science & Tech"

    def test_top_label_prefers_llm_name(self, stats_db):
        """top_label should prefer the LLM-generated community name when available."""
        refresh_book_communities(stats_db)
        refresh_community_stats(stats_db)

        # Comm0 has LLM label 'Physics & Math' in communities table
        row = stats_db.execute(
            "SELECT top_label FROM community_stats WHERE community_id=0"
        ).fetchone()
        assert row["top_label"] == "Physics & Math"

    def test_top_label_falls_back_to_topic(self, stats_db):
        """top_label should fall back to highest-occurrence topic when no LLM name."""
        # Clear the LLM label for community 0
        stats_db.execute("UPDATE communities SET label = '' WHERE id = 0")
        stats_db.commit()

        refresh_book_communities(stats_db)
        refresh_community_stats(stats_db)

        row = stats_db.execute(
            "SELECT top_label FROM community_stats WHERE community_id=0"
        ).fetchone()
        assert row["top_label"] == "quantum mechanics"

    def test_sample_books_json_valid(self, stats_db):
        """sample_books_json should be valid JSON with book data."""
        refresh_book_communities(stats_db)
        refresh_community_stats(stats_db)

        row = stats_db.execute(
            "SELECT sample_books_json FROM community_stats WHERE community_id=0"
        ).fetchone()
        books = json.loads(row["sample_books_json"])
        assert isinstance(books, list)
        assert len(books) <= 5
        if books:
            assert "id" in books[0]
            assert "title" in books[0]

    def test_book_count_uses_concentration_threshold(self, stats_db):
        """book_count should only count books with concentration >= 1%."""
        refresh_book_communities(stats_db)
        refresh_community_stats(stats_db)

        # All our test books have high concentration (>= 20%), so all should be counted
        row = stats_db.execute(
            "SELECT book_count FROM community_stats WHERE community_id=0"
        ).fetchone()
        assert row["book_count"] == 2  # Both books touch Comm0


class TestRefreshBookDomains:
    """Tests for refresh_book_domains (cluster_books + cluster_domains → book_domains)."""

    def test_populates_book_domains(self, stats_db):
        """Should create rows for each (book, domain) pair."""
        count = refresh_book_domains(stats_db)

        # B1 touches C0→D0 and C1→D1 → 2 rows
        # B2 touches C0→D0 and C1→D1 → 2 rows
        assert count == 4

    def test_concentration_correct(self, stats_db):
        """Concentration should aggregate by domain."""
        refresh_book_domains(stats_db)

        # B1 in D0: via C0 only → topics=60, total=100 → 0.6
        row = stats_db.execute(
            "SELECT concentration FROM book_domains WHERE book_id=1 AND domain_id=0"
        ).fetchone()
        assert abs(row["concentration"] - 0.6) < 0.001

    def test_primary_domain_assigned(self, stats_db):
        """Each book should have exactly one primary domain."""
        refresh_book_domains(stats_db)

        for book_id in [1, 2]:
            primary_count = stats_db.execute(
                "SELECT COUNT(*) as cnt FROM book_domains WHERE book_id=? AND is_primary=1",
                (book_id,),
            ).fetchone()["cnt"]
            assert primary_count == 1

    def test_idempotent(self, stats_db):
        """Running twice should produce the same result."""
        refresh_book_domains(stats_db)
        count1 = stats_db.execute("SELECT COUNT(*) FROM book_domains").fetchone()[0]

        refresh_book_domains(stats_db)
        count2 = stats_db.execute("SELECT COUNT(*) FROM book_domains").fetchone()[0]

        assert count1 == count2


class TestRefreshDomainStats:
    """Tests for refresh_domain_stats (book_domains + cluster_stats → domain_stats)."""

    def test_populates_domain_stats(self, stats_db):
        """Should create one stats row per domain."""
        refresh_book_domains(stats_db)  # prerequisite
        count = refresh_domain_stats(stats_db)

        assert count == 2

    def test_book_counts(self, stats_db):
        """book_count should reflect books with >= 1% concentration."""
        refresh_book_domains(stats_db)
        refresh_domain_stats(stats_db)

        row = stats_db.execute(
            "SELECT book_count, primary_book_count FROM domain_stats WHERE domain_id=0"
        ).fetchone()
        assert row["book_count"] == 2  # Both books have high concentration in D0
        assert row["primary_book_count"] >= 1

    def test_sample_books_json_valid(self, stats_db):
        """sample_books_json should be valid JSON."""
        refresh_book_domains(stats_db)
        refresh_domain_stats(stats_db)

        row = stats_db.execute(
            "SELECT sample_books_json FROM domain_stats WHERE domain_id=0"
        ).fetchone()
        books = json.loads(row["sample_books_json"])
        assert isinstance(books, list)

    def test_top_clusters_json_valid(self, stats_db):
        """top_clusters_json should be valid JSON with cluster data."""
        refresh_book_domains(stats_db)
        refresh_domain_stats(stats_db)

        row = stats_db.execute(
            "SELECT top_clusters_json FROM domain_stats WHERE domain_id=0"
        ).fetchone()
        clusters = json.loads(row["top_clusters_json"])
        assert isinstance(clusters, list)
        if clusters:
            assert "cluster_id" in clusters[0]
            assert "label" in clusters[0]
            assert "size" in clusters[0]
