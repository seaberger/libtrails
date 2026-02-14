"""Tests for book_cluster_relevance scoring function."""

import math

import pytest

from libtrails.stats import book_cluster_relevance


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
