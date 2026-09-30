import pytest

from modelforge.evaluation import ranking_metrics
from modelforge.retrieval import Retriever, cache_key, normalize_text


def test_rank_metrics_with_graded_relevance():
    values = ranking_metrics(["other", "primary", "secondary"],
                             {"primary": 2, "secondary": 1}, 3)
    assert values["recall_at_3"] == 1
    assert values["precision_at_3"] == pytest.approx(2 / 3)
    assert values["mrr_at_3"] == 0.5
    assert 0 < values["ndcg_at_3"] < 1


def test_lexical_search_and_cache_key_are_deterministic():
    retrieval = Retriever("data/sample", "lexical")
    found = retrieval.search("database connection pool saturated", 3)
    assert len(found) == 3
    assert found[0]["id"].startswith("db_pool-")
    assert cache_key(retrieval.rows, "model") == cache_key(retrieval.rows, "model")
    assert normalize_text(" two\n spaces ") == "two spaces"
