from pharma_attack.metrics import compute_retrieval_metrics, judge_retrieval_attack

def test_metrics_identify_attack_rank_2_as_success():
    hits = [
        {"rank": 1, "id": "seed::1", "metadata": {"source": "pubmed"}, "contains_canary": False},
        {"rank": 2, "id": "attack::x", "metadata": {"source": "attack_internal_docs"}, "contains_canary": True},
    ]
    metrics = compute_retrieval_metrics(hits, top_k=10)
    decision = judge_retrieval_attack(metrics)
    assert metrics.attack_in_top_k is True
    assert metrics.lowest_attack_rank == 2
    assert metrics.canary_seen is True
    assert decision.success is True

def test_metrics_identify_negative_result():
    hits = [{"rank": 1, "id": "seed::1", "metadata": {"source": "pubmed"}, "contains_canary": False}]
    metrics = compute_retrieval_metrics(hits, top_k=10)
    decision = judge_retrieval_attack(metrics)
    assert metrics.attack_in_top_k is False
    assert decision.success is False
    assert decision.partial_success is False
