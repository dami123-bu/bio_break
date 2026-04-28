from pharma_attack.payloads import CANARY_INSTRUCTION, CANARY_OVERRIDE, CANARY_VOLUME, build_a1a_passive_poison, build_a1b_active_instruction_poison, build_a1c_volume_poison

def test_a1a_payload_contains_passive_canary():
    payload = build_a1a_passive_poison("BF-042")
    assert CANARY_OVERRIDE in payload["content"]
    assert payload["metadata"]["attack_family"] == "a1_rag_retrieval_poisoning"
    assert payload["metadata"]["variant"] == "passive_factual_poison"

def test_a1b_payload_contains_instruction_canary():
    payload = build_a1b_active_instruction_poison("BF-099")
    assert CANARY_INSTRUCTION in payload["content"]
    assert payload["metadata"]["variant"] == "active_instruction_poison"

def test_a1c_volume_payload_count_and_canary():
    payloads = build_a1c_volume_poison("BF-042", count=5)
    assert len(payloads) == 5
    assert all(CANARY_VOLUME in payload["content"] for payload in payloads)
