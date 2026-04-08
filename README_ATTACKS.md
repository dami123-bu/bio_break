# PharmaHelp attacker patch (current public repo)

This patch is designed for the current public `pharma_help` repository.
It does **not** assume a fully wired online RAG query path. Instead, it focuses on the parts that are already present and verifiable now:

1. the current stub chat path (`scripts/chat_loop.py` + `src/pharma_help/agents/tools.py`), and
2. the persistent Chroma-backed storage layer created by `scripts/setup_kb.py`.

## Files in this patch

- `src/pharma_help/attacks/payloads.py` — lab-safe attack payloads with canary markers
- `src/pharma_help/attacks/stub_attack.py` — reproduces the current lexical retrieval logic and demonstrates rank-1 hijack
- `src/pharma_help/attacks/chroma_lab.py` — isolated Chroma attack lab; clones benign docs and injects poison docs
- `scripts/attack_rag_lab.py` — main CLI for running the scenarios
- `scripts/reset_attack_lab.py` — deletes the isolated attack-lab collection
- `tests/unit/test_attack_stub_scenarios.py` — unit tests for the stub hijack scenario
- `ATTACK_SCENARIOS_CURRENT_RAG.md` — scenario write-up for report/slides


## Scenarios implemented

### 1. `stub_keyword_hijack`
Attacks the **currently executable** chat path by adding one synthetic attacker document to a copy of the current lexical candidate set.

Example:
```bash
python scripts/attack_rag_lab.py \
  --scenario stub_keyword_hijack \
  --query "What is tamoxifen used for?" \
  --drug tamoxifen
```

### 2. `chroma_retrieval_bias`
Clones benign documents from `pubmed` into `pubmed_attack_lab`, injects attacker documents, and measures whether the attacker documents displace benign results in top-k.

Example:
```bash
python scripts/attack_rag_lab.py \
  --scenario chroma_retrieval_bias \
  --query "What is tamoxifen used for?" \
  --drug tamoxifen \
  --fresh \
  --output-json artifacts/chroma_retrieval_bias.json
```

### 3. `proto_context_poisoning`
Injects instruction-bearing synthetic chunks into the isolated lab collection. The current public repo does not yet assemble a real LLM context window from Chroma, so this scenario is framed as **proto-context poisoning**: it proves that poisoned text can enter the future context window once synthesis is wired.

Example:
```bash
python scripts/attack_rag_lab.py \
  --scenario proto_context_poisoning \
  --query "tamoxifen dosage and side effects" \
  --drug tamoxifen \
  --fresh \
  --emit-context-preview \
  --output-json artifacts/proto_context_poisoning.json
```

### 4. `persistence_check`
Re-queries the same isolated collection without reinjecting payloads. This shows whether poisoned content persists in the storage layer across runs.

Example:
```bash
python scripts/attack_rag_lab.py \
  --scenario persistence_check \
  --query "tamoxifen dosage and side effects" \
  --drug tamoxifen \
  --emit-context-preview
```

## Reset the isolated lab collection

```bash
python scripts/reset_attack_lab.py
```

## Run the added unit tests

```bash
python -m pytest -q tests/unit/test_attack_stub_scenarios.py
```
