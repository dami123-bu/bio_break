# Teammate Modification Guide

| Goal | File |
|---|---|
| Change attack text or canary | `src/pharma_attack/payloads.py` |
| Change success interpretation | `src/pharma_attack/metrics.py` |
| Change offline deterministic behavior | `src/pharma_attack/offline_lab.py` |
| Change Chroma-backed behavior | `src/pharma_attack/chroma_lab.py` |
| Add a CLI scenario | `scripts/attack_rag_lab.py` |
| Add a batch testbench case | `src/pharma_attack/testbench.py` |
| Add tests | `tests/` |

## How to add a new scenario

1. Add a payload builder in `payloads.py`.
2. Register it in `payloads_for_scenario()`.
3. Add a default query in `testbench.py` if it should run in batch mode.
4. Add the scenario name in `scripts/attack_rag_lab.py`.
5. Add one test that checks the payload and one test that checks the metrics.
6. Add one paragraph in `ATTACK_SCENARIOS.md`.

## How to make a payload rank higher

- For lexical/stub mode: repeat exact query terms in title and body.
- For Chroma mode: make the content semantically similar to the target corpus style.
- For passive poisoning: use plausible scientific language, not dramatic prompt-injection phrasing.
- For active injection: keep a canary and avoid real external actions.

## How to judge success

Use `success`, `partial_success`, `success_level`, and `metrics`. Do not judge by raw `document` text length or distance alone.
