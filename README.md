# PharmaAttack

Reproducible attacker-side testbench for PharmaHelp RAG/MCP exploit scenarios.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m pytest -q
python scripts/run_testbench.py --mode offline --output-dir artifacts/testbench_offline
```

For detailed testing instructions, see [`TESTBENCH.md`](TESTBENCH.md).
