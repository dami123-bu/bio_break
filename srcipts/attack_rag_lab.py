#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for c in (str(SRC), str(REPO_ROOT)):
    if c not in sys.path: sys.path.insert(0, c)
from pharma_attack.testbench import run_single_scenario, write_json

SCENARIOS = [
    "stub_keyword_hijack", "a1a_passive_rag_poison", "a1b_active_instruction_poison",
    "a1c_volume_poison", "a7_persistence_probe", "a10_semantic_obfuscation",
    "chroma_retrieval_bias", "proto_context_poisoning", "persistence_check"
]
ALIASES = {"chroma_retrieval_bias": "a1a_passive_rag_poison", "proto_context_poisoning": "a1b_active_instruction_poison", "persistence_check": "a7_persistence_probe"}

def main() -> None:
    p = argparse.ArgumentParser(description="Run a PharmaAttack scenario.")
    p.add_argument("--scenario", required=True, choices=SCENARIOS)
    p.add_argument("--query", required=True)
    p.add_argument("--drug", required=True)
    p.add_argument("--mode", choices=["offline", "chroma"], default="offline")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--volume-count", type=int, default=10)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--output-json")
    args = p.parse_args()
    scenario = ALIASES.get(args.scenario, args.scenario)
    mode = "offline" if args.scenario == "stub_keyword_hijack" else args.mode
    report = run_single_scenario(mode=mode, scenario=scenario, query=args.query, drug=args.drug, top_k=args.top_k, fresh=args.fresh, volume_count=args.volume_count)
    report["requested_scenario"] = args.scenario
    if args.output_json:
        write_json(Path(args.output_json), report)
        print(f"Saved JSON report to {args.output_json}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
