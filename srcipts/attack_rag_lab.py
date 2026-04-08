from __future__ import annotations

import argparse
import json
from pathlib import Path

from pharma_help.attacks.chroma_lab import dump_json_report, run_chroma_scenario
from pharma_help.attacks.stub_attack import run_stub_keyword_hijack_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attacker-side scenarios against the current PharmaHelp RAG build.")
    parser.add_argument(
        "--scenario",
        required=True,
        choices=[
            "stub_keyword_hijack",
            "chroma_retrieval_bias",
            "proto_context_poisoning",
            "persistence_check",
        ],
    )
    parser.add_argument("--query", required=True, help="User query to attack.")
    parser.add_argument("--drug", default="tamoxifen", help="Target drug keyword used in the lab payloads.")
    parser.add_argument("--lab-collection", default="pubmed_attack_lab", help="Isolated Chroma collection used for attacks.")
    parser.add_argument("--source-collection", default="pubmed", help="Benign source collection to clone into the lab.")
    parser.add_argument("--copy-count", type=int, default=60, help="How many benign seed docs to copy into the lab collection.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieval hits to inspect.")
    parser.add_argument("--fresh", action="store_true", help="Reset and reseed the isolated lab collection first.")
    parser.add_argument("--emit-context-preview", action="store_true", help="Include concatenated top-k context text in the report.")
    parser.add_argument("--output-json", help="Optional path for a saved JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.scenario == "stub_keyword_hijack":
        report = run_stub_keyword_hijack_demo(query=args.query, drug=args.drug)
    else:
        report = run_chroma_scenario(
            scenario=args.scenario,
            query=args.query,
            drug=args.drug,
            lab_collection_name=args.lab_collection,
            source_collection_name=args.source_collection,
            copy_count=args.copy_count,
            top_k=args.top_k,
            fresh=args.fresh,
            emit_context=args.emit_context_preview,
        )

    if args.output_json:
        output_path = dump_json_report(report, args.output_json)
        print(f"Saved JSON report to {output_path}")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
