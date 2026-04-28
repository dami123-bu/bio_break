#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for c in (str(SRC), str(REPO_ROOT)):
    if c not in sys.path: sys.path.insert(0, c)
from pharma_attack.testbench import run_testbench

def main() -> None:
    p = argparse.ArgumentParser(description="Run complete PharmaAttack testbench.")
    p.add_argument("--mode", choices=["offline", "chroma"], default="offline")
    p.add_argument("--output-dir", default="artifacts/testbench_offline")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--volume-count", type=int, default=10)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()
    reports = run_testbench(mode=args.mode, output_dir=args.output_dir, top_k=args.top_k, fresh=args.fresh, volume_count=args.volume_count)
    print(json.dumps([{"scenario": r.get("scenario"), "success": r.get("success", r.get("attack_success")), "partial_success": r.get("partial_success"), "success_level": r.get("success_level"), "success_reason": r.get("success_reason")} for r in reports], indent=2))
    print(f"\nWrote reports to: {args.output_dir}")
    print(f"Summary: {Path(args.output_dir) / 'summary.md'}")

if __name__ == "__main__":
    main()
