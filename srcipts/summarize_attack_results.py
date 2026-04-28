#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for c in (str(SRC), str(REPO_ROOT)):
    if c not in sys.path: sys.path.insert(0, c)
from pharma_attack.reporting import load_reports, write_summary_csv, write_summary_md

def main() -> None:
    p = argparse.ArgumentParser(description="Summarize PharmaAttack JSON reports.")
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default="artifacts/summary")
    args = p.parse_args()
    json_paths = []
    for raw in args.paths:
        path = Path(raw)
        json_paths.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    reports = load_reports(json_paths)
    out = Path(args.out)
    write_summary_csv(reports, out.with_suffix(".csv"))
    write_summary_md(reports, out.with_suffix(".md"))
    print(f"Wrote {out.with_suffix('.csv')} and {out.with_suffix('.md')}")

if __name__ == "__main__":
    main()
