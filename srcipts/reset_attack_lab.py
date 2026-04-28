#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for c in (str(SRC), str(REPO_ROOT)):
    if c not in sys.path: sys.path.insert(0, c)
from pharma_attack.chroma_lab import reset_collection
from pharma_attack.config_runtime import load_runtime_config

def main() -> None:
    p = argparse.ArgumentParser(description="Reset PharmaAttack Chroma lab collection.")
    p.add_argument("--collection", default=None)
    args = p.parse_args()
    config = load_runtime_config()
    collection = args.collection or config.lab_collection
    reset_collection(collection, config=config)
    print(f"Deleted lab collection if it existed: {collection}")

if __name__ == "__main__":
    main()
