from __future__ import annotations

import argparse
import json

from pharma_help.attacks.chroma_lab import reset_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete an isolated PharmaHelp attack lab collection.")
    parser.add_argument("--lab-collection", default="pubmed_attack_lab")
    args = parser.parse_args()
    print(json.dumps(reset_collection(args.lab_collection), indent=2))


if __name__ == "__main__":
    main()
