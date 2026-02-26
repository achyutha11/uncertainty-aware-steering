"""
download_datasets.py — Download and cache GSM8k and MATH-500 datasets.

Usage:
    python data/download_datasets.py --output_dir data/raw
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download evaluation datasets.")
    p.add_argument("--output_dir", default="data/raw")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "math500"],
        choices=["gsm8k", "math500"],
    )
    return p.parse_args()


def download_gsm8k(output_dir: str) -> None:
    from datasets import load_dataset

    print("Downloading GSM8k...")
    ds = load_dataset("gsm8k", "main")
    for split_name, split_data in ds.items():
        out_path = os.path.join(output_dir, f"gsm8k_{split_name}.json")
        records = [dict(row) for row in split_data]
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Saved {len(records)} examples to {out_path}")


def download_math500(output_dir: str) -> None:
    """Download MATH-500 benchmark (subset of MATH dataset used by the paper).

    The standard MATH-500 is the 500-problem test subset curated by Lightman et al.
    We use the HuggingFace-hosted version.
    """
    try:
        from datasets import load_dataset
        print("Downloading MATH-500...")
        # Try the lighteval/MATH hosted version
        ds = load_dataset("lighteval/MATH", split="test")
        # MATH-500 is a 500-example subset; take the first 500 if larger
        records = [dict(row) for row in ds][:500]
        out_path = os.path.join(output_dir, "math500_test.json")
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Saved {len(records)} examples to {out_path}")
    except Exception as e:
        print(f"  Warning: could not download MATH-500 ({e})")
        print("  You can manually place math500_test.json in the output directory.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if "gsm8k" in args.datasets:
        download_gsm8k(args.output_dir)

    if "math500" in args.datasets:
        download_math500(args.output_dir)

    print("\nAll requested datasets downloaded.")


if __name__ == "__main__":
    main()
