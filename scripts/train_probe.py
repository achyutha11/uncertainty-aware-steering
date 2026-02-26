"""
train_probe.py — Train the uncertainty probe from saved traces and directions.

Usage:
    python scripts/train_probe.py \
        --traces_path outputs/traces/gsm8k_train_traces.json \
        --directions_path outputs/directions/directions.npz \
        --output_path outputs/probes/probe.pkl \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.direction import load_directions
from src.model_utils import load_model
from src.probe import extract_think_activations, save_probe, train_probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the uncertainty probe.")
    p.add_argument("--traces_path", required=True)
    p.add_argument("--directions_path", required=True)
    p.add_argument("--output_path", default="outputs/probes/probe.pkl")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Fraction of traces held out for AUROC evaluation.")
    p.add_argument("--n_traces", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    print(f"Loading traces from {args.traces_path}")
    with open(args.traces_path) as f:
        traces = json.load(f)
    if args.n_traces:
        traces = traces[: args.n_traces]
    print(f"Using {len(traces)} traces")

    print(f"Loading directions from {args.directions_path}")
    direction_vectors = load_directions(args.directions_path)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)

    print("Extracting </think> token activations...")
    X, y = extract_think_activations(model, tokenizer, traces, direction_vectors)
    print(f"Feature matrix: {X.shape}, labels: {y.sum()} correct / {len(y)} total")

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print(f"Training probe on {len(X_train)} examples...")
    probe = train_probe(X_train, y_train)

    # Evaluate
    if len(np.unique(y_val)) > 1:
        probs = probe.predict_proba(X_val)[:, 1]
        auroc = roc_auc_score(y_val, probs)
        print(f"Held-out AUROC: {auroc:.4f}")
        if auroc < 0.7:
            print("  WARNING: AUROC < 0.7. Check directions quality and data quantity.")
    else:
        print("  Skipping AUROC: only one class in validation set.")

    # Save
    save_probe(probe, args.output_path)
    print(f"Probe saved to {args.output_path}")


if __name__ == "__main__":
    main()
