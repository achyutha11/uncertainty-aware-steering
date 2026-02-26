"""
extract_direction.py — Compute reflection direction vectors from saved traces.

Usage:
    python scripts/extract_direction.py \
        --traces_path outputs/traces/gsm8k_train_traces.json \
        --output_dir outputs/directions \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.direction import (
    collect_step_activations,
    compute_direction_vectors,
    get_steerable_layers,
    load_directions,
    save_directions,
)
from src.model_utils import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract reflection direction vectors.")
    p.add_argument("--traces_path", required=True, help="Path to traces JSON file.")
    p.add_argument("--output_dir", default="outputs/directions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument(
        "--n_traces",
        type=int,
        default=None,
        help="Limit the number of traces processed (default: all).",
    )
    p.add_argument(
        "--skip_first_layers",
        type=int,
        default=6,
        help="Number of early layers to skip when reporting steerable layers.",
    )
    p.add_argument(
        "--skip_last_layers",
        type=int,
        default=6,
        help="Number of final layers to skip.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load traces
    print(f"Loading traces from {args.traces_path}")
    with open(args.traces_path) as f:
        traces = json.load(f)

    if args.n_traces is not None:
        traces = traces[: args.n_traces]
    print(f"Using {len(traces)} traces")

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)

    # Collect activations
    print("Collecting step activations (one forward pass per trace)...")
    acts = collect_step_activations(model, tokenizer, traces)

    n_refl = len(acts["reflection"])
    n_non  = len(acts["non_reflection"])
    print(f"Collected: {n_refl} reflection traces, {n_non} non-reflection traces")

    if n_refl == 0 or n_non == 0:
        raise RuntimeError(
            "Insufficient data: need at least one reflection and one non-reflection trace. "
            "Check that traces contain reflection keywords (Wait, Hmm, Actually, etc.)."
        )

    # Compute directions
    print("Computing direction vectors...")
    direction_dict = compute_direction_vectors(acts["reflection"], acts["non_reflection"])
    print(f"Computed directions for {len(direction_dict)} layers")

    # Sanity check
    n_layers = len(model.model.layers)
    steerable = get_steerable_layers(n_layers, args.skip_first_layers, args.skip_last_layers)
    n_steerable = sum(1 for l in steerable if l in direction_dict)
    print(f"Steerable layers ({args.skip_first_layers} ≤ l < {n_layers - args.skip_last_layers}): "
          f"{len(steerable)} total, {n_steerable} with computed directions")

    # Verify directions are non-zero
    for layer_idx, subkeys in direction_dict.items():
        for subkey, vec in subkeys.items():
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                print(f"  WARNING: layer {layer_idx} {subkey} direction is near-zero (norm={norm:.2e})")

    # Save
    out_path = os.path.join(args.output_dir, "directions.npz")
    save_directions(direction_dict, out_path)
    print(f"Saved direction vectors to {out_path}")

    # Verify round-trip load
    loaded = load_directions(out_path)
    assert set(loaded.keys()) == set(direction_dict.keys()), "Round-trip load mismatch!"
    print("Round-trip load verification passed.")


if __name__ == "__main__":
    main()
