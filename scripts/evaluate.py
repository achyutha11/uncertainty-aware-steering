"""
evaluate.py — Evaluate baseline / steered / uncertainty-gated generation on GSM8k.

Modes:
    baseline           — plain model.generate(), no hooks
    steered            — ReflCtrl stepwise steering with fixed lambda
    uncertainty_gated  — probe gates lambda dynamically at each step boundary

Usage examples:
    python scripts/evaluate.py --mode baseline --n_samples 50
    python scripts/evaluate.py --mode steered --directions_path outputs/directions/directions.npz --lambda_coeff -1.0 --n_samples 50
    python scripts/evaluate.py --mode uncertainty_gated --directions_path outputs/directions/directions.npz --probe_path outputs/probes/probe.pkl --n_samples 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.direction import DirectionDict, load_directions
from src.model_utils import load_model, register_capture_hooks, register_steering_hooks
from src.probe import load_probe, score_uncertainty
from src.segmenter import DEFAULT_KEYWORDS, is_reflection_step, segment_steps
from src.steerer import (
    generate_baseline,
    generate_with_steering,
    get_step_delimiter_ids,
    _get_two_token_delimiter,
)


# ---------------------------------------------------------------------------
# Answer extraction (reuse from collect_traces.py)
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> Optional[str]:
    match = re.search(r"####\s*([\d,.-]+)", response)
    if match:
        return match.group(1).replace(",", "").strip()
    nums = re.findall(r"[\d,]+\.?\d*", response)
    return nums[-1].replace(",", "") if nums else None


def is_correct(response: str, gold: str) -> bool:
    pred = extract_answer(response)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()


def count_reflection_steps(response: str) -> int:
    steps = segment_steps(response)
    return sum(1 for s in steps if is_reflection_step(s))


# ---------------------------------------------------------------------------
# Few-shot prompt builder (same as collect_traces.py)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72",
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 × 50 = $10.\n#### 10",
    },
]


def build_prompt(question: str) -> str:
    shots = ""
    for ex in FEW_SHOT_EXAMPLES:
        shots += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    shots += f"Question: {question}\nAnswer:"
    return shots


# ---------------------------------------------------------------------------
# Uncertainty-gated generation
# ---------------------------------------------------------------------------

def generate_uncertainty_gated(
    model,
    tokenizer,
    prompt: str,
    direction_vectors: DirectionDict,
    probe,
    lambda_max: float = -2.0,
    uncertainty_threshold: float = 0.5,
    layers_to_skip_first: int = 6,
    layers_to_skip_last: int = 6,
    max_new_tokens: int = 8192,
) -> tuple[str, List[float]]:
    """Token-by-token generation with uncertainty-gated steering.

    At each step boundary:
        1. Score uncertainty using the probe on the text generated so far.
        2. If probe_score < threshold (model is uncertain), apply steering with
           effective_lambda = lambda_max * (1 - probe_score).
        3. If probe_score >= threshold (model is confident), skip steering.

    Returns:
        (generated_text, list_of_probe_scores_at_boundaries)
    """
    try:
        n_layers = len(model.model.layers)
    except AttributeError:
        n_layers = max(direction_vectors.keys()) + 1

    steerable = set(range(layers_to_skip_first, n_layers - layers_to_skip_last))

    delimiter_ids = get_step_delimiter_ids(tokenizer)
    two_token_delim = _get_two_token_delimiter(tokenizer)

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = input_ids.to(next(model.parameters()).device)

    generated_ids: List[int] = []
    past_key_values = None
    should_steer_next = False
    effective_lambda = 0.0
    prev_token_id: Optional[int] = None
    probe_scores: List[float] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if past_key_values is None:
                cur_input = input_ids
            else:
                cur_input = torch.tensor(
                    [[generated_ids[-1]]], device=input_ids.device
                )

            if should_steer_next and abs(effective_lambda) > 1e-6:
                with register_steering_hooks(
                    model, direction_vectors, effective_lambda, steerable
                ):
                    outputs = model(
                        input_ids=cur_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                should_steer_next = False
            else:
                outputs = model(
                    input_ids=cur_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                should_steer_next = False

            past_key_values = outputs.past_key_values
            next_token_id = int(outputs.logits[0, -1].argmax())
            generated_ids.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            # Detect step boundary
            boundary_hit = False
            if next_token_id in delimiter_ids:
                boundary_hit = True
            elif (
                two_token_delim is not None
                and prev_token_id == two_token_delim[0]
                and next_token_id == two_token_delim[1]
            ):
                boundary_hit = True

            if boundary_hit:
                # Score the current generation state
                partial = prompt + tokenizer.decode(generated_ids, skip_special_tokens=False)
                probe_score = score_uncertainty(probe, direction_vectors, model, tokenizer, partial)
                probe_scores.append(probe_score)

                if probe_score < uncertainty_threshold:
                    # Uncertain: apply steering proportional to uncertainty
                    effective_lambda = lambda_max * (1.0 - probe_score)
                else:
                    effective_lambda = 0.0
                should_steer_next = True

            prev_token_id = next_token_id

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, probe_scores


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)

    direction_vectors: Optional[DirectionDict] = None
    if args.directions_path:
        print(f"Loading directions from {args.directions_path}")
        direction_vectors = load_directions(args.directions_path)
        # Convert numpy arrays to torch tensors for hooks
        direction_tensors: DirectionDict = {
            layer: {sk: torch.from_numpy(v) for sk, v in subkeys.items()}
            for layer, subkeys in direction_vectors.items()
        }
    else:
        direction_tensors = None

    probe = None
    if args.probe_path:
        print(f"Loading probe from {args.probe_path}")
        probe = load_probe(args.probe_path)

    print(f"Loading GSM8k test split...")
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(min(args.n_samples, len(dataset))))

    results = []
    n_correct = 0
    total_tokens = 0
    total_refl_steps = 0

    for i, example in enumerate(tqdm(dataset, desc=f"[{args.mode}]")):
        question = example["question"]
        gold = example["answer"].split("####")[-1].strip().replace(",", "")
        prompt = build_prompt(question)

        t0 = time.time()
        extra_info = {}

        if args.mode == "baseline":
            response = generate_baseline(
                model, tokenizer, prompt, max_new_tokens=args.max_new_tokens
            )

        elif args.mode == "steered":
            if direction_tensors is None:
                raise ValueError("--directions_path required for steered mode.")
            response = generate_with_steering(
                model, tokenizer, prompt,
                direction_tensors,
                lambda_coeff=args.lambda_coeff,
                layers_to_skip_first=args.skip_first_layers,
                layers_to_skip_last=args.skip_last_layers,
                max_new_tokens=args.max_new_tokens,
            )

        elif args.mode == "uncertainty_gated":
            if direction_tensors is None or probe is None:
                raise ValueError(
                    "--directions_path and --probe_path required for uncertainty_gated mode."
                )
            response, probe_scores = generate_uncertainty_gated(
                model, tokenizer, prompt,
                direction_tensors, probe,
                lambda_max=args.lambda_coeff,
                uncertainty_threshold=args.uncertainty_threshold,
                layers_to_skip_first=args.skip_first_layers,
                layers_to_skip_last=args.skip_last_layers,
                max_new_tokens=args.max_new_tokens,
            )
            extra_info["probe_scores"] = probe_scores
            extra_info["mean_probe_score"] = float(np.mean(probe_scores)) if probe_scores else None

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        elapsed = time.time() - t0
        correct = is_correct(response, gold)
        n_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        n_refl = count_reflection_steps(response)

        n_correct += int(correct)
        total_tokens += n_tokens
        total_refl_steps += n_refl

        result = {
            "idx": i,
            "question": question,
            "gold_answer": gold,
            "response": response,
            "correct": correct,
            "n_tokens": n_tokens,
            "n_reflection_steps": n_refl,
            "elapsed_s": elapsed,
            **extra_info,
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            acc = n_correct / (i + 1)
            avg_tok = total_tokens / (i + 1)
            print(f"  [{i+1}/{len(dataset)}] acc={acc:.3f}, avg_tokens={avg_tok:.0f}")

    # Summary
    n = len(results)
    summary = {
        "mode": args.mode,
        "n_samples": n,
        "accuracy": n_correct / n if n else 0.0,
        "avg_tokens": total_tokens / n if n else 0.0,
        "avg_reflection_steps": total_refl_steps / n if n else 0.0,
        "lambda_coeff": args.lambda_coeff,
    }

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save results
    out_path = os.path.join(args.output_dir, f"eval_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate steering on GSM8k.")
    p.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "steered", "uncertainty_gated"],
    )
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--directions_path", default=None)
    p.add_argument("--probe_path", default=None)
    p.add_argument("--lambda_coeff", type=float, default=-1.0)
    p.add_argument("--uncertainty_threshold", type=float, default=0.5,
                   help="Probe score below this triggers steering.")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--skip_first_layers", type=int, default=6)
    p.add_argument("--skip_last_layers", type=int, default=6)
    p.add_argument("--output_dir", default="outputs/eval")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
