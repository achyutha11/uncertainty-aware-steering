"""
collect_traces.py — Run inference on GSM8k and save traces + metadata.

Usage:
    python scripts/collect_traces.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --output_dir outputs/traces \
        --n_samples 500 \
        --split train
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_utils import load_model
from src.steerer import generate_baseline


# ---------------------------------------------------------------------------
# Prompting
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
    """Build a few-shot chat prompt for DeepSeek-R1 style models."""
    shots = ""
    for ex in FEW_SHOT_EXAMPLES:
        shots += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    shots += f"Question: {question}\nAnswer:"
    return shots


def extract_gsm8k_answer(response: str) -> str | None:
    """Extract the numeric answer after '####' from a GSM8k response."""
    match = re.search(r"####\s*([\d,.-]+)", response)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last number in the response
    nums = re.findall(r"[\d,]+\.?\d*", response)
    return nums[-1].replace(",", "") if nums else None


def normalize_answer(ans: str) -> str:
    return ans.strip().rstrip(".")


def is_correct(response: str, gold_answer: str) -> bool:
    pred = extract_gsm8k_answer(response)
    if pred is None:
        return False
    try:
        return abs(float(normalize_answer(pred)) - float(normalize_answer(gold_answer))) < 1e-4
    except ValueError:
        return normalize_answer(pred) == normalize_answer(gold_answer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect reasoning traces from GSM8k.")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--output_dir", default="outputs/traces")
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)

    print(f"Loading GSM8k ({args.split} split)...")
    dataset = load_dataset("gsm8k", "main", split=args.split)
    dataset = dataset.select(range(min(args.n_samples, len(dataset))))

    traces = []
    n_correct = 0

    for i, example in enumerate(tqdm(dataset, desc="Generating")):
        question = example["question"]
        gold = example["answer"].split("####")[-1].strip().replace(",", "")
        prompt = build_prompt(question)

        response = generate_baseline(
            model, tokenizer, prompt, max_new_tokens=args.max_new_tokens
        )

        correct = is_correct(response, gold)
        n_correct += int(correct)

        # Store token IDs for efficient re-extraction later
        full_text = prompt + response
        token_ids = tokenizer.encode(full_text, add_special_tokens=False)

        trace = {
            "idx": i,
            "question": question,
            "gold_answer": gold,
            "prompt": prompt,
            "response": response,
            "correct": correct,
            "token_ids": token_ids,
        }
        traces.append(trace)

        if (i + 1) % 50 == 0:
            acc = n_correct / (i + 1)
            print(f"  [{i+1}/{len(dataset)}] Running accuracy: {acc:.3f}")
            # Checkpoint save
            ckpt_path = os.path.join(args.output_dir, f"traces_ckpt_{i+1}.json")
            with open(ckpt_path, "w") as f:
                json.dump(traces, f, indent=2)

    final_acc = n_correct / len(traces) if traces else 0.0
    print(f"\nFinal accuracy: {final_acc:.4f} ({n_correct}/{len(traces)})")

    out_path = os.path.join(args.output_dir, f"gsm8k_{args.split}_traces.json")
    with open(out_path, "w") as f:
        json.dump(traces, f, indent=2)
    print(f"Saved {len(traces)} traces to {out_path}")


if __name__ == "__main__":
    main()
