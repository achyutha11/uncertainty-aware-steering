"""
steerer.py — Token-by-token generation with stepwise reflection steering (ReflCtrl).

The core idea: maintain a `should_steer_next` flag. When the previous generated
token(s) constitute a step boundary (\\n\\n), set the flag. On the *next* forward
pass (first token of the new step), inject the steering direction and then clear
the flag. This ensures steering is applied only at the beginning of each step.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .direction import DirectionDict
from .model_utils import register_steering_hooks


def get_step_delimiter_ids(tokenizer: PreTrainedTokenizerBase) -> Set[int]:
    """Return the set of token IDs that can represent '\\n\\n' (step boundary).

    DeepSeek / Llama tokenizers may encode '\\n\\n' as one or two tokens.
    We collect all plausible single-token representations.
    """
    candidates = ["\n\n", " \n\n", "\n\n "]
    delimiter_ids: Set[int] = set()
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            delimiter_ids.add(ids[0])
    return delimiter_ids


def _get_two_token_delimiter(tokenizer: PreTrainedTokenizerBase) -> Optional[Tuple[int, int]]:
    """Return (id1, id2) if '\\n\\n' is encoded as exactly two tokens, else None."""
    ids = tokenizer.encode("\n\n", add_special_tokens=False)
    if len(ids) == 2:
        return (ids[0], ids[1])
    return None


def generate_with_steering(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    direction_vectors: DirectionDict,
    lambda_coeff: float = -1.0,
    layers_to_skip_first: int = 6,
    layers_to_skip_last: int = 6,
    max_new_tokens: int = 8192,
) -> str:
    """Token-by-token generation with stepwise reflection steering.

    Steering is applied on the forward pass that generates the first token of
    each new step (i.e., immediately after a \\n\\n boundary). The lambda_coeff
    scales the direction vector added to activations.

    Args:
        model: Eval-mode causal LM.
        tokenizer: Matching tokenizer.
        prompt: Input prompt text.
        direction_vectors: Precomputed direction dict from direction.py.
        lambda_coeff: Steering strength (negative suppresses reflection).
        layers_to_skip_first: Number of early layers not to steer.
        layers_to_skip_last: Number of final layers not to steer.
        max_new_tokens: Hard cap on generated tokens.

    Returns:
        Decoded generated text (excluding the prompt).
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

    generated_ids = []
    past_key_values = None
    should_steer_next = False
    prev_token_id: Optional[int] = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Build input for this step: on first step use full prompt,
            # thereafter use only the last generated token (KV cache active).
            if past_key_values is None:
                cur_input = input_ids
            else:
                cur_input = torch.tensor(
                    [[generated_ids[-1]]], device=input_ids.device
                )

            if should_steer_next:
                with register_steering_hooks(
                    model, direction_vectors, lambda_coeff, steerable
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

            past_key_values = outputs.past_key_values
            next_token_id = int(outputs.logits[0, -1].argmax())
            generated_ids.append(next_token_id)

            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                break

            # Detect step boundary to arm steering for the next token.
            # Single-token case:
            if next_token_id in delimiter_ids:
                should_steer_next = True
            # Two-token case: second token of pair just generated
            elif (
                two_token_delim is not None
                and prev_token_id == two_token_delim[0]
                and next_token_id == two_token_delim[1]
            ):
                should_steer_next = True

            prev_token_id = next_token_id

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 8192,
) -> str:
    """Plain greedy generation without any steering hooks."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = input_ids.to(next(model.parameters()).device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            max_length=input_ids.shape[1] + max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # Decode only the newly generated part
    new_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
