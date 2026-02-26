"""
direction.py — Extract reflection direction vectors from saved traces.

The direction vector for layer l is:
    d_l = mean(activations at reflection steps) - mean(activations at non-reflection steps)
computed separately for attention and MLP sub-modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


# Type aliases
# (Using Any for torch.Tensor to avoid import-time dependency)
ActivationDict = Dict[int, Dict[str, "torch.Tensor"]]  # layer -> {attn|mlp} -> [hidden]
DirectionDict  = Dict[int, Dict[str, np.ndarray]]       # layer -> {attn|mlp} -> [hidden_dim]


def collect_step_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    traces: List[dict],
    keywords: List[str] | None = None,
) -> Dict[str, List[Dict[int, Dict[str, np.ndarray]]]]:
    """Run one forward pass per trace; collect per-step activations.

    Args:
        model: Loaded causal LM (eval mode).
        tokenizer: Matching tokenizer.
        traces: List of dicts with at least a "response" key (the model's
                chain-of-thought text). Optionally "prompt" for full context.
        keywords: Reflection keywords; passed through to segmenter.

    Returns:
        Dict with keys "reflection" and "non_reflection", each a list of
        activation dicts: layer_idx -> {"attn": np.ndarray, "mlp": np.ndarray}
        where each array has shape [hidden_dim] (mean over the captured position).
    """
    import torch  # lazy import — not needed for pure-numpy functions below

    from .model_utils import register_capture_hooks
    from .segmenter import find_step_token_positions, label_steps, segment_steps

    reflection_acts: List[Dict[int, Dict[str, np.ndarray]]] = []
    non_reflection_acts: List[Dict[int, Dict[str, np.ndarray]]] = []

    for trace in traces:
        text = trace.get("prompt", "") + trace.get("response", trace.get("text", ""))
        steps = segment_steps(trace.get("response", trace.get("text", "")))
        if not steps:
            continue

        labeled = label_steps(steps, keywords)
        positions = find_step_token_positions(tokenizer, text, steps)

        # Separate valid positions by label
        refl_positions = [
            pos for (_, is_refl), pos in zip(labeled, positions)
            if is_refl and pos >= 0
        ]
        non_refl_positions = [
            pos for (_, is_refl), pos in zip(labeled, positions)
            if not is_refl and pos >= 0
        ]

        if not refl_positions and not non_refl_positions:
            continue

        all_positions = list(set(refl_positions + non_refl_positions))
        all_positions.sort()

        input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(next(model.parameters()).device)

        with torch.no_grad():
            with register_capture_hooks(model, target_positions=all_positions) as captures:
                model(input_ids=input_ids)

        # Convert captures to {layer: {attn|mlp: np.ndarray per position}}
        pos_to_idx = {pos: i for i, pos in enumerate(all_positions)}

        def _extract_mean(positions_list, subkey):
            """Mean activation over the given positions for one subkey."""
            acts_per_layer = {}
            for layer_idx, layer_caps in captures.items():
                if subkey not in layer_caps:
                    continue
                # layer_caps[subkey]: [n_captured_positions, hidden_dim]
                vecs = []
                for pos in positions_list:
                    if pos in pos_to_idx:
                        vec_idx = pos_to_idx[pos]
                        if vec_idx < layer_caps[subkey].shape[0]:
                            vecs.append(layer_caps[subkey][vec_idx].numpy())
                if vecs:
                    acts_per_layer[layer_idx] = {subkey: np.mean(vecs, axis=0)}
            return acts_per_layer

        def _merge(d1, d2):
            out = {}
            for k in set(list(d1.keys()) + list(d2.keys())):
                out[k] = {}
                if k in d1:
                    out[k].update(d1[k])
                if k in d2:
                    out[k].update(d2[k])
            return out

        if refl_positions:
            a = _extract_mean(refl_positions, "attn")
            m = _extract_mean(refl_positions, "mlp")
            reflection_acts.append(_merge(a, m))

        if non_refl_positions:
            a = _extract_mean(non_refl_positions, "attn")
            m = _extract_mean(non_refl_positions, "mlp")
            non_reflection_acts.append(_merge(a, m))

    return {"reflection": reflection_acts, "non_reflection": non_reflection_acts}


def compute_direction_vectors(
    reflection_acts: List[Dict[int, Dict[str, np.ndarray]]],
    non_reflection_acts: List[Dict[int, Dict[str, np.ndarray]]],
) -> DirectionDict:
    """Compute mean-difference direction vectors per layer and sub-module.

    d_l[subkey] = mean(reflection) - mean(non_reflection)

    Returns:
        dict[layer_idx][attn|mlp] -> np.ndarray of shape [hidden_dim]
    """
    # Collect all activation vectors per layer per subkey
    refl_by_layer: Dict[int, Dict[str, List[np.ndarray]]] = {}
    non_by_layer:  Dict[int, Dict[str, List[np.ndarray]]] = {}

    for acts_dict, storage in [(reflection_acts, refl_by_layer), (non_reflection_acts, non_by_layer)]:
        for act in acts_dict:
            for layer_idx, subkeys in act.items():
                if layer_idx not in storage:
                    storage[layer_idx] = {}
                for subkey, vec in subkeys.items():
                    storage[layer_idx].setdefault(subkey, []).append(vec)

    direction_dict: DirectionDict = {}
    all_layers = set(refl_by_layer.keys()) & set(non_by_layer.keys())

    for layer_idx in sorted(all_layers):
        direction_dict[layer_idx] = {}
        for subkey in ("attn", "mlp"):
            refl_vecs = refl_by_layer.get(layer_idx, {}).get(subkey, [])
            non_vecs  = non_by_layer.get(layer_idx, {}).get(subkey, [])
            if not refl_vecs or not non_vecs:
                continue
            mean_refl = np.mean(refl_vecs, axis=0)
            mean_non  = np.mean(non_vecs,  axis=0)
            direction_dict[layer_idx][subkey] = mean_refl - mean_non

    return direction_dict


def save_directions(direction_dict: DirectionDict, path: str) -> None:
    """Save direction vectors to a .npz file.

    Keys are formatted as "layer_{idx}_attn" and "layer_{idx}_mlp".
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {}
    for layer_idx, subkeys in direction_dict.items():
        for subkey, vec in subkeys.items():
            arrays[f"layer_{layer_idx}_{subkey}"] = vec
    np.savez(path, **arrays)


def load_directions(path: str) -> DirectionDict:
    """Load direction vectors from a .npz file produced by save_directions."""
    data = np.load(path)
    direction_dict: DirectionDict = {}
    for key in data.files:
        parts = key.split("_")  # ["layer", str(idx), subkey]
        layer_idx = int(parts[1])
        subkey = parts[2]
        direction_dict.setdefault(layer_idx, {})[subkey] = data[key]
    return direction_dict


def get_steerable_layers(
    n_total_layers: int,
    skip_first: int = 6,
    skip_last: int = 6,
) -> List[int]:
    """Return the list of layer indices to steer (skipping first/last N layers)."""
    return list(range(skip_first, n_total_layers - skip_last))
