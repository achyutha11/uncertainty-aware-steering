"""
model_utils.py — Load model/tokenizer and register forward hooks for
activation capture and steering.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_name: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokenizer.

    Args:
        model_name: HuggingFace model name or local path.
        device: Ignored when device_map='auto' (multi-GPU). Pass 'cpu' or
                'cuda' only for single-device, non-auto setups.
        dtype: Weight dtype; bfloat16 recommended for Llama-family models.

    Returns:
        (model, tokenizer) tuple. Model is in eval mode with grad disabled.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    # Override device_map for explicit single-device usage
    if device in ("cpu",):
        load_kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return model, tokenizer


def get_layer_modules(
    model: AutoModelForCausalLM,
) -> List[Tuple[torch.nn.Module, torch.nn.Module]]:
    """Return list of (attn_module, mlp_module) for each transformer layer.

    Handles Llama-style naming: model.model.layers[i].self_attn / .mlp.
    Raises AttributeError if the architecture is not recognised.
    """
    try:
        layers = model.model.layers
    except AttributeError as exc:
        raise AttributeError(
            "Cannot find model.model.layers — unsupported architecture."
        ) from exc

    result = []
    for layer in layers:
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if attn is None or mlp is None:
            raise AttributeError(
                f"Layer {layer} does not have expected self_attn/mlp sub-modules."
            )
        result.append((attn, mlp))
    return result


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def register_capture_hooks(
    model: AutoModelForCausalLM,
    target_positions: Optional[List[int]] = None,
):
    """Context manager that captures attn/mlp outputs for specified token positions.

    Yields a dict that will be populated after a forward pass:
        captures[layer_idx]["attn"]  -> tensor of shape [n_positions, hidden_dim]
        captures[layer_idx]["mlp"]   -> tensor of shape [n_positions, hidden_dim]

    If target_positions is None, captures the *last* token position.

    Usage::

        with register_capture_hooks(model, positions) as captures:
            model(input_ids=ids)
        attn_vec = captures[10]["attn"]  # layer 10, attn hidden states
    """
    layer_modules = get_layer_modules(model)
    captures: Dict[int, Dict[str, torch.Tensor]] = {}
    hooks = []

    def make_attn_hook(layer_idx: int):
        def hook(module, input, output):
            # Llama attn output: (hidden_states, attn_weights, past_kv)
            # hidden_states shape: [batch, seq_len, hidden_dim]
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.detach().float()
            if target_positions is None:
                vecs = hidden[0, -1:, :]  # last token
            else:
                indices = [p for p in target_positions if p < hidden.shape[1]]
                if not indices:
                    return
                vecs = hidden[0, indices, :]
            if layer_idx not in captures:
                captures[layer_idx] = {}
            captures[layer_idx]["attn"] = vecs.cpu()
        return hook

    def make_mlp_hook(layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.detach().float()
            if target_positions is None:
                vecs = hidden[0, -1:, :]
            else:
                indices = [p for p in target_positions if p < hidden.shape[1]]
                if not indices:
                    return
                vecs = hidden[0, indices, :]
            if layer_idx not in captures:
                captures[layer_idx] = {}
            captures[layer_idx]["mlp"] = vecs.cpu()
        return hook

    for idx, (attn_mod, mlp_mod) in enumerate(layer_modules):
        hooks.append(attn_mod.register_forward_hook(make_attn_hook(idx)))
        hooks.append(mlp_mod.register_forward_hook(make_mlp_hook(idx)))

    try:
        yield captures
    finally:
        for h in hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Steering hooks
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def register_steering_hooks(
    model: AutoModelForCausalLM,
    direction_vectors: Dict[int, Dict[str, torch.Tensor]],
    lambda_coeff: float,
    layers_to_steer: Optional[Set[int]] = None,
):
    """Context manager that adds λ * d_l to attn/mlp outputs during a forward pass.

    Args:
        direction_vectors: dict[layer_idx][attn|mlp] -> 1-D tensor [hidden_dim].
        lambda_coeff: Scalar multiplier (negative = suppress reflection).
        layers_to_steer: Set of layer indices to steer. If None, steers all
                         layers present in direction_vectors.
    """
    layer_modules = get_layer_modules(model)
    hooks = []

    active_layers = layers_to_steer if layers_to_steer is not None else set(direction_vectors.keys())

    def make_attn_steer(layer_idx: int, direction: torch.Tensor):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None
            d = direction.to(hidden.device, hidden.dtype)
            hidden = hidden + lambda_coeff * d
            if rest is not None:
                return (hidden,) + rest
            return hidden
        return hook

    def make_mlp_steer(layer_idx: int, direction: torch.Tensor):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None
            d = direction.to(hidden.device, hidden.dtype)
            hidden = hidden + lambda_coeff * d
            if rest is not None:
                return (hidden,) + rest
            return hidden
        return hook

    for idx, (attn_mod, mlp_mod) in enumerate(layer_modules):
        if idx not in active_layers or idx not in direction_vectors:
            continue
        layer_dirs = direction_vectors[idx]
        if "attn" in layer_dirs:
            d_attn = layer_dirs["attn"]
            hooks.append(attn_mod.register_forward_hook(make_attn_steer(idx, d_attn)))
        if "mlp" in layer_dirs:
            d_mlp = layer_dirs["mlp"]
            hooks.append(mlp_mod.register_forward_hook(make_mlp_steer(idx, d_mlp)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()
