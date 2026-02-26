"""
probe.py — Uncertainty probe: train a logistic regression on </think>-token
activations to predict whether the model will answer correctly.

The feature vector for each trace is:
    [cos(d_l_attn, z_l_attn), cos(d_l_mlp, z_l_mlp) for l in 0..L-1]

where z_l is the hidden state at the </think> boundary token.

Score interpretation: probe.predict_proba(x)[0, 1] ∈ [0, 1].
High value → model is confident / likely correct.
Low value  → model is uncertain / likely wrong.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .direction import DirectionDict
from .model_utils import register_capture_hooks


THINK_TOKEN = "</think>"


def _find_think_position(
    tokenizer: PreTrainedTokenizerBase,
    input_ids: List[int],
) -> int:
    """Return the token index of the *last* </think> token, or -1 if absent."""
    think_ids = tokenizer.encode(THINK_TOKEN, add_special_tokens=False)
    # Search backwards
    for i in range(len(input_ids) - len(think_ids), -1, -1):
        if input_ids[i : i + len(think_ids)] == think_ids:
            return i + len(think_ids) - 1  # index of the last token of </think>
    return -1


def _build_feature_vector(
    captures: Dict[int, Dict[str, torch.Tensor]],
    direction_vectors: DirectionDict,
    n_layers: int,
) -> np.ndarray:
    """Build the cosine-similarity feature vector from a single forward-pass capture.

    Feature layout:
        [cos(d_l_attn, z_l_attn), cos(d_l_mlp, z_l_mlp)]  for l in range(n_layers)

    Missing layers produce 0.0 features.

    Args:
        captures: Output of register_capture_hooks (each value is [n_pos, hidden_dim]).
        direction_vectors: Precomputed direction dict.
        n_layers: Total number of transformer layers.

    Returns:
        1-D numpy array of length 2 * n_layers.
    """
    feats = []
    for layer_idx in range(n_layers):
        for subkey in ("attn", "mlp"):
            cos_val = 0.0
            if (
                layer_idx in captures
                and subkey in captures[layer_idx]
                and layer_idx in direction_vectors
                and subkey in direction_vectors[layer_idx]
            ):
                z = captures[layer_idx][subkey][0].numpy()  # [hidden_dim]
                d = direction_vectors[layer_idx][subkey]
                norm_z = np.linalg.norm(z)
                norm_d = np.linalg.norm(d)
                if norm_z > 1e-8 and norm_d > 1e-8:
                    cos_val = float(np.dot(z, d) / (norm_z * norm_d))
            feats.append(cos_val)
    return np.array(feats, dtype=np.float32)


def extract_think_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    traces_with_labels: List[dict],
    direction_vectors: DirectionDict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and label vector y from a list of annotated traces.

    Each trace dict must have:
        "text" or ("prompt" + "response"): the full text.
        "correct": bool — whether the model answered correctly.

    Returns:
        X: np.ndarray of shape [n_valid_traces, 2 * n_layers]
        y: np.ndarray of shape [n_valid_traces], dtype int (1=correct, 0=wrong)
    """
    try:
        n_layers = len(model.model.layers)
    except AttributeError:
        n_layers = max(direction_vectors.keys()) + 1

    X_rows = []
    y_rows = []

    for trace in traces_with_labels:
        text = trace.get("prompt", "") + trace.get("response", trace.get("text", ""))
        correct = int(bool(trace.get("correct", False)))

        input_ids_list = tokenizer.encode(text, add_special_tokens=False)
        think_pos = _find_think_position(tokenizer, input_ids_list)
        if think_pos < 0:
            continue  # no </think> token found; skip

        input_ids = torch.tensor([input_ids_list], device=next(model.parameters()).device)

        with torch.no_grad():
            with register_capture_hooks(model, target_positions=[think_pos]) as captures:
                model(input_ids=input_ids)

        feat = _build_feature_vector(captures, direction_vectors, n_layers)
        X_rows.append(feat)
        y_rows.append(correct)

    if not X_rows:
        raise ValueError("No valid traces found (missing </think> token in all traces).")

    return np.vstack(X_rows), np.array(y_rows, dtype=int)


def train_probe(X: np.ndarray, y: np.ndarray, **lr_kwargs) -> LogisticRegression:
    """Fit a logistic regression probe on the given features and labels.

    Args:
        X: Feature matrix [n_samples, n_features].
        y: Binary labels [n_samples].
        **lr_kwargs: Keyword arguments forwarded to LogisticRegression.

    Returns:
        Fitted LogisticRegression instance.
    """
    defaults = dict(max_iter=1000, C=1.0, solver="lbfgs")
    defaults.update(lr_kwargs)
    probe = LogisticRegression(**defaults)
    probe.fit(X, y)
    return probe


def score_uncertainty(
    probe: LogisticRegression,
    direction_vectors: DirectionDict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    partial_text: str,
) -> float:
    """Score uncertainty for a partially generated text.

    Extracts activations at the last token of partial_text (used as a proxy
    for the current reasoning state), builds the feature vector, and returns
    the probe's predicted probability of being correct.

    Returns:
        Float in [0, 1]. High = confident (correct), low = uncertain (wrong).
    """
    try:
        n_layers = len(model.model.layers)
    except AttributeError:
        n_layers = max(direction_vectors.keys()) + 1

    input_ids_list = tokenizer.encode(partial_text, add_special_tokens=False)
    if not input_ids_list:
        return 0.5  # no information

    # Try to use the </think> position; fall back to last token
    think_pos = _find_think_position(tokenizer, input_ids_list)
    target_pos = think_pos if think_pos >= 0 else len(input_ids_list) - 1

    input_ids = torch.tensor([input_ids_list], device=next(model.parameters()).device)

    with torch.no_grad():
        with register_capture_hooks(model, target_positions=[target_pos]) as captures:
            model(input_ids=input_ids)

    feat = _build_feature_vector(captures, direction_vectors, n_layers)
    prob = probe.predict_proba(feat.reshape(1, -1))[0, 1]
    return float(prob)


def save_probe(probe: LogisticRegression, path: str) -> None:
    """Pickle the probe to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(probe, f)


def load_probe(path: str) -> LogisticRegression:
    """Load a pickled probe from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
