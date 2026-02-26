"""
segmenter.py — Segment reasoning traces into steps and identify reflection steps.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


DEFAULT_KEYWORDS = [
    "Wait",
    "Hmm",
    "Let me reconsider",
    "Let me think",
    "Actually",
    "wait",
    "Hmm,",
    "Wait,",
]


def segment_steps(text: str) -> List[str]:
    """Split reasoning text into steps by double-newline boundaries.

    Strips leading/trailing whitespace from each step and drops empty strings.
    """
    raw = re.split(r"\n\n+", text)
    return [s.strip() for s in raw if s.strip()]


def is_reflection_step(
    step_text: str,
    keywords: Optional[List[str]] = None,
) -> bool:
    """Return True if step_text begins with (or contains) a reflection keyword.

    Matching is done at the start of the step (after stripping) to avoid
    false positives from mid-sentence occurrences.
    """
    kws = keywords if keywords is not None else DEFAULT_KEYWORDS
    stripped = step_text.strip()
    for kw in kws:
        if stripped.startswith(kw):
            return True
    return False


def label_steps(
    steps: List[str],
    keywords: Optional[List[str]] = None,
) -> List[Tuple[str, bool]]:
    """Return list of (step_text, is_reflection) tuples."""
    return [(s, is_reflection_step(s, keywords)) for s in steps]


def find_step_token_positions(
    tokenizer: PreTrainedTokenizerBase,
    full_text: str,
    steps: List[str],
) -> List[int]:
    """Find the index of the first token of each step within the tokenized full_text.

    Strategy: encode the full text, then for each step encode its first ~10
    characters and search for the resulting token sub-sequence. Returns the
    index of the *first* token of the matched sub-sequence.

    If a step cannot be located, -1 is inserted for that position.

    Args:
        tokenizer: Tokenizer compatible with the model.
        full_text: The complete text (prompt + response).
        steps: Ordered list of step strings returned by segment_steps().

    Returns:
        List of integer token indices, same length as steps.
    """
    full_ids: List[int] = tokenizer.encode(full_text, add_special_tokens=False)
    positions: List[int] = []

    search_start = 0
    for step in steps:
        if not step:
            positions.append(-1)
            continue

        # Encode a prefix of the step to get a search target
        prefix = step[:60]  # enough to be unique
        step_ids = tokenizer.encode(prefix, add_special_tokens=False)

        # Slide over full_ids from search_start to find the sub-sequence
        found = -1
        for i in range(search_start, len(full_ids) - len(step_ids) + 1):
            if full_ids[i : i + len(step_ids)] == step_ids:
                found = i
                search_start = i + 1  # advance to avoid re-matching same location
                break

        positions.append(found)

    return positions
