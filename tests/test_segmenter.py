"""
Unit tests for src/segmenter.py — verify step segmentation and reflection labeling
on a realistic DeepSeek-R1 chain-of-thought trace.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.segmenter import (
    DEFAULT_KEYWORDS,
    is_reflection_step,
    label_steps,
    segment_steps,
)

# ---------------------------------------------------------------------------
# Sample DeepSeek-R1 style trace (abbreviated)
# ---------------------------------------------------------------------------

SAMPLE_TRACE = """\
Let me compute 15 * 7 step by step.

15 * 7 = 105. So the answer is 105.

Wait, let me double-check: 15 * 7 = 15 * 5 + 15 * 2 = 75 + 30 = 105. Yes, correct.

Actually, I can also verify: 7 * 15 = 7 * 10 + 7 * 5 = 70 + 35 = 105.

The answer is 105.
"""


def test_segment_steps_basic():
    steps = segment_steps(SAMPLE_TRACE)
    assert len(steps) == 5, f"Expected 5 steps, got {len(steps)}: {steps}"


def test_segment_steps_no_empty():
    steps = segment_steps(SAMPLE_TRACE)
    for s in steps:
        assert s.strip() != "", "segment_steps should not return empty strings"


def test_is_reflection_step_wait():
    assert is_reflection_step("Wait, let me double-check: 15 * 7 = 105.")


def test_is_reflection_step_actually():
    assert is_reflection_step("Actually, I can also verify: 7 * 15 = 105.")


def test_is_reflection_step_non_reflection():
    assert not is_reflection_step("15 * 7 = 105. So the answer is 105.")
    assert not is_reflection_step("The answer is 105.")


def test_label_steps_counts():
    steps = segment_steps(SAMPLE_TRACE)
    labeled = label_steps(steps)
    assert len(labeled) == len(steps)
    reflection_count = sum(1 for _, r in labeled if r)
    non_reflection_count = sum(1 for _, r in labeled if not r)
    # Trace has 2 explicit reflection steps (Wait, Actually)
    assert reflection_count == 2, f"Expected 2 reflection steps, got {reflection_count}"
    assert non_reflection_count == 3, f"Expected 3 non-reflection steps, got {non_reflection_count}"


def test_label_steps_order():
    steps = segment_steps(SAMPLE_TRACE)
    labeled = label_steps(steps)
    texts = [t for t, _ in labeled]
    # First step should NOT be reflection
    assert not labeled[0][1], "First step should not be a reflection step"
    # Third step (index 2) starts with "Wait"
    assert labeled[2][1], f"Third step should be reflection: {labeled[2][0][:40]}"
    # Fourth step (index 3) starts with "Actually"
    assert labeled[3][1], f"Fourth step should be reflection: {labeled[3][0][:40]}"


def test_is_reflection_step_custom_keywords():
    assert is_reflection_step("Hmm, this seems off.", keywords=["Hmm"])
    assert not is_reflection_step("Hmm, this seems off.", keywords=["Wait"])


def test_segment_steps_single_block():
    text = "Just one block with no double newlines."
    steps = segment_steps(text)
    assert steps == ["Just one block with no double newlines."]


def test_segment_steps_multiple_newlines():
    text = "Step one.\n\n\n\nStep two."
    steps = segment_steps(text)
    assert len(steps) == 2
    assert steps[0] == "Step one."
    assert steps[1] == "Step two."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
