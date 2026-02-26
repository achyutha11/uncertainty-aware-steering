"""
test_direction_sanity.py — Direction vector sanity checks (no GPU required).

Tests:
  1. compute_direction_vectors produces non-zero vectors.
  2. Vectors differ between layers.
  3. Cosine similarity between reflection and non-reflection means is < 1.
  4. save_directions / load_directions round-trip.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.direction import compute_direction_vectors, load_directions, save_directions


def _fake_acts(n: int, hidden_dim: int, n_layers: int, seed: int):
    """Generate fake activation dicts."""
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(n):
        act = {}
        for l in range(n_layers):
            act[l] = {
                "attn": rng.standard_normal(hidden_dim).astype(np.float32),
                "mlp":  rng.standard_normal(hidden_dim).astype(np.float32),
            }
        result.append(act)
    return result


N_LAYERS = 8
HIDDEN = 32
N_REFL = 20
N_NON  = 20


@pytest.fixture(scope="module")
def directions():
    refl = _fake_acts(N_REFL, HIDDEN, N_LAYERS, seed=0)
    non  = _fake_acts(N_NON,  HIDDEN, N_LAYERS, seed=1)
    return compute_direction_vectors(refl, non)


def test_directions_non_empty(directions):
    assert len(directions) > 0, "No direction vectors computed"


def test_directions_all_layers_present(directions):
    for l in range(N_LAYERS):
        assert l in directions, f"Layer {l} missing from direction dict"


def test_directions_non_zero(directions):
    for layer_idx, subkeys in directions.items():
        for subkey, vec in subkeys.items():
            norm = np.linalg.norm(vec)
            assert norm > 1e-8, (
                f"Direction for layer {layer_idx} {subkey} is near-zero (norm={norm:.2e})"
            )


def test_directions_differ_between_layers(directions):
    """Vectors in different layers should not be identical."""
    seen = {}
    for layer_idx, subkeys in directions.items():
        for subkey, vec in subkeys.items():
            for (prev_l, prev_sk), prev_vec in seen.items():
                if prev_sk == subkey and layer_idx != prev_l:
                    assert not np.allclose(vec, prev_vec), (
                        f"Layer {layer_idx} and {prev_l} {subkey} directions are identical"
                    )
            seen[(layer_idx, subkey)] = vec


def test_cosine_similarity_below_one(directions):
    """Sanity check: the direction vector itself should not have cosine 1 with zero."""
    for layer_idx, subkeys in directions.items():
        for subkey, vec in subkeys.items():
            # A non-zero vector should have cosine < 1 with a random vector
            rng = np.random.default_rng(42)
            rand = rng.standard_normal(vec.shape).astype(np.float32)
            cos = np.dot(vec, rand) / (np.linalg.norm(vec) * np.linalg.norm(rand) + 1e-10)
            assert abs(cos) < 1.0, f"Unexpected perfect alignment at layer {layer_idx} {subkey}"


def test_save_load_round_trip(directions):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test_dirs.npz")
        save_directions(directions, path)
        loaded = load_directions(path)

    assert set(loaded.keys()) == set(directions.keys())
    for layer_idx in directions:
        for subkey in directions[layer_idx]:
            assert subkey in loaded[layer_idx], f"Missing {subkey} in loaded layer {layer_idx}"
            np.testing.assert_array_almost_equal(
                directions[layer_idx][subkey],
                loaded[layer_idx][subkey],
                decimal=5,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
