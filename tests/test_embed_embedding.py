"""Tests for `Embedding` — the value object, which is pure numpy and needs no weights.

Every test here runs in the gate. That is the point of keeping `embedding.py` free of torch:
the thing a caller actually holds is testable on a machine with no CUDA, no `esm` package and
nothing in the HF cache.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from protein.embed import Embedding


def _embedding(
    rows: int = 33,
    columns: int = 960,
    *,
    source: str | None = "P12345",
    checkpoint: str = "300m",
    layer: int = 30,
) -> Embedding:
    return Embedding(np.zeros((rows, columns), dtype=np.float32), source, checkpoint, layer)


def test_len_is_the_residue_count_and_not_the_tokenised_length() -> None:
    # The sharp one, held here as well as in the model lane: BOS and EOS are stripped before
    # an `Embedding` is built, so 33 residues is 33 rows and never 35.
    assert len(_embedding(rows=33)) == 33


def test_shape_reaches_the_array_without_unwrapping_it() -> None:
    assert _embedding(rows=33, columns=960).shape == (33, 960)


def test_asarray_returns_the_stored_array_itself_when_nothing_needs_converting() -> None:
    embedding = _embedding()
    assert np.asarray(embedding) is embedding.array


def test_asarray_converts_when_a_dtype_is_asked_for() -> None:
    converted = np.asarray(_embedding(rows=2, columns=3), dtype=np.float64)
    assert converted.dtype == np.float64
    assert converted.shape == (2, 3)


def test_numpy_functions_take_an_embedding_directly() -> None:
    assert np.sum(np.asarray(_embedding(rows=4, columns=2))) == 0.0


def test_mean_is_the_per_residue_average_and_has_one_row_per_model_dimension() -> None:
    array = np.arange(6, dtype=np.float32).reshape(3, 2)
    pooled = Embedding(array, None, "300m", 30).mean()
    assert pooled.shape == (2,)
    np.testing.assert_array_equal(pooled, np.array([2.0, 3.0], dtype=np.float32))


def test_mean_needs_no_mask_because_every_row_is_a_residue() -> None:
    # #1 called a per-sequence vector "a masked mean you write". The mask exists for padded
    # batches; one un-padded sequence with BOS and EOS gone has nothing to mask, so the plain
    # mean is exact — asserted against an explicit average rather than against numpy itself.
    rng = np.random.default_rng(0)
    array = rng.standard_normal((7, 5)).astype(np.float32)
    expected = sum(array[row] for row in range(7)) / 7
    np.testing.assert_allclose(Embedding(array, None, "300m", 30).mean(), expected, rtol=1e-6)


def test_source_may_be_none_because_an_anonymous_sequence_still_embeds() -> None:
    assert _embedding(source=None).source is None


def test_an_embedding_cannot_be_edited_after_the_fact() -> None:
    embedding = _embedding()
    with pytest.raises(dataclasses.FrozenInstanceError):
        embedding.layer = 12  # type: ignore


def test_an_embedding_carries_no_instance_dict() -> None:
    # `slots=True`: the four fields are all of it, and nothing can be stapled on later.
    assert not hasattr(_embedding(), "__dict__")


def test_repr_names_the_four_facts_and_prints_no_numbers() -> None:
    assert repr(_embedding(rows=5, columns=4)) == "Embedding('P12345', 5 x 4, 300m layer 30)"


def test_as_json_is_the_provenance_and_leaves_the_array_out() -> None:
    written = _embedding().as_json()
    assert written == {"source": "P12345", "checkpoint": "300m", "layer": 30, "shape": [33, 960]}
    # It has to survive `json.dumps`, which numpy's own integer types do not.
    assert json.loads(json.dumps(written)) == written
