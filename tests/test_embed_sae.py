"""Tests for `SaeActivation` — the value object, which is pure numpy and needs no weights.

Every test here runs in the gate. That is the point of a peer type with nothing behind it:
what a caller actually holds is testable with no checkpoint anywhere, so the two arrays, the
dtypes, the reductions and the refusals are all pinned before a GPU is involved.
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from protein.embed import Embedding, SaeActivation

_SOURCE = Path(__file__).resolve().parents[1] / "src" / "protein" / "embed" / "esm" / "sae.py"

#: A full-variant slug, spelled once so the tests read as what they assert.
_SLUG = "300m-layer23-k64-cb16384"

#: What the value object may never reach for, in its module body or inside itself.
_BANNED = frozenset({"torch", "scipy"})


def _activation(
    rows: int = 3,
    *,
    codebook_size: int = 4,
    k: int = 2,
    source: str | None = "P12345",
    parent: str = "300m",
    layer: int = 23,
    sae: str = _SLUG,
    normalized: bool = False,
) -> SaeActivation:
    return SaeActivation(
        np.zeros((rows, k), dtype=SaeActivation.index_dtype(codebook_size)),
        np.zeros((rows, k), dtype=np.float16),
        np.zeros(rows, dtype=np.float32),
        source,
        parent,
        layer,
        sae,
        codebook_size,
        k,
        normalized,
    )


def _small() -> SaeActivation:
    """Two residues over a four-feature codebook, with every value exact in float16."""
    return SaeActivation(
        np.array([[2, 0], [1, 3]], dtype=np.uint8),
        np.array([[0.5, 0.25], [1.0, 0.75]], dtype=np.float16),
        np.array([0.125, 0.25], dtype=np.float32),
        "P12345",
        "300m",
        23,
        _SLUG,
        4,
        2,
    )


def _banned_imports(nodes: Iterable[ast.AST]) -> list[str]:
    """Return every banned top-level package name imported by ``nodes``."""
    imported: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    return [name for name in imported if name.split(".")[0] in _BANNED]


# --- a peer of Embedding, not one ----------------------------------------------


def test_an_activation_is_not_an_embedding() -> None:
    # `d_model` would be a lie on it and `.mean()` would mislead, so it inherits neither.
    assert not issubclass(SaeActivation, Embedding)
    assert not isinstance(_small(), Embedding)


def test_the_embedding_it_is_a_peer_of_was_left_exactly_as_it_was() -> None:
    # The lane grows by a peer type rather than by widening `Embedding` with a fourth fact.
    assert [field.name for field in dataclasses.fields(Embedding)] == [
        "array",
        "source",
        "checkpoint",
        "layer",
    ]
    assert hasattr(Embedding, "mean")
    assert hasattr(Embedding, "__array__")


def test_an_activation_cannot_be_edited_after_the_fact() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _small().layer = 12  # type: ignore


def test_an_activation_carries_no_instance_dict() -> None:
    # `slots=True`: nothing can be stapled on later.
    assert not hasattr(_small(), "__dict__")


# --- the two arrays -------------------------------------------------------------


def test_the_two_arrays_are_one_slot_table_each_and_the_same_shape() -> None:
    activation = _activation(rows=33, codebook_size=16384, k=64)
    assert activation.indices.shape == (33, 64)
    assert activation.values.shape == (33, 64)
    assert len(activation) == 33


def test_the_dense_shape_is_the_codebook_and_not_what_is_stored() -> None:
    activation = _activation(rows=33, codebook_size=16384, k=64)
    assert activation.shape == (33, 16384)
    assert activation.dense().shape == (33, 16384)


def test_exactly_k_slots_are_kept_per_residue_even_where_a_value_is_zero() -> None:
    # This is why the pair is lossless rather than an approximation: top-k returns k slots
    # per row whether or not all k are non-zero, so no row is ever short.
    activation = SaeActivation(
        np.array([[2, 0], [1, 3]], dtype=np.uint8),
        np.array([[0.5, 0.0], [0.0, 0.0]], dtype=np.float16),
        np.zeros(2, dtype=np.float32),
        None,
        "300m",
        23,
        _SLUG,
        4,
        2,
    )
    assert activation.indices.shape[1] == activation.k == 2
    assert activation.dense()[0].tolist() == [0.0, 0.0, 0.5, 0.0]


def test_the_pair_round_trips_through_the_dense_form_without_losing_a_value() -> None:
    activation = _small()
    dense = activation.dense(np.float16)
    for row in range(len(activation)):
        for slot in range(activation.k):
            assert dense[row, activation.indices[row, slot]] == activation.values[row, slot]


def test_the_reconstruction_loss_is_one_number_per_residue() -> None:
    assert _activation(rows=33, codebook_size=16384, k=64).reconstruction_loss.shape == (33,)


# --- the dtypes -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("codebook_size", "expected"),
    [(1, "uint8"), (256, "uint8"), (257, "uint16"), (16384, "uint16"), (65537, "uint32")],
)
def test_a_feature_index_is_stored_in_the_smallest_unsigned_type_that_holds_it(
    codebook_size: int, expected: str
) -> None:
    dtype = SaeActivation.index_dtype(codebook_size)
    assert dtype == np.dtype(expected)
    assert np.iinfo(dtype).max >= codebook_size - 1


def test_a_codebook_with_no_features_names_no_index_and_raises() -> None:
    with pytest.raises(ValueError, match="names no feature"):
        SaeActivation.index_dtype(0)


def test_the_three_arrays_come_back_in_the_dtypes_the_class_promises() -> None:
    activation = _activation(codebook_size=16384)
    assert activation.indices.dtype == np.dtype(np.uint16)
    assert activation.values.dtype == np.dtype(np.float16)
    assert activation.reconstruction_loss.dtype == np.dtype(np.float32)


@pytest.mark.parametrize(
    ("indices", "values", "loss"),
    [
        (np.int32, np.float16, np.float32),
        (np.uint32, np.float16, np.float32),
        (np.uint8, np.float32, np.float32),
        (np.uint8, np.float16, np.float64),
    ],
)
def test_an_array_in_the_wrong_dtype_is_refused_at_construction(
    indices: npt.DTypeLike, values: npt.DTypeLike, loss: npt.DTypeLike
) -> None:
    with pytest.raises(ValueError, match="this class stores"):
        SaeActivation(
            np.zeros((2, 2), dtype=indices),
            np.zeros((2, 2), dtype=values),
            np.zeros(2, dtype=loss),
            None,
            "300m",
            23,
            _SLUG,
            4,
            2,
        )


def test_two_arrays_that_disagree_with_each_other_are_refused() -> None:
    with pytest.raises(ValueError, match="must agree"):
        SaeActivation(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 3), dtype=np.float16),
            np.zeros(2, dtype=np.float32),
            None,
            "300m",
            23,
            _SLUG,
            4,
            3,
        )


def test_arrays_that_disagree_with_k_are_refused() -> None:
    with pytest.raises(ValueError, match="slots per residue"):
        SaeActivation(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float16),
            np.zeros(2, dtype=np.float32),
            None,
            "300m",
            23,
            _SLUG,
            4,
            64,
        )


def test_a_reconstruction_loss_that_is_not_one_number_per_residue_is_refused() -> None:
    with pytest.raises(ValueError, match="one number per"):
        SaeActivation(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float16),
            np.zeros(3, dtype=np.float32),
            None,
            "300m",
            23,
            _SLUG,
            4,
            2,
        )


# --- the reductions --------------------------------------------------------------


def test_max_is_the_per_sequence_vector_over_the_whole_codebook() -> None:
    pooled = _small().max()
    assert pooled.shape == (4,)
    assert pooled.dtype == np.dtype(np.float32)
    assert pooled.tolist() == [0.25, 1.0, 0.5, 0.75]


def test_a_feature_that_never_fired_pools_to_zero_because_the_encoder_is_a_relu() -> None:
    activation = SaeActivation(
        np.array([[0, 1]], dtype=np.uint8),
        np.array([[2.0, 3.0]], dtype=np.float16),
        np.zeros(1, dtype=np.float32),
        None,
        "300m",
        23,
        _SLUG,
        4,
        2,
    )
    assert activation.max().tolist() == [2.0, 3.0, 0.0, 0.0]


def test_max_agrees_with_the_dense_form_it_never_materialises() -> None:
    activation = _small()
    np.testing.assert_array_equal(activation.max(), activation.dense().max(axis=0))


def test_a_long_protein_does_not_dilute_a_feature_that_fired_at_one_residue() -> None:
    # The whole reason there is no `.mean()`: a mean over these hundred residues would
    # report 0.04 for a feature that fired at 4.0.
    rows = 100
    indices = np.zeros((rows, 2), dtype=np.uint8)
    indices[:, 1] = 1
    values = np.zeros((rows, 2), dtype=np.float16)
    values[0, 0] = 4.0
    activation = SaeActivation(
        indices, values, np.zeros(rows, dtype=np.float32), None, "300m", 23, _SLUG, 4, 2
    )
    assert activation.max()[0] == 4.0


def test_there_is_no_mean_at_all() -> None:
    # Deliberately absent, not overlooked: the misleading reduction should not be one
    # keystroke away from the honest one.
    assert not hasattr(SaeActivation, "mean")
    assert not hasattr(_small(), "mean")


def test_there_is_no_array_protocol_because_there_is_no_one_obvious_array() -> None:
    # Two `(L, k)` tables and a `(L,)` loss: `np.asarray(activation)` would have to pick.
    assert not hasattr(SaeActivation, "__array__")


# --- dense ------------------------------------------------------------------------


def test_dense_materialises_float32_unless_asked_otherwise() -> None:
    assert _small().dense().dtype == np.dtype(np.float32)
    assert _small().dense(np.float16).dtype == np.dtype(np.float16)


def test_dense_is_zero_everywhere_no_slot_named() -> None:
    assert _small().dense().tolist() == [[0.25, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.75]]


def test_the_dense_form_is_never_held() -> None:
    # Memory is spent when a caller asks and not before, so two calls are two arrays and
    # nothing on the object is codebook-wide.
    activation = _small()
    assert activation.dense() is not activation.dense()
    assert [field.name for field in dataclasses.fields(activation)] == [
        "indices",
        "values",
        "reconstruction_loss",
        "source",
        "parent",
        "layer",
        "sae",
        "codebook_size",
        "k",
        "normalized",
    ]


# --- what it carries ----------------------------------------------------------------


def test_it_carries_the_facts_that_make_two_sets_comparable_or_provably_not() -> None:
    activation = _activation(parent="6b", layer=60, sae="6b-layer60-k64-cb16384", normalized=True)
    assert activation.source == "P12345"
    assert activation.parent == "6b"
    assert activation.layer == 60
    assert activation.sae == "6b-layer60-k64-cb16384"
    assert activation.codebook_size == 4
    assert activation.k == 2
    assert activation.normalized is True


def test_normalisation_is_off_unless_it_was_asked_for() -> None:
    assert _activation().normalized is False


def test_source_may_be_none_because_an_anonymous_sequence_still_encodes() -> None:
    assert _activation(source=None).source is None


def test_repr_names_the_checkpoint_and_the_codebook_and_prints_no_numbers() -> None:
    assert repr(_small()) == (
        "SaeActivation('P12345', 2 x 4 top 2, 300m-layer23-k64-cb16384 on 300m layer 23)"
    )


# --- numpy only ----------------------------------------------------------------------


def test_the_module_body_imports_neither_torch_nor_scipy() -> None:
    # Against the syntax tree and not `sys.modules`, so this holds in an environment where
    # torch is absent and the import would have failed anyway.
    assert not _banned_imports(ast.parse(_SOURCE.read_text(encoding="utf-8")).body)


def test_nothing_inside_the_value_object_reaches_for_torch_or_scipy() -> None:
    # Scoped to the class and not the file: what loads weights may share this module, and
    # the value object still has to be numpy and nothing else.
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    found = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SaeActivation"
    ]
    assert len(found) == 1
    assert not _banned_imports(ast.walk(found[0]))
