"""Tests for `protein.embed.esm` that need no weights — the table, the range, the imports.

Everything here runs in the gate. `ESMC` is eager, so nothing constructs one; what is tested
is the part of the lane that answers before the weights are touched. The one real embedding
lives in `tests/test_embed_model.py` behind the `model` marker.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
from biotite.sequence import ProteinSequence

from protein import Protein
from protein.embed import CHECKPOINTS, ESMC, Embeddable
from protein.embed.esm.esmc import _layer_index

_SOURCE = Path(__file__).resolve().parents[1] / "src" / "protein" / "embed" / "esm" / "esmc.py"


def test_the_table_names_every_checkpoint_this_package_claims_to_know() -> None:
    assert CHECKPOINTS == {
        "300m": ("biohub/ESMC-300M", 960),
        "600m": ("biohub/ESMC-600M", 1152),
        "6b": ("biohub/ESMC-6B", 2560),
    }


def test_the_width_of_a_checkpoint_is_known_without_downloading_it() -> None:
    # The reason the table exists: construction is eager, so `ESMC("6b").d_model` would
    # otherwise cost the whole download to answer.
    assert CHECKPOINTS["6b"][1] == 2560


def test_an_unknown_slug_fails_by_name_and_lists_what_is_known() -> None:
    with pytest.raises(ValueError, match=r"unknown checkpoint 'esmc_300m'"):
        ESMC("esmc_300m")


def test_an_unknown_slug_fails_before_anything_heavy_is_imported() -> None:
    # The check is the first statement in `__init__`, ahead of `import torch`, which is what
    # makes a slug better than an HF id.
    with pytest.raises(ValueError, match=r"unknown checkpoint 'biohub/ESMC-300M'"):
        ESMC("biohub/ESMC-300M")
    assert "torch" not in sys.modules


def test_importing_the_embedding_lane_does_not_import_torch() -> None:
    import protein.embed
    import protein.embed.cli

    assert protein.embed.ESMC is ESMC
    assert protein.embed.cli.app is not None
    assert "torch" not in sys.modules


def test_the_module_body_of_esmc_py_imports_neither_torch_nor_esm() -> None:
    # Against the syntax tree and not `sys.modules`, so this holds even in an environment
    # where torch is absent and the import would have failed anyway.
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert not [name for name in imported if name.split(".")[0] in {"torch", "esm"}]


@pytest.mark.parametrize(
    ("layer", "index"),
    [(-1, 30), (0, 0), (30, 30), (-31, 0), (-2, 29), (12, 12)],
)
def test_a_layer_is_normalised_onto_a_non_negative_hidden_state_index(
    layer: int, index: int
) -> None:
    # `hidden_states` is `n_layers + 1` long: 0 is the embedding-layer output and the last
    # index is the last hidden state.
    assert _layer_index(layer, 30, "300m") == index


@pytest.mark.parametrize("layer", [31, -32, 100])
def test_a_layer_outside_the_range_names_the_range_it_fell_outside(layer: int) -> None:
    with pytest.raises(ValueError, match=r"30 transformer layers"):
        _layer_index(layer, 30, "300m")


def test_a_protein_is_something_esmc_can_embed() -> None:
    assert isinstance(Protein("MKTAY", id="P12345"), Embeddable)


def test_a_protein_with_no_accession_is_still_something_esmc_can_embed() -> None:
    assert isinstance(Protein("MKTAY"), Embeddable)


@pytest.mark.parametrize("item", ["MKTAY", ProteinSequence("MKTAY"), np.zeros(3), 5])
def test_a_thing_carrying_no_identity_is_not_something_esmc_can_embed(item: object) -> None:
    # The reason is `Embedding.source`: an embedding made from one of these could not say
    # afterwards what it embedded.
    assert not isinstance(item, Embeddable)
