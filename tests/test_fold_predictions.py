"""Tests for `protein.fold.predictions` — derived names and the refusal rule.

Driven by files staged in a temporary directory. No model, no GPU, and nothing here reaches
for the lab data root: an output directory is an argument, always.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from protein import Protein, Structure
from protein.fold import ChainRequest, FoldingRequest
from protein.fold.predictions import (
    Confidence,
    pairwise_path,
    prediction_name,
    prediction_path,
    stored_prediction,
)
from protein.io.structure import read_atoms, write_atoms

_SOURCE = Path(__file__).resolve().parents[1] / "src" / "protein" / "fold" / "predictions.py"

#: A structure the repo already carries, standing in for a written prediction. Its one chain
#: is labelled `A`, which is the label a one-chain request derives.
FIXTURE = Path(__file__).resolve().parent / "data" / "1ubq.cif.gz"

#: Ubiquitin's observed residues, read back off the fixture rather than written out here.
UBIQUITIN = str(Structure.from_file(FIXTURE)["A"].sequence)


def stage(directory: Path, name: str) -> Path:
    """Write the fixture into ``directory`` as the prediction called ``name``."""
    path = prediction_path(directory, name)
    write_atoms(path, read_atoms(FIXTURE))
    return path


def request_for(sequence: str, accession: str | None = None) -> FoldingRequest:
    """Return the one-chain request folding ``sequence``."""
    return FoldingRequest([ChainRequest("protein", sequence, accession=accession)])


def _imported(*, everywhere: bool = False) -> list[str]:
    """Return the modules the source imports — at the top of the file, or anywhere in it."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    nodes = ast.walk(tree) if everywhere else tree.body
    names: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return names


def test_a_user_given_name_wins() -> None:
    assert prediction_name(request_for(UBIQUITIN, "P0CG48"), "the mutant") == "the mutant"


def test_the_accession_names_the_prediction_when_the_caller_does_not() -> None:
    assert prediction_name(request_for(UBIQUITIN, "P0CG48")) == "P0CG48"


def test_one_accession_across_several_chains_still_names_it() -> None:
    homodimer = FoldingRequest(
        [ChainRequest("protein", "MKTAY", accession="P12345") for _ in range(2)]
    )
    assert prediction_name(homodimer) == "P12345"


def test_a_sequence_with_no_accession_gets_a_short_stable_hash() -> None:
    first = prediction_name(request_for("MKTAY"))
    assert first == prediction_name(request_for("MKTAY"))
    assert first != prediction_name(request_for("MKTAW"))
    assert len(first) == 16


def test_two_accessions_fall_back_to_the_hash_rather_than_picking_one() -> None:
    heterodimer = FoldingRequest(
        [
            ChainRequest("protein", "MKTAY", accession="P12345"),
            ChainRequest("protein", "MKTAW", accession="Q99999"),
        ]
    )
    assert prediction_name(heterodimer) not in {"P12345", "Q99999"}


def test_the_name_is_a_fact_about_the_molecule_and_takes_nothing_else() -> None:
    # No checkpoint, no seed, no sampler knob: neither says what was folded, so neither may
    # decide where it lands.
    assert list(inspect.signature(prediction_name).parameters) == ["request", "name"]


def test_the_output_directory_is_a_required_argument_that_defaults_nowhere() -> None:
    directory = inspect.signature(prediction_path).parameters["directory"]
    assert directory.default is inspect.Parameter.empty


def test_nothing_here_reaches_for_the_lab_data_root() -> None:
    # `LIULAB_DATA` holds reference and input data and never a user's outputs, so the module
    # that decides where a prediction lands never asks where the root is — not at the top of
    # the file and not inside a function body either.
    assert "protein.store" not in _imported(everywhere=True)


def test_a_prediction_lands_under_the_directory_it_was_given(tmp_path: Path) -> None:
    assert prediction_path(tmp_path, "P0CG48") == tmp_path / "P0CG48.cif"


def test_nothing_is_held_where_nothing_was_written(tmp_path: Path) -> None:
    path = prediction_path(tmp_path, "P0CG48")
    assert stored_prediction(path, request_for(UBIQUITIN, "P0CG48")) is None


def test_the_same_name_and_the_same_sequence_gives_back_what_is_there(tmp_path: Path) -> None:
    path = stage(tmp_path, "P0CG48")
    held = stored_prediction(path, request_for(UBIQUITIN, "P0CG48"))
    assert isinstance(held, Structure)
    assert held.path == path


def test_the_same_name_and_a_different_sequence_raises(tmp_path: Path) -> None:
    # The rule that earns its keep: `Protein("MQIFVKTLTG", id="P0CG48")` is legal, so a
    # mutant can arrive carrying a reference accession and land on the reference's file.
    path = stage(tmp_path, "P0CG48")
    mutant = Protein(UBIQUITIN.replace("MQIFV", "MQIFA"), id="P0CG48")
    with pytest.raises(FileExistsError, match=r"different sequence in chain 'A'"):
        stored_prediction(path, FoldingRequest([ChainRequest.of(mutant)]))


def test_a_sequence_of_a_different_length_under_a_held_name_raises(tmp_path: Path) -> None:
    path = stage(tmp_path, "P0CG48")
    with pytest.raises(FileExistsError, match=r"76 residues on disk against 10"):
        stored_prediction(path, request_for(UBIQUITIN[:10], "P0CG48"))


def test_a_different_chain_count_under_a_held_name_raises(tmp_path: Path) -> None:
    path = stage(tmp_path, "P0CG48")
    dimer = FoldingRequest([ChainRequest("protein", UBIQUITIN, accession="P0CG48") for _ in (0, 1)])
    with pytest.raises(FileExistsError, match=r"chains \['A'\] where this request folds"):
        stored_prediction(path, dimer)


def test_overwrite_is_how_a_caller_says_they_meant_it(tmp_path: Path) -> None:
    path = stage(tmp_path, "P0CG48")
    mutant = request_for(UBIQUITIN[:10], "P0CG48")
    assert stored_prediction(path, mutant, overwrite=True) is None


def test_the_stored_sequence_is_recovered_from_the_residues(tmp_path: Path) -> None:
    # The written file names no accession anywhere and carries no map — provenance does not
    # survive it (ADR-0005) — and the rule still holds a day later, because what is weighed
    # is the residues.
    path = stage(tmp_path, "run one")
    assert "P0CG48" not in path.read_text(encoding="utf-8")
    assert Structure.from_file(path).accessions is None
    assert stored_prediction(path, request_for(UBIQUITIN, "P0CG48")) is not None


def test_re_folding_with_a_different_seed_hits_the_cache(tmp_path: Path) -> None:
    # The accepted edge, pinned: settings are not in the path, so nothing about how a fold
    # is run can move it. Two folds of one sequence share a file whatever else differed.
    path = stage(tmp_path, "P0CG48")
    request = request_for(UBIQUITIN, "P0CG48")
    assert prediction_path(tmp_path, prediction_name(request)) == path
    assert stored_prediction(path, request) is not None
    # The two escapes.
    assert stored_prediction(path, request, overwrite=True) is None
    assert stored_prediction(prediction_path(tmp_path, "again"), request) is None


def test_a_cache_hit_carries_the_accessions_the_request_names(tmp_path: Path) -> None:
    # The accessions are the input, not something read back out of the file, so a prediction
    # handed back unfolded still answers with them rather than reaching for SIFTS.
    path = stage(tmp_path, "P0CG48")
    held = stored_prediction(path, request_for(UBIQUITIN, "P0CG48"))
    assert held is not None
    assert held.accessions == {"A": ("P0CG48",)}
    assert held["A"].uniprot == ("P0CG48",)


def test_a_cache_hit_carries_no_confidence_because_the_scalars_do_not_survive(
    tmp_path: Path,
) -> None:
    path = stage(tmp_path, "P0CG48")
    held = stored_prediction(path, request_for(UBIQUITIN, "P0CG48"))
    assert held is not None
    assert held.confidence is None


def test_the_pairwise_matrix_is_a_sibling_of_the_coordinates(tmp_path: Path) -> None:
    assert pairwise_path(tmp_path, "P0CG48").parent == prediction_path(tmp_path, "P0CG48").parent
    assert pairwise_path(tmp_path, "P0CG48").name == "P0CG48.pairwise.npy"


def test_confidence_is_frozen_so_a_measurement_cannot_be_edited_in_place() -> None:
    confidence = Confidence(plddt=0.93, ptm=0.88)
    with pytest.raises(AttributeError):
        confidence.plddt = 0.1  # type: ignore[misc]


def test_confidence_reads_the_pairwise_matrix_when_it_is_asked_and_not_before(
    tmp_path: Path,
) -> None:
    matrix = np.arange(9, dtype=np.float32).reshape(3, 3)
    written = pairwise_path(tmp_path, "P0CG48")
    np.save(written, matrix)
    confidence = Confidence(plddt=0.93, pairwise_file=written)
    written_again = confidence.pairwise()
    np.testing.assert_array_equal(written_again, matrix)


def test_confidence_says_so_when_no_pairwise_matrix_was_written() -> None:
    with pytest.raises(FileNotFoundError, match=r"reported no pairwise matrix"):
        Confidence(plddt=0.93).pairwise()


def test_a_structure_carries_a_confidence_and_one_read_off_disk_does_not(tmp_path: Path) -> None:
    path = stage(tmp_path, "P0CG48")
    confidence = Confidence(plddt=0.93, ptm=0.88)
    assert Structure("folded", path=path, confidence=confidence).confidence is confidence
    assert Structure.from_file(path).confidence is None


def test_the_module_body_imports_neither_torch_nor_esm() -> None:
    assert not [name for name in _imported() if name.split(".")[0] in {"torch", "esm"}]
