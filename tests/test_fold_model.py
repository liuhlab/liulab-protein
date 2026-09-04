"""The model lane: one real ESMFold2 fold, in the `esm` environment.

    pixi run -e esm pytest -m model

`-m "not model"` is in `addopts`, so plain `pytest` deselects every test here and the gate
stays green on a machine with no weights and no card. Selecting the lane by hand is the only
way in, and CI therefore sees none of it.

**Nothing here skips.** A lane that selects its tests, skips them all and exits 0 reports a
pass having folded nothing, so a missing `torch`, a missing `esm` or a cold HF cache fails
loudly instead.

**Quality against a floor, never coordinates.** Structure prediction is not reproducible
across processes and the spread grows with length, so a coordinate assertion is a coin flip.
What is asserted is lDDT-CA against a structure the repo already carries, and the confidence
scalars, with margins wide enough to clear the spread. That still catches `load_esmc=False`,
which sits far below any honest floor, where a shape-only check would not.

`HF_HUB_OFFLINE` makes "the model lane folds, it does not download" true rather than merely
intended: the suite's autouse network guard cannot be opted out of, and hub clients raise
straight through it rather than falling back to the cache. It is read once at the hub
client's import, so it is set here at collection and not in a fixture.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import lddt

from protein import Structure
from protein.fold import CHECKPOINTS, ESMFold2
from protein.fold.predictions import pairwise_path, prediction_path
from protein.fold.request import ChainRequest, FoldingRequest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

pytestmark = pytest.mark.model

#: The lane's one query: ubiquitin, whose deposited coordinates the repo already carries. No
#: reference structure is checked in for this — a binary that goes stale with the checkpoint
#: costs more than the test is worth.
FIXTURE = Path(__file__).resolve().parent / "data" / "1ubq.cif.gz"

#: What the fold is told it came from. SIFTS's accession for `1UBQ` chain A, so a prediction
#: that answered `()` would be indistinguishable from an entry nobody curated.
ACCESSION = "P0CG48"

#: Floors, not expectations. Each sits well below what was measured, because what is being
#: caught is a fold that went wrong and not a fold that came out a little differently.
LDDT_FLOOR = 0.85
PLDDT_FLOOR = 0.75
PTM_FLOOR = 0.70

#: What the B-factor column is scaled by, so a viewer's 0-100 reads as confidence.
B_FACTOR_SCALE = 100.0

_RECIPE = (
    "    export HF_HOME=/path/to/hf-cache\n"
    "    pixi run -e esm pytest -m model\n\n"
    "Refusing to skip. A skipped model lane is a green run that folded nothing, which is "
    "the failure this check exists to end."
)


def _require_the_esm_environment() -> None:
    """Fail, never skip, when this lane cannot do the thing it was selected to do."""
    missing = [name for name in ("torch", "esm") if importlib.util.find_spec(name) is None]
    if missing:
        pytest.fail(
            f"model lane: {', '.join(missing)} not importable. Both live in the `esm` "
            f"environment, which exists to hold them.\n\n{_RECIPE}"
        )
    from huggingface_hub import (  # pyright: ignore[reportMissingImports]
        try_to_load_from_cache,
    )

    # Offline mode is on, so a cold cache would otherwise fail deep inside `from_pretrained`
    # naming neither HF_HOME nor the fix.
    for repository in (CHECKPOINTS["ESMFold2-Fast"], "biohub/ESMC-6B"):
        if not isinstance(try_to_load_from_cache(repository, "config.json"), str):
            pytest.fail(
                f"model lane: {repository} is not in the Hugging Face cache this process "
                f"can see. Point HF_HOME at a cache holding it, or fetch it there once: "
                f"this lane itself never downloads.\n\n{_RECIPE}"
            )


def ubiquitin() -> str:
    """Return the residues `1UBQ` chain A was solved for."""
    return str(Structure.from_file(FIXTURE)["A"].sequence)


def alpha_carbons(structure: Structure, label: str = "A"):
    """Return one chain's alpha carbons, in residue order."""
    atoms = structure[label].atoms
    return atoms[atoms.atom_name == "CA"]


@pytest.fixture(scope="session")
def esmfold() -> ESMFold2:
    """One ESMFold2-Fast, loaded once for the whole lane rather than per test."""
    _require_the_esm_environment()
    return ESMFold2()


@pytest.fixture(scope="session")
def folds(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The directory this lane writes its one prediction into."""
    return tmp_path_factory.mktemp("folds")


@pytest.fixture(scope="session")
def request_for_ubiquitin() -> FoldingRequest:
    """The lane's one request: ubiquitin, carrying the accession it is folded from."""
    return FoldingRequest([ChainRequest("protein", ubiquitin(), accession=ACCESSION)])


@pytest.fixture(scope="session")
def prediction(esmfold: ESMFold2, folds: Path, request_for_ubiquitin: FoldingRequest) -> Structure:
    """One real fold, run once."""
    return esmfold.fold(request_for_ubiquitin, folds)


def test_the_lane_reports_which_device_it_actually_ran_on(
    esmfold: ESMFold2, capsys: pytest.CaptureFixture[str]
) -> None:
    # Reported, not asserted: the CPU is a supported fallback. But a run that took an hour
    # because it quietly landed there should say so in its own output.
    with capsys.disabled():
        print(f"\nESMFold2({esmfold.checkpoint!r}) is on {esmfold.device}")
    assert esmfold.device in {"cpu", "cuda"} or esmfold.device.startswith("cuda:")


def test_what_comes_back_is_a_structure_named_after_the_accession(
    prediction: Structure, folds: Path
) -> None:
    assert isinstance(prediction, Structure)
    assert prediction.id == ACCESSION
    assert prediction.path == prediction_path(folds, ACCESSION)
    assert prediction.path.is_file()


def test_the_prediction_answers_with_the_accession_it_was_folded_from(
    prediction: Structure,
) -> None:
    # Otherwise it reads exactly like a deposited entry SIFTS maps nothing to.
    assert prediction.accessions == {"A": (ACCESSION,)}
    assert prediction["A"].uniprot == (ACCESSION,)


def test_the_prediction_holds_the_residues_it_was_asked_for(prediction: Structure) -> None:
    assert str(prediction["A"].sequence) == ubiquitin()


def test_the_confidence_scalars_clear_their_floors(
    prediction: Structure, capsys: pytest.CaptureFixture[str]
) -> None:
    confidence = prediction.confidence
    assert confidence is not None
    with capsys.disabled():
        print(f"\nplddt {confidence.plddt:.3f}  ptm {confidence.ptm}  iptm {confidence.iptm}")
    assert confidence.plddt >= PLDDT_FLOOR
    assert confidence.ptm is not None
    assert confidence.ptm >= PTM_FLOOR


def test_the_fold_is_the_right_structure_and_not_merely_the_right_length(
    prediction: Structure, capsys: pytest.CaptureFixture[str]
) -> None:
    # THE assertion, and the one that catches `load_esmc=False`, which returns an mmCIF of
    # exactly the right length holding coordinates that are wrong.
    reference = alpha_carbons(Structure.from_file(FIXTURE))
    predicted = alpha_carbons(prediction)
    assert predicted.array_length() == reference.array_length()
    score = float(lddt(reference, predicted))
    with capsys.disabled():
        print(f"\nlDDT-CA against 1UBQ: {score:.3f}")
    assert score >= LDDT_FLOOR


def test_per_residue_confidence_rides_the_b_factor_column(prediction: Structure) -> None:
    confidence = prediction.confidence
    assert confidence is not None
    b_factor = np.asarray(prediction.atoms.b_factor, dtype=float)
    assert b_factor.min() > 0
    assert b_factor.max() <= B_FACTOR_SCALE
    # The scalar is the mean over residues and this is the mean over atoms, so they agree
    # only roughly — enough to say the column holds confidence and not something else.
    assert abs(b_factor.mean() / B_FACTOR_SCALE - confidence.plddt) < 0.05


def test_the_pairwise_matrix_is_a_sibling_file_read_when_it_is_asked_for(
    prediction: Structure, folds: Path
) -> None:
    confidence = prediction.confidence
    assert confidence is not None
    assert confidence.pairwise_file is not None
    assert confidence.pairwise_file == pairwise_path(folds, ACCESSION)
    assert confidence.pairwise_file.is_file()
    matrix = confidence.pairwise()
    residues = len(ubiquitin())
    assert matrix.shape == (residues, residues)


def test_folding_the_same_request_again_returns_what_is_there(
    esmfold: ESMFold2, folds: Path, request_for_ubiquitin: FoldingRequest, prediction: Structure
) -> None:
    again = esmfold.fold(request_for_ubiquitin, folds)
    assert again.path == prediction.path
    assert again.accessions == prediction.accessions
    # The scalars do not survive the file, so a prediction handed back carries none.
    assert again.confidence is None


def test_a_different_sequence_under_a_held_name_is_refused(
    esmfold: ESMFold2, folds: Path, prediction: Structure
) -> None:
    mutant = FoldingRequest([ChainRequest("protein", ubiquitin()[:20], accession=ACCESSION)])
    with pytest.raises(FileExistsError, match=r"different sequence"):
        esmfold.fold(mutant, folds)
