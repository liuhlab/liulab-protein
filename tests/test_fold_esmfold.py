"""Tests for `protein.fold.esmfold` that need no weights — the table, the guard, the imports.

Everything here runs in the gate. `ESMFold2` is eager, so nothing constructs one; what is
tested is the part of the lane that answers before the weights are touched. The one real
fold lives in `tests/test_fold_model.py` behind the `model` marker.
"""

from __future__ import annotations

import ast
import inspect
import sys
import warnings
from pathlib import Path

import pytest

from protein import Structure
from protein.fold import CHECKPOINTS, ESMFold2
from protein.fold.esmfold import DEFAULT_CHECKPOINT, DEFAULT_KERNEL_BACKEND, warn_about_esmc
from protein.fold.predictions import prediction_path
from protein.io.structure import read_atoms, write_atoms

_SOURCE = Path(__file__).resolve().parents[1] / "src" / "protein" / "fold" / "esmfold.py"
_FIXTURE = Path(__file__).resolve().parent / "data" / "1ubq.cif.gz"

#: The residues the fixture holds, so a request over them answers from the file.
UBIQUITIN = str(Structure.from_file(_FIXTURE)["A"].sequence)


def test_the_table_names_every_checkpoint_this_package_claims_to_know() -> None:
    assert CHECKPOINTS == {
        "ESMFold2-Fast": "biohub/ESMFold2-Fast",
        "ESMFold2": "biohub/ESMFold2",
    }


def test_both_checkpoints_are_reachable_by_slug_and_the_fast_one_is_the_default() -> None:
    assert DEFAULT_CHECKPOINT == "ESMFold2-Fast"
    assert DEFAULT_CHECKPOINT in CHECKPOINTS
    assert len(CHECKPOINTS) == 2


def test_an_unknown_slug_fails_by_name_and_lists_what_is_known() -> None:
    with pytest.raises(ValueError, match=r"unknown checkpoint 'esmfold-2'"):
        ESMFold2("esmfold-2")


def test_an_unknown_slug_fails_before_anything_is_downloaded() -> None:
    # The check is the first statement in `__init__`, ahead of `import torch`, which is what
    # makes a slug better than a repository name.
    with pytest.raises(ValueError, match=r"unknown checkpoint 'biohub/ESMFold2-Fast'"):
        ESMFold2("biohub/ESMFold2-Fast")
    assert "torch" not in sys.modules


def test_loading_the_language_model_warns_about_nothing() -> None:
    # The default path is quiet. Only the mode that returns a confidently wrong structure
    # has a guard.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_about_esmc(True)


def test_turning_the_language_model_off_warns_that_the_answer_will_be_wrong() -> None:
    with pytest.warns(UserWarning, match=r"not a memory option"):
        warn_about_esmc(False)


def test_the_guard_says_what_the_failure_looks_like_rather_than_naming_a_flag() -> None:
    with pytest.warns(UserWarning, match=r"load_esmc") as caught:
        warn_about_esmc(False)
    said = str(caught[0].message)
    assert "right length" in said
    assert "wrong structure" in said


def test_the_output_directory_is_required_and_defaults_nowhere() -> None:
    parameters = inspect.signature(ESMFold2.fold).parameters
    assert parameters["out"].default is inspect.Parameter.empty
    assert parameters["name"].default is None
    assert parameters["overwrite"].default is False


def test_fold_builds_the_request_itself_from_plain_python(tmp_path: Path) -> None:
    # Down the cache-hit path, which answers from the file and reaches neither the weights
    # nor `esm`. `object.__new__` because construction is eager and this test wants none of
    # it; what is under test is that `fold` coerces before it looks anything up.
    write_atoms(prediction_path(tmp_path, "P0CG48"), read_atoms(_FIXTURE))
    model = object.__new__(ESMFold2)
    held = model.fold([{"kind": "protein", "sequence": UBIQUITIN, "accession": "P0CG48"}], tmp_path)
    assert held.id == "P0CG48"
    assert held.accessions == {"A": ("P0CG48",)}


def test_the_upstream_schema_is_reachable_rather_than_curated() -> None:
    # Load-time arguments are named, everything else on `fold` is forwarded, and the loaded
    # model is an attribute — so no upstream knob is out of reach.
    fold = inspect.signature(ESMFold2.fold).parameters
    assert fold["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert inspect.signature(ESMFold2.__init__).parameters["kwargs"].kind is (
        inspect.Parameter.VAR_KEYWORD
    )
    assert "model : Any" in (ESMFold2.__doc__ or "")


def test_the_fused_kernels_are_what_a_fold_runs_unless_a_caller_says_otherwise() -> None:
    assert DEFAULT_KERNEL_BACKEND == "fused"
    assert inspect.signature(ESMFold2.__init__).parameters["kernel_backend"].default == "fused"


def test_importing_the_folding_lane_does_not_import_torch() -> None:
    import protein
    import protein.fold

    assert protein.fold.ESMFold2 is ESMFold2
    assert protein.ESMFold2 is ESMFold2
    assert "torch" not in sys.modules


def test_the_module_body_imports_neither_torch_nor_esm() -> None:
    # Against the syntax tree and not `sys.modules`, so this holds in an environment where
    # torch is absent and the import would have failed anyway.
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert not [name for name in imported if name.split(".")[0] in {"torch", "esm"}]
