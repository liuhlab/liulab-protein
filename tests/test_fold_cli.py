"""Tests for `protein fold structure` — what it writes, what it prints, what it exits with.

Driven against a stand-in for `ESMFold2` rather than the weights: what is under test is the
command, and the model has its own lane behind the `model` marker. The one path that runs the
real class is the unknown-slug refusal, which answers before anything is loaded.

Invoked through the root app — `protein fold structure ...` — and never through this sub-app
alone: a Typer app holding exactly one command collapses into that command, so invoking this
one directly would read `structure` as the query.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from protein import Structure
from protein.cli import app as root_app
from protein.fold.cli import app
from protein.fold.esmfold import DEFAULT_CHECKPOINT
from protein.fold.predictions import (
    Confidence,
    prediction_name,
    prediction_path,
    stored_prediction,
)
from protein.io.structure import read_atoms, write_atoms

from . import plain_text

_DATA = Path(__file__).resolve().parent / "data"
_FIXTURE = _DATA / "1ubq.cif.gz"

#: The residues the stand-in's coordinates hold, so a request over them is a cache hit.
UBIQUITIN = str(Structure.from_file(_FIXTURE)["A"].sequence)


class _StandInESMFold2:
    """Answers `fold()` with the fixture's coordinates, so the command runs with no weights."""

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, *, device: Any = None) -> None:
        self.checkpoint = checkpoint
        self.device = device if device is not None else "cpu"

    def fold(
        self,
        request: Any,
        out: Any,
        *,
        name: str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Structure:
        directory = Path(out)
        directory.mkdir(parents=True, exist_ok=True)
        called = prediction_name(request, name)
        path = prediction_path(directory, called)
        held = stored_prediction(path, request, overwrite=overwrite)
        if held is not None:
            return held
        write_atoms(path, read_atoms(_FIXTURE))
        return Structure(
            called,
            path=path,
            accessions=request.accessions,
            confidence=Confidence(plddt=0.93, ptm=0.88),
        )


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("protein.fold.cli._ESMFold2", _StandInESMFold2)


@pytest.fixture
def fasta(tmp_path: Path) -> Path:
    """A FASTA holding exactly one record: ubiquitin, under its accession."""
    path = tmp_path / "query.fasta"
    path.write_text(f">P0CG48\n{UBIQUITIN}\n", encoding="utf-8")
    return path


def test_the_command_is_registered_under_the_name_it_was_given() -> None:
    assert [command.name for command in app.registered_commands] == ["structure"]


def test_the_lane_is_mounted_on_the_root_app_as_fold() -> None:
    # `in`, not `==`: the other lanes mount themselves on the same list.
    assert "fold" in [group.name for group in root_app.registered_groups]
    result = CliRunner().invoke(root_app, ["fold", "--help"])
    assert result.exit_code == 0
    assert "structure" in plain_text(result.output)


def test_every_command_takes_json() -> None:
    # `plain_text`, not `result.output`: rich styles the first dash of `--json` separately,
    # so the raw output carries no such substring wherever colour is on.
    for command in app.registered_commands:
        assert command.name is not None
        result = CliRunner().invoke(root_app, ["fold", command.name, "--help"])
        assert "--json" in plain_text(result.output), command.name


def test_the_help_names_the_two_options_the_store_needs() -> None:
    help_text = plain_text(CliRunner().invoke(root_app, ["fold", "structure", "--help"]).output)
    assert "--name" in help_text
    assert "--overwrite" in help_text


def test_a_fasta_is_folded_into_the_directory_it_was_given(
    stand_in: None, fasta: Path, tmp_path: Path
) -> None:
    out = tmp_path / "folds"
    result = CliRunner().invoke(root_app, ["fold", "structure", str(fasta), str(out), "--json"])
    assert result.exit_code == 0, result.output
    written = json.loads(result.output)
    assert written["id"] == "P0CG48"
    assert Path(written["path"]) == out / "P0CG48.cif"
    assert Path(written["path"]).is_file()


def test_a_bare_sequence_is_folded_too(stand_in: None, tmp_path: Path) -> None:
    out = tmp_path / "folds"
    result = CliRunner().invoke(root_app, ["fold", "structure", UBIQUITIN, str(out), "--json"])
    assert result.exit_code == 0, result.output
    # No accession to name it after, so it lands under the hash of its own sequence.
    assert len(json.loads(result.output)["id"]) == 16


def test_what_it_prints_says_where_the_answer_went_and_how_good_it_is(
    stand_in: None, fasta: Path, tmp_path: Path
) -> None:
    result = CliRunner().invoke(root_app, ["fold", "structure", str(fasta), str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "id: P0CG48" in result.output
    assert "plddt: 0.93" in result.output
    assert "chains: A" in result.output


def test_a_name_the_caller_gave_wins(stand_in: None, fasta: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        root_app,
        ["fold", "structure", str(fasta), str(tmp_path), "--name", "the mutant", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["id"] == "the mutant"


def test_a_held_name_over_a_different_sequence_exits_one(
    stand_in: None, fasta: Path, tmp_path: Path
) -> None:
    first = ["fold", "structure", str(fasta), str(tmp_path), "--name", "held"]
    assert CliRunner().invoke(root_app, first).exit_code == 0
    other = tmp_path / "other.fasta"
    other.write_text(f">P0CG48\n{UBIQUITIN[:20]}\n", encoding="utf-8")
    second = ["fold", "structure", str(other), str(tmp_path), "--name", "held"]
    result = CliRunner().invoke(root_app, second)
    assert result.exit_code == 1
    assert "already holds" in result.output


def test_overwrite_is_how_a_caller_says_they_meant_it(
    stand_in: None, fasta: Path, tmp_path: Path
) -> None:
    first = ["fold", "structure", str(fasta), str(tmp_path), "--name", "held"]
    assert CliRunner().invoke(root_app, first).exit_code == 0
    other = tmp_path / "other.fasta"
    other.write_text(f">P0CG48\n{UBIQUITIN[:20]}\n", encoding="utf-8")
    second = ["fold", "structure", str(other), str(tmp_path), "--name", "held", "--overwrite"]
    assert CliRunner().invoke(root_app, second).exit_code == 0


def test_the_output_directory_is_required(stand_in: None, fasta: Path) -> None:
    result = CliRunner().invoke(root_app, ["fold", "structure", str(fasta)])
    assert result.exit_code == 2
    assert "Missing argument 'out'" in plain_text(result.output)


def test_an_unknown_checkpoint_exits_one_and_names_the_slugs_that_exist(
    fasta: Path, tmp_path: Path
) -> None:
    # Not stood in for: the real class refuses the slug before it imports torch.
    result = CliRunner().invoke(
        root_app,
        ["fold", "structure", str(fasta), str(tmp_path), "--checkpoint", "esmfold2-fast"],
    )
    assert result.exit_code == 1
    assert "unknown checkpoint 'esmfold2-fast'" in result.output
    assert "ESMFold2-Fast, ESMFold2" in result.output
    assert "torch" not in sys.modules


def test_a_query_that_is_neither_a_record_nor_a_sequence_exits_one(
    stand_in: None, tmp_path: Path
) -> None:
    result = CliRunner().invoke(root_app, ["fold", "structure", "MKT-VY*", str(tmp_path)])
    assert result.exit_code == 1
    assert "not in the protein alphabet" in result.output


def test_nothing_on_the_command_line_path_imports_torch(fasta: Path, tmp_path: Path) -> None:
    # Reading the help and refusing a slug are the whole of what the CLI does before a fold,
    # and neither may cost a torch import.
    CliRunner().invoke(root_app, ["fold", "structure", "--help"])
    CliRunner().invoke(
        root_app, ["fold", "structure", str(fasta), str(tmp_path), "--checkpoint", "nonesuch"]
    )
    assert "torch" not in sys.modules
