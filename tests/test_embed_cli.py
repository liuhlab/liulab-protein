"""Tests for `protein esm embed` — what it prints, what it writes, what it exits with.

Driven against a stand-in for `ESMC` rather than the weights: what is under test is the
command, and the model has its own lane behind the `model` marker. The one path that runs the
real class is the unknown-slug refusal, which answers before anything is loaded.

Invoked through the root app -- `protein esm embed ...` -- and never through this sub-app
alone. A Typer app holding exactly one command collapses into that command, so invoking this
one directly would read `embed` as the FASTA path. Mounting is what makes it a group, and
mounted is how anyone runs it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from protein.cli import app as root_app
from protein.embed import Embedding
from protein.embed.cli import app

from . import plain_text

_DATA = Path(__file__).resolve().parent / "data"
_FASTA = _DATA / "uniprot_p01308.fasta"
_THREE = _DATA / "uniprot_three.fasta"


class _StandInESMC:
    """Answers `embed()` with a small array, so the command can be tested with no weights."""

    def __init__(self, checkpoint: str = "300m", *, device: Any = None, token: Any = None) -> None:
        self.checkpoint = checkpoint
        self.device = device if device is not None else "cpu"

    def embed(self, item: Any, *, layer: int = -1) -> Embedding:
        array = np.arange(6, dtype=np.float32).reshape(3, 2)
        return Embedding(array, item.id, self.checkpoint, 30 if layer == -1 else layer)


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("protein.embed.cli._ESMC", _StandInESMC)


def test_the_command_is_registered_under_the_name_it_was_given() -> None:
    assert [command.name for command in app.registered_commands] == ["embed"]


def test_every_command_takes_json() -> None:
    # `plain_text`, not `result.output`: rich styles the first dash of `--json` separately,
    # so the raw output carries no such substring wherever colour is on. See tests/__init__.
    for command in app.registered_commands:
        assert command.name is not None
        result = CliRunner().invoke(root_app, ["esm", command.name, "--help"])
        assert "--json" in plain_text(result.output), command.name


def test_the_lane_is_mounted_on_the_root_app_as_esm() -> None:
    # `in`, not `==`: three more lanes mount themselves on the same list.
    assert "esm" in [group.name for group in root_app.registered_groups]
    result = CliRunner().invoke(root_app, ["esm", "--help"])
    assert result.exit_code == 0
    assert "embed" in result.output


def test_it_prints_the_provenance_of_what_it_embedded(stand_in: None) -> None:
    result = CliRunner().invoke(root_app, ["esm", "embed", str(_FASTA)])
    assert result.exit_code == 0
    assert "source: sp|P01308|INS_HUMAN" in result.output
    assert "checkpoint: 300m" in result.output
    assert "layer: 30" in result.output
    assert "shape: [3, 2]" in result.output


def test_it_answers_json_when_asked(stand_in: None) -> None:
    result = CliRunner().invoke(root_app, ["esm", "embed", str(_FASTA), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "source": "sp|P01308|INS_HUMAN",
        "checkpoint": "300m",
        "layer": 30,
        "shape": [3, 2],
        "out": None,
    }


def test_out_writes_the_array_and_json_says_where(stand_in: None, tmp_path: Path) -> None:
    destination = tmp_path / "insulin.npy"
    result = CliRunner().invoke(
        root_app, ["esm", "embed", str(_FASTA), "--out", str(destination), "--json"]
    )
    assert result.exit_code == 0
    assert json.loads(result.output)["out"] == str(destination)
    written = np.load(destination)
    assert written.shape == (3, 2)
    assert written.dtype == np.float32


def test_the_layer_it_was_given_reaches_the_model_and_the_provenance(stand_in: None) -> None:
    result = CliRunner().invoke(root_app, ["esm", "embed", str(_FASTA), "--layer", "12", "--json"])
    assert json.loads(result.output)["layer"] == 12


def test_an_unknown_checkpoint_exits_one_and_names_the_slugs_that_exist() -> None:
    # Not stood in for: the real class refuses the slug before it imports torch, which is
    # what makes a checkpoint table better than an arbitrary HF id.
    result = CliRunner().invoke(
        root_app, ["esm", "embed", str(_FASTA), "--checkpoint", "esmc_300m"]
    )
    assert result.exit_code == 1
    assert "unknown checkpoint 'esmc_300m'" in result.output
    assert "300m, 600m, 6b" in result.output


def test_a_fasta_holding_more_than_one_record_exits_one(stand_in: None) -> None:
    # One at a time, and the command needs no flag to say so: `Protein.from_fasta` refuses.
    result = CliRunner().invoke(root_app, ["esm", "embed", str(_THREE)])
    assert result.exit_code == 1
    assert "more than one FASTA record" in result.output


def test_a_missing_fasta_exits_one_rather_than_raising(stand_in: None, tmp_path: Path) -> None:
    result = CliRunner().invoke(root_app, ["esm", "embed", str(tmp_path / "nothing.fasta")])
    assert result.exit_code == 1
