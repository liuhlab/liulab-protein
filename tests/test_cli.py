"""Tests for the root CLI — the two commands belonging to no lane, and the conventions.

`doctor` is driven against a patched `protein.external.doctor` rather than the binaries:
what is under test is what the command prints and what it exits with, and the tools have
their own tests. The convention checks are here because four lanes will each mount a
sub-app against them.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from protein import __version__
from protein.cli import app
from protein.external import ToolNotFoundError

_TOOLS = {"mmseqs": "18.8cc5c", "foldseek": "10.941cd33"}


def test_a_bare_invocation_prints_help_instead_of_nothing() -> None:
    result = CliRunner().invoke(app, [])
    assert "version" in result.output
    assert "doctor" in result.output


def test_version_prints_what_the_installed_metadata_says() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_version_answers_json_when_asked() -> None:
    result = CliRunner().invoke(app, ["version", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"version": __version__}


def test_doctor_names_each_tool_and_what_it_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("protein.cli._doctor", lambda: dict(_TOOLS))
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert result.output.splitlines() == ["mmseqs: 18.8cc5c", "foldseek: 10.941cd33"]


def test_doctor_answers_json_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("protein.cli._doctor", lambda: dict(_TOOLS))
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == _TOOLS


def test_doctor_exits_one_and_prints_the_install_command_when_a_tool_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> dict[str, str]:
        raise ToolNotFoundError("foldseek is not installed. Add it with:\n    pixi add foldseek")

    monkeypatch.setattr("protein.cli._doctor", missing)
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "pixi add foldseek" in result.output


def test_every_command_is_registered_under_a_name_it_was_given() -> None:
    # The convention each lane's sub-app follows: names given explicitly, so `--help` and
    # the module docstring's doctest cannot drift from the functions.
    assert [command.name for command in app.registered_commands] == ["version", "doctor"]


def test_every_command_takes_json() -> None:
    for command in app.registered_commands:
        assert command.name is not None
        result = CliRunner().invoke(app, [command.name, "--help"])
        assert "--json" in result.output, command.name
