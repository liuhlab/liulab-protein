"""Tests for `protein db` — the four commands, their JSON, and their failures.

Everything goes through the root app, which is how a caller reaches this group and which
also proves the mount. No binary: an autouse fixture replaces
`ExternalTool.run`, and the databases are a handful of files.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from protein.cli import app
from protein.db import cli as db_cli
from protein.external import ExternalTool, InstalledTool

runner = CliRunner()


def _write_database(directory: Path, prefix: str, *, gpu: bool = False) -> Path:
    """Write a database-shaped file set and return its ffindex prefix."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / prefix).write_text("data", encoding="utf-8")
    (directory / f"{prefix}.dbtype").write_bytes(
        b"\x00\x00\x08\x00" if gpu else b"\x00\x00\x00\x00"
    )
    (directory / f"{prefix}.index").write_text("0\t0\t5\n", encoding="utf-8")
    (directory / f"{prefix}.lookup").write_text("0\tP12345\t0\n1\tP0A031\t0\n", encoding="utf-8")
    return directory / prefix


def _run(*args: str) -> tuple[int, str]:
    """Invoke `protein db ...` through the root app and return its exit code and output."""
    result = runner.invoke(app, ["db", *args])
    return result.exit_code, result.stdout


@pytest.fixture
def swissprot(liulab_data: Path) -> Path:
    """Write a Swiss-Prot-shaped, GPU-encoded database under this test's data root."""
    return _write_database(liulab_data / "protein" / "db" / "swissprot", "swissprot", gpu=True)


@pytest.fixture(autouse=True)
def no_tool(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace `ExternalTool.run` and record what a command asked the tool to do."""
    calls: list[list[str]] = []

    def record(
        self: ExternalTool, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True
    ) -> str:
        calls.append(list(args))
        if args and args[0] == "databases":
            target = Path(args[2])
            _write_database(target.parent, target.name)
        return ""

    monkeypatch.setattr(ExternalTool, "run", record)
    monkeypatch.setattr(InstalledTool, "_detect_version", lambda self: "18.8cc5c")
    return calls


# --- the mount ------------------------------------------------------------------


def test_the_sub_app_registers_its_commands_under_the_names_it_gave_them() -> None:
    assert [command.name for command in db_cli.app.registered_commands] == [
        "list",
        "adopt",
        "download",
        "status",
    ]


# --- list -----------------------------------------------------------------------


def test_list_names_what_can_be_fetched_even_when_nothing_is_registered() -> None:
    # On a fresh machine this list is the answer to "what can I download?".
    code, output = _run("list")
    assert code == 0
    assert "swissprot" in output
    assert "not registered" in output


def test_list_marks_a_registered_name_as_registered(swissprot: Path) -> None:
    _run("adopt", "swissprot", str(swissprot.parent))
    code, output = _run("list")
    assert code == 0
    assert "swissprot\tregistered" in output


def test_list_as_json_carries_one_row_per_name() -> None:
    code, output = _run("list", "--json")
    assert code == 0
    names = {row["name"] for row in json.loads(output)["databases"]}
    assert names == {"pdb", "swissprot"}


# --- adopt ----------------------------------------------------------------------


def test_adopting_registers_a_database_that_is_already_on_disk(swissprot: Path) -> None:
    code, output = _run("adopt", "swissprot", str(swissprot.parent), "--json")
    assert code == 0
    assert json.loads(output)["registered"] is True
    assert (swissprot.parent / ".completion.json").is_file()


def test_adopting_says_whether_the_residues_were_folded(swissprot: Path) -> None:
    code, output = _run("adopt", "swissprot", str(swissprot.parent), "--json")
    payload = json.loads(output)
    assert code == 0
    assert payload["is_gpu_encoded"] is True
    assert "B->D" in payload["residue_fold"]


def test_adopting_an_undeclared_name_needs_to_be_told_which_kind_it_is(liulab_data: Path) -> None:
    directory = liulab_data / "elsewhere" / "uniref50"
    _write_database(directory, "uniref50")
    code, _ = _run("adopt", "uniref50", str(directory))
    assert code == 1
    code, output = _run("adopt", "uniref50", str(directory), "--kind", "sequence", "--json")
    assert code == 0
    assert json.loads(output)["tool"] == "mmseqs"


def test_adopting_a_directory_that_holds_no_database_fails_with_a_message(tmp_path: Path) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()
    result = runner.invoke(app, ["db", "adopt", "swissprot", str(empty)])
    assert result.exit_code == 1
    assert "error:" in result.stderr


# --- download -------------------------------------------------------------------


def test_downloading_hands_the_tool_the_source_spelling_the_declaration_carries(
    no_tool: list[list[str]], liulab_data: Path
) -> None:
    code, _ = _run("download", "swissprot", "--json")
    assert code == 0
    assert no_tool[0][:2] == ["databases", "UniProtKB/Swiss-Prot"]
    assert (liulab_data / "protein" / "db" / "swissprot" / ".completion.json").is_file()


def test_downloading_an_undeclared_name_with_a_source_still_works(
    no_tool: list[list[str]],
) -> None:
    code, output = _run(
        "download", "uniref50", "--source", "UniRef50", "--kind", "sequence", "--json"
    )
    assert code == 0
    assert no_tool[0][:2] == ["databases", "UniRef50"]
    assert json.loads(output)["name"] == "uniref50"


def test_downloading_an_undeclared_name_with_no_source_says_what_to_do(
    no_tool: list[list[str]],
) -> None:
    result = runner.invoke(app, ["db", "download", "uniref50", "--kind", "sequence"])
    assert result.exit_code == 1
    assert "no download source" in result.stderr


# --- status ---------------------------------------------------------------------


def test_status_prints_one_key_per_line(swissprot: Path) -> None:
    code, output = _run("status", "swissprot")
    assert code == 0
    assert "name: swissprot" in output
    assert "index_entries: 1" in output
    assert "lookup_entries: 2" in output


def test_status_of_an_unregistered_declared_name_reports_rather_than_failing() -> None:
    code, output = _run("status", "pdb", "--json")
    assert code == 0
    payload = json.loads(output)
    assert payload["registered"] is False
    assert payload["tool"] == "foldseek"


def test_status_of_a_name_nothing_knows_about_says_which_kinds_there_are() -> None:
    result = runner.invoke(app, ["db", "status", "uniref50"])
    assert result.exit_code == 1
    assert "--kind" in result.stderr
