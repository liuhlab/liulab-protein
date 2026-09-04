"""Tests for `protein msa` — the sub-app, what reaches the lane, and the file it leaves.

Both verbs are patched at `protein.msa.search` and `protein.msa.align`: each has its own
tests, and what is under test here is what the command hands the lane, where it writes, and
what it prints. Nothing runs mmseqs or muscle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import protein.cli
from protein import MSA
from protein.msa.cli import app

from . import plain_text

_DATA = Path(__file__).resolve().parent / "data"
_FASTA = _DATA / "uniprot_three.fasta"

_INSULIN = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQ"

# The first record of the FASTA fixture, spelled here so a test need not read the file it
# hands the command.
_HUMAN_HEADER = "sp|P01308|INS_HUMAN Insulin OS=Homo sapiens OX=9606 GN=INS PE=1 SV=1"
_HUMAN_ID = "sp|P01308|INS_HUMAN"

# What both stand-ins answer with: depth 2, 5 match states, row 0 headed `P01308`.
_ALIGNMENT = MSA([("P01308", "MKTAY"), ("sp|P01315|INS_PIG key=9823", "MKTaAY")])


def _run(*args: str):
    """Invoke the lane the way a caller does — through the root app, as `protein msa`."""
    return CliRunner().invoke(protein.cli.app, ["msa", *args])


@pytest.fixture
def searches(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Answer every database search with one alignment, recording what it was asked."""
    asked: list[dict[str, Any]] = []

    def stand_in(sequence: str, database: Any, **kwargs: Any) -> MSA:
        asked.append({"sequence": sequence, "database": database, **kwargs})
        return _ALIGNMENT

    monkeypatch.setattr("protein.msa.search", stand_in)
    return asked


@pytest.fixture
def alignments(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Answer every MUSCLE alignment with one alignment, recording what it was asked."""
    asked: list[dict[str, Any]] = []

    def stand_in(sequences: Any, **kwargs: Any) -> MSA:
        asked.append({"sequences": list(sequences), **kwargs})
        return _ALIGNMENT

    monkeypatch.setattr("protein.msa.align", stand_in)
    return asked


# --- the sub-app -------------------------------------------------------------


def test_the_sub_app_registers_its_commands_under_the_names_it_gave_them() -> None:
    assert [command.name for command in app.registered_commands] == ["search", "align"]


def test_every_command_takes_json() -> None:
    # `plain_text`, not `result.output`: rich styles the first dash of `--json` separately,
    # so the raw output carries no such substring wherever colour is on.
    for command in app.registered_commands:
        assert command.name is not None
        assert "--json" in plain_text(_run(command.name, "--help").output), command.name


def test_the_root_app_mounts_the_lane_as_protein_msa() -> None:
    result = _run("--help")
    assert result.exit_code == 0
    assert "search" in result.output


def test_a_bare_invocation_prints_help_instead_of_nothing() -> None:
    assert "align" in _run().output


# --- searching a database ----------------------------------------------------


def test_the_sequence_and_the_database_reach_the_lane_as_they_were_typed(
    searches: list[dict[str, Any]], tmp_path: Path
) -> None:
    result = _run("search", _INSULIN, "swissprot", str(tmp_path / "insulin.a3m"))
    assert result.exit_code == 0
    assert searches[0]["sequence"] == _INSULIN
    assert searches[0]["database"] == "swissprot"


def test_the_alignment_is_written_to_the_path_that_was_given(
    searches: list[dict[str, Any]], tmp_path: Path
) -> None:
    out = tmp_path / "insulin.a3m"
    _run("search", _INSULIN, "swissprot", str(out))
    assert out.read_text(encoding="utf-8") == _ALIGNMENT.to_a3m()


def test_the_output_path_is_required(searches: list[dict[str, Any]]) -> None:
    result = _run("search", _INSULIN, "swissprot")
    assert result.exit_code == 2
    assert searches == []


def test_the_query_is_named_by_id_and_falls_back_to_the_default(
    searches: list[dict[str, Any]], tmp_path: Path
) -> None:
    _run("search", _INSULIN, "swissprot", str(tmp_path / "a.a3m"), "--id", "P01308")
    _run("search", _INSULIN, "swissprot", str(tmp_path / "b.a3m"))
    assert [asked["query_name"] for asked in searches] == ["P01308", "query"]


def test_the_knobs_reach_the_lane_and_the_unnamed_ones_stay_unnamed(
    searches: list[dict[str, Any]], tmp_path: Path
) -> None:
    _run("search", _INSULIN, "swissprot", str(tmp_path / "a.a3m"), "-s", "1.0", "--threads", "4")
    assert searches[0]["sensitivity"] == 1.0
    assert searches[0]["threads"] == 4
    assert searches[0]["evalue"] is None
    assert searches[0]["max_seqs"] is None


def test_what_it_prints_says_how_deep_the_alignment_is_and_where_it_went(
    searches: list[dict[str, Any]], tmp_path: Path
) -> None:
    out = tmp_path / "insulin.a3m"
    result = _run("search", _INSULIN, "swissprot", str(out))
    assert "depth: 2" in result.output
    assert f"path: {out}" in result.output


def test_search_json_carries_the_same_answer(
    searches: list[dict[str, Any]], tmp_path: Path
) -> None:
    out = tmp_path / "insulin.a3m"
    result = _run("search", _INSULIN, "swissprot", str(out), "--json")
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "database": "swissprot",
        "query": "P01308",
        "depth": 2,
        "match_states": 5,
        "path": str(out),
    }


def test_a_sequence_outside_the_alphabet_exits_one_before_any_tool_runs(
    searches: list[dict[str, Any]], tmp_path: Path
) -> None:
    result = _run("search", "MKT*AY", "swissprot", str(tmp_path / "a.a3m"))
    assert result.exit_code == 1
    assert "error:" in result.output
    assert searches == []


def test_a_database_nothing_is_registered_under_exits_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def missing(sequence: str, database: Any, **kwargs: Any) -> MSA:
        raise LookupError("'uniref50' is not a registered database.")

    monkeypatch.setattr("protein.msa.search", missing)
    result = _run("search", _INSULIN, "uniref50", str(tmp_path / "a.a3m"))
    assert result.exit_code == 1
    assert "error: 'uniref50' is not a registered database." in result.output


# --- aligning a FASTA --------------------------------------------------------


def test_the_records_reach_the_lane_with_their_headers_whole(
    alignments: list[dict[str, Any]], tmp_path: Path
) -> None:
    # Whole, because cutting the description off would take `OX=` with it, and a row with no
    # organism id is a row that cannot be paired.
    result = _run("align", str(_FASTA), str(tmp_path / "a.a3m"), "--query", _HUMAN_HEADER)
    assert result.exit_code == 0
    assert alignments[0]["sequences"][0][0] == _HUMAN_HEADER
    assert alignments[0]["query"] == _HUMAN_HEADER


def test_the_query_may_be_named_by_the_identifier_its_header_opens_with(
    alignments: list[dict[str, Any]], tmp_path: Path
) -> None:
    _run("align", str(_FASTA), str(tmp_path / "a.a3m"), "--query", _HUMAN_ID)
    assert alignments[0]["query"] == _HUMAN_HEADER


def test_align_writes_the_alignment_to_the_path_that_was_given(
    alignments: list[dict[str, Any]], tmp_path: Path
) -> None:
    out = tmp_path / "three.a3m"
    _run("align", str(_FASTA), str(out), "--query", _HUMAN_ID)
    assert out.read_text(encoding="utf-8") == _ALIGNMENT.to_a3m()


def test_align_requires_the_output_path(alignments: list[dict[str, Any]]) -> None:
    result = _run("align", str(_FASTA), "--query", _HUMAN_ID)
    assert result.exit_code == 2
    assert alignments == []


def test_align_requires_a_query_and_anchors_on_nothing_by_default(
    alignments: list[dict[str, Any]], tmp_path: Path
) -> None:
    result = _run("align", str(_FASTA), str(tmp_path / "a.a3m"))
    assert result.exit_code == 2
    assert "--query" in plain_text(result.output)
    assert alignments == []


def test_a_query_naming_no_record_exits_one(tmp_path: Path) -> None:
    # Unpatched on purpose: `align` refuses before it locates the binary, and its message is
    # the one that names the headers there are.
    result = _run("align", str(_FASTA), str(tmp_path / "a.a3m"), "--query", "P99999")
    assert result.exit_code == 1
    assert "error:" in result.output


def test_a_fasta_that_is_not_there_exits_one(
    alignments: list[dict[str, Any]], tmp_path: Path
) -> None:
    result = _run("align", str(tmp_path / "gone.fasta"), str(tmp_path / "a.a3m"), "--query", "x")
    assert result.exit_code == 1
    assert "error:" in result.output
    assert alignments == []


def test_align_json_carries_the_same_answer(
    alignments: list[dict[str, Any]], tmp_path: Path
) -> None:
    out = tmp_path / "three.a3m"
    result = _run("align", str(_FASTA), str(out), "--query", _HUMAN_ID, "--json")
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "sequences": str(_FASTA),
        "query": "P01308",
        "depth": 2,
        "match_states": 5,
        "path": str(out),
    }
