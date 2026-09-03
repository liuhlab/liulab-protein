"""Tests for `protein search` — the sub-app, its one command, and how it renders.

The search itself is patched out at `protein.search.mmseqs.search`, which is what the mixin
calls: the tool has its own tests, and what is under test here is what the command hands the
lane and what it prints. The frame it renders is real Foldseek-free mmseqs output from
`tests/data/`, so the rendering is exercised on numbers a tool actually wrote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import protein.cli
from protein.external import Mmseqs
from protein.search.cli import _Hits, _report, app
from protein.search.mmseqs import read_hits

_MMSEQS_HITS = Path(__file__).resolve().parent / "data" / "mmseqs_hits_p01308.tsv"

_INSULIN = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQ"


def _run(*args: str):
    """Invoke the lane the way a caller does — through the root app, as `protein search`.

    Not `invoke(app, ...)`: a Typer app holding exactly one command collapses into that
    command, so `seq` was read as the sequence argument until #15 added `struct`. It parses
    either way now, and going through the root app is still how a caller reaches it.
    """
    return CliRunner().invoke(protein.cli.app, ["search", *args])


@pytest.fixture
def searches(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Answer every search with a real hit table, recording what it was asked."""
    asked: list[dict[str, Any]] = []

    def stand_in(sequence: str, database: Any, **kwargs: Any) -> Any:
        asked.append({"sequence": sequence, "database": database, **kwargs})
        return read_hits(_MMSEQS_HITS, Mmseqs())

    monkeypatch.setattr("protein.search.mmseqs.search", stand_in)
    return asked


def test_the_sub_app_registers_its_commands_under_the_names_it_gave_them() -> None:
    assert [command.name for command in app.registered_commands] == ["seq", "struct"]


def test_every_command_takes_json() -> None:
    for command in app.registered_commands:
        assert command.name is not None
        assert "--json" in _run(command.name, "--help").output, command.name


def test_the_root_app_mounts_the_lane_as_protein_search() -> None:
    result = _run("--help")
    assert result.exit_code == 0
    assert "seq" in result.output


def test_a_bare_invocation_prints_help_instead_of_nothing() -> None:
    assert "seq" in _run().output


def test_the_sequence_and_the_database_reach_the_lane_as_they_were_typed(
    searches: list[dict[str, Any]],
) -> None:
    result = _run("seq", _INSULIN, "swissprot")
    assert result.exit_code == 0
    assert searches[0]["sequence"] == _INSULIN
    assert searches[0]["database"] == "swissprot"


def test_the_query_is_named_by_id_and_falls_back_to_the_default(
    searches: list[dict[str, Any]],
) -> None:
    _run("seq", _INSULIN, "swissprot", "--id", "P01308")
    _run("seq", _INSULIN, "swissprot")
    assert [asked["query_name"] for asked in searches] == ["P01308", "query"]


def test_the_knobs_reach_the_lane_and_the_unnamed_ones_stay_unnamed(
    searches: list[dict[str, Any]],
) -> None:
    _run("seq", _INSULIN, "swissprot", "-s", "1.0", "--threads", "4")
    assert searches[0]["sensitivity"] == 1.0
    assert searches[0]["threads"] == 4
    assert searches[0]["evalue"] is None
    assert searches[0]["max_seqs"] is None


def test_the_hits_are_printed_one_row_per_line(searches: list[dict[str, Any]]) -> None:
    result = _run("seq", _INSULIN, "swissprot", "--id", "P01308")
    assert result.exit_code == 0
    assert "P01308\tQ6YK33\t100.0\t" in result.output


def test_json_carries_the_same_answer_keyed_by_column(searches: list[dict[str, Any]]) -> None:
    result = _run("seq", _INSULIN, "swissprot", "--id", "P01308", "--json")
    assert result.exit_code == 0
    answer = json.loads(result.output)
    assert answer["query"] == "P01308"
    assert answer["database"] == "swissprot"
    assert answer["columns"] == list(Mmseqs().format_columns)
    assert len(answer["hits"]) == 20
    assert answer["hits"][0]["target"] == "Q6YK33"
    assert answer["hits"][0]["alnlen"] == 110


def test_a_sequence_outside_the_alphabet_exits_one_before_any_tool_runs(
    searches: list[dict[str, Any]],
) -> None:
    result = _run("seq", "MKT*AY", "swissprot")
    assert result.exit_code == 1
    assert "error:" in result.output
    assert searches == []


def test_a_database_nothing_is_registered_under_exits_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(sequence: str, database: Any, **kwargs: Any) -> Any:
        raise LookupError("'uniref50' is not a registered database.")

    monkeypatch.setattr("protein.search.mmseqs.search", missing)
    result = _run("seq", _INSULIN, "uniref50")
    assert result.exit_code == 1
    assert "error: 'uniref50' is not a registered database." in result.output


def test_the_header_goes_to_stderr_so_the_rows_pipe(capsys: pytest.CaptureFixture[str]) -> None:
    hits = _Hits.of(read_hits(_MMSEQS_HITS, Mmseqs()), query="P01308", database="swissprot")
    _report(hits)
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 20
    assert captured.err.startswith("query\ttarget\tpident\t")
    assert "20 hits for P01308 in swissprot" in captured.err


def test_every_cell_of_the_json_is_a_plain_python_value() -> None:
    hits = _Hits.of(read_hits(_MMSEQS_HITS, Mmseqs()), query="P01308", database="swissprot")
    # `json.dumps` is the assertion: a numpy scalar left in a row raises here.
    assert json.loads(json.dumps(hits.as_json()))["hits"][0]["bits"] == 231.0
