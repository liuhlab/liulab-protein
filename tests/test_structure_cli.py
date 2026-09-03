"""Tests for the two command lines this lane adds: `protein structure` and `search struct`.

Both go through the root app, which is how a caller reaches them. The one exception is the
test that a sub-app holding two commands no longer collapses into one — that is the finding
`#14` left behind, and it is checked against the sub-app directly because that is where it
was visible.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from protein import sifts
from protein.cli import app as root_app
from protein.external import ExternalTool
from protein.search.cli import app as search_app
from protein.structure.cli import app as structure_app

_DATA = Path(__file__).resolve().parent / "data"
_UBQ = _DATA / "1ubq.cif.gz"
_BNA = _DATA / "1bna.cif.gz"
_HITS = _DATA / "foldseek_hits_1ubq.tsv"

runner = CliRunner()


@pytest.fixture
def cached_ubq(liulab_data: Path) -> Path:
    """`1UBQ` in the coordinate cache, so a bare entry id resolves offline."""
    directory = liulab_data / "protein" / "structures"
    directory.mkdir(parents=True)
    target = directory / "1ubq.cif"
    target.write_bytes(gzip.decompress(_UBQ.read_bytes()))
    return target


@pytest.fixture
def pdb_database(liulab_data: Path) -> Path:
    """A pdb100-shaped database registered under this test's own data root."""
    directory = liulab_data / "protein" / "db" / "pdb"
    directory.mkdir(parents=True)
    (directory / "pdb100.dbtype").write_bytes(b"\x00\x00\x00\x00")
    return directory / "pdb100"


@pytest.fixture
def no_sifts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer every SIFTS lookup with one accession, so `show` needs no prepared map."""
    monkeypatch.setattr(sifts, "accessions_for", lambda pdb, chain: ("P0CG48",))


@pytest.fixture
def hits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every `easy-search` write a real Foldseek hit table and run nothing."""

    def record(
        self: ExternalTool, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True
    ) -> str:
        if args and args[0] == "easy-search":
            Path(args[3]).write_text(_HITS.read_text(encoding="utf-8"), encoding="utf-8")
        return ""

    monkeypatch.setattr(ExternalTool, "run", record)


# --- protein structure fetch ---------------------------------------------------


def test_fetch_reports_the_file_that_is_already_cached(cached_ubq: Path) -> None:
    result = runner.invoke(root_app, ["structure", "fetch", "1UBQ"])
    assert result.exit_code == 0
    assert f"path: {cached_ubq}" in result.stdout


def test_fetch_json_carries_the_same_answer(cached_ubq: Path) -> None:
    result = runner.invoke(root_app, ["structure", "fetch", "1UBQ", "--json"])
    payload = json.loads(result.stdout)
    assert payload["id"] == "1UBQ"
    assert payload["path"] == str(cached_ubq)
    assert payload["bytes"] == cached_ubq.stat().st_size


def test_fetch_exits_one_when_it_cannot_reach_rcsb(liulab_data: Path) -> None:
    # The suite's network guard is what stands in for a compute node with no route out.
    result = runner.invoke(root_app, ["structure", "fetch", "1UBQ"])
    assert result.exit_code == 1
    assert "error:" in result.stderr


# --- protein structure show ----------------------------------------------------


def test_show_lists_a_chain_per_row_with_the_header_on_stderr(no_sifts: None) -> None:
    result = runner.invoke(root_app, ["structure", "show", str(_UBQ)])
    assert result.exit_code == 0
    assert result.stdout.splitlines() == ["A\tprotein\t76\t660\tP0CG48"]
    assert result.stderr.startswith("chain\tkind\tresidues\tatoms\tuniprot")


def test_show_leaves_the_residue_count_empty_for_a_chain_that_is_not_protein(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sifts, "accessions_for", lambda pdb, chain: ())
    result = runner.invoke(root_app, ["structure", "show", str(_BNA)])
    assert result.stdout.splitlines() == ["A\tnucleic\t\t280\t", "B\tnucleic\t\t286\t"]


def test_show_takes_a_cached_entry_id_as_readily_as_a_path(
    cached_ubq: Path, no_sifts: None
) -> None:
    result = runner.invoke(root_app, ["structure", "show", "1ubq"])
    assert result.exit_code == 0
    assert result.stdout.splitlines() == ["A\tprotein\t76\t660\tP0CG48"]


def test_show_json_names_the_file_it_read(no_sifts: None) -> None:
    result = runner.invoke(root_app, ["structure", "show", str(_UBQ), "--json"])
    payload = json.loads(result.stdout)
    assert payload["path"] == str(_UBQ)
    assert payload["chains"][0] == {
        "chain": "A",
        "kind": "protein",
        "residues": 76,
        "atoms": 660,
        "uniprot": "P0CG48",
    }


def test_show_exits_one_when_the_map_nobody_prepared_is_asked_for() -> None:
    result = runner.invoke(root_app, ["structure", "show", str(_UBQ)])
    assert result.exit_code == 1
    assert "protein sifts prepare" in result.stderr


# --- protein search struct -----------------------------------------------------


def test_a_structural_search_prints_the_hits(pdb_database: Path, hits: None) -> None:
    result = runner.invoke(root_app, ["search", "struct", str(_UBQ), "pdb"])
    assert result.exit_code == 0
    assert result.stdout.splitlines()[0].startswith("1ubq\t2n2k-assembly1_A\t0.973")
    assert "20 hits for 1ubq in pdb" in result.stderr


def test_a_structural_search_names_the_structural_columns(pdb_database: Path, hits: None) -> None:
    result = runner.invoke(root_app, ["search", "struct", str(_UBQ), "pdb", "--json"])
    payload = json.loads(result.stdout)
    assert payload["columns"][2] == "fident"
    assert payload["columns"][-2:] == ["alntmscore", "lddt"]
    assert "q3di" not in payload["columns"]


def test_a_structural_search_can_be_pointed_at_one_chain(pdb_database: Path, hits: None) -> None:
    result = runner.invoke(root_app, ["search", "struct", str(_UBQ), "pdb", "--chain", "A"])
    assert result.exit_code == 0
    assert "hits for 1ubq_A in pdb" in result.stderr


def test_a_chain_the_structure_does_not_have_is_reported_without_the_repr_quotes(
    pdb_database: Path, hits: None
) -> None:
    result = runner.invoke(root_app, ["search", "struct", str(_UBQ), "pdb", "--chain", "Z"])
    assert result.exit_code == 1
    assert "error: 1ubq has no chain 'Z'" in result.stderr


def test_a_database_nothing_is_registered_under_exits_one(liulab_data: Path) -> None:
    result = runner.invoke(root_app, ["search", "struct", str(_UBQ), "nonesuch"])
    assert result.exit_code == 1
    assert "not a registered database" in result.stderr


# --- the sub-apps themselves ---------------------------------------------------


def test_the_search_sub_app_no_longer_collapses_into_its_only_command(
    pdb_database: Path, hits: None
) -> None:
    # A Typer app holding exactly one command becomes that command, which is why `#14`'s
    # tests had to go through the root app. `struct` is the second, so this now works.
    result = runner.invoke(search_app, ["struct", str(_UBQ), "pdb"])
    assert result.exit_code == 0
    assert "20 hits for 1ubq in pdb" in result.stderr


def test_every_command_of_both_sub_apps_is_named_explicitly() -> None:
    assert [command.name for command in structure_app.registered_commands] == ["fetch", "show"]
    assert [command.name for command in search_app.registered_commands] == ["seq", "struct"]


def test_the_structure_sub_app_is_mounted_on_the_root_app() -> None:
    result = runner.invoke(root_app, ["structure", "--help"])
    assert result.exit_code == 0
    assert "fetch" in result.stdout
    assert "show" in result.stdout
