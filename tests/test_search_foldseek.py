"""Tests for the structural half of the search lane.

Nothing on `Protein` reaches this module — a protein has no coordinates — so what is under
test is what `Structure` and `Chain` will call when #15 builds them: the query goes through
as the file it already is, and the frame comes back through the same parser the sequence
half uses. `tests/data/foldseek_hits_1ubq.tsv` is real Foldseek output against pdb100; its
provenance is in `tests/data/README.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from protein.external import ExternalTool, Foldseek
from protein.search import foldseek
from protein.search.mmseqs import empty_hits, read_hits

#: A real pdb100 search, run by hand on GPU71FM — see `tests/data/README.md`.
_FOLDSEEK_HITS = Path(__file__).resolve().parent / "data" / "foldseek_hits_1ubq.tsv"


@dataclass
class _Runs:
    """Every call an `easy-search` made."""

    calls: list[list[str]] = field(default_factory=list)
    #: Written into each search's output file, so the parse can be exercised too.
    hits: str = ""


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> _Runs:
    """Replace `ExternalTool.run` on the base class and record what a search did."""
    recorded = _Runs()

    def record(
        self: ExternalTool, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True
    ) -> str:
        recorded.calls.append(list(args))
        if args and args[0] == "easy-search" and recorded.hits:
            Path(args[3]).write_text(recorded.hits, encoding="utf-8")
        return ""

    monkeypatch.setattr(ExternalTool, "run", record)
    return recorded


@pytest.fixture
def pdb(liulab_data: Path) -> Path:
    """Register a pdb100-shaped database under this test's own data root."""
    directory = liulab_data / "protein" / "db" / "pdb"
    directory.mkdir(parents=True)
    for stem in ("pdb100", "pdb100_h", "pdb100_ca", "pdb100_ss"):
        (directory / f"{stem}.dbtype").write_bytes(b"\x00\x00\x00\x00")
    return directory / "pdb100"


def test_foldseek_reports_the_two_structural_columns_mmseqs_cannot() -> None:
    assert Foldseek().format_columns[-2:] == ("alntmscore", "lddt")


def test_foldseek_is_never_asked_for_q3di() -> None:
    # Measured on 10-941cd33: the code is in no `--format-output` list, and asking for it
    # fails the whole search with `Format code q3di does not exist.`
    assert "q3di" not in Foldseek().format_columns


def test_a_real_foldseek_table_reads_back_with_its_columns_named() -> None:
    frame = read_hits(_FOLDSEEK_HITS, Foldseek())
    assert list(frame.columns) == list(Foldseek().format_columns)
    assert len(frame) == 20
    assert frame.loc[0, "target"] == "2n2k-assembly1_A"


def test_a_real_foldseek_table_reports_identity_as_a_fraction_and_not_a_percentage() -> None:
    frame = read_hits(_FOLDSEEK_HITS, Foldseek())
    assert frame["fident"].max() <= 1.0
    assert frame.loc[0, "fident"] == pytest.approx(0.973)


def test_a_real_foldseek_table_reads_the_structural_columns_as_numbers() -> None:
    frame = read_hits(_FOLDSEEK_HITS, Foldseek())
    assert str(frame.dtypes["alntmscore"]) == "float64"
    assert frame.loc[0, "lddt"] == pytest.approx(0.9839)


def test_an_empty_foldseek_table_still_carries_the_structural_columns() -> None:
    assert list(empty_hits(Foldseek()).columns)[-2:] == ["alntmscore", "lddt"]


@pytest.fixture
def query(tmp_path: Path) -> Path:
    """A structure file on disk, which is the only shape a Foldseek query comes in."""
    coordinates = tmp_path / "coordinates"
    coordinates.mkdir()
    structure = coordinates / "1ubq.cif"
    structure.write_text("data_1UBQ\n", encoding="utf-8")
    return structure


def test_a_structural_search_passes_the_query_file_through_unchanged(
    pdb: Path, query: Path, runs: _Runs
) -> None:
    foldseek.search(query, "pdb")
    verb, passed, target, _output, _work, flag, columns = runs.calls[0]
    assert (verb, flag) == ("easy-search", "--format-output")
    assert passed == str(query)
    assert target == str(pdb)
    assert columns == Foldseek().format_output


def test_a_structural_search_writes_nothing_beside_the_query(
    pdb: Path, query: Path, runs: _Runs
) -> None:
    # A sequence query is written into the scratch directory; a structural one is already a
    # file, so this half of the lane creates nothing at all.
    foldseek.search(query, "pdb")
    assert [entry.name for entry in query.parent.iterdir()] == ["1ubq.cif"]


def test_a_structural_search_takes_the_same_knobs_the_sequence_half_does(
    pdb: Path, query: Path, runs: _Runs
) -> None:
    foldseek.search(query, "pdb", sensitivity=1.0, threads=4)
    assert runs.calls[0][-4:] == ["-s", "1.0", "--threads", "4"]


def test_a_structural_search_returns_the_table_foldseek_wrote(
    pdb: Path, query: Path, runs: _Runs
) -> None:
    runs.hits = _FOLDSEEK_HITS.read_text(encoding="utf-8")
    frame = foldseek.search(query, "pdb")
    assert frame.loc[0, "query"] == "1ubq"
    assert frame.loc[0, "alntmscore"] == pytest.approx(0.9885)


def test_a_structural_search_takes_a_database_object_as_readily_as_a_name(
    pdb: Path, query: Path, runs: _Runs
) -> None:
    class Registered:
        path = pdb

    foldseek.search(query, Registered())
    assert runs.calls[0][2] == str(pdb)
