"""Tests for the sequence half of the search lane, and the hit table it owns.

No binary and no database. Every search here rides on one
`monkeypatch.setattr(ExternalTool, "run", ...)`, which is the property `run_to` was written
for: it catches both adapters and both tools, so what reaches the command line, what the
query FASTA held and what the frame came back as are all observable without mmseqs. The
tables that get parsed are real output — `tests/data/mmseqs_hits_p01308.tsv` came off
GPU71FM, and its provenance is in `tests/data/README.md`.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from protein import Protein
from protein.external import ExternalTool, Foldseek, Mmseqs, MmseqsLikeTool
from protein.search import mmseqs
from protein.search.mmseqs import (
    COLUMN_DTYPES,
    DEFAULT_QUERY_NAME,
    database_path,
    empty_hits,
    hit_dtypes,
    read_hits,
    search_flags,
)

#: The lane's source directory, found from this file so the test moves with the repo.
_LANE = Path(__file__).resolve().parents[1] / "src" / "protein" / "search"

#: A real Swiss-Prot search, run by hand on GPU71FM — see `tests/data/README.md`.
_MMSEQS_HITS = Path(__file__).resolve().parent / "data" / "mmseqs_hits_p01308.tsv"

_INSULIN = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQ"


@dataclass
class _Runs:
    """Every call an `easy-search` made, and the query FASTA it was pointed at."""

    calls: list[list[str]] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
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
        if args and args[0] == "easy-search":
            query = Path(args[1])
            recorded.queries.append(query.read_text(encoding="utf-8") if query.is_file() else "")
            if recorded.hits:
                Path(args[3]).write_text(recorded.hits, encoding="utf-8")
        return ""

    monkeypatch.setattr(ExternalTool, "run", record)
    return recorded


@pytest.fixture
def swissprot(liulab_data: Path) -> Path:
    """Register a Swiss-Prot-shaped database under this test's own data root."""
    directory = liulab_data / "protein" / "db" / "swissprot"
    directory.mkdir(parents=True)
    for stem in ("swissprot", "swissprot_h"):
        (directory / stem).write_text("ffindex", encoding="utf-8")
        (directory / f"{stem}.dbtype").write_bytes(b"\x00\x00\x08\x00")
    return directory / "swissprot"


# --- the column table --------------------------------------------------------


def test_the_column_table_types_every_column_either_tool_asks_for() -> None:
    asked = {*Mmseqs().format_columns, *Foldseek().format_columns}
    assert asked <= set(COLUMN_DTYPES)


def test_each_tool_keeps_its_own_identity_column_and_neither_is_renamed() -> None:
    # A percentage and a fraction. Flattening them would make two frames comparable that
    # are not, so the column a caller has is what says which number it holds.
    assert hit_dtypes(Mmseqs())["pident"] == "float64"
    assert hit_dtypes(Foldseek())["fident"] == "float64"
    assert "fident" not in hit_dtypes(Mmseqs())
    assert "pident" not in hit_dtypes(Foldseek())


def test_the_columns_come_back_in_the_order_the_tool_was_asked_for_them() -> None:
    assert tuple(hit_dtypes(Foldseek())) == Foldseek().format_columns


def test_a_column_the_table_does_not_type_fails_and_names_it() -> None:
    class _AsksForSomethingElse(MmseqsLikeTool):
        IDENTITY_COLUMN = "q3di"

    with pytest.raises(KeyError, match="q3di"):
        hit_dtypes(_AsksForSomethingElse("mmseqs"))


# --- reading a hit table -----------------------------------------------------


def test_a_search_that_found_nothing_answers_the_columns_with_no_rows() -> None:
    frame = empty_hits(Mmseqs())
    assert frame.empty
    assert list(frame.columns) == list(Mmseqs().format_columns)


def test_an_output_that_was_never_written_reads_as_no_hits(tmp_path: Path) -> None:
    assert read_hits(tmp_path / "absent.tsv", Mmseqs()).empty


def test_an_output_the_tool_left_empty_reads_as_no_hits(tmp_path: Path) -> None:
    output = tmp_path / "hits.tsv"
    output.touch()
    assert read_hits(output, Mmseqs()).empty


def test_a_real_mmseqs_table_reads_back_with_its_columns_named() -> None:
    frame = read_hits(_MMSEQS_HITS, Mmseqs())
    assert list(frame.columns) == list(Mmseqs().format_columns)
    assert len(frame) == 20
    assert frame.loc[0, "query"] == "P01308"
    assert frame.loc[0, "target"] == "Q6YK33"


def test_a_real_mmseqs_table_reads_the_dtypes_the_table_declares() -> None:
    frame = read_hits(_MMSEQS_HITS, Mmseqs())
    assert str(frame.dtypes["target"]) == "string"
    assert str(frame.dtypes["alnlen"]) == "int64"
    assert str(frame.dtypes["evalue"]) == "float64"
    # pident is a percentage, and mmseqs really does write 100.000 for an identical hit.
    assert frame.loc[0, "pident"] == pytest.approx(100.0)


# --- resolving a database ----------------------------------------------------


def test_a_database_answers_with_its_own_path(tmp_path: Path) -> None:
    class Registered:
        path = tmp_path / "swissprot"

    assert database_path(Registered()) == tmp_path / "swissprot"


def test_a_registered_name_resolves_to_the_ffindex_database_inside_its_directory(
    swissprot: Path,
) -> None:
    assert database_path("swissprot") == swissprot


def test_a_directory_whose_database_is_spelled_differently_resolves_to_the_shortest_stem(
    liulab_data: Path,
) -> None:
    # Measured on GPU71FM: `db/pdb/` holds `pdb100` and its `_h`, `_ca`, `_clu` and `_seq`
    # siblings, so the registered name and the ffindex prefix are not the same string.
    directory = liulab_data / "protein" / "db" / "pdb"
    directory.mkdir(parents=True)
    for stem in ("pdb100", "pdb100_h", "pdb100_ca", "pdb100_clu", "pdb100_seq", "pdb100_seq_h"):
        (directory / f"{stem}.dbtype").write_bytes(b"\x00\x00\x00\x00")
    assert database_path("pdb") == directory / "pdb100"


def test_a_name_nothing_is_registered_under_raises_and_names_what_is(swissprot: Path) -> None:
    with pytest.raises(LookupError, match="swissprot") as failed:
        database_path("uniref50")
    assert "uniref50" in str(failed.value)


def test_a_directory_holding_no_ffindex_database_raises(liulab_data: Path) -> None:
    (liulab_data / "protein" / "db" / "empty").mkdir(parents=True)
    with pytest.raises(LookupError, match="no ffindex database"):
        database_path("empty")


# --- the flags ---------------------------------------------------------------


def test_only_the_knobs_that_were_named_reach_the_command_line() -> None:
    assert search_flags(sensitivity=1.0) == ["-s", "1.0"]
    assert search_flags() == []


def test_every_knob_maps_to_the_argument_both_tools_spell_the_same_way() -> None:
    assert search_flags(sensitivity=4.0, evalue=1e-3, max_seqs=20, threads=4) == [
        "-s",
        "4.0",
        "-e",
        "0.001",
        "--max-seqs",
        "20",
        "--threads",
        "4",
    ]


def test_extra_arguments_come_last_and_are_passed_through_unread() -> None:
    assert search_flags(threads=2, extra=["--comp-bias-corr", "0"]) == [
        "--threads",
        "2",
        "--comp-bias-corr",
        "0",
    ]


# --- the search --------------------------------------------------------------


def test_a_search_asks_mmseqs_for_every_column_in_one_order(swissprot: Path, runs: _Runs) -> None:
    mmseqs.search(_INSULIN, "swissprot")
    verb, _query, target, _output, _work, flag, columns = runs.calls[0]
    assert (verb, flag) == ("easy-search", "--format-output")
    assert target == str(swissprot)
    assert columns == Mmseqs().format_output


def test_the_query_fasta_holds_the_sequence_under_the_name_it_was_given(
    swissprot: Path, runs: _Runs
) -> None:
    mmseqs.search(_INSULIN, "swissprot", query_name="P01308")
    assert runs.queries == [f">P01308\n{_INSULIN}\n"]


def test_the_query_fasta_lives_in_the_scratch_directory_and_goes_with_it(
    swissprot: Path, runs: _Runs
) -> None:
    mmseqs.search(_INSULIN, "swissprot")
    query = Path(runs.calls[0][1])
    assert query.parent.name.startswith("mmseqs-search-")
    assert not query.exists()
    assert not query.parent.exists()


def test_the_flags_a_search_was_given_reach_the_tool(swissprot: Path, runs: _Runs) -> None:
    mmseqs.search(_INSULIN, "swissprot", sensitivity=1.0, threads=4)
    assert runs.calls[0][-4:] == ["-s", "1.0", "--threads", "4"]


def test_a_search_returns_the_table_the_tool_wrote(swissprot: Path, runs: _Runs) -> None:
    runs.hits = _MMSEQS_HITS.read_text(encoding="utf-8")
    frame = mmseqs.search(_INSULIN, "swissprot")
    assert list(frame.columns) == list(Mmseqs().format_columns)
    assert frame["target"].tolist()[:2] == ["Q6YK33", "P01308"]


def test_a_search_that_wrote_nothing_still_answers_a_named_table(
    swissprot: Path, runs: _Runs
) -> None:
    frame = mmseqs.search(_INSULIN, "swissprot")
    assert frame.empty
    assert list(frame.columns) == list(Mmseqs().format_columns)


def test_a_search_against_an_unregistered_name_never_starts_the_tool(
    liulab_data: Path, runs: _Runs
) -> None:
    with pytest.raises(LookupError):
        mmseqs.search(_INSULIN, "swissprot")
    assert runs.calls == []


# --- the mixin ---------------------------------------------------------------


def test_a_protein_searches_under_its_own_accession(swissprot: Path, runs: _Runs) -> None:
    Protein(_INSULIN, id="P01308").search("swissprot")
    assert runs.queries == [f">P01308\n{_INSULIN}\n"]


def test_a_protein_without_an_accession_searches_under_the_default_query_name(
    swissprot: Path, runs: _Runs
) -> None:
    Protein(_INSULIN).search("swissprot")
    assert runs.queries == [f">{DEFAULT_QUERY_NAME}\n{_INSULIN}\n"]


def test_a_caller_that_names_the_query_itself_is_not_overruled(
    swissprot: Path, runs: _Runs
) -> None:
    Protein(_INSULIN, id="P01308").search("swissprot", query_name="mine")
    assert runs.queries == [f">mine\n{_INSULIN}\n"]


def test_a_protein_forwards_the_search_knobs_it_was_given(swissprot: Path, runs: _Runs) -> None:
    Protein(_INSULIN, id="P01308").search("swissprot", max_seqs=20)
    assert runs.calls[0][-2:] == ["--max-seqs", "20"]


def test_a_protein_hands_back_the_hits_as_they_were_parsed(swissprot: Path, runs: _Runs) -> None:
    runs.hits = _MMSEQS_HITS.read_text(encoding="utf-8")
    frame = Protein(_INSULIN, id="P01308").search("swissprot")
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc[0, "pident"] == pytest.approx(100.0)


def test_no_module_in_the_search_lane_imports_pandas_at_module_level() -> None:
    # `protein.core` imports the mixin, so anything at the top of a lane module is paid for
    # by every `import protein` — including by a caller that never searches anything.
    offenders = sorted(
        source.name for source in _lane_modules() if "pandas" in _module_level_imports(source)
    )
    assert offenders == []


def _lane_modules() -> Iterator[Path]:
    """Yield every Python module in the search lane."""
    return _LANE.glob("*.py")


def _module_level_imports(source: Path) -> set[str]:
    """Return the top-level packages ``source`` imports at module level."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            names.add(node.module.split(".")[0])
    return names
