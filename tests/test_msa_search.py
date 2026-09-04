"""`Protein.msa(db)`: the four invocations, the headers they carry, and the value back.

No binary and no database. Every run rides on one
`monkeypatch.setattr(ExternalTool, "run", ...)`, so the flags reaching the boundary, the
query FASTA and the alignment that comes back are all observable with mmseqs absent. The
alignment text below is what `result2msa` writes: a hit's own header, whole, and one column
per query residue.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from protein import MSA, Protein
from protein.external import ExternalTool
from protein.msa import mmseqs as lane
from protein.msa.mmseqs import organism_id, with_organism_key

_INSULIN = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQ"

_PIG_HEADER = "sp|P01315|INS_PIG Insulin OS=Sus scrofa OX=9823 GN=INS PE=1 SV=1"
_MOUSE_HEADER = "UniRef100_A0A0 Cluster: Insulin n=2 Tax=Mus musculus TaxID=10090 RepID=X"
_ANONYMOUS_HEADER = "somedb|XYZ Uncharacterized protein"

#: What one `unpackdb` leaves behind, headers and all.
_UNPACKED = (
    f">P01308\n{_INSULIN}\n"
    f">{_PIG_HEADER}\n{_INSULIN.replace('MALWM', 'MALWT')}\n"
    f">{_MOUSE_HEADER}\n{_INSULIN.replace('MALWMR', 'MA--MR')}\n"
    f">{_ANONYMOUS_HEADER}\n{_INSULIN}\n"
)


@dataclass
class _Runs:
    """Every invocation the lane made, and what the query FASTA held."""

    calls: list[list[str]] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    #: Written into the unpack directory, so the parse and the rewriting run too.
    unpacked: str = ""


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> _Runs:
    """Replace `ExternalTool.run` on the base class and record what the lane did."""
    recorded = _Runs()

    def record(
        self: ExternalTool, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True
    ) -> str:
        recorded.calls.append(list(args))
        if args and args[0] == "createdb":
            query = Path(args[1])
            recorded.queries.append(query.read_text(encoding="utf-8") if query.is_file() else "")
        if args and args[0] == "unpackdb" and recorded.unpacked:
            (Path(args[2]) / "0.a3m").write_text(recorded.unpacked, encoding="utf-8")
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


def _call(runs: _Runs, verb: str) -> list[str]:
    """Return the one invocation of `verb`."""
    return next(call for call in runs.calls if call[0] == verb)


# --- the flags reaching the boundary -----------------------------------------


def test_the_lane_is_four_invocations_in_one_order(swissprot: Path, runs: _Runs) -> None:
    lane.search(_INSULIN, "swissprot")
    assert [call[0] for call in runs.calls] == ["createdb", "search", "result2msa", "unpackdb"]


def test_the_search_runs_against_the_database_the_caller_named(
    swissprot: Path, runs: _Runs
) -> None:
    lane.search(_INSULIN, "swissprot")
    assert _call(runs, "search")[2] == str(swissprot)
    assert _call(runs, "result2msa")[2] == str(swissprot)


def test_the_query_fasta_holds_the_sequence_under_the_name_it_was_given(
    swissprot: Path, runs: _Runs
) -> None:
    lane.search(_INSULIN, "swissprot", query_name="P01308")
    assert runs.queries == [f">P01308\n{_INSULIN}\n"]


def test_result2msa_is_asked_for_the_format_whose_headers_survive(
    swissprot: Path, runs: _Runs
) -> None:
    # The A3M modes replace a hit's header with its accession alone, which drops the organism
    # id that pairs the chains of a complex.
    lane.search(_INSULIN, "swissprot")
    call = _call(runs, "result2msa")
    assert call[call.index("--msa-format-mode") + 1] == "2"


def test_the_alignment_is_unpacked_by_key_with_an_a3m_suffix(swissprot: Path, runs: _Runs) -> None:
    lane.search(_INSULIN, "swissprot")
    call = _call(runs, "unpackdb")
    assert call[call.index("--unpack-suffix") + 1] == ".a3m"
    assert call[call.index("--unpack-name-mode") + 1] == "0"


def test_the_search_knobs_reach_the_search_verb_and_nothing_else(
    swissprot: Path, runs: _Runs
) -> None:
    lane.search(_INSULIN, "swissprot", sensitivity=7.5, threads=4)
    assert _call(runs, "search")[-4:] == ["-s", "7.5", "--threads", "4"]
    assert "-s" not in _call(runs, "result2msa")


def test_a_database_nothing_is_registered_under_never_starts_the_tool(
    liulab_data: Path, runs: _Runs
) -> None:
    with pytest.raises(LookupError):
        lane.search(_INSULIN, "swissprot")
    assert runs.calls == []


# --- the scratch directory ---------------------------------------------------


def test_every_intermediate_lives_in_the_scratch_directory_and_goes_with_it(
    swissprot: Path, runs: _Runs
) -> None:
    runs.unpacked = _UNPACKED
    lane.search(_INSULIN, "swissprot")
    work = Path(_call(runs, "unpackdb")[2]).parent
    assert work.name.startswith("mmseqs-msa-")
    assert not work.exists()


def test_nothing_durable_lands_in_the_data_dir(
    swissprot: Path, runs: _Runs, liulab_data: Path
) -> None:
    runs.unpacked = _UNPACKED
    lane.search(_INSULIN, "swissprot")
    assert list((liulab_data / "protein" / ".work").iterdir()) == []
    assert list(liulab_data.rglob("*.a3m")) == []


# --- what comes back ---------------------------------------------------------


def test_the_alignment_comes_back_in_memory(swissprot: Path, runs: _Runs) -> None:
    runs.unpacked = _UNPACKED
    found = lane.search(_INSULIN, "swissprot", query_name="P01308")
    assert isinstance(found, MSA)
    assert found.depth == 4
    assert found.query_header == "P01308"
    assert found.query == _INSULIN


def test_a_search_that_unpacked_nothing_answers_the_query_alone(
    swissprot: Path, runs: _Runs
) -> None:
    # A thin alignment is not refused; the lane returns what the search found.
    found = lane.search(_INSULIN, "swissprot", query_name="P01308")
    assert found.rows == (("P01308", _INSULIN),)


# --- the taxonomy the folding lane pairs on ----------------------------------


def test_a_uniprot_header_gains_the_key_the_folding_lane_reads(
    swissprot: Path, runs: _Runs
) -> None:
    runs.unpacked = _UNPACKED
    headers = [header for header, _ in lane.search(_INSULIN, "swissprot").rows]
    assert headers[1] == f"{_PIG_HEADER} key=9823"


def test_a_uniref_header_gains_one_too(swissprot: Path, runs: _Runs) -> None:
    runs.unpacked = _UNPACKED
    headers = [header for header, _ in lane.search(_INSULIN, "swissprot").rows]
    assert headers[2] == f"{_MOUSE_HEADER} key=10090"


def test_the_key_lands_where_the_folding_lane_looks_for_it(swissprot: Path, runs: _Runs) -> None:
    # ESMFold2 pairs chains by a `key=<int>` match over the header. A row without one is
    # unpaired and folds block-diagonal with nothing raised.
    runs.unpacked = _UNPACKED
    headers = [header for header, _ in lane.search(_INSULIN, "swissprot").rows]
    assert [re.search(r"key=(\d+)", header) is not None for header in headers] == [
        False,
        True,
        True,
        False,
    ]


def test_a_header_naming_no_organism_is_left_exactly_as_it_was(
    swissprot: Path, runs: _Runs
) -> None:
    runs.unpacked = _UNPACKED
    headers = [header for header, _ in lane.search(_INSULIN, "swissprot").rows]
    assert headers[3] == _ANONYMOUS_HEADER


def test_the_header_the_search_wrote_is_kept_in_front_of_the_key(
    swissprot: Path, runs: _Runs
) -> None:
    runs.unpacked = _UNPACKED
    headers = [header for header, _ in lane.search(_INSULIN, "swissprot").rows]
    assert headers[1].startswith(_PIG_HEADER)


def test_organism_id_reads_the_three_spellings_a_search_hands_back() -> None:
    assert organism_id(_PIG_HEADER) == 9823
    assert organism_id(_MOUSE_HEADER) == 10090
    assert organism_id("101 key=9606") == 9606
    assert organism_id(_ANONYMOUS_HEADER) is None


def test_a_header_that_already_carries_a_key_is_not_given_a_second_one() -> None:
    assert with_organism_key("101 key=9606") == "101 key=9606"


def test_a_field_that_merely_ends_in_an_equals_sign_is_not_an_organism() -> None:
    assert organism_id("UniRef100_X Cluster: y n=2 RepID=Z SV=1 PE=4") is None


# --- the mixin's peer on Protein ---------------------------------------------


def test_a_protein_searches_under_its_own_accession(swissprot: Path, runs: _Runs) -> None:
    Protein(_INSULIN, id="P01308").msa("swissprot")
    assert runs.queries == [f">P01308\n{_INSULIN}\n"]


def test_a_protein_without_an_accession_uses_the_default_query_name(
    swissprot: Path, runs: _Runs
) -> None:
    Protein(_INSULIN).msa("swissprot")
    assert runs.queries == [f">query\n{_INSULIN}\n"]


def test_a_protein_forwards_the_knobs_it_was_given(swissprot: Path, runs: _Runs) -> None:
    Protein(_INSULIN, id="P01308").msa("swissprot", max_seqs=5000)
    assert _call(runs, "search")[-2:] == ["--max-seqs", "5000"]


def test_a_protein_hands_back_the_alignment_it_parsed(swissprot: Path, runs: _Runs) -> None:
    runs.unpacked = _UNPACKED
    found = Protein(_INSULIN, id="P01308").msa("swissprot")
    assert isinstance(found, MSA)
    assert found.depth == 4


def test_the_database_is_required_and_defaults_to_nothing() -> None:
    with pytest.raises(TypeError):
        Protein(_INSULIN, id="P01308").msa()  # pyright: ignore[reportCallIssue]


def test_there_is_no_output_path_argument(swissprot: Path, runs: _Runs) -> None:
    # An alignment is a value, like a hit table. `MSA.write` is how one is kept.
    with pytest.raises(TypeError):
        Protein(_INSULIN, id="P01308").msa("swissprot", output="out.a3m")
