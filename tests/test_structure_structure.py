"""Tests for `Structure` — the cache, the lazy parse, the chains, and the search.

The coordinate cache is exercised for real: the suite's data root is this test's own
directory, so a fixture copied into `<root>/protein/structures/` is a cache hit and the
network is never reached. What a miss does is checked by patching `rcsb.fetch`, the one call
this package makes to RCSB.
"""

from __future__ import annotations

import gzip
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from biotite.database import rcsb

from protein import Structure
from protein.external import ExternalTool, Foldseek
from protein.structure import CoordinatesNotDownloadedError, cached_path, fetch
from protein.structure import structure as lane

_DATA = Path(__file__).resolve().parent / "data"
_UBQ = _DATA / "1ubq.cif.gz"
_BNA = _DATA / "1bna.cif.gz"
_TRP = _DATA / "1l2y_2models.pdb.gz"


@pytest.fixture
def cache(liulab_data: Path) -> Path:
    """The coordinate cache directory for this test, made but empty."""
    directory = liulab_data / "protein" / "structures"
    directory.mkdir(parents=True)
    return directory


@pytest.fixture
def cached_ubq(cache: Path) -> Path:
    """`1UBQ` in the cache, decompressed, under the name a fetch would have written."""
    target = cache / "1ubq.cif"
    target.write_bytes(gzip.decompress(_UBQ.read_bytes()))
    return target


@dataclass
class _Runs:
    """Every call an `easy-search` made."""

    calls: list[list[str]] = field(default_factory=list)
    queries: list[bytes] = field(default_factory=list)


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> _Runs:
    """Replace `ExternalTool.run` and record what a search was asked to do."""
    recorded = _Runs()

    def record(
        self: ExternalTool, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True
    ) -> str:
        recorded.calls.append(list(args))
        if args and args[0] == "easy-search":
            # Read while the scratch directory is still there; it is removed when the search
            # returns.
            recorded.queries.append(Path(args[1]).read_bytes())
        return ""

    monkeypatch.setattr(ExternalTool, "run", record)
    return recorded


@pytest.fixture
def pdb_database(liulab_data: Path) -> Path:
    """A pdb100-shaped database registered under this test's own data root."""
    directory = liulab_data / "protein" / "db" / "pdb"
    directory.mkdir(parents=True)
    for stem in ("pdb100", "pdb100_ss", "pdb100_ca"):
        (directory / f"{stem}.dbtype").write_bytes(b"\x00\x00\x00\x00")
    return directory / "pdb100"


# --- construction --------------------------------------------------------------


def test_constructing_a_structure_reads_nothing_and_fetches_nothing() -> None:
    # The network guard is autouse, so a constructor that reached RCSB would fail here.
    assert repr(Structure("1UBQ")) == "Structure('1UBQ')"


def test_the_id_is_kept_exactly_as_it_was_given() -> None:
    assert Structure("1UBQ").id == "1UBQ"
    assert Structure("1ubq").id == "1ubq"


def test_from_file_names_the_structure_after_the_file() -> None:
    assert Structure.from_file(_UBQ).id == "1ubq"


def test_from_file_takes_an_id_when_the_file_is_not_named_after_its_entry() -> None:
    assert Structure.from_file(_TRP, id="1L2Y").id == "1L2Y"


def test_from_file_refuses_a_path_that_is_not_there(tmp_path: Path) -> None:
    # A lazy parse would otherwise report a typo'd path at whichever line first asked for an
    # atom, nowhere near where it was typed.
    with pytest.raises(FileNotFoundError, match="is not a file"):
        Structure.from_file(tmp_path / "nothing.cif")


# --- the accessions a structure was produced from ------------------------------


def test_a_structure_carries_no_accessions_unless_it_is_given_some() -> None:
    assert Structure("1UBQ").accessions is None


def test_a_structure_read_off_disk_carries_no_accession_map() -> None:
    # A documented limit and not an oversight: provenance does not survive the file, so a
    # reopened prediction falls back to SIFTS like any other entry (ADR-0005).
    structure = Structure.from_file(_UBQ)
    assert structure.accessions is None
    assert structure.id == "1ubq"


def test_the_accessions_a_structure_was_produced_from_are_frozen_into_tuples() -> None:
    structure = Structure("folded", accessions={"A": ["P12345", "P67890"], "B": []})
    assert structure.accessions == {"A": ("P12345", "P67890"), "B": ()}


def test_a_chain_mapped_to_a_bare_string_is_refused_rather_than_read_letter_by_letter() -> None:
    with pytest.raises(TypeError, match=r"\['A'\] map to a str"):
        Structure("folded", accessions={"A": "P12345"})


# --- the coordinate cache ------------------------------------------------------


def test_the_cache_lives_beside_the_other_prepared_things(liulab_data: Path) -> None:
    assert lane.structure_data_dir() == liulab_data / "protein" / "structures"


def test_a_cached_entry_is_found_whatever_case_the_id_was_asked_in(cached_ubq: Path) -> None:
    assert cached_path("1UBQ") == cached_ubq
    assert cached_path("1ubq") == cached_ubq


def test_a_gzipped_cache_entry_is_found_too_which_is_what_an_rsync_mirror_leaves(
    cache: Path,
) -> None:
    packed = cache / "1ubq.cif.gz"
    packed.write_bytes(_UBQ.read_bytes())
    assert cached_path("1UBQ") == packed


def test_an_empty_cache_file_reads_as_absent(cache: Path) -> None:
    # How an interrupted download is repaired.
    (cache / "1ubq.cif").write_bytes(b"")
    assert cached_path("1UBQ") is None


def test_nothing_cached_means_nothing_found_rather_than_an_error(cache: Path) -> None:
    assert cached_path("1ubq") is None


def test_a_cache_hit_reaches_no_further(cached_ubq: Path) -> None:
    assert fetch("1UBQ") == cached_ubq


def test_a_miss_asks_rcsb_for_the_lower_cased_id_and_caches_it_there(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: dict[str, object] = {}

    def fake_fetch(entry: str, suffix: str, target_path: str = "", **kwargs: object) -> str:
        asked.update(id=entry, format=suffix, target=target_path)
        written = Path(target_path) / f"{entry}.{suffix}"
        written.write_bytes(gzip.decompress(_UBQ.read_bytes()))
        return str(written)

    monkeypatch.setattr(rcsb, "fetch", fake_fetch)
    assert fetch("1UBQ") == cache / "1ubq.cif"
    assert asked == {"id": "1ubq", "format": "cif", "target": str(cache)}


def test_a_fetch_that_cannot_happen_names_the_command_that_fixes_it(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> str:
        raise OSError("no route to host")

    monkeypatch.setattr(rcsb, "fetch", refuse)
    with pytest.raises(CoordinatesNotDownloadedError, match="protein structure fetch 1UBQ"):
        fetch("1UBQ")


def test_a_structure_with_no_cached_file_really_does_try_the_network(cache: Path) -> None:
    # The guard blocks it and names the URL, which is the proof that a miss is a download
    # rather than a silent empty answer.
    with pytest.raises(RuntimeError, match="blocked network call"):
        _ = Structure("1ubq").atoms


# --- the lazy parse ------------------------------------------------------------


def test_the_path_of_a_bare_id_is_resolved_through_the_cache(cached_ubq: Path) -> None:
    assert Structure("1UBQ").path == cached_ubq


def test_the_atoms_are_parsed_once_and_then_held() -> None:
    structure = Structure.from_file(_UBQ)
    assert structure.atoms is structure.atoms


def test_the_first_model_is_what_atoms_means() -> None:
    assert Structure.from_file(_TRP).atoms.array_length() == 304


def test_the_models_are_a_stack_of_their_own_and_not_a_slice_of_atoms() -> None:
    structure = Structure.from_file(_TRP)
    assert structure.models.stack_depth() == 2
    assert structure.models is structure.models


# --- chains --------------------------------------------------------------------


def test_the_chains_of_an_entry_are_listed_once_each_in_file_order() -> None:
    assert Structure.from_file(_BNA).chain_ids == ("A", "B")


def test_indexing_reaches_a_chain() -> None:
    structure = Structure.from_file(_UBQ)
    assert structure["A"].chain_id == "A"


def test_a_label_this_entry_does_not_carry_names_the_ones_it_does() -> None:
    structure = Structure.from_file(_BNA, id="1bna")
    with pytest.raises(KeyError, match=r"1bna has no chain 'Z'.*'A', 'B'"):
        _ = structure["Z"]


def test_a_label_of_the_wrong_case_is_a_different_chain() -> None:
    # `10EG` carries both an `A` and an `a`, so folding here would answer for the wrong one.
    with pytest.raises(KeyError):
        _ = Structure.from_file(_BNA)["a"]


def test_membership_answers_without_indexing() -> None:
    structure = Structure.from_file(_BNA)
    assert "A" in structure
    assert "Z" not in structure


def test_the_chains_property_is_one_chain_per_label() -> None:
    chains = Structure.from_file(_BNA).chains
    assert [chain.chain_id for chain in chains] == ["A", "B"]


# --- search --------------------------------------------------------------------


def test_a_structure_search_passes_its_own_file_and_runs_once(
    pdb_database: Path, runs: _Runs
) -> None:
    # One invocation and not a loop: Foldseek fans a multi-chain query out itself.
    structure = Structure.from_file(_BNA)
    structure.search("pdb")
    assert len(runs.calls) == 1
    verb, query, target = runs.calls[0][:3]
    assert (verb, query, target) == ("easy-search", str(_BNA), str(pdb_database))


def test_a_structure_search_forwards_the_knobs(pdb_database: Path, runs: _Runs) -> None:
    Structure.from_file(_UBQ).search("pdb", sensitivity=1.0, threads=4)
    assert runs.calls[0][-4:] == ["-s", "1.0", "--threads", "4"]


def test_a_structure_search_takes_the_tool_it_is_given(pdb_database: Path, runs: _Runs) -> None:
    Structure.from_file(_UBQ).search("pdb", tool=Foldseek())
    assert runs.calls[0][6] == Foldseek().format_output
