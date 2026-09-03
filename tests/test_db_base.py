"""Tests for the Database base: the layout, the record, and the two ways in.

No binary and no gigabytes. Every database here is a handful of files with the shapes the
real ones have — including pdb100's `_ss`, `_ca`, `_clu` and split `_seq` siblings, and the
symlink among them. Every tool call rides on one autouse
`monkeypatch.setattr(ExternalTool, "run", ...)`, which catches both adapters and both tools;
the version probe is patched beside it, because that one reaches the binary through
`_execute` rather than through `run`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from genome.store import completion

from protein.db import base
from protein.db.base import (
    GPU_ENCODED_BIT,
    Database,
    DatabaseStatus,
    SequenceDatabase,
    StructureDatabase,
    database_data_dir,
    database_files,
    database_path,
    is_gpu_encoded,
    registered_names,
)
from protein.external import ExternalTool, Foldseek, InstalledTool, ToolCall
from protein.search import mmseqs as search_mmseqs

#: A `db/pdb/` listing: the flat database, the siblings a search needs, the cluster
#: database, and the split full tier.
_PDB_STEMS = (
    "pdb100",
    "pdb100_h",
    "pdb100_ss",
    "pdb100_ca",
    "pdb100_clu",
    "pdb100_seq",
    "pdb100_seq_h",
    "pdb100_seq_ss",
    "pdb100_seq_ca",
)

_HEADER = "sp|P12345|AATM_RABIT Aspartate aminotransferase OS=Oryctolagus cuniculus OX=9986"


def _silent(call: ToolCall) -> str:
    """Answer any call with nothing, which is what most verbs write to stdout."""
    return ""


@dataclass
class _Runs:
    """Every call the tools were asked to make, and what stdout they get back.

    Attributes
    ----------
    calls : list of ToolCall
        In order, with the `capture` flag each was made under.
    answer : callable
        Given the call, returns its stdout. Reassign it to make a stand-in leave behind the
        files a real run would have written, or to answer a `view`.
    """

    calls: list[ToolCall] = field(default_factory=list)
    answer: Callable[[ToolCall], str] = _silent


@pytest.fixture(autouse=True)
def runs(monkeypatch: pytest.MonkeyPatch) -> _Runs:
    """Replace the process boundary for every tool, so nothing in this file needs a binary."""
    recorded = _Runs()

    def record(
        self: ExternalTool, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True
    ) -> str:
        call = ToolCall(tuple(args), cwd, capture)
        recorded.calls.append(call)
        return recorded.answer(call)

    monkeypatch.setattr(ExternalTool, "run", record)
    # `version` reaches `_execute` rather than `run`, and a record notes it.
    monkeypatch.setattr(InstalledTool, "_detect_version", lambda self: "18.8cc5c")
    return recorded


def _write_database(
    directory: Path,
    prefix: str,
    *,
    gpu: bool = False,
    entries: tuple[tuple[str, str], ...] = (("0", "P12345"),),
) -> Path:
    """Write a database-shaped file set and return its ffindex prefix."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / prefix).write_text("data", encoding="utf-8")
    (directory / f"{prefix}.dbtype").write_bytes(
        GPU_ENCODED_BIT.to_bytes(4, "little") if gpu else b"\x00\x00\x00\x00"
    )
    (directory / f"{prefix}.index").write_text(
        "".join(f"{key}\t0\t5\n" for key, _ in entries), encoding="utf-8"
    )
    (directory / f"{prefix}.lookup").write_text(
        "".join(f"{key}\t{name}\t0\n" for key, name in entries), encoding="utf-8"
    )
    (directory / f"{prefix}_h").write_text("headers", encoding="utf-8")
    (directory / f"{prefix}_h.dbtype").write_bytes(b"\x0c\x00\x00\x00")
    return directory / prefix


@pytest.fixture
def db_root(liulab_data: Path) -> Path:
    """Return the registry root under this test's own data dir."""
    return liulab_data / "protein" / "db"


@pytest.fixture
def swissprot(db_root: Path) -> Path:
    """Write a Swiss-Prot-shaped, GPU-encoded database and return its prefix."""
    return _write_database(db_root / "swissprot", "swissprot", gpu=True)


class _Sequence(SequenceDatabase):
    """A sequence database with no declaration behind it, for the base class's own tests."""


# --- the layout ---------------------------------------------------------------


def test_the_registry_lives_under_this_packages_own_root(liulab_data: Path) -> None:
    assert database_data_dir() == liulab_data / "protein" / "db"


def test_nothing_is_registered_when_the_root_does_not_exist() -> None:
    assert registered_names() == []


def test_every_directory_under_the_root_is_a_name(swissprot: Path, db_root: Path) -> None:
    (db_root / "pdb").mkdir()
    assert registered_names() == ["pdb", "swissprot"]


def test_a_directory_named_after_its_database_resolves_to_the_exact_spelling(
    swissprot: Path,
) -> None:
    assert database_path("swissprot") == swissprot


def test_a_directory_whose_database_is_spelled_differently_resolves_to_the_shortest_stem(
    db_root: Path,
) -> None:
    # Every other stem here is `pdb100` plus a suffix, so the shortest one is the database.
    directory = db_root / "pdb"
    directory.mkdir(parents=True)
    for stem in _PDB_STEMS:
        (directory / f"{stem}.dbtype").write_bytes(b"\x00\x00\x00\x00")
    assert database_path("pdb") == directory / "pdb100"


def test_a_name_nothing_is_registered_under_raises_and_names_what_is(swissprot: Path) -> None:
    with pytest.raises(LookupError, match="swissprot") as failed:
        database_path("uniref50")
    assert "uniref50" in str(failed.value)


def test_a_directory_holding_no_ffindex_database_raises(db_root: Path) -> None:
    (db_root / "empty").mkdir(parents=True)
    with pytest.raises(LookupError, match="no ffindex database"):
        database_path("empty")


def test_the_search_lane_resolves_a_name_through_this_module(swissprot: Path) -> None:
    assert search_mmseqs.database_path("swissprot") == swissprot


def test_a_database_is_a_search_target_and_answers_with_its_own_prefix(swissprot: Path) -> None:
    # One read-only `path` is the whole of `SearchTarget`, so `p.search(SwissProt())` and
    # `p.search("swissprot")` reach the same file.
    assert search_mmseqs.database_path(_Sequence("swissprot")) == swissprot


# --- the four dbtype bytes ------------------------------------------------------


def test_a_database_built_with_the_gpu_flag_reads_as_gpu_encoded(swissprot: Path) -> None:
    assert (swissprot.parent / "swissprot.dbtype").read_bytes() == b"\x00\x00\x08\x00"
    assert is_gpu_encoded(swissprot)


def test_a_plain_database_reads_as_not_gpu_encoded(db_root: Path) -> None:
    prefix = _write_database(db_root / "pdb", "pdb100")
    assert is_gpu_encoded(prefix) is False


def test_a_dbtype_that_is_not_there_reads_as_not_gpu_encoded(tmp_path: Path) -> None:
    assert is_gpu_encoded(tmp_path / "absent") is False


# --- what a record claims -------------------------------------------------------


def test_a_record_claims_the_siblings_a_structural_search_cannot_run_without(
    db_root: Path,
) -> None:
    directory = db_root / "pdb"
    _write_database(directory, "pdb100")
    for stem in ("pdb100_ss", "pdb100_ca"):
        (directory / stem).write_text("sibling", encoding="utf-8")
    claimed = {path.name for path in database_files(directory)}
    assert {"pdb100_ss", "pdb100_ca"} <= claimed


def test_a_record_does_not_claim_the_bookkeeping_beside_the_database(db_root: Path) -> None:
    directory = db_root / "swissprot"
    _write_database(directory, "swissprot")
    (directory / ".completion.json").write_text("{}", encoding="utf-8")
    (directory / ".work").mkdir()
    (directory / ".work" / "leftover").write_text("x", encoding="utf-8")
    claimed = {path.name for path in database_files(directory)}
    assert ".completion.json" not in claimed
    assert "leftover" not in claimed


def test_a_symlinked_split_file_is_claimed_as_the_file_it_resolves_to(db_root: Path) -> None:
    # Recorded at the size it resolves to, a tree copied with `rsync -a` still agrees.
    directory = db_root / "pdb"
    _write_database(directory, "pdb100")
    (directory / "pdb100_seq.0").symlink_to(directory / "pdb100")
    claimed = {path.name: path.stat().st_size for path in database_files(directory)}
    assert claimed["pdb100_seq.0"] == claimed["pdb100"]


# --- adopt ----------------------------------------------------------------------


def test_adopting_a_directory_already_in_place_writes_a_record_beside_it(
    swissprot: Path,
) -> None:
    record = _Sequence.adopt("swissprot", swissprot.parent).record
    assert record is not None
    assert record.kind == "database"
    assert record.name == "swissprot"
    assert "swissprot.index" in record.files


def test_adopting_records_the_tool_the_database_is_searched_with(swissprot: Path) -> None:
    record = _Sequence.adopt("swissprot", swissprot.parent).record
    assert record is not None
    assert record.details["tool"] == "mmseqs"
    assert record.details["kind"] == "sequence"
    assert record.details["prefix"] == "swissprot"


def test_adopting_records_that_the_residues_were_folded(swissprot: Path) -> None:
    record = _Sequence.adopt("swissprot", swissprot.parent).record
    assert record is not None
    assert record.details["gpu_encoded"] is True


def test_adopting_notes_the_version_of_the_tool_it_was_handed(swissprot: Path) -> None:
    # The seam is the object's own tool, not a second one reached through liulab-genome.
    record = _Sequence.adopt("swissprot", swissprot.parent, tool=Foldseek()).record
    assert record is not None
    assert record.details["tool_version"] == "18.8cc5c"


def test_adopting_accepts_the_ffindex_prefix_as_well_as_the_directory(swissprot: Path) -> None:
    assert _Sequence.adopt("swissprot", swissprot).is_registered


def test_adopting_a_database_that_lives_elsewhere_makes_the_name_address_it(
    tmp_path: Path, db_root: Path
) -> None:
    elsewhere = tmp_path / "shared" / "swissprot"
    _write_database(elsewhere, "swissprot")
    database = _Sequence.adopt("swissprot", elsewhere)
    assert (db_root / "swissprot").is_symlink()
    assert database.path == db_root / "swissprot" / "swissprot"
    assert (elsewhere / ".completion.json").is_file()


def test_adopting_over_a_name_that_already_addresses_something_else_refuses(
    tmp_path: Path, swissprot: Path
) -> None:
    elsewhere = tmp_path / "shared" / "swissprot"
    _write_database(elsewhere, "swissprot")
    with pytest.raises(FileExistsError, match="already exists"):
        _Sequence.adopt("swissprot", elsewhere)


def test_adopting_a_database_that_is_already_registered_writes_nothing_new(
    swissprot: Path,
) -> None:
    _Sequence.adopt("swissprot", swissprot.parent)
    marker = completion.record_path(swissprot.parent)
    written = marker.stat().st_mtime_ns
    assert _Sequence.adopt("swissprot", swissprot.parent).is_registered
    assert marker.stat().st_mtime_ns == written


def test_adopting_a_database_whose_files_changed_underneath_the_record_refuses(
    swissprot: Path,
) -> None:
    _Sequence.adopt("swissprot", swissprot.parent)
    swissprot.write_text("a longer data file than the record claims", encoding="utf-8")
    with pytest.raises(completion.RegistrationMismatchError, match="disagrees"):
        _Sequence.adopt("swissprot", swissprot.parent)


def test_forcing_an_adoption_writes_a_record_over_the_one_that_disagreed(
    swissprot: Path,
) -> None:
    _Sequence.adopt("swissprot", swissprot.parent)
    swissprot.write_text("a longer data file than the record claims", encoding="utf-8")
    record = _Sequence.adopt("swissprot", swissprot.parent, force=True).record
    assert record is not None
    assert record.files["swissprot"] == swissprot.stat().st_size


def test_adopting_a_directory_that_holds_no_database_raises(tmp_path: Path) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(LookupError, match="no ffindex database"):
        _Sequence.adopt("swissprot", empty)


def test_adopting_a_path_that_is_neither_a_directory_nor_a_prefix_raises(tmp_path: Path) -> None:
    with pytest.raises(LookupError, match="neither a directory"):
        _Sequence.adopt("swissprot", tmp_path / "absent")


# --- download -------------------------------------------------------------------


@pytest.fixture
def downloads(runs: _Runs) -> _Runs:
    """Make every `databases` call leave a database where it was told to put one."""

    def build(call: ToolCall) -> str:
        if call.args[0] == "databases":
            target = Path(call.args[2])
            _write_database(target.parent, target.name)
        return ""

    runs.answer = build
    return runs


def test_downloading_hands_the_tool_its_own_spelling_and_a_work_directory(
    downloads: _Runs, db_root: Path
) -> None:
    _Sequence.download("swissprot", source="UniProtKB/Swiss-Prot")
    verb, source, target, work = downloads.calls[0].args
    assert (verb, source) == ("databases", "UniProtKB/Swiss-Prot")
    assert Path(target) == db_root / "swissprot" / "swissprot"
    assert Path(work) == db_root / "swissprot" / ".work"


def test_downloading_streams_the_tools_progress_rather_than_capturing_it(
    downloads: _Runs,
) -> None:
    # Hours and gigabytes: the tool's own progress belongs on the terminal.
    _Sequence.download("swissprot", source="UniProtKB/Swiss-Prot")
    assert downloads.calls[0].capture is False


def test_downloading_writes_a_record_and_then_clears_the_work_directory(
    downloads: _Runs, db_root: Path
) -> None:
    database = _Sequence.download("swissprot", source="UniProtKB/Swiss-Prot")
    assert database.is_registered
    assert not (db_root / "swissprot" / ".work").exists()


def test_downloading_a_name_that_is_already_registered_runs_nothing(
    swissprot: Path, downloads: _Runs
) -> None:
    _Sequence.adopt("swissprot", swissprot.parent)
    downloads.calls.clear()
    _Sequence.download("swissprot", source="UniProtKB/Swiss-Prot")
    assert downloads.calls == []


def test_forcing_a_download_runs_the_tool_again(swissprot: Path, downloads: _Runs) -> None:
    _Sequence.adopt("swissprot", swissprot.parent)
    downloads.calls.clear()
    _Sequence.download("swissprot", source="UniProtKB/Swiss-Prot", force=True)
    assert downloads.calls[0].args[0] == "databases"


def test_downloading_a_name_with_no_source_says_what_to_do_instead() -> None:
    with pytest.raises(ValueError, match="no download source"):
        _Sequence.download("uniref50")


def test_a_database_with_no_default_name_and_none_given_refuses_to_be_built() -> None:
    with pytest.raises(ValueError, match="no default name"):
        _Sequence()


# --- status ---------------------------------------------------------------------


def test_status_of_a_name_that_is_not_here_says_so_without_raising() -> None:
    found = _Sequence("uniref50").status()
    assert found.registered is False
    assert found.path is None
    assert found.index_entries is None


def test_status_reports_both_entry_counts_and_names_each_file_it_counted(
    db_root: Path,
) -> None:
    # A structure database names far more chains than it makes searchable, and reporting one
    # count without saying which is how the wrong number gets quoted.
    directory = db_root / "pdb"
    prefix = _write_database(
        directory, "pdb100", entries=(("0", "1ubq-assembly1_A"), ("1", "201l-assembly1_A"))
    )
    (directory / "pdb100.lookup").write_text(
        "0\t1ubq-assembly1_A\t0\n1\t201l-assembly1_A\t0\n2\t201l-assembly2_B\t0\n",
        encoding="utf-8",
    )
    found = StructureDatabase("pdb").status()
    assert found.index_entries == 2
    assert found.lookup_entries == 3
    assert found.path == str(prefix)


def test_status_carries_the_fold_in_the_callers_terms_when_the_database_is_gpu_encoded(
    swissprot: Path,
) -> None:
    found = _Sequence("swissprot").status()
    assert found.is_gpu_encoded is True
    assert found.residue_fold is not None
    assert "B->D" in found.residue_fold


def test_status_says_nothing_about_a_fold_that_did_not_happen(db_root: Path) -> None:
    _write_database(db_root / "pdb", "pdb100")
    found = StructureDatabase("pdb").status()
    assert found.is_gpu_encoded is False
    assert found.residue_fold is None


def test_status_counts_the_files_a_finished_registration_claims(swissprot: Path) -> None:
    _Sequence.adopt("swissprot", swissprot.parent)
    found = _Sequence("swissprot").status()
    assert found.registered is True
    assert found.files == 6
    assert found.completed_at is not None


def test_status_counts_what_is_there_before_anything_is_registered(swissprot: Path) -> None:
    found = _Sequence("swissprot").status()
    assert found.registered is False
    assert found.files == 6
    assert found.bytes is not None


def test_a_status_renders_as_json_with_one_key_per_attribute() -> None:
    payload = DatabaseStatus(name="swissprot", directory="/d", tool="mmseqs").as_json()
    assert payload["name"] == "swissprot"
    assert payload["lookup_entries"] is None


# --- one entry out of a sequence database ----------------------------------------


@pytest.fixture
def views(runs: _Runs) -> _Runs:
    """Answer `view` with a header or a sequence, as the flags ask for one or the other."""
    runs.answer = lambda call: (
        f"{_HEADER}\n" if "--idx-entry-type" in call.args else "MALLHSARVLSG\n"
    )
    return runs


def test_an_accession_resolves_to_the_opaque_numeric_key_the_lookup_carries(
    db_root: Path,
) -> None:
    # `createdb` shuffles, so the key has no relation to the accession.
    _write_database(
        db_root / "swissprot", "swissprot", entries=(("7", "P0A031"), ("415743", "P12345"))
    )
    assert _Sequence("swissprot").key_for("P12345") == "415743"


def test_an_accession_the_database_does_not_carry_raises_rather_than_answering_empty(
    swissprot: Path,
) -> None:
    # mmseqs warns on stderr and exits 0 for a missing name, so absence is detected here.
    with pytest.raises(KeyError, match="Q99999"):
        _Sequence("swissprot").key_for("Q99999")


def test_membership_is_answered_from_the_lookup_alone(swissprot: Path, runs: _Runs) -> None:
    database = _Sequence("swissprot")
    assert "P12345" in database
    assert "Q99999" not in database
    assert runs.calls == []


def test_retrieving_one_entry_asks_for_the_header_by_key_and_then_the_sequence(
    swissprot: Path, views: _Runs
) -> None:
    header, sequence = _Sequence("swissprot").entry("P12345")
    assert header.startswith("sp|P12345|AATM_RABIT")
    assert sequence == "MALLHSARVLSG"
    first, second = (call.args for call in views.calls)
    assert first[:4] == ("view", str(swissprot), "--id-list", "0")
    assert first[4:] == ("--idx-entry-type", "2")
    assert second == ("view", str(swissprot), "--id-list", "0")


def test_one_entry_comes_back_as_a_protein_carrying_its_header(
    swissprot: Path, views: _Runs
) -> None:
    entry = _Sequence("swissprot")["P12345"]
    assert str(entry.sequence) == "MALLHSARVLSG"
    assert entry.id == "sp|P12345|AATM_RABIT"
    assert entry.metadata["database"] == "swissprot"


# --- what the surface is, and is not ----------------------------------------------


def _surface(cls: type) -> set[str]:
    """Return the public names a class offers."""
    return {name for name in dir(cls) if not name.startswith("_")}


def test_a_database_offers_no_verb_that_would_change_one() -> None:
    # These are immutable: the index holds byte offsets into the data file, so editing the
    # data breaks every offset. The surface is pinned, so adding a verb is a decision.
    assert _surface(Database) == {
        "NAME",
        "SOURCE",
        "adopt",
        "directory",
        "download",
        "is_gpu_encoded",
        "is_registered",
        "path",
        "record",
        "status",
        "tool",
    }


def test_a_structure_database_adds_its_tool_and_nothing_else() -> None:
    # No `__getitem__`: coordinates come from `protein.structure.fetch`, not from here.
    assert _surface(StructureDatabase) - _surface(Database) == {"KIND", "TOOL_NAME"}
    assert not hasattr(StructureDatabase, "__getitem__")


def test_a_sequence_database_adds_retrieval_because_mmseqs_supports_it_directly() -> None:
    added = _surface(SequenceDatabase) - _surface(Database)
    assert added == {"KIND", "TOOL_NAME", "entry", "key_for"}
    assert hasattr(SequenceDatabase, "__getitem__")


def test_the_layout_is_spelled_once_and_the_search_lane_no_longer_spells_it() -> None:
    assert base.DATABASE_SUBDIR == "db"
    assert not hasattr(search_mmseqs, "DATABASE_SUBDIR")
