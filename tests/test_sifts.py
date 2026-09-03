"""Tests for protein.sifts — the PDB-UniProt map, and what each direction answers with.

Everything runs over `tests/data/sifts_pdb_chain_uniprot_slice.tsv`, real rows whose release
is named in `tests/data/README.md`. `genome.store.fetch.fetch_url` is monkeypatched through
its module, so the pipeline runs whole against the slice and the network is never reached.

The claims worth holding are that the reader keeps seven columns and records which release
it read, that a chain answers with a tuple and an accession with a frame, that both ranges
come back verbatim, and that an unprepared set raises rather than answering nothing.
"""

from __future__ import annotations

import gzip
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from genome.store import completion, fetch
from typer.testing import CliRunner

from protein import sifts
from protein.cli import app as root_app
from protein.core import Protein
from protein.sifts import SiftsFormatError, SiftsNotDownloadedError, app

from . import plain_text

_SLICE = Path(__file__).parent / "data" / "sifts_pdb_chain_uniprot_slice.tsv"

#: The release the slice was cut from, verbatim from its own first line.
_RELEASE_LINE = "# 2026/08/30 - 13:24 | PDB: 35.26 | UniProt: 2026.03"

#: The publisher's column header, tab-separated, as the reader checks it.
_COLUMN_LINE = "PDB\tCHAIN\tSP_PRIMARY\tRES_BEG\tRES_END\tPDB_BEG\tPDB_END\tSP_BEG\tSP_END"


def _lines(*rows: str) -> Iterator[str]:
    """Return the publisher's line stream for a file made of ``rows``, CRLF and all."""
    return iter([f"{_RELEASE_LINE}\n", f"{_COLUMN_LINE}\r\n", *(f"{row}\r\n" for row in rows)])


@pytest.fixture
def publisher_file(tmp_path: Path) -> Path:
    """The committed slice, gzipped the way the publisher serves it."""
    packed = tmp_path / "publisher" / "pdb_chain_uniprot.tsv.gz"
    packed.parent.mkdir(parents=True)
    with gzip.GzipFile(packed, "wb", mtime=0) as out:
        out.write(_SLICE.read_bytes())
    return packed


@pytest.fixture
def prepared_sifts(monkeypatch: pytest.MonkeyPatch, publisher_file: Path) -> Path:
    """Run the whole prepared-set pipeline against the slice, and return the stored file."""

    def fake_fetch(url: str, dest_dir: Path, **kwargs: Any) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / (kwargs.get("fname") or "pdb_chain_uniprot.tsv.gz")
        shutil.copyfile(publisher_file, target)
        return target

    monkeypatch.setattr(fetch, "fetch_url", fake_fetch)
    return sifts.prepare(progressbar=False).path


# --- the source ---------------------------------------------------------------


def test_the_set_lands_in_its_own_directory_under_this_packages_root(tmp_path: Path) -> None:
    assert sifts.sifts_data_dir() == tmp_path / "protein" / "sifts"
    assert sifts.source().path.name == sifts.STORED_NAME


def test_the_source_pins_nothing_because_the_publisher_overwrites_in_place() -> None:
    # There is no archive: the weekly file is replaced under the same name, so a digest
    # taken today would reject every release after it.
    assert sifts.source().checksum is None


def test_the_repair_is_delete_and_rebuild_and_names_the_prepare_command() -> None:
    assert sifts.source().repair.endswith(sifts.PREPARE_COMMAND)


# --- the reader ---------------------------------------------------------------


def test_the_reader_records_which_release_it_read(tmp_path: Path) -> None:
    measured = sifts.read_sifts(
        _lines("101m\tA\tP02185\t1\t154\t0\t153\t1\t154"),
        tmp_path / "out.tsv.gz",
        origin="slice",
    )
    assert measured["sifts_header"] == _RELEASE_LINE
    assert measured["sifts_released"] == "2026/08/30 - 13:24"
    assert measured["pdb_release"] == "35.26"
    assert measured["uniprot_release"] == "2026.03"
    assert measured["rows"] == 1


def test_the_reader_keeps_seven_columns_and_drops_the_author_numbering(
    prepared_sifts: Path,
) -> None:
    assert tuple(sifts.table().columns) == sifts.COLUMNS
    assert "pdb_beg" not in sifts.table().columns
    assert "pdb_end" not in sifts.table().columns


def test_the_reader_strips_the_carriage_returns_the_publisher_writes(
    prepared_sifts: Path,
) -> None:
    # Line endings are mixed in the source. Left on, a CR would ride the last column into
    # the store and `sp_end` would not parse as an integer.
    frame = sifts.table()
    for column in ("pdb", "chain", "accession"):
        assert not frame[column].str.contains("\r").any()


def test_the_stored_frame_is_sorted_by_accession_then_entry_then_chain(
    prepared_sifts: Path,
) -> None:
    frame = sifts.table()
    keys = list(zip(*(frame[name] for name in sifts.SORT_COLUMNS), strict=True))
    assert keys == sorted(keys)


def test_the_reader_refuses_a_file_that_does_not_begin_with_a_release_line(
    tmp_path: Path,
) -> None:
    with pytest.raises(SiftsFormatError, match="release line"):
        sifts.read_sifts(iter(["PDB\tCHAIN\n"]), tmp_path / "out.tsv.gz", origin="slice")


def test_the_reader_refuses_a_file_whose_columns_were_re_shaped(tmp_path: Path) -> None:
    with pytest.raises(SiftsFormatError, match="by position"):
        sifts.read_sifts(
            iter([f"{_RELEASE_LINE}\n", "PDB\tCHAIN\tSP_PRIMARY\r\n"]),
            tmp_path / "out.tsv.gz",
            origin="slice",
        )


def test_the_reader_refuses_a_release_that_carries_no_rows(tmp_path: Path) -> None:
    with pytest.raises(SiftsFormatError, match="no mapping rows"):
        sifts.read_sifts(_lines(), tmp_path / "out.tsv.gz", origin="slice")


def test_the_reader_refuses_a_residue_bound_that_is_not_an_integer(tmp_path: Path) -> None:
    with pytest.raises(SiftsFormatError, match="res_beg"):
        sifts.read_sifts(
            _lines("101m\tA\tP02185\t\t154\t0\t153\t1\t154"),
            tmp_path / "out.tsv.gz",
            origin="slice",
        )


def test_the_reader_refuses_a_row_with_the_wrong_number_of_fields(tmp_path: Path) -> None:
    with pytest.raises(SiftsFormatError, match="fields"):
        sifts.read_sifts(_lines("101m\tA\tP02185"), tmp_path / "out.tsv.gz", origin="slice")


def test_nothing_is_placed_when_the_reader_refuses(tmp_path: Path) -> None:
    staged = tmp_path / "out.tsv.gz"
    with pytest.raises(SiftsFormatError):
        sifts.read_sifts(_lines("101m\tA"), staged, origin="slice")
    assert not staged.exists()


# --- the pipeline -------------------------------------------------------------


def test_the_marker_records_the_release_so_any_result_can_name_its_sifts(
    prepared_sifts: Path,
) -> None:
    record = completion.read_record(sifts.sifts_data_dir())
    assert record is not None
    assert record.kind == "sifts"
    assert record.source_url == sifts.SIFTS_URL
    assert record.details["pdb_release"] == "35.26"
    assert record.details["uniprot_release"] == "2026.03"
    assert record.details["rows"] == 91


def test_preparing_a_set_that_is_already_here_fetches_nothing(
    monkeypatch: pytest.MonkeyPatch, prepared_sifts: Path
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("a prepared set must not be fetched again")

    monkeypatch.setattr(fetch, "fetch_url", refuse)
    assert sifts.prepare(progressbar=False).path == prepared_sifts


def test_a_fetch_this_machine_cannot_make_names_the_login_node_and_the_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreachable(*args: Any, **kwargs: Any) -> Path:
        raise OSError("no route to host")

    monkeypatch.setattr(fetch, "fetch_url", unreachable)
    with pytest.raises(SiftsNotDownloadedError) as raised:
        sifts.prepare(progressbar=False)
    assert sifts.PREPARE_COMMAND in str(raised.value)
    assert "login node" in str(raised.value)


# --- accessions_for: a chain to its proteins -----------------------------------


def test_a_chain_answers_with_the_accession_sifts_curated_not_the_one_the_mmcif_froze(
    prepared_sifts: Path,
) -> None:
    # 1UBQ chain A is P62988 in the file's own `_struct_ref_seq` and P0CG48 here.
    assert sifts.accessions_for("1ubq", "A") == ("P0CG48",)


def test_an_entry_id_is_folded_to_the_lower_case_sifts_stores(prepared_sifts: Path) -> None:
    assert sifts.accessions_for("1UBQ", "A") == sifts.accessions_for("1ubq", "A")


def test_a_chain_label_is_not_folded_because_case_is_part_of_the_name(
    prepared_sifts: Path,
) -> None:
    # `10eg` carries both an `A` and an `a`, so case is part of the name.
    assert sifts.accessions_for("10eg", "a") == ("P0A405",)
    assert sifts.accessions_for("10lk", "Q1") == ("Q72RA0",)
    assert sifts.accessions_for("10lk", "q1") == ()


def test_a_chain_labelled_na_is_a_chain_and_not_a_missing_value(prepared_sifts: Path) -> None:
    # `NA` is a real chain label. Read with pandas' default missing-value list it becomes
    # null and the chain is unreachable; `na_filter=False` does not fix it, because the
    # pyarrow engine ignores that spelling, so the loader passes `keep_default_na=False`.
    assert sifts.accessions_for("9on4", "NA") == ("P06702",)
    assert int(sifts.table()["chain"].isna().sum()) == 0


def test_a_chain_carrying_four_accessions_answers_with_all_four(prepared_sifts: Path) -> None:
    # Accession order, which is the order the stored slice is sorted in — not the
    # publisher's row order, which is by residue range.
    assert sifts.accessions_for("8uqe", "B") == ("P04908", "P61077", "Q16778", "Q8IYW5")


def test_several_segments_of_one_chain_answer_with_one_accession_not_several(
    prepared_sifts: Path,
) -> None:
    # A tuple of accessions is not a tuple of rows.
    assert sifts.accessions_for("102l", "A") == ("P00720",)


def test_a_chain_sifts_does_not_carry_answers_empty_rather_than_raising(
    prepared_sifts: Path,
) -> None:
    # What a chain of a `Structure.from_file` that is no PDB entry gets, and what every
    # nucleic-acid and ligand chain gets. It is an answer, not a failure.
    assert sifts.accessions_for("1ubq", "Z") == ()
    assert sifts.accessions_for("zzzz", "A") == ()


# --- structures_for: a protein to its chains -----------------------------------


def test_an_accession_answers_with_every_segment_it_reaches(prepared_sifts: Path) -> None:
    frame = sifts.structures_for("P0CG48")
    assert tuple(frame.columns) == sifts.COLUMNS
    assert set(frame["pdb"]) == {"11sy", "1cmx", "1ubq"}


def test_an_accession_is_folded_to_the_upper_case_sifts_stores(prepared_sifts: Path) -> None:
    assert sifts.structures_for("p0cg48").equals(sifts.structures_for("P0CG48"))


def test_several_segments_of_one_triple_are_several_rows(prepared_sifts: Path) -> None:
    frame = sifts.structures_for("P00720")
    segments = frame[(frame["pdb"] == "102l") & (frame["chain"] == "A")]
    assert list(segments["res_beg"]) == [1, 42]
    assert list(segments["res_end"]) == [40, 165]


def test_both_ranges_come_back_verbatim_when_no_offset_is_definable(
    prepared_sifts: Path,
) -> None:
    # The two ranges are different lengths, so no single integer shift exists. Some segments
    # are always like this, which is why neither range is adjusted.
    row = sifts.structures_for("Q12791").iloc[0]
    assert (row["res_beg"], row["res_end"]) == (1, 1113)
    assert (row["sp_beg"], row["sp_end"]) == (66, 1236)
    assert row["res_end"] - row["res_beg"] != row["sp_end"] - row["sp_beg"]


def test_an_accession_sifts_does_not_carry_answers_an_empty_frame(prepared_sifts: Path) -> None:
    empty = sifts.structures_for("P99999")
    assert empty.empty
    assert tuple(empty.columns) == sifts.COLUMNS


# --- the table and its cache ---------------------------------------------------


def test_the_table_is_read_once_and_then_held(prepared_sifts: Path) -> None:
    assert sifts.table() is sifts.table()


def test_clearing_the_cache_re_reads_from_disk(prepared_sifts: Path) -> None:
    first = sifts.table()
    sifts.clear_cache()
    assert sifts.table() is not first


def test_an_unprepared_set_raises_rather_than_answering_nothing() -> None:
    # The distinction the whole module rests on: `()` means this chain has no protein, and
    # a missing set means nobody has prepared it. A script must not confuse the two.
    with pytest.raises(SiftsNotDownloadedError):
        sifts.accessions_for("1ubq", "A")
    with pytest.raises(SiftsNotDownloadedError):
        sifts.structures_for("P0CG48")


def test_the_missing_set_error_names_the_command_that_prepares_it() -> None:
    with pytest.raises(SiftsNotDownloadedError) as raised:
        sifts.table()
    assert sifts.PREPARE_COMMAND in str(raised.value)
    assert "login node" in str(raised.value)


# --- status --------------------------------------------------------------------


def test_status_says_nothing_is_prepared_when_nothing_is(tmp_path: Path) -> None:
    found = sifts.status()
    assert found.prepared is False
    assert found.rows is None
    assert found.path == tmp_path / "protein" / "sifts" / sifts.STORED_NAME


def test_status_reads_the_marker_and_names_the_release(prepared_sifts: Path) -> None:
    found = sifts.status()
    assert found.prepared is True
    assert found.pdb_release == "35.26"
    assert found.uniprot_release == "2026.03"
    assert found.released == "2026/08/30 - 13:24"
    assert found.rows == 91
    assert found.completed_at is not None


def test_status_reads_the_marker_and_not_the_slice(
    monkeypatch: pytest.MonkeyPatch, prepared_sifts: Path
) -> None:
    # Offline and cheap: nothing about a status answer needs the table opened.
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("status must not read the table")

    monkeypatch.setattr(sifts, "_read_table", refuse)
    assert sifts.status().rows == 91


# --- Protein.structures ---------------------------------------------------------


def test_a_protein_answers_with_what_sifts_maps_its_accession_to(prepared_sifts: Path) -> None:
    protein = Protein("MQIFVKTLTG", id="P0CG48")
    assert protein.structures.equals(sifts.structures_for("P0CG48"))


def test_a_protein_reaches_sifts_through_the_module_so_one_patch_takes_it_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    monkeypatch.setattr(sifts, "structures_for", lambda accession: called.append(accession))
    _ = Protein("MKTAY", id="P12345").structures
    assert called == ["P12345"]


def test_a_protein_with_no_accession_cannot_be_joined() -> None:
    with pytest.raises(ValueError, match="no id"):
        _ = Protein("MKTAY").structures


def test_a_protein_with_an_accession_but_no_prepared_set_raises() -> None:
    with pytest.raises(SiftsNotDownloadedError):
        _ = Protein("MKTAY", id="P12345").structures


# --- the CLI ---------------------------------------------------------------------


def test_every_command_is_registered_under_a_name_it_was_given() -> None:
    assert [command.name for command in app.registered_commands] == ["prepare", "status"]


def test_every_command_takes_json() -> None:
    # `plain_text`, not `result.output`: rich styles the first dash of `--json` separately,
    # so the raw output carries no such substring wherever colour is on. See tests/__init__.
    for command in app.registered_commands:
        assert command.name is not None
        result = CliRunner().invoke(app, [command.name, "--help"])
        assert "--json" in plain_text(result.output), command.name


def test_a_bare_invocation_prints_help_instead_of_nothing() -> None:
    result = CliRunner().invoke(app, [])
    assert "prepare" in result.output
    assert "status" in result.output


def test_the_sub_app_is_mounted_on_the_root_cli() -> None:
    result = CliRunner().invoke(root_app, ["sifts", "--help"])
    assert result.exit_code == 0
    assert "prepare" in result.output


def test_prepare_stores_the_set_and_then_prints_which_release_it_is(
    monkeypatch: pytest.MonkeyPatch, publisher_file: Path
) -> None:
    def fake_fetch(url: str, dest_dir: Path, **kwargs: Any) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / "pdb_chain_uniprot.tsv.gz"
        shutil.copyfile(publisher_file, target)
        return target

    monkeypatch.setattr(fetch, "fetch_url", fake_fetch)
    result = CliRunner().invoke(app, ["prepare", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["pdb_release"] == "35.26"


def test_prepare_exits_one_and_says_where_to_run_it_when_the_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreachable(*args: Any, **kwargs: Any) -> Path:
        raise OSError("no route to host")

    monkeypatch.setattr(fetch, "fetch_url", unreachable)
    result = CliRunner().invoke(app, ["prepare"])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert sifts.PREPARE_COMMAND in result.output


def test_status_answers_json_when_asked(prepared_sifts: Path) -> None:
    result = CliRunner().invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == sifts.status().as_json()


def test_status_prints_one_line_per_field_when_not_asked_for_json(
    prepared_sifts: Path,
) -> None:
    result = CliRunner().invoke(app, ["status"])
    assert result.exit_code == 0
    assert "pdb_release: 35.26" in result.output
    assert "rows: 91" in result.output
