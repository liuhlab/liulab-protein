"""mmCIF and PDB in and out: the format branch, gzip, and the three array operations.

Everything runs over the real entries in `tests/data`, whose provenance is in
`tests/data/README.md`. The claims worth holding are that both formats and both spellings
read, that a format this package does not read says so by name, that a chain list is the
distinct labels rather than the segment boundaries, that a chain label of more than one
character survives a round trip, and that the two annotations Foldseek needs are read.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest

from protein.io import structure as io

_DATA = Path(__file__).resolve().parent / "data"
_UBQ = _DATA / "1ubq.cif.gz"
_BNA = _DATA / "1bna.cif.gz"
_TRP = _DATA / "1l2y_2models.pdb.gz"


@pytest.fixture
def plain_ubq(tmp_path: Path) -> Path:
    """`1UBQ` decompressed, because the committed fixtures are all gzipped."""
    target = tmp_path / "1ubq.cif"
    target.write_bytes(gzip.decompress(_UBQ.read_bytes()))
    return target


# --- naming --------------------------------------------------------------------


def test_the_format_suffix_ignores_gzip_and_case() -> None:
    assert io.format_suffix("1ubq.cif.gz") == ".cif"
    assert io.format_suffix("model.PDB") == ".pdb"
    assert io.format_suffix("coordinates") == ""


def test_an_entry_name_drops_both_suffixes_and_keeps_the_case() -> None:
    # Nothing here folds an identifier somebody chose, and this string is what Foldseek
    # reports the query under.
    assert io.entry_name("/data/1ubq.cif.gz") == "1ubq"
    assert io.entry_name("104L.pdb") == "104L"


# --- reading -------------------------------------------------------------------


def test_a_gzipped_mmcif_reads_every_atom_of_the_first_model() -> None:
    assert io.read_atoms(_UBQ).array_length() == 660


def test_the_same_file_decompressed_reads_the_same_atoms(plain_ubq: Path) -> None:
    assert io.read_atoms(plain_ubq).array_length() == io.read_atoms(_UBQ).array_length()


def test_a_pdb_file_reads_as_readily_as_an_mmcif() -> None:
    assert io.read_atoms(_TRP).array_length() == 304


def test_reading_the_models_of_an_nmr_entry_gives_a_stack() -> None:
    assert io.read_models(_TRP).stack_depth() == 2


def test_a_single_model_entry_is_a_stack_of_depth_one_and_not_a_special_case() -> None:
    assert io.read_models(_UBQ).stack_depth() == 1


def test_a_later_model_can_be_asked_for_by_number() -> None:
    first = io.read_atoms(_TRP, model=1)
    second = io.read_atoms(_TRP, model=2)
    assert first.array_length() == second.array_length()
    assert not (first.coord == second.coord).all()


def test_a_format_this_package_does_not_read_names_the_ones_it_does(tmp_path: Path) -> None:
    unknown = tmp_path / "coordinates.xyz"
    unknown.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.cif, \.mmcif, \.pdbx, \.pdb, \.ent"):
        io.read_atoms(unknown)


def test_binarycif_says_it_is_deferred_rather_than_unknown(tmp_path: Path) -> None:
    # A format biotite reads and this package chooses not to is a different answer from a
    # suffix nobody recognises.
    deferred = tmp_path / "1ubq.bcif"
    deferred.write_bytes(b"")
    with pytest.raises(ValueError, match="BinaryCIF is deferred"):
        io.read_atoms(deferred)


# --- chains --------------------------------------------------------------------


def test_the_chain_list_is_the_distinct_labels_and_not_the_segment_boundaries() -> None:
    # `get_chains` answers with the label at every chain segment start -- here the waters
    # open one per letter -- which is the whole reason this package does not use it.
    atoms = io.read_atoms(_BNA)
    assert len(struc.get_chains(atoms)) == 4
    assert io.chain_ids(atoms) == ("A", "B")


def test_a_repeated_label_is_listed_once_and_in_first_appearance_order() -> None:
    atoms = struc.AtomArray(3)
    atoms.chain_id = np.array(["A", "B", "A"])
    assert io.chain_ids(atoms) == ("A", "B")


def test_a_chain_selection_takes_every_atom_with_that_label() -> None:
    atoms = io.read_atoms(_BNA)
    a, b = (io.chain_atoms(atoms, label) for label in ("A", "B"))
    assert a.array_length() + b.array_length() == atoms.array_length()
    assert io.chain_ids(a) == ("A",)


def test_a_label_nothing_carries_selects_nothing_rather_than_raising() -> None:
    assert io.chain_atoms(io.read_atoms(_BNA), "Z").array_length() == 0


# --- writing -------------------------------------------------------------------


def test_an_mmcif_round_trip_keeps_a_chain_label_of_more_than_one_character(
    tmp_path: Path,
) -> None:
    # mmCIF is the format that can hold a label longer than a character, which is why the
    # coordinate cache and every written query are mmCIF.
    atoms = _relabelled(io.read_atoms(_UBQ), "Q1")
    written = tmp_path / "relabelled.cif"
    io.write_atoms(written, atoms)
    assert io.chain_ids(io.read_atoms(written)) == ("Q1",)


def test_a_written_file_reads_back_with_the_atoms_it_was_given(tmp_path: Path) -> None:
    atoms = io.read_atoms(_UBQ)
    written = tmp_path / "1ubq_A.cif"
    io.write_atoms(written, atoms)
    assert io.read_atoms(written).array_length() == atoms.array_length()


def test_the_two_fields_foldseek_needs_are_written_back_out(tmp_path: Path) -> None:
    # Foldseek reads no mmCIF whose atom_site lacks these two, so a round trip that dropped
    # them would be unsearchable.
    written = tmp_path / "1ubq_A.cif"
    io.write_atoms(written, io.read_atoms(_UBQ))
    text = written.read_text(encoding="utf-8")
    assert "_atom_site.B_iso_or_equiv" in text
    assert "_atom_site.occupancy" in text


def test_serialising_to_text_gives_what_writing_to_a_file_gives(tmp_path: Path) -> None:
    # One serialiser, two destinations: a viewer takes a string and Foldseek takes a file,
    # and neither may see a different document.
    atoms = io.read_atoms(_UBQ)
    written = tmp_path / "1ubq.cif"
    io.write_atoms(written, atoms)
    assert io.to_text(atoms, name="1ubq") == written.read_text(encoding="utf-8")


def test_text_reads_back_as_the_atoms_it_was_given(tmp_path: Path) -> None:
    atoms = _relabelled(io.read_atoms(_BNA), "Q1")
    path = tmp_path / "round_trip.cif"
    path.write_text(io.to_text(atoms, name="round_trip"), encoding="utf-8")
    assert io.read_atoms(path).array_length() == atoms.array_length()
    assert io.chain_ids(io.read_atoms(path)) == ("Q1",)


def test_the_data_block_is_named_by_the_caller_rather_than_by_a_file() -> None:
    # A chain has no file to take a name from, so the name is an argument here.
    assert io.to_text(io.read_atoms(_UBQ), name="1UBQ_A").startswith("data_1UBQ_A")


def test_a_pdb_file_is_read_and_not_written(tmp_path: Path) -> None:
    # Nothing in this package needs to write one, and a PDB file cannot spell every chain
    # label the archive uses.
    assert io.read_atoms(_TRP).array_length() == 304
    with pytest.raises(ValueError, match="can write"):
        io.write_atoms(tmp_path / "out.pdb", io.read_atoms(_UBQ))


def test_a_written_file_is_gzipped_when_its_name_says_so(tmp_path: Path) -> None:
    written = tmp_path / "1ubq.cif.gz"
    io.write_atoms(written, io.read_atoms(_UBQ))
    assert written.read_bytes()[:2] == b"\x1f\x8b"
    assert io.read_atoms(written).array_length() == 660


def test_writing_a_format_this_package_does_not_write_says_which_it_does(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"can write \.cif, \.mmcif, \.pdbx"):
        io.write_atoms(tmp_path / "out.xyz", io.read_atoms(_UBQ))


def test_the_b_factors_come_back_from_the_file_and_not_from_a_default() -> None:
    b_factors = np.asarray(io.read_atoms(_UBQ).b_factor)
    assert b_factors[0] == pytest.approx(9.67)


def _relabelled(atoms: struc.AtomArray, label: str) -> struc.AtomArray:
    """Return a copy of `atoms` with every chain label replaced."""
    copy = atoms.copy()
    copy.chain_id = np.full(copy.array_length(), label, dtype="U4")
    return copy
