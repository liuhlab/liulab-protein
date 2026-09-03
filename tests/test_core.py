"""The Protein class: what it holds, what it refuses, and what it is not."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from biotite.sequence import ProteinSequence

from protein import Protein
from protein.search import SearchMixin
from protein.seq import InvalidResidueError, ResidueCoercionWarning

_DATA = Path(__file__).resolve().parent / "data"

# The single-record fixture, spelled here so a test need not read the file it is testing the
# reader against.
_INSULIN_ID = "sp|P01308|INS_HUMAN"
_INSULIN_DESCRIPTION = "Insulin OS=Homo sapiens OX=9606 GN=INS PE=1 SV=1"
_INSULIN_LENGTH = 110


# --- what it holds -----------------------------------------------------------


def test_a_protein_holds_every_field_it_was_given() -> None:
    p = Protein("MKTAY", id="P12345", name="INS_HUMAN", description="a short one")
    assert p.id == "P12345"
    assert p.name == "INS_HUMAN"
    assert p.description == "a short one"


def test_every_optional_field_defaults_to_none() -> None:
    p = Protein("MKTAY")
    assert (p.id, p.name, p.description) == (None, None, None)


def test_metadata_is_eager_and_total_rather_than_none() -> None:
    assert Protein("MKTAY").metadata == {}


def test_metadata_is_copied_so_the_caller_cannot_change_it_afterwards() -> None:
    given = {"organism": "Homo sapiens"}
    p = Protein("MKTAY", metadata=given)
    given["organism"] = "Mus musculus"
    assert p.metadata == {"organism": "Homo sapiens"}


# --- the sequence is biotite's type, not a str -------------------------------


def test_the_sequence_is_a_biotite_protein_sequence() -> None:
    assert isinstance(Protein("MKTAY").sequence, ProteinSequence)


def test_the_sequence_does_not_compare_equal_to_the_string_it_was_built_from() -> None:
    # The consequence of holding biotite's type: `str(...)` is how a tokenizer or a
    # subprocess gets its string.
    p = Protein("MKTAY")
    assert p.sequence != "MKTAY"
    assert str(p.sequence) == "MKTAY"


def test_lowercase_input_comes_back_uppercase() -> None:
    assert str(Protein("mktay").sequence) == "MKTAY"


# --- ADR-0001: it validates at construction ----------------------------------


def test_a_sequence_outside_the_alphabet_raises_at_construction() -> None:
    with pytest.raises(InvalidResidueError) as raised:
        Protein("MK*T-AY", id="P12345")
    assert raised.value.offenders == [(2, "*"), (4, "-")]
    assert raised.value.name == "P12345"


def test_the_construction_error_names_the_accession() -> None:
    with pytest.raises(InvalidResidueError, match="P12345"):
        Protein("MK*T", id="P12345")


def test_the_three_codes_biotite_cannot_store_are_folded_with_a_warning() -> None:
    with pytest.warns(ResidueCoercionWarning, match="P12345"):
        p = Protein("MKUOJ", id="P12345")
    assert str(p.sequence) == "MKXXX"


def test_the_check_runs_before_the_fold_so_a_stop_symbol_still_raises() -> None:
    with pytest.raises(InvalidResidueError):
        Protein("MKU*")


# --- length and indexing -----------------------------------------------------


def test_length_and_len_are_the_same_residue_count() -> None:
    p = Protein("MKTAY")
    assert p.length == 5
    assert len(p) == 5


def test_indexing_one_residue_returns_a_str() -> None:
    p = Protein("MKTAY")
    assert p[0] == "M"
    assert p[-1] == "Y"


def test_a_slice_of_a_protein_is_a_str_and_not_a_protein() -> None:
    sliced = Protein("MKTAY", id="P12345")[1:3]
    assert sliced == "KT"
    assert not isinstance(sliced, Protein)


def test_an_index_past_the_end_raises() -> None:
    with pytest.raises(IndexError):
        Protein("MKTAY")[99]


def test_repr_names_the_accession_and_the_residue_count() -> None:
    assert repr(Protein("MKTAY", id="P12345")) == "Protein('P12345', 5 aa)"


def test_repr_of_an_unnamed_protein_says_so_rather_than_inventing_an_id() -> None:
    assert repr(Protein("MKTAY")) == "Protein(None, 5 aa)"


# --- FASTA ------------------------------------------------------------------


def test_from_fasta_reads_the_one_record_in_a_single_record_file() -> None:
    p = Protein.from_fasta(_DATA / "uniprot_p01308.fasta")
    assert p.id == _INSULIN_ID
    assert p.description == _INSULIN_DESCRIPTION
    assert p.length == _INSULIN_LENGTH


def test_from_fasta_refuses_a_file_holding_more_than_one_record() -> None:
    with pytest.raises(ValueError, match="read_proteins"):
        Protein.from_fasta(_DATA / "uniprot_three.fasta")


def test_from_fasta_refuses_a_file_holding_no_record(tmp_path: Path) -> None:
    empty = tmp_path / "empty.fasta"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no FASTA record"):
        Protein.from_fasta(empty)


def test_to_fasta_renders_the_header_from_the_id_and_the_description() -> None:
    p = Protein("MKTAY", id="P12345", description="a short one")
    assert p.to_fasta() == ">P12345 a short one\nMKTAY\n"


def test_to_fasta_wraps_at_the_line_width_it_is_given() -> None:
    assert Protein("MKTAY", id="P12345").to_fasta(line_width=3) == ">P12345\nMKT\nAY\n"


def test_to_fasta_writes_the_same_text_it_returns(tmp_path: Path) -> None:
    p = Protein("MKTAY", id="P12345", description="a short one")
    written = tmp_path / "one.fasta"
    text = p.to_fasta(written)
    assert written.read_text(encoding="utf-8") == text


def test_a_real_record_round_trips_through_to_fasta_and_from_fasta(tmp_path: Path) -> None:
    original = Protein.from_fasta(_DATA / "uniprot_p01308.fasta")
    written = tmp_path / "again.fasta"
    original.to_fasta(written, line_width=60)
    again = Protein.from_fasta(written)
    assert (again.id, again.description) == (original.id, original.description)
    assert str(again.sequence) == str(original.sequence)
    assert written.read_bytes() == (_DATA / "uniprot_p01308.fasta").read_bytes()


def test_the_name_and_the_metadata_are_not_written_to_a_fasta_header() -> None:
    p = Protein("MKTAY", id="P12345", name="INS_HUMAN", metadata={"organism": "Homo sapiens"})
    assert p.to_fasta() == ">P12345\nMKTAY\n"


# --- what this class is not --------------------------------------------------


def test_the_search_mixin_comes_first_in_the_bases() -> None:
    assert Protein.__mro__[1] is SearchMixin


@pytest.mark.parametrize("absent", ["embed", "structure", "foldseek_search", "from_structure"])
def test_protein_carries_neither_coordinates_nor_the_weights(absent: str) -> None:
    # Two rules, one assertion: foldseek takes a structure and a Protein has none, and
    # ESM-C's weights need a lifetime the caller can see, so `ESMC` is an object.
    assert not hasattr(Protein, absent)


def test_importing_protein_does_not_import_torch() -> None:
    import protein

    assert protein.Protein is Protein
    assert "torch" not in sys.modules
