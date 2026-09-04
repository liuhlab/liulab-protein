"""The MSA class: what it holds, what it refuses, and what it deliberately does not check."""

from __future__ import annotations

from pathlib import Path

import pytest
from biotite.sequence.align import Alignment

from protein import MSA
from protein.msa import InvalidAlignmentError, count_match_states

_DATA = Path(__file__).resolve().parent / "data"

_PAIRED = _DATA / "colabfold_pair.a3m"
_HHBLITS = _DATA / "hhblits_5ahw_slice.a3m"

# The paired fixture, spelled here so a test need not read the file it is testing the reader
# against: two chains of 12 and 10 residues, three rows.
_PAIRED_MATCH_STATES = 22
_PAIRED_DEPTH = 3


# --- what it holds -----------------------------------------------------------


def test_the_rows_arrive_as_pairs_in_the_order_they_were_given() -> None:
    msa = MSA([("query", "MKTAY"), ("hit", "MKTaAY")])
    assert msa.rows == (("query", "MKTAY"), ("hit", "MKTaAY"))


def test_case_is_preserved_exactly_as_read() -> None:
    # Case IS the match-state bit, so uppercasing a row would change what the file means.
    msa = MSA.from_a3m(_HHBLITS)
    assert "dvinttielgvtpsvrqeqefaveikerr" in msa.rows[1][1]


def test_a_generator_of_rows_is_consumed_once_and_kept() -> None:
    msa = MSA((header, "MKT") for header in ("query", "hit"))
    assert msa.depth == 2
    assert msa.rows[1] == ("hit", "MKT")


def test_the_comment_line_is_carried_verbatim() -> None:
    assert MSA.from_a3m(_PAIRED).comment == "#12,10\t1,1"


def test_a_file_with_no_comment_line_carries_none() -> None:
    assert MSA.from_a3m(_HHBLITS).comment is None


def test_the_query_row_is_distinguished_from_the_rest() -> None:
    msa = MSA.from_a3m(_PAIRED)
    assert msa.query_header == "101"
    assert msa.query == "MKTAYIAKQRQISHFSRQLEER"


# --- rows are text, not typed sequences --------------------------------------


def test_a_row_is_a_plain_string_and_not_a_typed_sequence() -> None:
    header, row = MSA([("query", "MKTAY")]).rows[0]
    assert type(header) is str
    assert type(row) is str


def test_nothing_held_is_a_biotite_alignment() -> None:
    # Disqualified, not merely worse: an Alignment uppercases and renders every gap as `-`.
    msa = MSA.from_a3m(_PAIRED)
    assert not any(isinstance(value, Alignment) for value in vars(msa).values())


def test_residues_are_not_validated_so_a_row_may_spell_anything() -> None:
    # An MSA is a file's content. `U` is what a database entry spells, and `protein.seq`
    # would reject `*`; neither is this class's business.
    msa = MSA([("query", "MUKTAY"), ("hit", "MUKTA*")])
    assert msa.rows[1][1] == "MUKTA*"


# --- the A3M invariant, checked at construction ------------------------------


def test_lowercase_in_row_zero_raises_the_named_error() -> None:
    with pytest.raises(InvalidAlignmentError, match="row 0") as raised:
        MSA([("query", "MKTaY")])
    assert raised.value.row == 0


def test_rows_that_disagree_on_the_match_state_count_raise_the_named_error() -> None:
    with pytest.raises(InvalidAlignmentError, match="match states") as raised:
        MSA([("query", "MKTAY"), ("hit", "MKTA")])
    assert raised.value.row == 1


def test_an_alignment_with_no_rows_has_nothing_to_be_anchored_on() -> None:
    with pytest.raises(InvalidAlignmentError):
        MSA([])


def test_the_error_is_its_own_class_and_not_a_bare_value_error() -> None:
    with pytest.raises(InvalidAlignmentError) as raised:
        MSA([("query", "MKTaY")])
    assert type(raised.value) is not ValueError
    assert isinstance(raised.value, ValueError)


def test_the_check_holds_generated_rows_to_the_same_rule_as_parsed_ones() -> None:
    # Why it lives at construction rather than in the reader.
    parsed = MSA.from_a3m(_PAIRED)
    with pytest.raises(InvalidAlignmentError):
        MSA([*parsed.rows, ("invented", "MKT")])


def test_an_insertion_does_not_change_the_match_state_count() -> None:
    assert MSA([("query", "MKTAY"), ("hit", "MKTaaaAY")]).match_states == 5


def test_a_gap_does_occupy_a_match_state() -> None:
    assert count_match_states("MK--Y") == 5
    assert count_match_states("MKtaY") == 3


# --- depth, match states and repr --------------------------------------------


def test_depth_and_len_are_the_same_row_count() -> None:
    msa = MSA.from_a3m(_PAIRED)
    assert msa.depth == _PAIRED_DEPTH
    assert len(msa) == _PAIRED_DEPTH


def test_the_match_state_count_is_the_query_length_not_the_longest_row() -> None:
    msa = MSA.from_a3m(_PAIRED)
    assert msa.match_states == _PAIRED_MATCH_STATES
    assert max(len(row) for _, row in msa.rows) > _PAIRED_MATCH_STATES


def test_repr_names_the_depth_and_the_match_state_count() -> None:
    assert repr(MSA.from_a3m(_PAIRED)) == "MSA(depth 3, 22 match states)"


# --- compress ----------------------------------------------------------------


def test_compress_demotes_the_anchor_rows_gap_columns_to_insertions() -> None:
    compressed = MSA([("query", "MK-TAY"), ("hit", "MKWTAY")]).compress()
    assert compressed.rows == (("query", "MKTAY"), ("hit", "MKwTAY"))


def test_compress_drops_a_column_where_both_the_anchor_and_the_row_are_gaps() -> None:
    compressed = MSA([("query", "MK-TAY"), ("hit", "MK-TAY")]).compress()
    assert compressed.rows[1] == ("hit", "MKTAY")


def test_compress_leaves_a_deletion_in_a_match_column_alone() -> None:
    compressed = MSA([("query", "MKTAY"), ("hit", "MK--Y")]).compress()
    assert compressed.rows[1] == ("hit", "MK--Y")


def test_the_designated_row_leads_the_result_and_the_rest_keep_their_order() -> None:
    symmetric = MSA([("a", "MK-T"), ("b", "MKWT"), ("c", "MK-T")])
    assert [header for header, _ in symmetric.compress(1).rows] == ["b", "a", "c"]


def test_compressing_on_a_gapped_row_leaves_that_row_without_gaps() -> None:
    compressed = MSA([("query", "M-K-T"), ("hit", "MAKGT")]).compress()
    assert compressed.query == "MKT"
    assert compressed.match_states == 3


def test_a_compressed_alignment_is_checked_like_any_other() -> None:
    # It comes back through the constructor, so a caller cannot be handed a bad shape.
    compressed = MSA([("query", "MK-TAY"), ("hit", "MKWTAY")]).compress()
    assert isinstance(compressed, MSA)
    assert compressed.match_states == count_match_states(compressed.query)


def test_compress_carries_the_comment_line() -> None:
    assert MSA([("q", "MK-T"), ("h", "MKWT")], comment="#4").compress().comment == "#4"


def test_compressing_an_a3m_raises_because_it_is_not_a_symmetric_alignment() -> None:
    with pytest.raises(InvalidAlignmentError, match="symmetric") as raised:
        MSA([("query", "MKTAY"), ("hit", "MKTaaAY")]).compress()
    assert raised.value.row == 1


def test_compress_on_a_row_that_is_not_there_raises() -> None:
    with pytest.raises(IndexError):
        MSA([("query", "MKTAY"), ("hit", "MKTAY")]).compress(7)


def test_there_is_no_expand() -> None:
    # Declined on purpose: nothing in this package needs a rectangular matrix back.
    assert not hasattr(MSA([("query", "MKTAY")]), "expand")


# --- the two exits -----------------------------------------------------------


def test_to_a3m_returns_the_text_the_file_held() -> None:
    assert MSA.from_a3m(_PAIRED).to_a3m() == _PAIRED.read_text(encoding="utf-8")


def test_a_real_alignment_round_trips_byte_for_byte(tmp_path: Path) -> None:
    written = MSA.from_a3m(_HHBLITS).write(tmp_path / "again.a3m")
    assert written.read_bytes() == _HHBLITS.read_bytes()


def test_the_comment_and_the_key_headers_survive_the_round_trip(tmp_path: Path) -> None:
    written = MSA.from_a3m(_PAIRED).write(tmp_path / "again.a3m")
    assert written.read_bytes() == _PAIRED.read_bytes()


def test_write_returns_the_path_it_was_given(tmp_path: Path) -> None:
    destination = tmp_path / "out.a3m"
    assert MSA([("query", "MKTAY")]).write(str(destination)) == destination


def test_write_requires_a_path_and_defaults_nowhere() -> None:
    # Nothing durable lands in the Data dir without the caller saying where.
    with pytest.raises(TypeError):
        MSA([("query", "MKTAY")]).write()  # pyright: ignore[reportCallIssue]


def test_write_and_from_a3m_round_trip_a_generated_alignment(tmp_path: Path) -> None:
    original = MSA([("query", "MKTAY"), ("hit", "MKTaAY")], comment="#5\t1")
    again = MSA.from_a3m(original.write(tmp_path / "generated.a3m"))
    assert again.rows == original.rows
    assert again.comment == original.comment
