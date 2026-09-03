"""The alphabet guard: what it accepts, what it folds, and what it refuses."""

import string
import warnings

import pytest
from biotite.sequence import ProteinSequence

from protein.seq import (
    ALPHABET,
    AMBIGUOUS,
    COERCED,
    STANDARD,
    STORED,
    UNKNOWN,
    InvalidResidueError,
    ResidueCoercionWarning,
    _biotite_symbols,
    check_alphabet,
    offending_positions,
    outside_alphabet,
    to_protein_sequence,
)


def test_the_accepted_alphabet_is_every_ascii_letter() -> None:
    # The claim the module docstring rests on: no letter is left for the check to reject.
    assert len(STANDARD) == 20
    assert len(AMBIGUOUS) == 6
    assert frozenset(string.ascii_uppercase) == ALPHABET


def test_what_biotite_stores_is_read_from_biotite_and_drops_the_stop_symbol() -> None:
    assert _biotite_symbols() == STORED | {"*"}
    assert frozenset("ACDEFGHIKLMNPQRSTVWYBZX") == STORED


def test_only_the_three_codes_biotite_cannot_store_are_coerced() -> None:
    assert frozenset("UOJ") == COERCED
    assert UNKNOWN in STORED


def test_outside_alphabet_reports_each_offender_once_and_sorted() -> None:
    assert outside_alphabet("MK*T-*") == ["*", "-"]


def test_outside_alphabet_accepts_every_code_including_the_three_biotite_lacks() -> None:
    assert outside_alphabet("MKTVUOJBZX") == []


def test_case_is_not_an_offence() -> None:
    assert outside_alphabet("mktvuoj") == []
    assert offending_positions("mktvuoj") == []


def test_offending_positions_are_zero_based_offsets_in_order() -> None:
    assert offending_positions("MK*T-") == [(2, "*"), (4, "-")]


def test_offending_positions_reports_every_occurrence_not_every_character() -> None:
    assert offending_positions("--") == [(0, "-"), (1, "-")]


def test_an_empty_sequence_offends_nothing() -> None:
    assert outside_alphabet("") == []
    assert offending_positions("") == []
    assert str(to_protein_sequence("")) == ""


def test_check_alphabet_passes_a_sequence_of_letters_and_says_nothing() -> None:
    assert check_alphabet("MKTVUOJBZX") is None


def test_check_alphabet_rejects_the_stop_symbol_biotite_would_accept() -> None:
    # The point of the check: biotite stores `*`, and a stop reaching a tokenizer fails far
    # from its cause.
    assert str(ProteinSequence("MK*")) == "MK*"
    with pytest.raises(InvalidResidueError):
        check_alphabet("MK*")


@pytest.mark.parametrize("offender", ["-", " ", "1", ".", "\n"])
def test_check_alphabet_rejects_gaps_whitespace_digits_and_punctuation(offender: str) -> None:
    with pytest.raises(InvalidResidueError):
        check_alphabet(f"MK{offender}T")


def test_the_error_carries_the_offenders_as_data_so_no_caller_parses_the_message() -> None:
    with pytest.raises(InvalidResidueError) as raised:
        check_alphabet("MK*T-", name="P12345")
    assert raised.value.offenders == [(2, "*"), (4, "-")]
    assert raised.value.name == "P12345"
    assert isinstance(raised.value, ValueError)


def test_the_error_names_the_sequence_only_when_it_was_given_a_name() -> None:
    with pytest.raises(InvalidResidueError, match=r"^P12345: not in the protein alphabet"):
        check_alphabet("MK*", name="P12345")
    with pytest.raises(InvalidResidueError, match=r"^not in the protein alphabet"):
        check_alphabet("MK*")


def test_the_error_message_lists_five_positions_and_counts_the_rest() -> None:
    with pytest.raises(InvalidResidueError) as raised:
        check_alphabet("-" * 8)
    message = str(raised.value)
    assert message.count(" at ") == 5
    assert message.endswith("and 3 more (8 in total)")
    # Capped in the message, complete in the attribute.
    assert len(raised.value.offenders) == 8


def test_a_sequence_carrying_u_warns_and_yields_x() -> None:
    with pytest.warns(ResidueCoercionWarning):
        sequence = to_protein_sequence("MKU")
    assert isinstance(sequence, ProteinSequence)
    assert str(sequence) == "MKX"


def test_the_coercion_warning_names_the_accession_and_counts_what_it_folded() -> None:
    with pytest.warns(ResidueCoercionWarning, match=r"P0CG48: coerced 1 O, 2 U to X"):
        assert str(to_protein_sequence("MUOU", name="P0CG48")) == "MXXX"


def test_a_sequence_carrying_a_stop_symbol_raises_rather_than_coercing() -> None:
    with pytest.raises(InvalidResidueError):
        to_protein_sequence("MK*U")


def test_the_codes_biotite_can_store_reach_it_unfolded() -> None:
    # B and Z survive: only what biotite cannot hold is folded.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert str(to_protein_sequence("MKTBZX")) == "MKTBZX"


def test_lowercase_input_comes_back_uppercase() -> None:
    with pytest.warns(ResidueCoercionWarning):
        assert str(to_protein_sequence("mku")) == "MKX"
