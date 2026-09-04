"""The alphabet guards: what they accept, what they fold, and what they refuse."""

import string
import warnings

import pytest
from biotite.sequence import NucleotideSequence, ProteinSequence

from protein.seq import (
    ALPHABET,
    AMBIGUOUS,
    COERCED,
    NUCLEIC_ALPHABET,
    NUCLEIC_AMBIGUOUS,
    NUCLEIC_COERCED,
    NUCLEIC_STANDARD,
    NUCLEIC_STORED,
    STANDARD,
    STORED,
    THYMINE,
    UNKNOWN,
    InvalidResidueError,
    ResidueCoercionWarning,
    _biotite_nucleotide_symbols,
    _biotite_symbols,
    check_alphabet,
    offending_positions,
    outside_alphabet,
    to_nucleotide_sequence,
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


# --- the nucleic alphabet ------------------------------------------------------


def test_the_nucleic_alphabet_is_the_four_bases_the_eleven_codes_and_uracil() -> None:
    assert len(NUCLEIC_STANDARD) == 4
    assert len(NUCLEIC_AMBIGUOUS) == 11
    assert frozenset("ACGTU") | NUCLEIC_AMBIGUOUS == NUCLEIC_ALPHABET


def test_what_biotite_stores_is_read_from_biotite_and_holds_no_uracil() -> None:
    assert _biotite_nucleotide_symbols() == NUCLEIC_STORED
    assert NUCLEIC_STORED == NUCLEIC_STANDARD | NUCLEIC_AMBIGUOUS
    assert "U" not in NUCLEIC_STORED


def test_uracil_is_the_one_code_that_is_coerced() -> None:
    assert frozenset("U") == NUCLEIC_COERCED
    assert THYMINE in NUCLEIC_STORED


def test_the_four_bases_and_every_ambiguity_code_are_accepted() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert str(to_nucleotide_sequence("ACGTRYWSMKHBVDN")) == "ACGTRYWSMKHBVDN"


def test_a_nucleic_sequence_comes_back_as_biotites_type_and_not_a_string() -> None:
    sequence = to_nucleotide_sequence("ACGT")
    assert isinstance(sequence, NucleotideSequence)
    assert sequence != "ACGT"
    assert str(sequence) == "ACGT"


def test_lowercase_nucleic_input_comes_back_uppercase() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert str(to_nucleotide_sequence("acgn")) == "ACGN"


def test_uracil_folds_to_thymine_and_warns() -> None:
    with pytest.warns(ResidueCoercionWarning):
        sequence = to_nucleotide_sequence("ACGU")
    assert str(sequence) == "ACGT"


def test_the_nucleic_coercion_warning_names_the_sequence_and_counts_what_it_folded() -> None:
    with pytest.warns(ResidueCoercionWarning, match=r"a transcript: coerced 2 U to T"):
        assert str(to_nucleotide_sequence("UACGU", name="a transcript")) == "TACGT"


def test_the_nucleic_coercion_warning_names_the_alphabet_that_cannot_store_it() -> None:
    with pytest.warns(ResidueCoercionWarning, match=r"biotite's nucleic alphabet"):
        to_nucleotide_sequence("acgu")


@pytest.mark.parametrize("offender", ["-", " ", "1", ".", "\n", "X", "E"])
def test_the_nucleic_check_refuses_gaps_whitespace_digits_and_every_other_letter(
    offender: str,
) -> None:
    with pytest.raises(InvalidResidueError):
        to_nucleotide_sequence(f"AC{offender}GT")


def test_the_nucleic_error_carries_its_offenders_as_data_exactly_as_the_protein_one_does() -> None:
    with pytest.raises(InvalidResidueError) as raised:
        to_nucleotide_sequence("AC-GT1", name="a transcript")
    assert raised.value.offenders == [(2, "-"), (5, "1")]
    assert raised.value.name == "a transcript"
    assert isinstance(raised.value, ValueError)


def test_the_error_says_which_alphabet_refused_the_sequence() -> None:
    with pytest.raises(InvalidResidueError, match=r"not in the nucleic alphabet") as nucleic:
        to_nucleotide_sequence("AC-GT")
    assert nucleic.value.alphabet == "nucleic"
    with pytest.raises(InvalidResidueError, match=r"not in the protein alphabet") as protein:
        check_alphabet("MK-T")
    assert protein.value.alphabet == "protein"


def test_an_empty_nucleic_sequence_offends_nothing() -> None:
    assert str(to_nucleotide_sequence("")) == ""


def test_biotite_picks_the_alphabet_and_a_caller_never_does() -> None:
    # The four-letter alphabet where four letters suffice, the fifteen-letter one otherwise.
    assert to_nucleotide_sequence("ACGT").get_alphabet() == NucleotideSequence.alphabet_unamb
    assert to_nucleotide_sequence("ACGN").get_alphabet() == NucleotideSequence.alphabet_amb
