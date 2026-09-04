"""The two alphabets this package accepts, and the guard in front of biotite's.

Pure: no I/O, no subprocess, no network.

:class:`biotite.sequence.ProteinSequence` cannot store ``U``, ``O`` or ``J``, and does store
``*``, which no protein carries; :class:`biotite.sequence.NucleotideSequence` cannot store
``U`` either. This module is the two-sided guard those mismatches need: the check rejects
what neither alphabet should hold, and :func:`to_protein_sequence` and
:func:`to_nucleotide_sequence` fold what biotite cannot store, loudly, rather than letting
biotite's own converters write a different residue in silence (ADR-0002).

**Know what the protein check is worth.** :data:`ALPHABET` is every ASCII letter, because the
six ambiguity codes fill exactly the six gaps the twenty leave. So the check catches gaps,
stops, digits, whitespace and punctuation — a stray ``*`` reaching a tokenizer fails far from
its cause — and it **cannot** catch a misspelled residue. :data:`NUCLEIC_ALPHABET` is sixteen
letters, so the nucleic check refuses the other ten as well.

Examples
--------
>>> outside_alphabet("MKT-V*")
['*', '-']
>>> offending_positions("MKT-V*")
[(3, '-'), (5, '*')]
>>> str(to_protein_sequence("MUOJK"))
'MXXXK'
>>> str(to_nucleotide_sequence("ACGU"))
'ACGT'
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Mapping
from typing import cast

from biotite.sequence import Alphabet, NucleotideSequence, ProteinSequence

__all__ = [
    "ALPHABET",
    "AMBIGUOUS",
    "COERCED",
    "NUCLEIC_ALPHABET",
    "NUCLEIC_AMBIGUOUS",
    "NUCLEIC_COERCED",
    "NUCLEIC_STANDARD",
    "NUCLEIC_STORED",
    "STANDARD",
    "STORED",
    "THYMINE",
    "UNKNOWN",
    "InvalidResidueError",
    "ResidueCoercionWarning",
    "check_alphabet",
    "offending_positions",
    "outside_alphabet",
    "to_nucleotide_sequence",
    "to_protein_sequence",
]

#: The twenty proteinogenic residues, one letter each.
STANDARD: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: The six codes that name something other than one of the twenty: ``X`` any residue, ``B``
#: aspartate-or-asparagine, ``Z`` glutamate-or-glutamine, ``J`` leucine-or-isoleucine, ``U``
#: selenocysteine, ``O`` pyrrolysine.
AMBIGUOUS: frozenset[str] = frozenset("XBZJUO")

#: **What this package accepts as a protein sequence** — every ASCII letter. Wider than what
#: biotite stores, which is what :data:`COERCED` exists to bridge.
ALPHABET: frozenset[str] = STANDARD | AMBIGUOUS


def _biotite_symbols() -> frozenset[str]:
    """Return every symbol biotite's protein alphabet holds, the stop symbol included.

    The cast is biotite's shape, not a doubt about it: ``ProteinSequence.alphabet`` shadows a
    property of the same name on the base class, so it answers correctly at runtime while
    every static reading of it is ``property``.
    """
    alphabet = cast("Alphabet", ProteinSequence.alphabet)
    return frozenset(str(symbol) for symbol in alphabet.get_symbols())


#: **What biotite stores**, read from biotite rather than written out here, so an upgrade
#: cannot silently disagree with us. ``*`` is dropped: a protein sequence reaching this
#: package must not carry a stop symbol.
STORED: frozenset[str] = _biotite_symbols() - {"*"}

#: Accepted here, unstorable there: the codes :func:`to_protein_sequence` folds to ``X``.
#: Derived rather than listed, so whatever biotite gains, this loses.
COERCED: frozenset[str] = ALPHABET - STORED

#: What an unstorable code becomes. "Unknown" is true of a residue biotite cannot name; the
#: ``C`` and ``K`` its own converters would write are not.
UNKNOWN = "X"

#: Positions the message lists before it counts the rest.
#: :attr:`InvalidResidueError.offenders` is not capped.
_MAX_LISTED = 5

_COERCION = str.maketrans(dict.fromkeys(COERCED, UNKNOWN))

#: The four bases, one letter each.
NUCLEIC_STANDARD: frozenset[str] = frozenset("ACGT")

#: The eleven IUPAC codes for a base that is not fully determined: ``N`` any base, ``R``
#: purine, ``Y`` pyrimidine, and the eight naming two or three bases.
NUCLEIC_AMBIGUOUS: frozenset[str] = frozenset("RYWSMKHBVDN")

#: **What this package accepts as a nucleic sequence** — the four bases, the eleven codes,
#: and ``U``, which :data:`NUCLEIC_COERCED` exists to bridge.
NUCLEIC_ALPHABET: frozenset[str] = NUCLEIC_STANDARD | NUCLEIC_AMBIGUOUS | {"U"}


def _biotite_nucleotide_symbols() -> frozenset[str]:
    """Return every symbol biotite's ambiguous nucleotide alphabet holds.

    The ambiguous one of the two, because it is the wider, and because a caller never picks
    between them: ``NucleotideSequence`` does that from the symbols it is given.
    """
    return frozenset(str(symbol) for symbol in NucleotideSequence.alphabet_amb.get_symbols())


#: **What biotite stores**, read from biotite rather than written out here, so an upgrade
#: cannot silently disagree with us.
NUCLEIC_STORED: frozenset[str] = _biotite_nucleotide_symbols()

#: Accepted here, unstorable there: ``U``, which :func:`to_nucleotide_sequence` folds to
#: :data:`THYMINE`. Derived rather than listed, so whatever biotite gains, this loses.
NUCLEIC_COERCED: frozenset[str] = NUCLEIC_ALPHABET - NUCLEIC_STORED

#: What ``U`` becomes. Thymine sits where uracil does, so the base survives and only the
#: spelling is lost — which is why the fold warns.
THYMINE = "T"

_NUCLEIC_COERCION = str.maketrans(dict.fromkeys(NUCLEIC_COERCED, THYMINE))


class ResidueCoercionWarning(UserWarning):
    """Warns that codes biotite cannot store were replaced by ``X``.

    Its own category so one targeted ``filterwarnings`` entry can tolerate this warning,
    raised on purpose, without tolerating any other.
    """


class InvalidResidueError(ValueError):
    """Raised when a sequence holds characters its alphabet excludes.

    A :class:`ValueError`, unlike biotite's ``AlphabetError``, and carrying the offenders as
    data so a caller repairing input never has to parse the message.

    Attributes
    ----------
    offenders : list of tuple of (int, str)
        Every offending ``(index, character)``, zero-based and in order of appearance.
        Uncapped, where the message lists five.
    name : str or None
        The accession or identifier the caller named the sequence by, if any.
    alphabet : str
        Which alphabet refused it — ``"protein"`` or ``"nucleic"``.
    """

    def __init__(
        self,
        offenders: list[tuple[int, str]],
        *,
        name: str | None = None,
        alphabet: str = "protein",
    ) -> None:
        """Build the error from the offending positions, whose they are, and which alphabet."""
        self.offenders = offenders
        self.name = name
        self.alphabet = alphabet
        super().__init__(_invalid_message(offenders, name, alphabet))


def _offenders(sequence: str, accepted: frozenset[str]) -> list[tuple[int, str]]:
    """Return every ``(index, character)`` of ``sequence`` that ``accepted`` excludes."""
    return [
        (index, character)
        for index, character in enumerate(sequence)
        if character.upper() not in accepted
    ]


def outside_alphabet(sequence: str) -> list[str]:
    """Return the distinct characters of ``sequence`` that :data:`ALPHABET` excludes.

    Case is not an offence: lowercase input is accepted here and comes back uppercase from
    :func:`to_protein_sequence`.

    Parameters
    ----------
    sequence : str
        Text to weigh against the alphabet.

    Returns
    -------
    list of str
        The offending characters, distinct and sorted, in the case they arrived in. Empty when
        nothing offends.

    Examples
    --------
    >>> outside_alphabet("MKTVU")
    []
    >>> outside_alphabet("mktvu")
    []
    >>> outside_alphabet("MK*T-*")
    ['*', '-']
    """
    return sorted({character for _index, character in _offenders(sequence, ALPHABET)})


def offending_positions(sequence: str) -> list[tuple[int, str]]:
    """Return every ``(index, character)`` of ``sequence`` that :data:`ALPHABET` excludes.

    One entry per occurrence, in order. Indices count from zero and are offsets into
    ``sequence``, not residue numbers.

    Parameters
    ----------
    sequence : str
        Text to weigh against the alphabet.

    Returns
    -------
    list of tuple of (int, str)
        The offending positions and what sits at each. Empty when nothing offends.

    Examples
    --------
    >>> offending_positions("MK*T-")
    [(2, '*'), (4, '-')]
    >>> offending_positions("MKTV")
    []
    """
    return _offenders(sequence, ALPHABET)


def check_alphabet(sequence: str, *, name: str | None = None) -> None:
    """Raise :class:`InvalidResidueError` if ``sequence`` holds anything outside the alphabet.

    Stricter than biotite in one place and one only: ``*`` is in biotite's alphabet and is
    rejected here.

    Parameters
    ----------
    sequence : str
        Text to weigh against :data:`ALPHABET`.
    name : str, optional
        The accession or identifier this sequence belongs to, named in the error.

    Raises
    ------
    InvalidResidueError
        If anything in ``sequence`` is outside :data:`ALPHABET`. The message names at most
        five positions and counts the rest; ``.offenders`` carries them all.

    Examples
    --------
    >>> check_alphabet("MKTVU")
    >>> check_alphabet("MK*T", name="P12345")
    Traceback (most recent call last):
        ...
    protein.seq.InvalidResidueError: P12345: not in the protein alphabet: '*' at 2
    """
    offenders = offending_positions(sequence)
    if offenders:
        raise InvalidResidueError(offenders, name=name)


def to_protein_sequence(sequence: str, *, name: str | None = None) -> ProteinSequence:
    """Check ``sequence``, fold what biotite cannot store, and return biotite's type.

    The one door from a :class:`str` to a :class:`~biotite.sequence.ProteinSequence` in this
    package (ADR-0002). The check runs first, so a ``*`` raises rather than being folded.

    Parameters
    ----------
    sequence : str
        The residues, in either case.
    name : str, optional
        The accession or identifier this sequence belongs to, named in both the warning and
        the error.

    Returns
    -------
    ProteinSequence
        Uppercase, with every character of :data:`COERCED` replaced by :data:`UNKNOWN`.

    Warns
    -----
    ResidueCoercionWarning
        When anything was folded.

    Raises
    ------
    InvalidResidueError
        If anything in ``sequence`` is outside :data:`ALPHABET`.

    Examples
    --------
    >>> str(to_protein_sequence("MKTV"))
    'MKTV'
    >>> str(to_protein_sequence("mktv"))
    'MKTV'
    >>> str(to_protein_sequence("MKUV", name="P0CG48"))
    'MKXV'
    """
    check_alphabet(sequence, name=name)
    upper = sequence.upper()
    folded = upper.translate(_COERCION)
    if folded != upper:
        counts = Counter(character for character in upper if character in COERCED)
        warnings.warn(
            _coercion_message(counts, name, UNKNOWN, "protein"),
            ResidueCoercionWarning,
            stacklevel=2,
        )
    return ProteinSequence(folded)


def to_nucleotide_sequence(sequence: str, *, name: str | None = None) -> NucleotideSequence:
    """Check ``sequence``, fold ``U`` to ``T``, and return biotite's type.

    The one door from a :class:`str` to a :class:`~biotite.sequence.NucleotideSequence` in
    this package, and the nucleic half of ADR-0002. There is no ``DNA`` class and no ``RNA``
    class: neither has an accession to be identified by, and this package holds biotite's
    types rather than wrapping them.

    Which of biotite's two nucleotide alphabets the result carries is biotite's own choice —
    the four-letter one where four letters suffice, the fifteen-letter one otherwise.

    Parameters
    ----------
    sequence : str
        The bases, in either case.
    name : str, optional
        The identifier this sequence belongs to, named in both the warning and the error.

    Returns
    -------
    NucleotideSequence
        Uppercase, with every ``U`` replaced by :data:`THYMINE`.

    Warns
    -----
    ResidueCoercionWarning
        When anything was folded.

    Raises
    ------
    InvalidResidueError
        If anything in ``sequence`` is outside :data:`NUCLEIC_ALPHABET`.

    Examples
    --------
    >>> str(to_nucleotide_sequence("ACGT"))
    'ACGT'
    >>> str(to_nucleotide_sequence("acgn"))
    'ACGN'
    >>> str(to_nucleotide_sequence("ACGU", name="a transcript"))
    'ACGT'
    """
    offenders = _offenders(sequence, NUCLEIC_ALPHABET)
    if offenders:
        raise InvalidResidueError(offenders, name=name, alphabet="nucleic")
    upper = sequence.upper()
    folded = upper.translate(_NUCLEIC_COERCION)
    if folded != upper:
        counts = Counter(character for character in upper if character in NUCLEIC_COERCED)
        warnings.warn(
            _coercion_message(counts, name, THYMINE, "nucleic"),
            ResidueCoercionWarning,
            stacklevel=2,
        )
    return NucleotideSequence(folded)


def _prefix(name: str | None) -> str:
    """Return ``'<name>: '`` when there is a sequence to blame by name, else ``''``."""
    return f"{name}: " if name else ""


def _invalid_message(offenders: list[tuple[int, str]], name: str | None, alphabet: str) -> str:
    """Return the error text: up to five offending positions, then a count of the rest."""
    listed = ", ".join(f"{character!r} at {index}" for index, character in offenders[:_MAX_LISTED])
    hidden = len(offenders) - _MAX_LISTED
    rest = f", and {hidden} more ({len(offenders)} in total)" if hidden > 0 else ""
    return f"{_prefix(name)}not in the {alphabet} alphabet: {listed}{rest}"


def _coercion_message(
    counts: Mapping[str, int], name: str | None, replacement: str, alphabet: str
) -> str:
    """Return the warning text: which codes were folded, how many of each, and whose."""
    detail = ", ".join(f"{count} {code}" for code, count in sorted(counts.items()))
    return (
        f"{_prefix(name)}coerced {detail} to {replacement}; "
        f"biotite's {alphabet} alphabet cannot store them"
    )
