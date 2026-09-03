"""The alphabet this package accepts, and the guard in front of biotite's.

Pure: no I/O, no subprocess, no network.

:class:`biotite.sequence.ProteinSequence` stores the twenty residues plus ``B``, ``Z``, ``X``
and ``*``, and nothing else. Three of the codes UniProt publishes are missing from it — ``U``
selenocysteine, ``O`` pyrrolysine, ``J`` leucine-or-isoleucine — while ``*`` is present though
no protein carries one. This module is the two-sided guard that mismatch needs:
:func:`check_alphabet` rejects what neither alphabet should hold, and
:func:`to_protein_sequence` folds ``U``, ``O`` and ``J`` to ``X``, loudly, before biotite ever
sees the string.

**Never fold by way of biotite.** ``fasta.get_sequence`` and ``structure.to_sequence`` map
``U`` to ``C`` and ``O`` to ``K`` — a different residue, claimed with no signal (``O`` with
none at all). ``X`` means unknown, which is true, and it is what ``mmseqs databases`` already
writes for those two codes. That rule is ADR-0002; this module is how it is kept.

**Know what the check is worth.** The six ambiguity codes fill exactly the six gaps the twenty
leave, so :data:`ALPHABET` is every ASCII letter. The check therefore catches gaps, stops,
digits, whitespace and punctuation — a stray ``*`` or ``-`` reaching a tokenizer fails far from
its cause, which is the whole point — and it **cannot** catch a misspelled residue. There is no
letter left for it to reject.

Examples
--------
>>> outside_alphabet("MKT-V*")
['*', '-']
>>> offending_positions("MKT-V*")
[(3, '-'), (5, '*')]
>>> str(to_protein_sequence("MUOJK"))
'MXXXK'
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Mapping
from typing import cast

from biotite.sequence import Alphabet, ProteinSequence

__all__ = [
    "ALPHABET",
    "AMBIGUOUS",
    "COERCED",
    "STANDARD",
    "STORED",
    "UNKNOWN",
    "InvalidResidueError",
    "ResidueCoercionWarning",
    "check_alphabet",
    "offending_positions",
    "outside_alphabet",
    "to_protein_sequence",
]

#: The twenty proteinogenic residues, one letter each.
STANDARD: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: The six codes that name something other than one of the twenty: ``X`` any residue, ``B``
#: aspartate-or-asparagine, ``Z`` glutamate-or-glutamine, ``J`` leucine-or-isoleucine, ``U``
#: selenocysteine, ``O`` pyrrolysine. UniProt publishes all six.
AMBIGUOUS: frozenset[str] = frozenset("XBZJUO")

#: **What this package accepts as input** — every ASCII letter, since the six above fill the
#: six gaps the twenty leave. Wider than what biotite stores, which is what :data:`COERCED`
#: exists to bridge.
ALPHABET: frozenset[str] = STANDARD | AMBIGUOUS


def _biotite_symbols() -> frozenset[str]:
    """Return every symbol biotite's protein alphabet holds, the stop symbol included.

    The cast is biotite's shape, not a doubt about it: ``ProteinSequence.alphabet`` is a class
    attribute shadowing a property of the same name on the base class, so it answers correctly
    at runtime while every static reading of it — the class attribute and the return of
    ``get_alphabet`` alike — is ``property``.
    """
    alphabet = cast("Alphabet", ProteinSequence.alphabet)
    return frozenset(str(symbol) for symbol in alphabet.get_symbols())


#: **What biotite stores**, read from biotite rather than written out here, so an upgrade
#: cannot silently disagree with us. ``*`` is dropped: biotite carries a stop symbol and a
#: protein sequence reaching this package must not.
STORED: frozenset[str] = _biotite_symbols() - {"*"}

#: Accepted here, unstorable there: the codes :func:`to_protein_sequence` folds to ``X``.
#: Derived rather than listed, so whatever biotite gains, this loses.
COERCED: frozenset[str] = ALPHABET - STORED

#: What an unstorable code becomes. "Unknown" is true of a residue biotite cannot name; ``C``
#: and ``K``, which biotite's own converters would write, are not.
UNKNOWN = "X"

#: Positions the message lists before it gives up and counts the rest. A caller repairing
#: input reads :attr:`InvalidResidueError.offenders`, which is not capped.
_MAX_LISTED = 5

_COERCION = str.maketrans(dict.fromkeys(COERCED, UNKNOWN))


class ResidueCoercionWarning(UserWarning):
    """Warns that codes biotite cannot store were replaced by ``X``.

    Its own category because the gate turns every warning into an error: a category of its own
    is what lets one targeted ``filterwarnings`` entry tolerate this warning, raised on
    purpose, without tolerating any other.
    """


class InvalidResidueError(ValueError):
    """Raised when a sequence holds characters :data:`ALPHABET` excludes.

    A :class:`ValueError`, unlike biotite's ``AlphabetError``, and carrying the offenders as
    data so a caller repairing input never has to parse the message.

    Attributes
    ----------
    offenders : list of tuple of (int, str)
        Every offending ``(index, character)``, zero-based and in order of appearance — what
        :func:`offending_positions` returned. Uncapped, where the message lists five.
    name : str or None
        The accession or identifier the caller named the sequence by, if any.
    """

    def __init__(self, offenders: list[tuple[int, str]], *, name: str | None = None) -> None:
        """Build the error from the offending positions and, if known, whose they are."""
        self.offenders = offenders
        self.name = name
        super().__init__(_invalid_message(offenders, name))


def outside_alphabet(sequence: str) -> list[str]:
    """Return the distinct characters of ``sequence`` that :data:`ALPHABET` excludes.

    Case is not an offence. biotite uppercases what it stores, so lowercase input is accepted
    here and comes back uppercase from :func:`to_protein_sequence`.

    Parameters
    ----------
    sequence : str
        Text to weigh against the alphabet, before any ``ProteinSequence`` exists.

    Returns
    -------
    list of str
        The offending characters, distinct and sorted, in the case they arrived in. Empty when
        nothing offends, which is the whole of what a caller must test.

    Examples
    --------
    >>> outside_alphabet("MKTVU")
    []
    >>> outside_alphabet("mktvu")
    []
    >>> outside_alphabet("MK*T-*")
    ['*', '-']
    """
    return sorted({character for character in sequence if character.upper() not in ALPHABET})


def offending_positions(sequence: str) -> list[tuple[int, str]]:
    """Return every ``(index, character)`` of ``sequence`` that :data:`ALPHABET` excludes.

    One entry per occurrence, in order, so a caller can point at the input rather than search
    it. Indices count from zero and are offsets into ``sequence``, not residue numbers.

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
    return [
        (index, character)
        for index, character in enumerate(sequence)
        if character.upper() not in ALPHABET
    ]


def check_alphabet(sequence: str, *, name: str | None = None) -> None:
    """Raise :class:`InvalidResidueError` if ``sequence`` holds anything outside the alphabet.

    Stricter than biotite in one place and one only: ``*`` is in biotite's alphabet and is
    rejected here, because catching a stray stop or gap before it reaches a tokenizer is what
    this check is for.

    Parameters
    ----------
    sequence : str
        Text to weigh against :data:`ALPHABET`.
    name : str, optional
        The accession or identifier this sequence belongs to, named in the error so a reader
        knows which protein went wrong.

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
    package. Reaching around it means reaching for ``fasta.get_sequence`` or
    ``structure.to_sequence``, which fold ``U`` to ``C`` and ``O`` to ``K`` silently — see
    ADR-0002. The check runs first, so a ``*`` raises rather than being folded.

    Parameters
    ----------
    sequence : str
        The residues, in either case.
    name : str, optional
        The accession or identifier this sequence belongs to, named in both the warning and
        the error so a caller reading either knows which protein it was.

    Returns
    -------
    ProteinSequence
        Uppercase, with every character of :data:`COERCED` replaced by :data:`UNKNOWN`.

    Warns
    -----
    ResidueCoercionWarning
        When anything was folded. Measured in Swiss-Prot: 256 of 575,503 entries carry ``U``
        and 29 carry ``O``; none carries ``J``.

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
        warnings.warn(_coercion_message(counts, name), ResidueCoercionWarning, stacklevel=2)
    return ProteinSequence(folded)


def _prefix(name: str | None) -> str:
    """Return ``'<name>: '`` when there is a sequence to blame by name, else ``''``."""
    return f"{name}: " if name else ""


def _invalid_message(offenders: list[tuple[int, str]], name: str | None) -> str:
    """Return the error text: up to five offending positions, then a count of the rest."""
    listed = ", ".join(f"{character!r} at {index}" for index, character in offenders[:_MAX_LISTED])
    hidden = len(offenders) - _MAX_LISTED
    rest = f", and {hidden} more ({len(offenders)} in total)" if hidden > 0 else ""
    return f"{_prefix(name)}not in the protein alphabet: {listed}{rest}"


def _coercion_message(counts: Mapping[str, int], name: str | None) -> str:
    """Return the warning text: which codes were folded, how many of each, and whose."""
    detail = ", ".join(f"{count} {code}" for code, count in sorted(counts.items()))
    return (
        f"{_prefix(name)}coerced {detail} to {UNKNOWN}; "
        "biotite's protein alphabet cannot store them"
    )
