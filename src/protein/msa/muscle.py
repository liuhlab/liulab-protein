"""Lining up sequences the caller already holds, by driving MUSCLE through biotite.

No **Database** is involved and none is searched — this is the verb for homologues that came
from a paper, a colleague or an earlier search. biotite's ``Muscle5App`` owns the temporary
files, the arguments and the parsing; this package locates the binary and hands the sequences
over.

The application layer is imported inside :func:`align` rather than at module scope, so
``import protein`` does not pay for it.

Examples
--------
>>> from protein.msa import align
>>> align({"P01308": "MKTAYIAK", "Q6YK33": "MKTYIAK"}, query="P01308")  # doctest: +SKIP
MSA(depth 2, 8 match states)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from protein import seq
from protein.external import Muscle
from protein.msa.msa import MSA

if TYPE_CHECKING:
    from collections.abc import Iterable

    from protein.external import ExternalTool

__all__ = ["align"]

#: What MUSCLE needs before it will align anything.
_MIN_ALIGNED = 2


def align(
    sequences: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    query: str,
    tool: ExternalTool | None = None,
) -> MSA:
    """Align sequences the caller already holds, and anchor the result on ``query``.

    A function and not a method: MUSCLE takes a **set**, and this package has no class for
    one.

    ``Muscle5App`` — version 5, not the ``MuscleApp`` that wraps version 3 — is what drives
    the binary. This package builds the sequences through
    :func:`protein.seq.to_protein_sequence` and hands both them and the located path over.
    MUSCLE aligns symmetrically, so the result is anchored by
    :meth:`~protein.msa.msa.MSA.compress` before it is returned.

    Parameters
    ----------
    sequences : mapping of str to str, or iterable of tuple of (str, str)
        ``(header, residues)`` pairs, which is what :func:`protein.io.fasta.read_records`
        yields, or a mapping of the same. Residues are ungapped.
    query : str
        The header of the sequence to anchor on. It becomes row 0.
    tool : protein.external.ExternalTool, optional
        Where the binary is. Defaults to :class:`~protein.external.Muscle`.

    Returns
    -------
    MSA
        Query-anchored, in the order the sequences arrived except that the query leads.

    Raises
    ------
    ValueError
        If fewer than two sequences were given. MUSCLE aligns a set, not a sequence.
    LookupError
        If no sequence carries the ``query`` header. The message names the headers there are.
    protein.seq.InvalidResidueError
        If a sequence holds anything outside :data:`protein.seq.ALPHABET`.
    protein.external.ToolNotFoundError
        If ``muscle`` is not installed.

    Warns
    -----
    protein.seq.ResidueCoercionWarning
        If a sequence holds ``U``, ``O`` or ``J``, which this package folds to ``X``.

    Examples
    --------
    >>> msa = align({"P01308": "MKTAYIAK", "Q6YK33": "MKTYIAK"}, query="P01308")  # doctest: +SKIP
    >>> msa.query_header                                                          # doctest: +SKIP
    'P01308'
    """
    from biotite.application.muscle import Muscle5App

    records = _records(sequences)
    if len(records) < _MIN_ALIGNED:
        raise ValueError(
            f"align takes at least {_MIN_ALIGNED} sequences and was given {len(records)}; "
            f"one sequence is not an alignment."
        )
    headers = [header for header, _ in records]
    if query not in headers:
        raise LookupError(
            f"no sequence is headed {query!r}, so there is nothing to anchor on. The "
            f"headers given are {headers}."
        )
    typed = [seq.to_protein_sequence(residues, name=header) for header, residues in records]
    located = tool if tool is not None else Muscle()
    alignment = Muscle5App.align(typed, bin_path=located.path)
    rows = zip(headers, alignment.get_gapped_sequences(), strict=True)
    return MSA(rows).compress(headers.index(query))


def _records(sequences: Mapping[str, str] | Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return the ``(header, residues)`` pairs, from a mapping or from pairs.

    The casts are the argument's shape and not a doubt about it: a ``Mapping`` is also an
    ``Iterable``, so no static reading can subtract one branch from the other.
    """
    if isinstance(sequences, Mapping):
        return list(cast("Mapping[str, str]", sequences).items())
    return list(cast("Iterable[tuple[str, str]]", sequences))
