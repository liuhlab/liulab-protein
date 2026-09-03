"""The :class:`Protein` class — one amino-acid sequence and what is known about it.

A **Protein** is identified by a UniProt accession. A PDB id addresses a **Structure**
instead, and SIFTS joins the two many-to-many in both directions, so neither owns the other.
This class therefore **carries no coordinates**: there is no ``.structure``, no
``foldseek_search()`` and no ``from_structure(path)``. Foldseek takes a structure, so its
search belongs to the classes that have one, and a file on disk becomes one through
``Structure.from_file``.

There is no ``embed()`` either. ESM-C holds 1.33 GB of weights across calls, so it is an
object a caller constructs and keeps — ``ESMC`` in ``protein.embed`` — rather than a method
with nowhere honest to put them. :meth:`SearchMixin.search` stays, because mmseqs is a
subprocess that holds nothing between calls and reaches its database by path. That asymmetry
is a rule of this package, *resident state gets an object and a subprocess does not*, and not
an oversight.

:attr:`Protein.sequence` is a :class:`~biotite.sequence.ProteinSequence` and **not a**
``str``. So ``p.sequence == "MKT"`` is ``False``, and ``str(p.sequence)`` is how ESM-C and
mmseqs get their string. That diverges from `liulab-genome`, whose ``DNA`` is a typed ``str``
subclass, and it diverges deliberately: genome slices whole chromosomes and needs ``str``
cheapness, while a protein carries a few hundred residues into typed biotite calls.

The sequence is **checked and folded at construction**, through
:func:`protein.seq.to_protein_sequence` and never ``ProteinSequence`` directly — ADR-0001
says why this repo validates where `liulab-genome`'s ADR-0005 declines to.

Nothing here imports torch, at module level or in a method body, and a test asserts it.

Examples
--------
>>> from protein import Protein
>>> p = Protein("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", id="P12345")
>>> p
Protein('P12345', 33 aa)
>>> p.length
33
>>> p[:3]
'MKT'
>>> p.sequence == "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
False
>>> str(p.sequence)[:3]
'MKT'
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING, Any, Self

from protein import seq
from protein.search import SearchMixin

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from biotite.sequence import ProteinSequence

__all__ = ["Protein"]


class Protein(SearchMixin):
    """One protein: its residues, the names it goes by, and whatever else is known.

    Parameters
    ----------
    sequence : str
        The residues, in either case. Checked against :data:`protein.seq.ALPHABET` and
        folded to what biotite can store before anything else happens, so a constructed
        ``Protein`` never holds a stop symbol, a gap, a digit or a space (ADR-0001).
    id : str, optional
        The UniProt accession, e.g. ``"P12345"`` — what this protein is addressed by, and
        what names it in an error, a warning and :meth:`__repr__`.
    name : str, optional
        The entry name, e.g. ``"INS_HUMAN"``. Distinct from ``id``: an accession is stable
        and an entry name is not. Nothing in v1 fills this from a FASTA header — see
        :func:`protein.io.fasta.split_header`.
    description : str, optional
        Free text, e.g. what follows the accession in a FASTA header.
    metadata : collections.abc.Mapping, optional
        Anything else known about this protein. Copied, so a later change to the mapping
        passed in does not reach the protein.

    Attributes
    ----------
    sequence : biotite.sequence.ProteinSequence
        The residues. **Not a** ``str`` — see this module's docstring.
    id : str or None
        The UniProt accession.
    name : str or None
        The entry name.
    description : str or None
        Free text.
    metadata : dict
        Eager and total: always a mapping, empty when nothing was given, never ``None``.

    Raises
    ------
    protein.seq.InvalidResidueError
        If ``sequence`` holds anything outside :data:`protein.seq.ALPHABET`. Its
        ``.offenders`` carries every offending position, so a caller repairing input need
        not parse the message.

    Warns
    -----
    protein.seq.ResidueCoercionWarning
        If ``sequence`` holds ``U``, ``O`` or ``J``, which biotite cannot store and this
        package folds to ``X``.

    Examples
    --------
    >>> p = Protein("MKTAY", id="P12345", description="a short one")
    >>> len(p), p.length
    (5, 5)
    >>> print(p.to_fasta(), end="")
    >P12345 a short one
    MKTAY
    """

    def __init__(
        self,
        sequence: str,
        *,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.description = description
        # Eager and total, and a copy: a `Protein` that shares a caller's dict would change
        # under it. Empty is the answer for "nothing is known", never `None`.
        self.metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        # The one door from text to a `ProteinSequence` (ADR-0002). Called with the
        # accession so the error and the coercion warning both name which protein it was.
        self.sequence: ProteinSequence = seq.to_protein_sequence(sequence, name=id)

    @classmethod
    def from_fasta(cls, path: str | Path) -> Self:
        """Read the one record in the FASTA file at ``path``.

        For a file holding several, use :func:`protein.io.fasta.read_proteins`, which yields
        one ``Protein`` per record. This constructor refuses rather than taking the first,
        because taking the first is the failure nobody notices.

        Parameters
        ----------
        path : str or pathlib.Path
            The FASTA file. A name ending ``.gz`` is decompressed.

        Returns
        -------
        Protein
            Built from the record: the header's first token becomes :attr:`id` and the rest
            becomes :attr:`description`.

        Raises
        ------
        ValueError
            If the file holds no record, or more than one.
        protein.seq.InvalidResidueError
            If the record's sequence holds anything outside :data:`protein.seq.ALPHABET`.

        Examples
        --------
        >>> Protein.from_fasta("tests/data/uniprot_p01308.fasta")  # doctest: +SKIP
        Protein('sp|P01308|INS_HUMAN', 110 aa)
        """
        from protein.io import fasta

        records = list(islice(fasta.read_records(path), 2))
        if not records:
            raise ValueError(f"{path}: holds no FASTA record.")
        if len(records) > 1:
            raise ValueError(
                f"{path}: holds more than one FASTA record, and Protein.from_fasta reads "
                f"exactly one. Use protein.io.fasta.read_proteins to read them all."
            )
        header, sequence = records[0]
        identifier, description = fasta.split_header(header)
        return cls(sequence, id=identifier, description=description)

    def to_fasta(self, path: str | Path | None = None, *, line_width: int = 80) -> str:
        """Return this protein as FASTA text, and write it to ``path`` when one is given.

        The header is :attr:`id` and :attr:`description` joined by one space, which is what
        :meth:`from_fasta` splits — so a header of that shape round-trips, and one padded
        with a run of spaces comes back with a single one. :attr:`name` and :attr:`metadata`
        are not written: a FASTA header has one field and this package does not spell a
        UniProt one.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Where to write. A name ending ``.gz`` is compressed. Omitted, nothing is
            written and only the text comes back.
        line_width : int, default 80
            Residues per line, biotite's default.

        Returns
        -------
        str
            The record, newline-terminated.

        Examples
        --------
        >>> print(Protein("MKTAY", id="P12345").to_fasta(line_width=3), end="")
        >P12345
        MKT
        AY
        """
        from protein.io import fasta

        record = (fasta.join_header(self.id, self.description), str(self.sequence))
        text = fasta.format_records([record], line_width=line_width)
        if path is not None:
            fasta.write_records(path, [record], line_width=line_width)
        return text

    @property
    def length(self) -> int:
        """Residue count — the same number :func:`len` gives.

        Examples
        --------
        >>> Protein("MKTAY").length
        5
        """
        return len(self.sequence)

    def __len__(self) -> int:
        """Return the residue count."""
        return len(self.sequence)

    def __getitem__(self, key: int | slice) -> str:
        """Return residues as a ``str`` — **a slice of a protein is not a protein**.

        A subsequence has no accession, no description and no metadata, so returning a
        ``Protein`` would invent one. A ``str`` is what the caller wanted anyway; wrap it in
        :class:`Protein` yourself if a slice really is its own protein.

        Parameters
        ----------
        key : int or slice
            A residue offset, zero-based, or a range of them.

        Returns
        -------
        str
            One residue for an integer key, the subsequence for a slice.

        Examples
        --------
        >>> p = Protein("MKTAY")
        >>> p[0], p[-1], p[1:3]
        ('M', 'Y', 'KT')
        """
        return str(self.sequence[key])

    def __repr__(self) -> str:
        """Return e.g. ``Protein('P12345', 214 aa)``.

        Examples
        --------
        >>> Protein("MKTAY", id="P12345")
        Protein('P12345', 5 aa)
        >>> Protein("MKTAY")
        Protein(None, 5 aa)
        """
        return f"{type(self).__name__}({self.id!r}, {len(self)} aa)"
