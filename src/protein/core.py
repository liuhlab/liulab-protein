"""The :class:`Protein` class — one amino-acid sequence and what is known about it.

A **Protein** is identified by a UniProt accession and **carries no coordinates**: SIFTS
joins it to a **Structure** many-to-many in both directions, so neither owns the other. There
is no ``embed()`` either, and ``CONTEXT.md`` carries both rules.

:attr:`Protein.sequence` is a :class:`~biotite.sequence.ProteinSequence` and **not a**
``str``, so ``p.sequence == "MKT"`` is ``False`` and ``str(p.sequence)`` is what a tokenizer
or a subprocess gets. It is checked and folded at construction, by
:func:`protein.seq.to_protein_sequence` and never ``ProteinSequence`` directly (ADR-0001).

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

    import pandas as pd
    from biotite.sequence import ProteinSequence

    from protein.msa import MSA
    from protein.search.mmseqs import SearchTarget

__all__ = ["Protein"]


class Protein(SearchMixin):
    """One protein: its residues, the names it goes by, and whatever else is known.

    Parameters
    ----------
    sequence : str
        The residues, in either case. Checked and folded before anything else happens, so a
        constructed ``Protein`` never holds a stop symbol, a gap, a digit or a space.
    id : str, optional
        The UniProt accession, which names this protein in an error, a warning and
        :meth:`__repr__`.
    name : str, optional
        The entry name, e.g. ``"INS_HUMAN"``. Distinct from ``id``: an accession is stable
        and an entry name is not.
    description : str, optional
        Free text, e.g. what follows the accession in a FASTA header.
    metadata : collections.abc.Mapping, optional
        Anything else known about this protein. Copied.

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
        Always a mapping, empty when nothing was given, never ``None``.

    Raises
    ------
    protein.seq.InvalidResidueError
        If ``sequence`` holds anything outside :data:`protein.seq.ALPHABET`. Its
        ``.offenders`` carries every offending position.

    Warns
    -----
    protein.seq.ResidueCoercionWarning
        If ``sequence`` holds ``U``, ``O`` or ``J``, which this package folds to ``X``.

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
        # A copy: a `Protein` sharing a caller's dict would change under it.
        self.metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        # The one door from text to a `ProteinSequence` (ADR-0002), named so the error and
        # the coercion warning both say which protein it was.
        self.sequence: ProteinSequence = seq.to_protein_sequence(sequence, name=id)

    @classmethod
    def from_fasta(cls, path: str | Path) -> Self:
        """Read the one record in the FASTA file at ``path``.

        For a file holding several, use :func:`protein.io.fasta.read_proteins`. This
        constructor refuses rather than silently taking the first.

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
        :meth:`from_fasta` splits, so a header of that shape round-trips. :attr:`name` and
        :attr:`metadata` are not written.

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

    def msa(self, database: SearchTarget | str, **kwargs: Any) -> MSA:
        """Search ``database`` and return the alignment, in memory.

        :meth:`search` exactly, one step further along: the same tool, the same database, and
        a value back rather than a file. **There is no output path** — an alignment is a
        value, like a hit table, and :meth:`MSA.write` is how one is kept.

        ``database`` is required and nothing is shipped or adopted behind it. A shallow set
        quietly standing in for a deep one is a wrong answer that looks right.

        A thin entry point onto :func:`protein.msa.search`, where the knobs are documented.

        Parameters
        ----------
        database : protein.search.mmseqs.SearchTarget or str
            What to search against: a **Database**, or the name of a registered one.
        **kwargs : Any
            Forwarded to :func:`protein.msa.search`. ``query_name`` is filled in from
            :attr:`id` unless it is given here.

        Returns
        -------
        protein.msa.MSA
            Query-anchored, this protein's sequence in row 0. Depth 1 when the search found
            nothing.

        Raises
        ------
        LookupError
            If ``database`` names nothing registered.
        protein.external.ToolNotFoundError
            If ``mmseqs`` is not installed.
        RuntimeError
            If the search exits non-zero.

        Examples
        --------
        >>> from protein import Protein
        >>> p = Protein("MKTAYIAKQRQISFVKSHFSRQ", id="P12345")
        >>> p.msa("uniref30").depth                             # doctest: +SKIP
        1281
        """
        from protein import msa
        from protein.search import mmseqs

        kwargs.setdefault("query_name", self.id or mmseqs.DEFAULT_QUERY_NAME)
        return msa.search(str(self.sequence), database, **kwargs)

    @property
    def structures(self) -> pd.DataFrame:
        """Every PDB chain segment SIFTS maps this protein's accession to.

        The reverse direction of ``Chain.uniprot``, over the same table, and it reads SIFTS
        alone — never a structure file's own ``_struct_ref_seq``, which disagrees. A frame
        rather than ids, because an entry id alone does not say which chain to fetch
        coordinates for.

        Returns
        -------
        pandas.DataFrame
            :data:`protein.sifts.COLUMNS`, one row per mapped segment, empty when SIFTS
            carries nothing for this accession. ``res_beg``/``res_end`` and
            ``sp_beg``/``sp_end`` are both verbatim; no offset is computed.

        Raises
        ------
        ValueError
            If this protein has no :attr:`id`. An accession is what SIFTS is keyed on, so
            there is nothing to look up.
        protein.sifts.SiftsNotDownloadedError
            If the map is not prepared on this machine. Distinct from an empty frame, which
            means SIFTS genuinely maps this accession to nothing.

        Examples
        --------
        >>> Protein("MQIFVKTLTG", id="P0CG48").structures.iloc[0]["pdb"]  # doctest: +SKIP
        '11sy'
        """
        from protein import sifts

        if self.id is None:
            raise ValueError(
                "this Protein has no id, and SIFTS is keyed on a UniProt accession, so "
                "there is nothing to look up. Construct it with `Protein(..., id=...)`."
            )
        return sifts.structures_for(self.id)

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
        ``Protein`` would invent one. Wrap it yourself where a slice really is its own.

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
