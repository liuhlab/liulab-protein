"""The :class:`Chain` class — one chain of one structure, and the four things known about it.

A **Chain** is identified by a structure and a label, and it is reached through
``structure["A"]`` rather than built by hand. It is where the two namespaces meet:
:attr:`~Chain.sequence` and :attr:`~Chain.atoms` come from the structure file,
:attr:`~Chain.uniprot` comes from SIFTS, and :attr:`~Chain.kind` says whether either of the
first two means anything.

**The sequence is built here, over ``res_name``.** ``structure.to_sequence`` is banned by
ADR-0002 and would not work anyway: it raises ``BadStructureError`` on every real entry
tested, because a water block is its own chain segment holding neither amino acids nor
nucleotides. The residue-to-letter step is
:func:`biotite.structure.info.one_letter_code`, which is CCD-backed and **truthful** —
``MSE`` is ``M``, ``MLY`` is ``K``, ``SEC`` is ``U`` — and the fold from there into what
biotite can store is :func:`protein.seq.to_protein_sequence`, which writes ``X`` and warns.

One measurement on biotite 1.4.0 is the whole of why that converter and not the other:
``info.one_letter_code("SEC")`` is ``'U'`` and
``ProteinSequence.convert_letter_3to1("SEC")`` is ``'C'``. The second is not one of the
three names ADR-0002 lists and it has exactly the defect ADR-0002 is about — selenocysteine
reported as cysteine, a different residue, with no signal. It also raises ``KeyError`` for
every modified residue, ``MSE`` and ``SEP`` and ``MLY`` among them, so it is wrong for this
in both directions. Do not reach for it.

**A sequence here is the observed one.** It is built from the residues that have
coordinates, so a disordered loop is absent rather than filled in from ``SEQRES``. That is
what Foldseek reads too, and what a residue-level SIFTS join is against.

**A chain is not always a protein.** :attr:`~Chain.kind` is ``"protein"``, ``"nucleic"`` or
``"other"``, and :attr:`~Chain.sequence` refuses on the last two rather than answering with
something a tokenizer would accept — see :meth:`Chain.sequence`.

Examples
--------
>>> from protein import Structure
>>> chain = Structure("1UBQ")["A"]                        # doctest: +SKIP
>>> chain.kind, chain.id                                  # doctest: +SKIP
('protein', '1UBQ_A')
>>> chain.uniprot                                         # doctest: +SKIP
('P0CG48',)
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import biotite.structure as struc
from biotite.structure import info as struc_info

from protein import seq
from protein.io import structure as _io

if TYPE_CHECKING:
    import pandas as pd
    from biotite.sequence import ProteinSequence
    from biotite.structure import AtomArray

    from protein.search.mmseqs import SearchTarget
    from protein.structure.structure import Structure

__all__ = ["KINDS", "Chain", "ChainKind"]

#: What :attr:`Chain.kind` answers with.
ChainKind = Literal["protein", "nucleic", "other"]

#: The three kinds, in one place so a caller can test membership without spelling them.
KINDS: tuple[ChainKind, ...] = ("protein", "nucleic", "other")

#: What a residue with no one-letter code in the Chemical Component Dictionary becomes.
#: ``X`` means unknown, which is true of it.
_UNKNOWN_RESIDUE = "X"

#: What a chain's query file is written as. mmCIF, because it is the format that can hold a
#: chain label of more than one character and the only one this package writes.
_QUERY_FORMAT = "cif"

#: What joins an entry id to a chain label to make a chain's key. SIFTS and Foldseek both
#: spell it this way, so ``1ubq_A`` from a hit table and ``chain.id`` are one string.
_KEY_SEPARATOR = "_"


class Chain:
    """One chain of one structure: its atoms, what it is, what it reads as, and whose it is.

    Reached through ``structure[label]``, which is where a label is checked. Constructing
    one directly for a label the structure does not carry gives a chain with no atoms rather
    than an error.

    Parameters
    ----------
    structure : protein.structure.Structure
        The entry this chain belongs to. Held rather than copied from: the file, the id and
        the parsed atoms all stay in one place.
    chain_id : str
        The author chain label — ``auth_asym_id``, which is what SIFTS keys on — exactly as
        the file spells it. **Not folded**: ``10EG`` carries both an ``A`` and an ``a``, so
        case is part of the name. **Not one character either**: 12% of the archive's labels
        are longer.

    Attributes
    ----------
    structure : protein.structure.Structure
        The entry this chain belongs to.
    chain_id : str
        The author chain label.

    Examples
    --------
    >>> from protein import Structure
    >>> chain = Structure("1UBQ")["A"]                    # doctest: +SKIP
    >>> chain                                             # doctest: +SKIP
    Chain('1UBQ_A', protein, 76 residues)
    """

    def __init__(self, structure: Structure, chain_id: str) -> None:
        self.structure = structure
        self.chain_id = chain_id

    @property
    def id(self) -> str:
        """The chain's key, ``<entry>_<label>`` — e.g. ``"1UBQ_A"``.

        The convention SIFTS and Foldseek both use, so a hit table's ``query`` column and
        this string are comparable. Neither half is folded: the entry is as the structure
        spells it and the label is as the file does.

        It is also what :attr:`protein.embed.Embedding.source` records, which is the reason
        it is spelled ``id`` — :class:`~protein.core.Protein` spells its accession that way,
        and one :class:`~protein.embed.esm.Embeddable` protocol over both is the point.

        Returns
        -------
        str
            The key.

        Examples
        --------
        >>> Structure("1UBQ")["A"].id                     # doctest: +SKIP
        '1UBQ_A'
        """
        return f"{self.structure.id}{_KEY_SEPARATOR}{self.chain_id}"

    @cached_property
    def atoms(self) -> AtomArray:
        """Every atom carrying this chain's label, in file order.

        The whole chain in one array however many segments the file splits it into — waters
        and ligands sharing the label included, since a chain is not filtered by what a
        caller happens to want from it.

        Returns
        -------
        biotite.structure.AtomArray
            The chain's atoms, out of the structure's first model.

        Examples
        --------
        >>> Structure("1UBQ")["A"].atoms.array_length()   # doctest: +SKIP
        660
        """
        return _io.chain_atoms(self.structure.atoms, self.chain_id)

    @cached_property
    def kind(self) -> ChainKind:
        """What this chain is: ``"protein"``, ``"nucleic"`` or ``"other"``.

        Both filters are biotite's and both are backed by the PDB Chemical Component
        Dictionary, so a modified residue — ``MSE``, ``SEP``, ``TPO`` — counts as the amino
        acid it is.

        **Whichever filter matches more atoms wins**, rather than *all* of them: a chain
        carries the waters and ligands modelled against it, so ``1BNA`` chain A is 243
        nucleotide atoms and 37 water atoms, and a rule demanding purity would call it
        ``"other"``.

        Returns
        -------
        {"protein", "nucleic", "other"}
            ``"other"`` when neither filter matches anything — a ligand-only or water-only
            chain.

        Examples
        --------
        >>> Structure("1BNA")["A"].kind                   # doctest: +SKIP
        'nucleic'
        """
        atoms = self.atoms
        amino = int(struc.filter_amino_acids(atoms).sum())
        nucleic = int(struc.filter_nucleotides(atoms).sum())
        if amino == 0 and nucleic == 0:
            return "other"
        return "protein" if amino >= nucleic else "nucleic"

    @cached_property
    def sequence(self) -> ProteinSequence:
        """The residues this chain was solved for, as biotite's ``ProteinSequence``.

        Built from the amino-acid residues' ``res_name`` — see this module's docstring for
        which converter does the three-letter step and which two must not. **Observed, not
        ``SEQRES``**: a residue with no coordinates is not in here.

        Returns
        -------
        biotite.sequence.ProteinSequence
            One symbol per residue, in file order. A residue the Chemical Component
            Dictionary gives no one-letter code for becomes ``X``.

        Raises
        ------
        ValueError
            If :attr:`kind` is not ``"protein"``. A nucleic-acid or ligand chain has no
            protein sequence, and this is the one place that can say so: the
            :class:`~protein.embed.esm.Embeddable` protocol ``ESMC.embed`` checks knows
            nothing of :attr:`kind`, so refusing here is what stops a DNA chain reaching the
            tokenizer.
        protein.seq.InvalidResidueError
            If a residue's one-letter code is outside :data:`protein.seq.ALPHABET`.

        Warns
        -----
        protein.seq.ResidueCoercionWarning
            If the chain holds ``SEC`` or ``PYL``, whose truthful codes ``U`` and ``O``
            biotite cannot store and this package folds to ``X`` — loudly, where
            ``to_sequence`` would have written ``C`` and ``K`` in silence.

        Examples
        --------
        >>> str(Structure("1UBQ")["A"].sequence)[:5]      # doctest: +SKIP
        'MQIFV'
        """
        if self.kind != "protein":
            raise ValueError(
                f"chain {self.id} is {self.kind}, not protein, so it has no amino-acid "
                f"sequence. Check `.kind` before asking; `.atoms` is what a non-protein "
                f"chain answers with."
            )
        atoms = self.atoms
        residues = atoms[struc.filter_amino_acids(atoms)]
        _, names = struc.get_residues(residues)
        letters = "".join(
            struc_info.one_letter_code(str(name)) or _UNKNOWN_RESIDUE for name in names
        )
        return seq.to_protein_sequence(letters, name=self.id)

    @property
    def uniprot(self) -> tuple[str, ...]:
        """The UniProt accessions SIFTS maps this chain to.

        **SIFTS and never the structure file.** Every mmCIF carries its own
        ``_struct_ref_seq`` and the two disagree — ``1UBQ`` chain A is ``P62988`` in the
        file and ``P0CG48`` here — because the file holds the depositor's reference frozen
        at deposition and SIFTS holds PDBe's re-curated one. Reading the file would break
        the round trip with :attr:`protein.core.Protein.structures`.

        Returns
        -------
        tuple of str
            The accessions, in accession order. A tuple and never a scalar: 1.00% of chains
            carry more than one, up to four. ``()`` is a real answer and is **not** ``None``
            — a nucleic-acid chain, a ligand chain, an entry SIFTS never curated, and any id
            SIFTS does not carry, which is what a chain of a :meth:`Structure.from_file`
            that is no PDB entry gets.

        Raises
        ------
        protein.sifts.SiftsNotDownloadedError
            If the map is not prepared on this machine. Distinct from ``()``, and not caught
            into it: one means *nobody built the map here* and the other means *this chain
            has no protein*.

        Examples
        --------
        >>> Structure("1UBQ")["A"].uniprot                # doctest: +SKIP
        ('P0CG48',)
        """
        # The module and not the function: one `monkeypatch.setattr(sifts, ...)` then takes
        # every caller offline, which is how this lane is tested. Deferred as well, because
        # `protein.sifts` reads a 41 MB table and imports pandas to do it.
        from protein import sifts

        return sifts.accessions_for(self.structure.id, self.chain_id)

    def search(self, database: SearchTarget | str, **kwargs: Any) -> pd.DataFrame:
        """Search this one chain against ``database`` with Foldseek.

        Foldseek reads a file, so the chain's coordinates are written to one first — inside
        the tool's own scratch directory, which is removed however the search ends.

        **The file is named** :attr:`id`, because a one-chain query is reported under its
        file's stem: measured on 10-941cd33, ``zzz_Q1.pdb`` holding a chain labelled ``A``
        comes back as query ``zzz_Q1``. So the ``query`` column says ``1UBQ_A`` and joins
        straight onto SIFTS.

        **It is mmCIF**, which is also the format a chain label of more than one character
        survives — ``foldseek convert2pdb`` truncates such a label in silence, and a PDB
        file has one column for it.

        Parameters
        ----------
        database : protein.search.mmseqs.SearchTarget or str
            What to search against: a **Database**, or the name of a registered one.
        **kwargs : Any
            Forwarded to :func:`protein.search.foldseek.search` — ``sensitivity``,
            ``evalue``, ``max_seqs``, ``threads``, ``extra`` and ``tool``.

        Returns
        -------
        pandas.DataFrame
            One row per hit, in Foldseek's column order, with ``fident`` — a **fraction** —
            as the identity column. Its ``query`` column is :attr:`id`.

        Raises
        ------
        LookupError
            If ``database`` names nothing registered.
        protein.external.ToolNotFoundError
            If ``foldseek`` is not installed.

        Examples
        --------
        >>> Structure("1UBQ")["A"].search("pdb").loc[0, "query"]   # doctest: +SKIP
        '1UBQ_A'
        """
        from protein.external import Foldseek
        from protein.search import foldseek

        tool = kwargs.pop("tool", None)
        if tool is None:
            tool = Foldseek()
        with tool.scratch_dir("chain") as work:
            query = work / f"{self.id}.{_QUERY_FORMAT}"
            _io.write_atoms(query, self.atoms)
            return foldseek.search(query, database, tool=tool, **kwargs)

    def __len__(self) -> int:
        """Return the number of atoms this chain has.

        Atoms and not residues: a chain *is* its atoms, and a residue count for a chain that
        is half ligand would be a claim about which residues count.

        Examples
        --------
        >>> len(Structure("1UBQ")["A"])                   # doctest: +SKIP
        660
        """
        return self.atoms.array_length()

    def __repr__(self) -> str:
        """Return e.g. ``Chain('1UBQ_A', protein, 660 atoms)``.

        Unlike :meth:`Structure.__repr__` this one parses, because a chain cannot be built
        without the structure having been read.

        Examples
        --------
        >>> Structure("1UBQ")["A"]                        # doctest: +SKIP
        Chain('1UBQ_A', protein, 660 atoms)
        """
        return f"{type(self).__name__}({self.id!r}, {self.kind}, {len(self)} atoms)"
