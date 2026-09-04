"""The :class:`Chain` class — one chain of one structure, and the four things known about it.

A **Chain** is identified by a structure and a label, and it is reached through
``structure["A"]`` rather than built by hand. It is where the two namespaces meet:
:attr:`~Chain.sequence` and :attr:`~Chain.atoms` come from the structure file,
:attr:`~Chain.uniprot` comes from SIFTS or from what the structure was produced from, and
:attr:`~Chain.kind` says whether either of the first two means anything.

**The sequence is built here, over ``res_name``.** ``structure.to_sequence`` is banned by
ADR-0002 and would not work anyway: it raises ``BadStructureError`` on an entry whose water
block is its own chain segment. The residue-to-letter step is
:func:`biotite.structure.info.one_letter_code`, which is CCD-backed and **truthful**, and the
fold from there into what biotite can store is :func:`protein.seq.to_protein_sequence` or
:func:`protein.seq.to_nucleotide_sequence`, which write ``X`` and ``T`` and warn.
``ProteinSequence.convert_letter_3to1`` is the other converter ADR-0002 rules out; do not
reach for it.

**A sequence here is the observed one.** It is built from the residues that have
coordinates, so a disordered loop is absent rather than filled in from ``SEQRES``. That is
what Foldseek reads too, and what a residue-level SIFTS join is against.

**A chain is not always a protein.** :attr:`~Chain.kind` is ``"protein"``, ``"nucleic"`` or
``"other"``, and it is what a caller checks first, because it says which of biotite's two
sequence types :attr:`~Chain.sequence` will answer with. A ligand-only or water-only chain
has neither, so it refuses.

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
    import numpy as np
    import pandas as pd
    import py3Dmol
    from biotite.sequence import NucleotideSequence, ProteinSequence
    from biotite.structure import AtomArray
    from numpy.typing import NDArray

    from protein.search.target import SearchTarget
    from protein.structure.structure import Structure

__all__ = ["KINDS", "Chain", "ChainKind"]

#: What :attr:`Chain.kind` answers with.
ChainKind = Literal["protein", "nucleic", "other"]

#: The three kinds, in one place so a caller can test membership without spelling them.
KINDS: tuple[ChainKind, ...] = ("protein", "nucleic", "other")

#: What a residue with no one-letter code in the Chemical Component Dictionary becomes.
#: ``X`` means unknown, which is true of it.
_UNKNOWN_RESIDUE = "X"

#: The same, for a nucleotide. ``N`` is the nucleic alphabet's unknown; ``X`` is not in it.
_UNKNOWN_NUCLEOTIDE = "N"

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
        case is part of the name. **Not one character either**: plenty of the archive's
        labels are longer.

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

        It is spelled ``id`` because :class:`~protein.core.Protein` spells its accession that
        way, and one :class:`~protein.embed.esm.esmc.Embeddable` protocol over both is the point.

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
        carries the waters and ligands modelled against it, and a rule demanding purity would
        call a solvated DNA chain ``"other"``.

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
    def sequence(self) -> ProteinSequence | NucleotideSequence:
        """The residues this chain was solved for, as one of biotite's two sequence types.

        Built from the polymer residues' ``res_name`` — see this module's docstring for which
        converter does the three-letter step and ADR-0002 for which must not. **Observed, not
        ``SEQRES``**: a residue with no coordinates is not in here.

        Returns
        -------
        biotite.sequence.ProteinSequence or biotite.sequence.NucleotideSequence
            One symbol per residue, in file order, and which type it is follows
            :attr:`kind`. A residue the Chemical Component Dictionary gives no one-letter
            code for becomes ``X`` in a protein and ``N`` in a nucleic acid.

        Raises
        ------
        ValueError
            If :attr:`kind` is ``"other"`` — a ligand-only or water-only chain has no
            polymer to read.
        protein.seq.InvalidResidueError
            If a residue's one-letter code is outside the alphabet for its kind.

        Warns
        -----
        protein.seq.ResidueCoercionWarning
            If a protein chain holds ``SEC`` or ``PYL``, or a nucleic one holds uracil.
            biotite can store none of the three, and this package folds them loudly where
            ADR-0002's banned converters are silent.

        Examples
        --------
        >>> str(Structure("1UBQ")["A"].sequence)[:5]      # doctest: +SKIP
        'MQIFV'
        >>> str(Structure("1BNA")["A"].sequence)          # doctest: +SKIP
        'CGCGAATTCGCG'
        """
        if self.kind == "other":
            raise ValueError(
                f"chain {self.id} is other, so it is ligand or solvent and has no polymer "
                f"sequence. Check `.kind` before asking; `.atoms` is what such a chain "
                f"answers with."
            )
        atoms = self.atoms
        if self.kind == "protein":
            letters = self._letters(struc.filter_amino_acids(atoms), _UNKNOWN_RESIDUE)
            return seq.to_protein_sequence(letters, name=self.id)
        letters = self._letters(struc.filter_nucleotides(atoms), _UNKNOWN_NUCLEOTIDE)
        return seq.to_nucleotide_sequence(letters, name=self.id)

    def _letters(self, mask: NDArray[np.bool_], unknown: str) -> str:
        """Return one Chemical Component Dictionary letter per residue the mask keeps."""
        _, names = struc.get_residues(self.atoms[mask])
        return "".join(struc_info.one_letter_code(str(name)) or unknown for name in names)

    @property
    def uniprot(self) -> tuple[str, ...]:
        """The UniProt accessions this chain belongs to: its structure's, else SIFTS'.

        **Where the structure was produced from an accession, that is the answer.** A
        prediction carries the accessions it was folded from, and returning them is what
        keeps it from reading like a deposited entry SIFTS maps nothing to. The map is
        read whole: a structure that carries one answers every chain from it and never asks
        SIFTS, so a folded complex's DNA chains do not send the id of something that is no
        PDB entry off to a map that may not be prepared.

        **Otherwise SIFTS, and never the structure file.** Every mmCIF carries its own
        ``_struct_ref_seq`` and the two disagree — ``1UBQ`` chain A is ``P62988`` in the file
        and ``P0CG48`` here — because the file holds the depositor's reference frozen at
        deposition and SIFTS holds PDBe's re-curated one. Reading the file would break the
        round trip with :attr:`protein.core.Protein.structures`, which is why a written
        prediction's accessions are not read back either (ADR-0005).

        Returns
        -------
        tuple of str
            The accessions, in accession order from SIFTS and as given from a structure's
            own map. A tuple and never a scalar, because a chain may carry more than one.
            ``()`` is a real answer and is **not** ``None`` — a nucleic-acid chain, a ligand
            chain, an entry SIFTS never curated, any id SIFTS does not carry, and a chain its
            structure's map does not name.

        Raises
        ------
        protein.sifts.SiftsNotDownloadedError
            If SIFTS was asked and the map is not prepared on this machine. Distinct from
            ``()``, and not caught into it: one means *nobody built the map here* and the
            other means *this chain has no protein*.

        Examples
        --------
        >>> Structure("1UBQ")["A"].uniprot                # doctest: +SKIP
        ('P0CG48',)
        """
        produced_from = self.structure.accessions
        if produced_from is not None:
            return produced_from.get(self.chain_id, ())

        # The module and not the function, so one `monkeypatch.setattr(sifts, ...)` reaches
        # every caller. Deferred as well: `protein.sifts` imports pandas.
        from protein import sifts

        return sifts.accessions_for(self.structure.id, self.chain_id)

    def search(self, database: SearchTarget | str, **kwargs: Any) -> pd.DataFrame:
        """Search this one chain against ``database`` with Foldseek.

        Foldseek reads a file, so the chain's coordinates are written to one first — inside
        the tool's own scratch directory, which is removed however the search ends.

        **The file is named** :attr:`id`, because Foldseek reports a one-chain query under
        its file's stem, so the ``query`` column says ``1UBQ_A`` and joins straight onto
        SIFTS.

        **It is mmCIF**, which is also the format a chain label of more than one character
        survives — ``foldseek convert2pdb`` truncates such a label in silence, and a PDB file
        has one column for it.

        Parameters
        ----------
        database : protein.search.target.SearchTarget or str
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

    def view(self, **kwargs: Any) -> py3Dmol.view:
        """Return a 3D viewer holding this chain's atoms, and no others.

        The chain is serialised on its own rather than filtered in the viewer, so what is
        drawn is what :attr:`atoms` holds. Nothing renders by itself — see
        :meth:`Structure.view`.

        Parameters
        ----------
        **kwargs : Any
            Forwarded to :func:`protein.structure.view.view` — ``width``, ``height`` and
            ``style``. The name is this chain's :attr:`id`, so a saved page says which
            chain it holds.

        Returns
        -------
        py3Dmol.view
            A ribbon coloured N to C. A ligand or water chain carries no cartoon and draws
            nothing until a caller names another style.

        Examples
        --------
        >>> with open("1ubq_a.html", "w") as page:                    # doctest: +SKIP
        ...     Structure("1UBQ")["A"].view().write_html(page)
        """
        from protein.structure import view as _view

        return _view.view(self.atoms, name=self.id, **kwargs)

    def __len__(self) -> int:
        """Return the number of atoms this chain has.

        Atoms and not residues: a residue count for a chain that is half ligand would be a
        claim about which residues count.

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
