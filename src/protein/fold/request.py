"""The :class:`FoldingRequest` — what goes in, and every check upstream does not make.

A request is **transient**: a caller builds one at the call site, folds it and drops it. It
is named for that job rather than for a molecule, because what separates it from a
:class:`~protein.structure.Structure` is lifetime. It holds one :class:`ChainRequest` per
chain and **no output path** — where the answer is written is not an input, so one request
can be folded to two destinations.

**The checks live here, and they are ours to make.** Upstream requires neither that an
alignment's row 0 equals the query nor that its length agrees, and past its own sibling-row
check everything degrades in silence: a long row is cut, a short one gap-filled, a short
alignment has its last column repeated across the tail. That is ADR-0002's failure shape —
a confidently wrong answer where an error belongs.

**An alignment is for protein chains only.** ``DNAInput`` carries no such field and
``RNAInput``'s is read by nothing, so an alignment on a nucleic chain is accepted, carried
and dropped without a word. It is refused here instead.

**A protein chain always has an alignment.** Given none, the depth-1 alignment on the chain's
own sequence is built here, which is what upstream would have built one level down — after
warning once per chain, which ``filterwarnings = ["error"]`` would turn into an exception.

Examples
--------
>>> from protein import MSA
>>> from protein.fold import ChainRequest, FoldingRequest
>>> request = FoldingRequest([ChainRequest("protein", "MKTAY", accession="P12345")])
>>> request
FoldingRequest(1 chain, 5 residues)
>>> request.chain_ids
('A',)
>>> request.accessions
{'A': ('P12345',)}
>>> request.chains[0].alignment
MSA(depth 1, 5 match states)
>>> ChainRequest("dna", "ACGT", alignment=MSA([("q", "MKTAY")]))
Traceback (most recent call last):
    ...
ValueError: a dna chain takes no alignment: upstream drops it without a word.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

from protein import seq
from protein.msa import MSA

if TYPE_CHECKING:
    from collections.abc import Iterable

    from biotite.sequence import NucleotideSequence, ProteinSequence

    from protein.core import Protein

__all__ = ["POLYMERS", "ChainRequest", "FoldingRequest", "PolymerKind"]

#: What a chain of a request can be. Three and not :data:`protein.structure.chain.KINDS`'
#: three: DNA and RNA are one kind to a solved structure and two different inputs to the
#: model, which tokenizes them against different alphabets.
PolymerKind = Literal["protein", "dna", "rna"]

#: The three kinds, in one place so a caller can test membership without spelling them.
POLYMERS: tuple[PolymerKind, ...] = get_args(PolymerKind)

#: What the header of a derived depth-1 alignment says when the chain has no accession.
_ANONYMOUS = "query"

#: How RNA is spelled to the tokenizer. biotite stores thymine where RNA writes uracil, and
#: the model's RNA alphabet holds no ``T`` — an unspelt one tokenizes as an unknown residue.
_URACIL = "U"


class ChainRequest:
    """One chain to fold: what it is, what it reads, whose it is and what it aligns to.

    Parameters
    ----------
    kind : {"protein", "dna", "rna"}
        Which of the three the chain is. It decides the alphabet the sequence is checked
        against and whether an alignment is allowed at all.
    sequence : str or biotite.sequence.ProteinSequence or biotite.sequence.NucleotideSequence
        The residues, in either case. Checked and folded at construction by
        :func:`protein.seq.to_protein_sequence` or
        :func:`protein.seq.to_nucleotide_sequence` (ADR-0002, ADR-0004).
    accession : str, optional
        The UniProt accession this chain was taken from. Provenance, carried onto the
        prediction and never read back out of it (ADR-0005).
    alignment : protein.msa.MSA, optional
        The alignment to condition on. **Protein chains only.** Its query row must be this
        chain's sequence. Omitted on a protein chain, the depth-1 alignment on that sequence
        is built here.

    Attributes
    ----------
    kind : {"protein", "dna", "rna"}
        As given.
    sequence : biotite.sequence.ProteinSequence or biotite.sequence.NucleotideSequence
        The residues, as biotite stores them.
    accession : str or None
        As given.
    alignment : protein.msa.MSA or None
        The alignment for a protein chain, given or derived; ``None`` for a nucleic one.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three, if a nucleic chain is given an alignment, or if
        an alignment's query row is not this chain's sequence.
    protein.seq.InvalidResidueError
        If ``sequence`` holds anything outside the alphabet for its kind.

    Warns
    -----
    protein.seq.ResidueCoercionWarning
        If a protein sequence holds ``U``, ``O`` or ``J``, or a nucleic one holds ``U``.

    Examples
    --------
    >>> chain = ChainRequest("protein", "MKTAY")
    >>> chain
    ChainRequest('protein', 5 residues)
    >>> chain.alignment.depth
    1
    >>> ChainRequest("rna", "ACGU").residues
    'ACGU'
    """

    def __init__(
        self,
        kind: PolymerKind,
        sequence: str | ProteinSequence | NucleotideSequence,
        *,
        accession: str | None = None,
        alignment: MSA | None = None,
    ) -> None:
        if kind not in POLYMERS:
            raise ValueError(f"unknown chain kind {kind!r}. Known kinds: {', '.join(POLYMERS)}.")
        self.kind: PolymerKind = kind
        self.accession = accession
        text = str(sequence)
        self.sequence: ProteinSequence | NucleotideSequence = (
            seq.to_protein_sequence(text, name=accession)
            if kind == "protein"
            else seq.to_nucleotide_sequence(text, name=accession)
        )
        if kind != "protein":
            if alignment is not None:
                raise ValueError(
                    f"a {kind} chain takes no alignment: upstream drops it without a word."
                )
            self.alignment: MSA | None = None
            return
        self.alignment = self._checked(alignment)

    @classmethod
    def of(cls, protein: Protein, *, alignment: MSA | None = None) -> ChainRequest:
        """Return the protein chain that folds ``protein``, carrying its accession.

        Parameters
        ----------
        protein : protein.core.Protein
            What to fold. Its ``id`` becomes the chain's :attr:`accession`, which is what
            ``Chain.uniprot`` answers with on the prediction.
        alignment : protein.msa.MSA, optional
            As for the constructor.

        Returns
        -------
        ChainRequest
            A protein chain.

        Examples
        --------
        >>> from protein import Protein
        >>> ChainRequest.of(Protein("MKTAY", id="P12345")).accession
        'P12345'
        """
        return cls("protein", protein.sequence, accession=protein.id, alignment=alignment)

    @property
    def residues(self) -> str:
        """What the tokenizer reads: the sequence, with an RNA chain spelt back in ``U``.

        Returns
        -------
        str
            Uppercase. ``T`` becomes ``U`` on an RNA chain and nowhere else.

        Examples
        --------
        >>> ChainRequest("dna", "ACGT").residues
        'ACGT'
        >>> ChainRequest("rna", "ACGT").residues
        'ACGU'
        """
        text = str(self.sequence)
        return text.replace(seq.THYMINE, _URACIL) if self.kind == "rna" else text

    def _checked(self, alignment: MSA | None) -> MSA:
        """Return the alignment this chain folds with: the given one, checked, or a new one."""
        residues = str(self.sequence)
        if alignment is None:
            return MSA([(self.accession or _ANONYMOUS, residues)])
        if alignment.match_states != len(residues):
            raise ValueError(
                f"the alignment occupies {alignment.match_states} match states and the "
                f"chain has {len(residues)} residues. Upstream cuts the long one and "
                f"gap-fills the short one rather than saying so."
            )
        if alignment.query != residues:
            raise ValueError(
                f"the alignment's query row is not this chain's sequence "
                f"({_first_difference(alignment.query, residues)}). An alignment conditions "
                f"the chain it was built for; upstream never checks which."
            )
        return alignment

    def __len__(self) -> int:
        """Return the residue count.

        Examples
        --------
        >>> len(ChainRequest("protein", "MKTAY"))
        5
        """
        return len(self.sequence)

    def __repr__(self) -> str:
        """Return e.g. ``ChainRequest('protein', 76 residues)``.

        Examples
        --------
        >>> ChainRequest("dna", "ACGT")
        ChainRequest('dna', 4 residues)
        """
        return f"{type(self).__name__}({self.kind!r}, {len(self)} residues)"


class FoldingRequest:
    """One structure prediction's input: its chains, and the labels they will carry.

    Parameters
    ----------
    chains : iterable of ChainRequest
        One entry per chain, in the order the complex is built. At least one.

    Attributes
    ----------
    chains : tuple of ChainRequest
        The entries, in order.

    Raises
    ------
    ValueError
        If ``chains`` is empty.

    Examples
    --------
    >>> request = FoldingRequest(
    ...     [ChainRequest("protein", "MKTAY", accession="P12345"), ChainRequest("dna", "ACGT")]
    ... )
    >>> request
    FoldingRequest(2 chains, 9 residues)
    >>> request.chain_ids
    ('A', 'B')
    >>> request.accessions
    {'A': ('P12345',), 'B': ()}
    """

    def __init__(self, chains: Iterable[ChainRequest]) -> None:
        self.chains: tuple[ChainRequest, ...] = tuple(chains)
        if not self.chains:
            raise ValueError("a folding request holds at least one chain, and this one holds none.")

    @property
    def chain_ids(self) -> tuple[str, ...]:
        """The label each chain will carry in the prediction, derived from its position.

        Returns
        -------
        tuple of str
            ``('A', 'B', ...)``, continuing ``'AA'``, ``'AB'`` past twenty-six. Nobody named
            these chains, so position is the only honest name for them.

        Examples
        --------
        >>> FoldingRequest([ChainRequest("protein", "MKTAY")]).chain_ids
        ('A',)
        """
        return tuple(_label(index) for index in range(len(self.chains)))

    @property
    def accessions(self) -> dict[str, tuple[str, ...]]:
        """The accessions this request folds from, keyed by chain label.

        What :class:`~protein.structure.Structure` is built with, so ``Chain.uniprot``
        answers with the input the file was written from rather than asking SIFTS about an
        id that is no PDB entry (ADR-0005).

        Returns
        -------
        dict of str to tuple of str
            Every label, with ``()`` for a chain that names no accession.

        Examples
        --------
        >>> FoldingRequest([ChainRequest("dna", "ACGT")]).accessions
        {'A': ()}
        """
        return {
            label: () if chain.accession is None else (chain.accession,)
            for label, chain in zip(self.chain_ids, self.chains, strict=True)
        }

    def __len__(self) -> int:
        """Return the chain count.

        Examples
        --------
        >>> len(FoldingRequest([ChainRequest("protein", "MKTAY")]))
        1
        """
        return len(self.chains)

    def __repr__(self) -> str:
        """Return e.g. ``FoldingRequest(2 chains, 9 residues)``.

        Examples
        --------
        >>> FoldingRequest([ChainRequest("protein", "MKTAY")])
        FoldingRequest(1 chain, 5 residues)
        """
        count = len(self.chains)
        residues = sum(len(chain) for chain in self.chains)
        plural = "" if count == 1 else "s"
        return f"{type(self).__name__}({count} chain{plural}, {residues} residues)"


def _label(index: int) -> str:
    """Return the spreadsheet label for a zero-based position: ``A``, ``Z``, ``AA``, ``AB``."""
    label = ""
    while True:
        index, remainder = divmod(index, 26)
        label = chr(ord("A") + remainder) + label
        if index == 0:
            return label
        index -= 1


def _first_difference(query: str, residues: str) -> str:
    """Return where two equal-length strings first disagree, for the error to quote."""
    for index, (left, right) in enumerate(zip(query, residues, strict=True)):
        if left != right:
            return f"{left!r} against {right!r} at {index}"
    return "no difference"
