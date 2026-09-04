"""The :class:`Structure` class — one set of coordinates, and where they came from.

A **Structure** is named by an id its constructor does not police, and a **PDB id** is the
ordinary case: give one and the coordinates arrive from the cache or RCSB. It is not
protein-specific and it is not owned by a protein: an entry also carries nucleic acids,
ligands and water, none of which has a :class:`~protein.core.Protein` at all. The join to
that namespace is SIFTS, it is chain-level, and it therefore attaches to
:class:`~protein.structure.chain.Chain` rather than here.

**A deposited entry holds its asymmetric unit, and not a biological assembly** — a
consequence rather than a taste. SIFTS keys on AU author chains, so ``Chain.uniprot`` only
answers against the AU, and many entries have more than one assembly, which would leave
``Structure("1UBQ")`` undefined.

**A structure may carry the accessions it was produced from**, one per chain, and
``Chain.uniprot`` answers from those rather than asking SIFTS. That is provenance and not a
join (ADR-0005): it is an input the file was written from, never a cross-reference read back
out of the file. Nothing here reads ``_struct_ref``. A prediction may carry its
:class:`~protein.fold.predictions.Confidence` the same way, and for the same reason: it is
what the model reported, not something read back out of what it wrote.

**The path is held and the parse is lazy.** Foldseek needs a file on disk whatever else
happens, so the file is what a structure *is*; :attr:`~Structure.atoms` and
:attr:`~Structure.models` parse on first use and are then cached on the instance. Nothing is
read, fetched or looked up by constructing one, and :func:`repr` stays free.

**Coordinates: local first, RCSB on a miss, one cache.** :func:`fetch` looks under
:func:`structure_data_dir` and, finding nothing, asks ``files.rcsb.org`` and leaves the file
there. Look-local-first is ``biotite.database.rcsb.fetch``'s own, so this module supplies the
root and the gzipped spelling and adopts the rest. A bulk rsync mirror is therefore optional:
an operator may fill the cache ahead of time, and no API changes if they do not.

**pdb100 is a search target and never a coordinate source.** It is C-alpha only, it indexes
a fraction of named chains, and ``convert2pdb`` renumbers residues from 1, which destroys the
residue-level SIFTS join the offsets are carried for.

Examples
--------
>>> import os
>>> from protein import Structure
>>> os.environ["LIULAB_DATA"] = "/scratch/liulab"
>>> structure_data_dir()
PosixPath('/scratch/liulab/protein/structures')
>>> del os.environ["LIULAB_DATA"]
>>> Structure("1UBQ")
Structure('1UBQ')
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

from protein.io import structure as _io

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import pandas as pd
    from biotite.structure import AtomArray, AtomArrayStack

    from protein.fold.predictions import Confidence
    from protein.search.mmseqs import SearchTarget
    from protein.structure.chain import Chain

__all__ = [
    "COORDINATE_FORMAT",
    "FETCH_COMMAND",
    "STRUCTURE_SUBDIR",
    "CoordinatesNotDownloadedError",
    "Structure",
    "cached_path",
    "fetch",
    "structure_data_dir",
]

#: Where cached coordinates live under :func:`protein.store.protein_data_dir`. A sibling of
#: ``db/`` and ``sifts/``: these are neither an ffindex **Database** nor a **Prepared set**,
#: but one file per entry, fetched on demand and never all at once.
STRUCTURE_SUBDIR = "structures"

#: What is fetched and what the cache is keyed by. mmCIF, because it is what RCSB serves and
#: the only one of the two formats that can spell a chain label of more than one character.
COORDINATE_FORMAT = "cif"

#: The call that fills the cache where there is a network, quoted into the error a machine
#: without one raises.
FETCH_COMMAND = "protein structure fetch"

#: How many chain labels an error lists before it counts the rest. A ribosome has hundreds
#: and a message carrying all of them is not read.
_MAX_LISTED = 12


class CoordinatesNotDownloadedError(RuntimeError):
    """The coordinates are not in the cache and this machine could not fetch them.

    One class for both ways that happens — a compute node with no network, and an id RCSB
    does not serve — because the caller's next move is the same. The message names the id,
    the cache directory and the command that fills it from a login node.

    Examples
    --------
    >>> issubclass(CoordinatesNotDownloadedError, RuntimeError)
    True
    """


def structure_data_dir() -> Path:
    """Return the coordinate cache's directory under the lab **Data dir**.

    Returns
    -------
    pathlib.Path
        ``<LIULAB_DATA>/protein/structures``. Nothing is created by asking.

    Examples
    --------
    >>> import os
    >>> os.environ["LIULAB_DATA"] = "/scratch/liulab"
    >>> structure_data_dir()
    PosixPath('/scratch/liulab/protein/structures')
    >>> del os.environ["LIULAB_DATA"]
    """
    from protein.store import protein_data_dir

    return protein_data_dir() / STRUCTURE_SUBDIR


def cached_path(pdb_id: str) -> Path | None:
    """Return the cached coordinate file for ``pdb_id``, or ``None`` when it is not here.

    **Offline, always.** This is the half of :func:`fetch` that never touches the network,
    so a caller that must not reach out can ask what is already local.

    Parameters
    ----------
    pdb_id : str
        A PDB entry id, in either case. The cache is keyed lower-case, which is how RCSB's
        own bulk tree and SIFTS both spell an entry.

    Returns
    -------
    pathlib.Path or None
        ``<cache>/<id>.cif`` when it is there and non-empty, else ``<cache>/<id>.cif.gz`` on
        the same terms, else ``None``. A zero-byte file reads as absent, which is how an
        interrupted download is repaired.

    Examples
    --------
    >>> cached_path("1UBQ")                               # doctest: +SKIP
    PosixPath('/scratch/liulab/protein/structures/1ubq.cif')
    """
    directory = structure_data_dir()
    entry = pdb_id.strip().lower()
    for name in (f"{entry}.{COORDINATE_FORMAT}", f"{entry}.{COORDINATE_FORMAT}.gz"):
        candidate = directory / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def fetch(pdb_id: str) -> Path:
    """Return the coordinates for ``pdb_id``, downloading them once if they are not here.

    Parameters
    ----------
    pdb_id : str
        A PDB entry id, in either case.

    Returns
    -------
    pathlib.Path
        The cached mmCIF file. The directory is created if the download runs.

    Raises
    ------
    CoordinatesNotDownloadedError
        If nothing is cached and the download did not happen — no network, or no such entry.

    Examples
    --------
    >>> fetch("1UBQ")                                     # doctest: +SKIP
    PosixPath('/scratch/liulab/protein/structures/1ubq.cif')
    """
    local = cached_path(pdb_id)
    if local is not None:
        return local

    # Deferred, because reading a cached file should not pay for an HTTP client. The module
    # rather than the function, so a test can monkeypatch `rcsb.fetch` and be seen here.
    from biotite.database import RequestError, rcsb

    directory = structure_data_dir()
    entry = pdb_id.strip().lower()
    try:
        written = rcsb.fetch(entry, COORDINATE_FORMAT, target_path=str(directory))
        return Path(cast("str", written))
    except (OSError, RequestError) as error:
        from genome.store import prepared

        raise CoordinatesNotDownloadedError(
            f"no coordinates for {pdb_id!r}: nothing under {directory} and RCSB could not be "
            f"read ({error}). {prepared.login_node_help(f'{FETCH_COMMAND} {pdb_id}')}"
        ) from error


def _frozen_accessions(accessions: Mapping[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    """Freeze a per-chain accession map, refusing a bare string where a sequence belongs.

    ``{"A": "P12345"}`` is the mistake worth catching: a ``str`` is an iterable of ``str``,
    so it satisfies the annotation and would be read one character per accession.
    """
    loose = sorted(label for label, value in accessions.items() if isinstance(value, str))
    if loose:
        raise TypeError(
            f"accessions maps a chain label to a sequence of accessions, and {loose} map to "
            f"a str, which would be read one character per accession. Wrap each in a tuple."
        )
    return {label: tuple(value) for label, value in accessions.items()}


class Structure:
    """One set of coordinates: its file, its atoms and its chains.

    Parameters
    ----------
    id : str
        What this structure is called — a PDB entry id, e.g. ``"1UBQ"``, in the ordinary
        case and in either case. It is **not folded** and **not checked**: it is what
        :meth:`__repr__` and every chain key spell, and the coordinate cache and SIFTS
        lower-case it themselves.
    path : str or pathlib.Path, optional
        The coordinate file, when it is already known. Omitted, it is resolved on first use
        by :func:`fetch` — the cache, then RCSB.
    accessions : mapping of str to iterable of str, optional
        The UniProt accessions this structure was **produced from**, keyed by chain label.
        Given, it is what :attr:`~protein.structure.chain.Chain.uniprot` answers with, for
        every chain and not only the ones it names, so SIFTS is never asked. Provenance and
        not a join (ADR-0005): nothing puts a deposited entry's own cross-reference here.
    confidence : protein.fold.predictions.Confidence, optional
        What the model reported about a prediction. ``None`` for a deposited entry and for
        anything read off disk, which is the same limit the accession map has.

    Attributes
    ----------
    id : str
        The id, as it was given.
    accessions : dict of str to tuple of str, or None
        The map, with each entry frozen into a tuple. ``None`` where none was given, which
        is what every structure read off disk carries.
    confidence : protein.fold.predictions.Confidence or None
        As given.

    Raises
    ------
    TypeError
        If ``accessions`` maps a chain to a bare :class:`str`, which would be read one
        character per accession.

    Examples
    --------
    >>> from protein import Structure
    >>> s = Structure("1UBQ")
    >>> s
    Structure('1UBQ')
    >>> s.chain_ids                                       # doctest: +SKIP
    ('A',)
    >>> s["A"].sequence[:5]                               # doctest: +SKIP
    ProteinSequence("MQIFV")
    >>> Structure("folded", accessions={"A": ["P12345"]}).accessions
    {'A': ('P12345',)}
    """

    def __init__(
        self,
        id: str,
        *,
        path: str | Path | None = None,
        accessions: Mapping[str, Iterable[str]] | None = None,
        confidence: Confidence | None = None,
    ) -> None:
        self.id = id
        self._path = Path(path) if path is not None else None
        self.accessions = None if accessions is None else _frozen_accessions(accessions)
        self.confidence = confidence

    @classmethod
    def from_file(cls, path: str | Path, *, id: str | None = None) -> Self:
        """Build a structure from a coordinate file, wherever it came from.

        What replaces ``Protein.from_structure(path)``, which does not exist: a file gives
        you a structure, and ``structure["A"].sequence`` gives you the sequence. A
        ``Protein`` needs an accession, which a file does not carry.

        Parameters
        ----------
        path : str or pathlib.Path
            An mmCIF or PDB file, optionally gzipped. It must exist.
        id : str, optional
            What to call it. Defaults to the file's own stem, which is also the name
            **Foldseek** reports the query under, so a hit table and ``chain.id`` agree by
            construction. Give one when the file is named something other than its entry.

        Returns
        -------
        Structure
            Holding this file, and **carrying neither an** :attr:`accessions` **map nor a**
            :attr:`confidence`. Nothing is parsed yet.

            That is a limit rather than an oversight: provenance does not survive the file
            and nothing is built to make it, so a prediction reopened from disk answers
            ``()`` for its accessions like any other uncurated entry (ADR-0005).

        Raises
        ------
        FileNotFoundError
            If ``path`` is not a file. A lazy parse would otherwise report a typo'd path at
            whichever line first asked for an atom.

        Examples
        --------
        >>> Structure.from_file("tests/data/1ubq.cif.gz")   # doctest: +SKIP
        Structure('1ubq')
        """
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"{file} is not a file, so there are no coordinates in it.")
        return cls(id if id is not None else _io.entry_name(file), path=file)

    @property
    def path(self) -> Path:
        """The coordinate file on disk, fetched and cached on first use.

        Returns
        -------
        pathlib.Path
            What was given to the constructor, or what :func:`fetch` resolved.

        Raises
        ------
        CoordinatesNotDownloadedError
            If nothing is cached for this id and the download did not happen.

        Examples
        --------
        >>> Structure("1UBQ").path                        # doctest: +SKIP
        PosixPath('/scratch/liulab/protein/structures/1ubq.cif')
        """
        if self._path is None:
            self._path = fetch(self.id)
        return self._path

    @cached_property
    def atoms(self) -> AtomArray:
        """Every atom of the first model, parsed once and then held.

        Returns
        -------
        biotite.structure.AtomArray
            Protein, nucleic acid, ligand and water alike — nothing is filtered out. Use
            :attr:`models` for every model of an NMR entry.

        Examples
        --------
        >>> Structure("1UBQ").atoms.array_length()        # doctest: +SKIP
        660
        """
        return _io.read_atoms(self.path)

    @cached_property
    def models(self) -> AtomArrayStack:
        """Every model in the file, parsed once and then held.

        A separate parse from :attr:`atoms` rather than its source, so an entry whose models
        do not share one atom set — which no stack can hold — still answers ``.atoms``.

        Returns
        -------
        biotite.structure.AtomArrayStack
            Depth one for an X-ray entry, one per deposited model for NMR.

        Examples
        --------
        >>> Structure("1L2Y").models.stack_depth()        # doctest: +SKIP
        38
        """
        return _io.read_models(self.path)

    @cached_property
    def chain_ids(self) -> tuple[str, ...]:
        """Every chain label in the first model, once each, in file order.

        Returns
        -------
        tuple of str
            The labels as the file spells them. **Not all are one character**, so nothing may
            index one by position.

        Examples
        --------
        >>> Structure("1BNA").chain_ids                   # doctest: +SKIP
        ('A', 'B')
        """
        return _io.chain_ids(self.atoms)

    @property
    def chains(self) -> tuple[Chain, ...]:
        """Every chain, in file order.

        Returns
        -------
        tuple of Chain
            One per label in :attr:`chain_ids`.

        Examples
        --------
        >>> [chain.kind for chain in Structure("1BNA").chains]    # doctest: +SKIP
        ['nucleic', 'nucleic']
        """
        return tuple(self[label] for label in self.chain_ids)

    def search(self, database: SearchTarget | str, **kwargs: Any) -> pd.DataFrame:
        """Search this whole structure against ``database`` with Foldseek.

        **One invocation, not a loop over chains**: Foldseek fans a multi-chain query out
        itself and reports each chain in the ``query`` column as ``<entry>_<chain>``, the
        same convention SIFTS keys on.

        **The file goes to Foldseek as it is**, unparsed: the two tools do not read the same
        set of formats, and refusing here what Foldseek would have read is this package
        inventing a limit.

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
            One row per hit, in Foldseek's column order. Identity is ``fident``, a
            **fraction**, where MMseqs2's ``pident`` is a percentage.

        Raises
        ------
        LookupError
            If ``database`` names nothing registered.
        CoordinatesNotDownloadedError
            If this structure's file is neither cached nor fetchable.
        protein.external.ToolNotFoundError
            If ``foldseek`` is not installed.

        Examples
        --------
        >>> Structure("1UBQ").search("pdb").loc[0, "target"]       # doctest: +SKIP
        '2n2k-assembly1_A'
        """
        from protein.search import foldseek

        return foldseek.search(self.path, database, **kwargs)

    def __getitem__(self, key: str) -> Chain:
        """Return the chain labelled ``key``.

        Parameters
        ----------
        key : str
            A chain label, exactly as the file spells it. Case is part of a chain name.

        Returns
        -------
        Chain
            The chain. This is the door a chain is reached through, and the one place a label
            is checked.

        Raises
        ------
        KeyError
            If no chain carries that label. The message names the ones that are there.

        Examples
        --------
        >>> Structure("1UBQ")["A"]                        # doctest: +SKIP
        Chain('1UBQ_A', protein, 76 residues)
        """
        from protein.structure.chain import Chain

        if key not in self.chain_ids:
            raise KeyError(self._no_such_chain(key))
        return Chain(self, key)

    def __contains__(self, key: object) -> bool:
        """Return whether a chain carries this label.

        Examples
        --------
        >>> "A" in Structure("1UBQ")                      # doctest: +SKIP
        True
        """
        return key in self.chain_ids

    def __repr__(self) -> str:
        """Return e.g. ``Structure('1UBQ')`` — the id, and deliberately nothing else.

        A chain or atom count would parse the file and could fetch it, so printing a
        structure in a debugger would reach the network.

        Examples
        --------
        >>> Structure("1UBQ")
        Structure('1UBQ')
        """
        return f"{type(self).__name__}({self.id!r})"

    def _no_such_chain(self, key: str) -> str:
        """Return the text of the :class:`KeyError` for a label this structure lacks."""
        present = self.chain_ids
        listed = ", ".join(repr(label) for label in present[:_MAX_LISTED])
        hidden = len(present) - _MAX_LISTED
        rest = f", and {hidden} more" if hidden > 0 else ""
        return (
            f"{self.id} has no chain {key!r}. Its chains are {listed}{rest} — and case is "
            f"part of a chain name, so 'a' and 'A' are two of them."
        )
