"""mmCIF and PDB, in two layers: biotite's files below, this package's atom arrays above.

The **file layer** is adopted whole — ``pdbx.CIFFile``, ``pdb.PDBFile`` and
``get_structure``, which gives one model as an :class:`~biotite.structure.AtomArray` or every
model as an :class:`~biotite.structure.AtomArrayStack`. What is added here is the format
branch, gzip, and the three array operations :class:`protein.structure.Structure` and
:class:`protein.structure.Chain` are built from.

**Two formats are read and one is written.** mmCIF and PDB both read; only mmCIF is written,
because nothing in this package needs a PDB file and a PDB file cannot spell every chain
label the archive uses. BinaryCIF is read by biotite and deliberately not here, so ``.bcif``
says *deferred* rather than falling through to *unknown suffix*. gzip is one branch rather
than a second set of functions, because RCSB's bulk tree serves ``.cif.gz`` and biotite reads
and writes any text handle.

**``b_factor`` and ``occupancy`` are read on every file**, which would otherwise look
arbitrary: Foldseek will not read an mmCIF whose ``atom_site`` category lacks them.

**``get_chains`` is not the chain list a caller means.** It is the label at every chain
*segment* boundary, so an entry's protein, ligand and water records each inflate the count.
:func:`chain_ids` is the order-preserving unique instead, and :func:`chain_atoms` is the
selection ``get_chains`` gets reached for by mistake.

**Nothing here turns a residue into a letter.** That is
:attr:`protein.structure.Chain.sequence`'s job.

Examples
--------
>>> entry_name("/data/protein/structures/1ubq.cif.gz")
'1ubq'
>>> format_suffix("104L.PDB")
'.pdb'
"""

from __future__ import annotations

import gzip
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from biotite.structure.io import pdb, pdbx

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import TextIO

    from biotite.structure import AtomArray, AtomArrayStack

__all__ = [
    "CIF_SUFFIXES",
    "DEFERRED_SUFFIXES",
    "EXTRA_FIELDS",
    "PDB_SUFFIXES",
    "chain_atoms",
    "chain_ids",
    "entry_name",
    "format_suffix",
    "read_atoms",
    "read_models",
    "to_text",
    "write_atoms",
]

#: What is read as mmCIF. All three name one format.
CIF_SUFFIXES: tuple[str, ...] = (".cif", ".mmcif", ".pdbx")

#: What is read as a PDB file. ``.ent`` is the PDB archive's own spelling of it.
PDB_SUFFIXES: tuple[str, ...] = (".pdb", ".ent")

#: Read by biotite, deliberately not read here — so the refusal says *deferred* rather than
#: *unknown*.
DEFERRED_SUFFIXES: tuple[str, ...] = (".bcif",)

#: The annotations read beyond biotite's defaults. Not a convenience: **Foldseek will not
#: read an mmCIF that carries neither**, so a file this module writes would be unsearchable
#: without them.
EXTRA_FIELDS: tuple[str, ...] = ("b_factor", "occupancy")

#: The one compression this package unpacks, which is the one RCSB's bulk tree uses.
_GZIP_SUFFIX = ".gz"


def format_suffix(path: str | Path) -> str:
    """Return the suffix of ``path`` that names its format, lower-cased and without ``.gz``.

    Parameters
    ----------
    path : str or pathlib.Path
        A structure file's name. It need not exist.

    Returns
    -------
    str
        What decides the parser, e.g. ``".cif"`` for ``1ubq.cif.gz``. Empty when the name
        carries no suffix at all.

    Examples
    --------
    >>> format_suffix("1ubq.cif.gz")
    '.cif'
    >>> format_suffix("coordinates")
    ''
    """
    name = Path(path)
    if name.suffix.lower() == _GZIP_SUFFIX:
        name = name.with_suffix("")
    return name.suffix.lower()


def entry_name(path: str | Path) -> str:
    """Return what ``path`` calls its entry: the file name with its suffixes taken off.

    This is also **the name Foldseek reports a one-chain query under**, which is why
    :meth:`protein.structure.Chain.search` writes its query under the chain's own key.

    Parameters
    ----------
    path : str or pathlib.Path
        A structure file's name. It need not exist.

    Returns
    -------
    str
        The stem, with ``.gz`` and the format suffix removed. Case is kept — ``104L.pdb`` is
        ``104L`` — because nothing here folds an identifier somebody chose.

    Examples
    --------
    >>> entry_name("/data/1ubq.cif.gz")
    '1ubq'
    >>> entry_name("my model.pdb")
    'my model'
    """
    name = Path(path)
    if name.suffix.lower() == _GZIP_SUFFIX:
        name = name.with_suffix("")
    return name.stem


def read_atoms(path: str | Path, *, model: int = 1) -> AtomArray:
    """Read one model of the structure at ``path``.

    Parameters
    ----------
    path : str or pathlib.Path
        An mmCIF or PDB file, optionally gzipped.
    model : int, default 1
        Which model to read, **one-based**, as biotite numbers them.

    Returns
    -------
    biotite.structure.AtomArray
        Every atom of that model — protein, nucleic acid, ligand and water alike. Nothing is
        filtered here: a structure is not protein-specific. :data:`EXTRA_FIELDS` are read
        alongside biotite's own annotations.

    Raises
    ------
    ValueError
        If the suffix names no format this package reads. BinaryCIF says it is deferred.
    OSError
        If the file cannot be read.

    Examples
    --------
    >>> read_atoms("tests/data/1ubq.cif.gz").array_length()   # doctest: +SKIP
    660
    """
    with _open_text(Path(path)) as handle:
        return cast("AtomArray", _get_structure(Path(path), handle, model=model))


def read_models(path: str | Path) -> AtomArrayStack:
    """Read every model of the structure at ``path``.

    An entry carrying a single model comes back as a stack of depth one rather than as a
    special case.

    Parameters
    ----------
    path : str or pathlib.Path
        An mmCIF or PDB file, optionally gzipped.

    Returns
    -------
    biotite.structure.AtomArrayStack
        The models, in deposition order.

    Raises
    ------
    ValueError
        If the suffix names no format this package reads.
    biotite.structure.BadStructureError
        If the models do not share one atom set, which a stack cannot hold. Read a single
        model with :func:`read_atoms` instead.

    Examples
    --------
    >>> read_models("tests/data/1l2y_2models.pdb.gz").stack_depth()   # doctest: +SKIP
    2
    """
    with _open_text(Path(path)) as handle:
        return cast("AtomArrayStack", _get_structure(Path(path), handle, model=None))


def write_atoms(path: str | Path, atoms: AtomArray) -> None:
    """Write ``atoms`` to the mmCIF file at ``path``.

    **mmCIF and nothing else** — see this module's docstring. The data block is named after
    the file, though what Foldseek reports a query under is the file's own name rather than
    the block's.

    Parameters
    ----------
    path : str or pathlib.Path
        Where to write. A ``.gz`` name is compressed. The parent directory must exist;
        nothing here creates one.
    atoms : biotite.structure.AtomArray
        What to write. It should carry :data:`EXTRA_FIELDS` if Foldseek is to read it back.

    Raises
    ------
    ValueError
        If the suffix names no format this package writes, PDB included.

    Examples
    --------
    >>> write_atoms("/tmp/chain.cif", atoms)                  # doctest: +SKIP
    """
    target = Path(path)
    suffix = format_suffix(target)
    if suffix not in CIF_SUFFIXES:
        raise _unsupported(target, suffix, verb="write")
    with _open_text(target, mode="wt") as handle:
        handle.write(to_text(atoms, name=entry_name(target)))


def to_text(atoms: AtomArray, *, name: str) -> str:
    """Return ``atoms`` as mmCIF text — what :func:`write_atoms` puts in a file.

    Text rather than a path because a viewer takes a string, and because **a chain has no
    file of its own** to hand one.

    Parameters
    ----------
    atoms : biotite.structure.AtomArray
        What to serialise. It should carry :data:`EXTRA_FIELDS` if Foldseek is to read it
        back, and a viewer wants ``b_factor`` to colour a prediction by.
    name : str
        What to call the data block.

    Returns
    -------
    str
        One mmCIF document, holding ``atom_site`` and no secondary structure — a viewer
        computes its own.

    Examples
    --------
    >>> to_text(atoms, name="1ubq")                           # doctest: +SKIP
    """
    written = pdbx.CIFFile()
    pdbx.set_structure(written, atoms, data_block=name)
    handle = StringIO()
    written.write(handle)
    return handle.getvalue()


def chain_ids(atoms: AtomArray) -> tuple[str, ...]:
    """Return each chain label in ``atoms`` once, in the order the file lists them.

    Not ``get_chains``, which answers with the label at every chain **segment** boundary and
    so over-reports — see this module's docstring.

    Parameters
    ----------
    atoms : biotite.structure.AtomArray
        One model's atoms.

    Returns
    -------
    tuple of str
        The labels as the file spells them. **Not all of them are one character**, so nothing
        downstream may assume one or write such a label into a PDB file unchanged.

    Examples
    --------
    >>> chain_ids(atoms)                                      # doctest: +SKIP
    ('A', 'B')
    """
    # Cast because biotite types every annotation as optional; a parser never produces an
    # `AtomArray` without a `chain_id`.
    labels = cast("Iterable[str]", atoms.chain_id)
    return tuple(dict.fromkeys(str(label) for label in labels))


def chain_atoms(atoms: AtomArray, chain_id: str) -> AtomArray:
    """Return every atom of ``atoms`` whose chain label is ``chain_id``.

    The whole chain in one array, however many segments the file splits it into — which is
    what asking for a chain means, and what ``chain_iter`` does not answer.

    Parameters
    ----------
    atoms : biotite.structure.AtomArray
        One model's atoms.
    chain_id : str
        The label, exactly as the file spells it. Case is part of it: ``10EG`` carries both
        an ``A`` and an ``a``.

    Returns
    -------
    biotite.structure.AtomArray
        The chain's atoms, in file order. Empty when no atom carries that label.

    Examples
    --------
    >>> chain_atoms(atoms, "A").array_length()                # doctest: +SKIP
    660
    """
    return cast("AtomArray", atoms[atoms.chain_id == chain_id])


def _get_structure(path: Path, handle: TextIO, *, model: int | None) -> object:
    """Parse an open structure file, dispatching on ``path``'s format suffix.

    ``model=None`` gives an ``AtomArrayStack`` and an integer an ``AtomArray``; the two
    public readers name which they got.
    """
    suffix = format_suffix(path)
    fields = list(EXTRA_FIELDS)
    if suffix in CIF_SUFFIXES:
        return pdbx.get_structure(pdbx.CIFFile.read(handle), model=model, extra_fields=fields)
    if suffix in PDB_SUFFIXES:
        return pdb.get_structure(pdb.PDBFile.read(handle), model=model, extra_fields=fields)
    raise _unsupported(path, suffix, verb="read")


def _unsupported(path: Path, suffix: str, *, verb: str) -> ValueError:
    """Return the refusal for a suffix this package does not handle."""
    handled = CIF_SUFFIXES if verb == "write" else (*CIF_SUFFIXES, *PDB_SUFFIXES)
    if suffix in DEFERRED_SUFFIXES:
        return ValueError(
            f"{path}: BinaryCIF is deferred, not unknown. biotite reads it and this package "
            f"does not, because nothing in it writes one. Ask RCSB for mmCIF instead."
        )
    return ValueError(
        f"{path}: {suffix or 'no suffix'} names no structure format this package can "
        f"{verb}. It can {verb} {', '.join(handled)}, each optionally gzipped."
    )


def _open_text(path: Path, mode: Literal["rt", "wt"] = "rt") -> TextIO:
    """Open ``path`` as UTF-8 text, decompressing or compressing when it ends ``.gz``."""
    if path.suffix.lower() == _GZIP_SUFFIX:
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")
