"""The **Database** lane: what is registered, what this package knows how to fetch.

Three things live here and nothing else does: the lane's public names, the small
**declaration table** of databases this package can download by itself, and
:func:`open_database`, which turns a name into the right class.

**A declaration, not a subclass.** ``liulab-genome``'s rule holds: anything whose only
distinguishing facts are a name and a source is a row in a table. ``pdb`` is such a row —
once ``pdb["1UBQ"]`` moved to :func:`protein.structure.fetch`, nothing was left for a ``PDB``
class to do that :class:`~protein.db.base.StructureDatabase` does not. ``swissprot`` is not:
:class:`~protein.db.swissprot.SwissProt` reads UniProt headers, so it is a class, and the
table names it as that row's factory.

**Registration is not centralised and this table is not a registry.** A name is registered
when its directory holds a **Completion marker**; :func:`~protein.db.base.registered_names`
reads the disk. This table only says which names ``protein db download`` knows a source for.
A name that is not in it is adopted, or downloaded with an explicit ``source``.

Examples
--------
>>> from protein.db import DECLARED, open_database
>>> sorted(DECLARED)
['pdb', 'swissprot']
>>> DECLARED["swissprot"].source
'UniProtKB/Swiss-Prot'
>>> open_database("pdb")
StructureDatabase('pdb')
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from protein.db.base import (
    DATABASE_KIND,
    DATABASE_SUBDIR,
    GPU_ENCODED_BIT,
    GPU_RESIDUE_FOLD,
    Database,
    DatabaseStatus,
    SequenceDatabase,
    StructureDatabase,
    database_data_dir,
    database_files,
    database_path,
    ffindex_prefix,
    is_gpu_encoded,
    registered_help,
    registered_names,
)
from protein.db.swissprot import SwissProt, UniProtHeader, parse_uniprot_header

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "DATABASE_KIND",
    "DATABASE_SUBDIR",
    "DECLARED",
    "GPU_ENCODED_BIT",
    "GPU_RESIDUE_FOLD",
    "KINDS",
    "Database",
    "DatabaseStatus",
    "Declaration",
    "SequenceDatabase",
    "StructureDatabase",
    "SwissProt",
    "UniProtHeader",
    "database_class",
    "database_data_dir",
    "database_files",
    "database_path",
    "ffindex_prefix",
    "is_gpu_encoded",
    "open_database",
    "parse_uniprot_header",
    "registered_help",
    "registered_names",
]


@dataclass(frozen=True)
class Declaration:
    """One database this package can download by itself.

    Attributes
    ----------
    name : str
        The registered name — a filesystem-safe slug, which is what a caller types.
    source : str
        The **tool's** own spelling, which is what ``mmseqs databases`` or ``foldseek
        databases`` is handed. It is not the name because it cannot be: ``UniProtKB/Swiss-
        Prot`` carries a slash.
    factory : type
        The :class:`~protein.db.base.Database` subclass to build. A row whose class adds
        nothing points at the base class for its kind.
    description : str
        One line, printed by ``protein db list``.

    Examples
    --------
    >>> DECLARED["pdb"].factory.__name__
    'StructureDatabase'
    """

    name: str
    source: str
    factory: type[Database]
    description: str


#: The two databases v1 can fetch, with the sizes they really land at rather than the ones
#: the publishers bill: Swiss-Prot 1.1G, pdb100 4.3G. The 824 MB of ``swissprot_taxonomy``
#: `mmseqs databases` also writes is one of the eleven files **inside** that 1.1G, not a sum
#: on top of it — ``du -sb`` on the real directory is 1,149,140,927 bytes.
#: The registry must eventually carry names at UniRef50 and AlphaFold DB scale — 8.8 GB and a
#: measured 491 GB of source — so nothing here assumes a database is small, and neither is
#: declared, because neither has been measured on this cluster.
DECLARED: Mapping[str, Declaration] = MappingProxyType(
    {
        "swissprot": Declaration(
            name="swissprot",
            source="UniProtKB/Swiss-Prot",
            factory=SwissProt,
            description="UniProtKB reviewed entries, searched with MMseqs2 (1.1G on disk)",
        ),
        "pdb": Declaration(
            name="pdb",
            source="PDB",
            factory=StructureDatabase,
            description="Foldseek's pdb100, searched with Foldseek (4.3G on disk)",
        ),
    }
)

#: What ``--kind`` accepts, for a name this package has no declaration for. Written down
#: because a database adopted from disk records its kind, and reopening it has to read that
#: word back.
KINDS: Mapping[str, type[Database]] = MappingProxyType(
    {"sequence": SequenceDatabase, "structure": StructureDatabase}
)


def database_class(name: str, *, kind: str | None = None) -> type[Database]:
    """Return the :class:`~protein.db.base.Database` subclass ``name`` should be opened as.

    Three sources, in order: an explicit ``kind``, then :data:`DECLARED`, then the
    **Completion marker** an ``adopt`` already wrote. The last is what makes an undeclared
    name reopenable — ``adopt`` records which tool searches it, so the answer is on disk.

    Parameters
    ----------
    name : str
        The registered name.
    kind : str, optional
        ``"sequence"`` or ``"structure"``, overriding everything else.

    Returns
    -------
    type
        The class to construct.

    Raises
    ------
    LookupError
        If ``kind`` is not one this package knows, or if nothing says what ``name`` is.

    Examples
    --------
    >>> database_class("swissprot").__name__
    'SwissProt'
    >>> database_class("uniref50", kind="sequence").__name__
    'SequenceDatabase'
    """
    if kind is not None:
        if kind not in KINDS:
            raise LookupError(f"{kind!r} is not a database kind. Pass one of {sorted(KINDS)}.")
        return KINDS[kind]

    declared = DECLARED.get(name)
    if declared is not None:
        return declared.factory

    from genome.store import completion

    record = completion.read_record(database_data_dir() / name)
    recorded = record.details.get("kind") if record is not None else None
    if isinstance(recorded, str) and recorded in KINDS:
        return KINDS[recorded]

    raise LookupError(
        f"nothing says whether {name!r} is a sequence database or a structure one: it is not "
        f"declared here and no completion record names its kind. Say which with "
        f"`--kind sequence` or `--kind structure`."
    )


def open_database(name: str, *, kind: str | None = None) -> Database:
    """Return a **Database** for ``name``, of whatever class :func:`database_class` picks.

    Nothing is read and nothing is created: a **Database** is a name and a path until it is
    asked a question.

    Parameters
    ----------
    name : str
        The registered name.
    kind : str, optional
        ``"sequence"`` or ``"structure"``, for a name nothing else identifies.

    Returns
    -------
    Database
        The object, carrying the declared source when there is one.

    Raises
    ------
    LookupError
        If nothing says what ``name`` is.

    Examples
    --------
    >>> open_database("swissprot")
    SwissProt('swissprot')
    >>> open_database("uniref50", kind="sequence")
    SequenceDatabase('uniref50')
    """
    declared = DECLARED.get(name)
    source = declared.source if declared is not None else None
    return database_class(name, kind=kind)(name, source=source)
