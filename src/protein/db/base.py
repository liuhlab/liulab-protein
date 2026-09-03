"""What a **Database** is here: a directory of ffindex files, a name, and a record.

A **Database** is a large local thing this package searches and does **not** manage. Two
verbs point at one, and neither of them is a downloader we wrote:

- :meth:`Database.adopt` takes a database already on disk and writes a record for it. That
  is the common case on a cluster, where the gigabytes arrived by someone else's ``rsync``.
- :meth:`Database.download` delegates to ``mmseqs databases`` / ``foldseek databases``,
  which do this well, and writes a record afterwards.

Registration is `liulab-genome`'s and is not reimplemented: **a directory plus a completion
record is the registration; a name addresses a directory; nothing is persisted centrally.**
The name is a filesystem-safe slug — ``swissprot``, never ``UniProtKB/Swiss-Prot``, which
carries a slash — and the directory is ``<protein_data_dir()>/db/<name>``.

**The ffindex prefix inside that directory is not always the name.** Measured on GPU71FM,
``db/swissprot/`` holds ``swissprot`` and ``db/pdb/`` holds ``pdb100``, so
:func:`ffindex_prefix` looks it up rather than assuming: the exact spelling when it is
there, else the **shortest** stem with a ``.dbtype`` beside it. Every derived sibling is that
stem plus a suffix — ``_h``, ``_ca``, ``_ss``, ``_clu``, ``_seq`` — so the shortest is the
database itself. :attr:`Database.path` is that prefix, which is what makes a
**Database** a :class:`~protein.search.mmseqs.SearchTarget`.

**A record claims the whole directory, not a flat file set.** #1 described ``<name>``,
``.index``, ``.dbtype``, ``.lookup`` and ``<name>_h``; pdb100 also has ``_ss`` (3Di), ``_ca``
(coordinates), ``_clu`` and a ``pdb100_seq*`` split database, and structural search cannot
run without ``_ss`` and ``_ca``. So :func:`database_files` claims every file it finds rather
than a list this package would have to keep in step with two tools.

**These databases are immutable and nothing here offers to change one.** The index holds
byte offsets into the data file, so editing the data breaks every offset; every real
mutation makes a *new* database (``createsubdb``, ``filterdb``, ``concatdbs``). ``adopt``,
``download`` and ``status`` are the whole surface, and a test pins that.

**A downloaded MMseqs2 database may have folded five residue codes**, because ``mmseqs
databases`` hardcodes ``createdb --gpu 1``. That is labelled rather than hidden:
:attr:`Database.is_gpu_encoded` reads the four ``.dbtype`` bytes and :meth:`Database.status`
carries the consequence. ADR-0003 says why the fold is accepted and how to reverse it.

Examples
--------
>>> from protein.db.base import DATABASE_SUBDIR, SequenceDatabase
>>> DATABASE_SUBDIR
'db'
>>> SequenceDatabase("uniref50").name
'uniref50'
>>> SequenceDatabase("uniref50").TOOL_NAME
'mmseqs'
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

from genome.store import completion

from protein import store

if TYPE_CHECKING:
    from genome.store.completion import CompletionRecord

    from protein.core import Protein
    from protein.external import MmseqsLikeTool

__all__ = [
    "DATABASE_KIND",
    "DATABASE_SUBDIR",
    "GPU_ENCODED_BIT",
    "GPU_RESIDUE_FOLD",
    "Database",
    "DatabaseStatus",
    "SequenceDatabase",
    "StructureDatabase",
    "database_data_dir",
    "database_files",
    "database_path",
    "ffindex_prefix",
    "is_gpu_encoded",
    "registered_names",
]

#: This package's subdirectory of registered **Database**s, under
#: :func:`protein.store.protein_data_dir`. Spelled **once**, here, because this lane owns the
#: database layout — :func:`protein.search.mmseqs.database_path` asks this module rather than
#: keeping a second copy of the string.
DATABASE_SUBDIR = "db"

#: What the **Completion marker** calls what it recorded, for every database of either kind.
#: One value, because registration does not vary by tool: the record's ``details`` carry which
#: tool searches it.
DATABASE_KIND = "database"

#: The bit ``mmseqs createdb --gpu 1`` stamps into the four little-endian ``.dbtype`` bytes:
#: ``00 00 08 00`` is GPU-extended, ``00 00 00 00`` is plain. Both tools write a ``.dbtype``,
#: so this is a property of the format rather than of either tool — pdb100 answers ``False``
#: and that is a real answer. Read the mechanism, never a claim about the source FASTA.
GPU_ENCODED_BIT = 8 << 16

#: What a GPU-encoded database means for the residues that come back out, in the caller's
#: terms rather than in the format's. ADR-0003 accepts the fold and says how to reverse it.
GPU_RESIDUE_FOLD = (
    "residues were encoded against a 21-letter table: B->D, Z->E, U/O->X. The standard 20 "
    "are untouched, so 575,303 of Swiss-Prot's 575,503 entries are byte-perfect and 200 are "
    "not. See ADR-0003."
)

#: What ``mmseqs view --idx-entry-type`` calls the parallel header database. ``0`` is the
#: sequence database, which is the default and is therefore left unspelled.
_HEADER_ENTRY_TYPE = 2


def database_data_dir() -> Path:
    """Return the directory registered **Database**s live in.

    Returns
    -------
    pathlib.Path
        ``<LIULAB_DATA>/protein/db``. Nothing is created by asking.

    Examples
    --------
    >>> import os
    >>> os.environ["LIULAB_DATA"] = "/scratch/liulab"
    >>> database_data_dir()
    PosixPath('/scratch/liulab/protein/db')
    >>> del os.environ["LIULAB_DATA"]
    """
    return store.protein_data_dir() / DATABASE_SUBDIR


def registered_names() -> list[str]:
    """Return every name that addresses a directory under :func:`database_data_dir`.

    A name here has a directory; whether that directory finished registering is
    :attr:`Database.is_registered`, which reads the record. The two are separate on purpose:
    an interrupted download leaves the first true and the second false, and a listing that
    hid it would hide the thing a caller has to repair.

    Returns
    -------
    list of str
        The directory names, sorted. Empty when nothing is registered, including when the
        root does not exist.

    Examples
    --------
    >>> registered_names()                                       # doctest: +SKIP
    ['pdb', 'swissprot']
    """
    root = database_data_dir()
    if not root.is_dir():
        return []
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir())


def ffindex_prefix(directory: Path) -> Path:
    """Return the ffindex prefix inside ``directory`` — the path a tool is pointed at.

    The exact spelling when ``<directory>/<directory.name>.dbtype`` is there, else the
    **shortest** stem with a ``.dbtype`` beside it. Measured on GPU71FM: ``db/swissprot/``
    holds ``swissprot``, so the first rule answers; ``db/pdb/`` holds ``pdb100`` and its
    ``_h``, ``_ca``, ``_ss``, ``_clu`` and ``_seq`` siblings, so the second does. Every
    derived sibling is the database's own stem plus a suffix, which is why the shortest one
    is the database.

    Parameters
    ----------
    directory : pathlib.Path
        A registered database's directory.

    Returns
    -------
    pathlib.Path
        ``<directory>/<prefix>``. Both tools find ``.index``, ``.dbtype`` and ``.lookup``
        beside it themselves.

    Raises
    ------
    LookupError
        If nothing in ``directory`` has a ``.dbtype`` file beside it, so it holds no ffindex
        database at all.

    Examples
    --------
    >>> from pathlib import Path
    >>> ffindex_prefix(Path("/data/protein/db/pdb"))             # doctest: +SKIP
    PosixPath('/data/protein/db/pdb/pdb100')
    """
    exact = directory / f"{directory.name}.dbtype"
    if exact.is_file():
        return directory / directory.name

    stems = sorted(
        (marker.with_suffix("") for marker in directory.glob("*.dbtype")),
        key=lambda stem: (len(stem.name), stem.name),
    )
    if not stems:
        raise LookupError(
            f"{directory} holds no ffindex database: nothing in it has a .dbtype file beside "
            f"it. {registered_help()}"
        )
    return stems[0]


def database_path(name: str) -> Path:
    """Return the ffindex prefix registered under ``name``.

    The one place a name becomes a path. :func:`protein.search.mmseqs.database_path` calls
    it, so ``p.search("swissprot")`` and ``SwissProt().path`` resolve the same way.

    Parameters
    ----------
    name : str
        A registered name — a filesystem-safe slug, never the tool's own spelling.

    Returns
    -------
    pathlib.Path
        The ffindex prefix. Nothing is created and no completion record is read: a search
        against a half-finished download is a failure the tool reports, and refusing it here
        would make every search pay for a record read.

    Raises
    ------
    LookupError
        If ``name`` has no directory, or the directory holds no ffindex database. The
        message names what *is* registered.

    Examples
    --------
    >>> database_path("swissprot")                               # doctest: +SKIP
    PosixPath('/scratch/zhoulab/hanliu/protein/db/swissprot/swissprot')
    """
    directory = database_data_dir() / name
    if not directory.is_dir():
        raise LookupError(f"{name!r} is not a registered database. {registered_help()}")
    return ffindex_prefix(directory)


def registered_help() -> str:
    """Return a sentence naming the registered databases, for the end of a failure message.

    Returns
    -------
    str
        What is registered and where, or that nothing is.

    Examples
    --------
    >>> registered_help()                                        # doctest: +SKIP
    'Registered under /scratch/liulab/protein/db: pdb, swissprot.'
    """
    root = database_data_dir()
    names = registered_names()
    if not names:
        return f"Nothing is registered under {root}."
    return f"Registered under {root}: {', '.join(names)}."


def is_gpu_encoded(prefix: Path) -> bool:
    """Return whether the database at ``prefix`` was built with ``createdb --gpu 1``.

    Read from the four ``.dbtype`` bytes and nothing else — see :data:`GPU_ENCODED_BIT`. It
    is named for the **mechanism** rather than for fidelity: the bytes prove how the database
    was encoded and say nothing about the FASTA behind it, so ``is_lossless`` would be a lie
    for a database built cleanly from damaged input.

    Parameters
    ----------
    prefix : pathlib.Path
        The ffindex prefix, as :func:`ffindex_prefix` gives it.

    Returns
    -------
    bool
        ``True`` for ``00 00 08 00``, ``False`` for ``00 00 00 00`` and for a ``.dbtype``
        that is absent or too short to read.

    Examples
    --------
    >>> from pathlib import Path
    >>> is_gpu_encoded(Path("/data/protein/db/pdb/pdb100"))      # doctest: +SKIP
    False
    """
    marker = prefix.parent / f"{prefix.name}.dbtype"
    try:
        raw = marker.read_bytes()[:4]
    except OSError:
        return False
    if len(raw) < 4:
        return False
    return bool(int.from_bytes(raw, "little") & GPU_ENCODED_BIT)


def database_files(directory: Path) -> list[Path]:
    """Return every file a record for ``directory`` should claim.

    Everything under it that is not hidden. Hidden entries are bookkeeping — the record
    itself and the working area — and the rest is the database, whatever shape the tool gave
    it. That is deliberate rather than lazy: a hand-written list of suffixes would have
    tracked ``<name>``, ``.index``, ``.dbtype``, ``.lookup`` and ``_h`` and **missed pdb100's
    ``_ss`` and ``_ca``**, without which structural search cannot run.

    Symlinks are claimed as the files they are. ``pdb100_seq.0`` is a link to ``pdb100``, and
    recording the size it resolves to is what lets a caller copy the tree with ``rsync -a``
    and have the record still agree.

    Parameters
    ----------
    directory : pathlib.Path
        A database's directory.

    Returns
    -------
    list of pathlib.Path
        Absolute paths, sorted. Empty when the directory is absent or holds no file.

    Examples
    --------
    >>> from pathlib import Path
    >>> database_files(Path("/data/protein/db/swissprot"))       # doctest: +SKIP
    [PosixPath('/data/protein/db/swissprot/swissprot'), ...]
    """
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )


def _count_lines(path: Path) -> int | None:
    """Return how many newlines ``path`` holds, or ``None`` when it is not readable.

    Read in blocks rather than whole: ``pdb100.lookup`` is 50 MB and ``pdb100_seq.index``
    is 32 MB, and a status report is not a reason to hold either in memory.
    """
    try:
        with path.open("rb") as handle:
            return sum(block.count(b"\n") for block in iter(lambda: handle.read(1 << 20), b""))
    except OSError:
        return None


@dataclass(frozen=True)
class DatabaseStatus:
    """What :meth:`Database.status` found on disk, without touching the network.

    Attributes
    ----------
    name : str
        The registered name.
    directory : str
        Where it lives, whether or not it exists.
    path : str or None
        The ffindex prefix, or ``None`` when the directory holds no ffindex database.
    tool : str
        The **External tool** this database is searched with.
    registered : bool
        Whether a **Completion marker** is here — the only thing that says a build finished.
    is_gpu_encoded : bool or None
        Whether ``createdb --gpu 1`` built it; ``None`` when there is no ``.dbtype`` to read.
    residue_fold : str or None
        What that costs the caller, spelled out, or ``None`` when nothing was folded.
    index_entries : int or None
        Rows in ``<prefix>.index`` — **the searchable set**, which is what a search can hit.
        For pdb100 that is 324,204 representatives.
    lookup_entries : int or None
        Rows in ``<prefix>.lookup`` — every *named* entry, which for pdb100 is 1,562,678
        chains, five times the searchable count. Both are reported because reporting one
        without saying which invites the wrong one to be quoted.
    files : int or None
        How many files the record claims, or how many are there when it is not registered.
    bytes : int or None
        Their total size in bytes. **Not ``du``**: a split database's symlinks are counted
        as the files they resolve to, so pdb100 reports 5.99 GB where the tree occupies
        4.3 GB. That is the right number for a record, which claims each name it can be
        asked about, and the wrong one for a disk budget.
    completed_at : str or None
        When registration finished, ISO-8601 in UTC.

    Examples
    --------
    >>> DatabaseStatus(name="swissprot", directory="/d", tool="mmseqs").registered
    False
    """

    name: str
    directory: str
    tool: str
    path: str | None = None
    registered: bool = False
    is_gpu_encoded: bool | None = None
    residue_fold: str | None = None
    index_entries: int | None = None
    lookup_entries: int | None = None
    files: int | None = None
    bytes: int | None = None
    completed_at: str | None = None

    def as_json(self) -> dict[str, Any]:
        """Return this status as the mapping ``--json`` prints.

        Returns
        -------
        dict
            One key per attribute, in the order they are declared.

        Examples
        --------
        >>> DatabaseStatus(name="pdb", directory="/d", tool="foldseek").as_json()["tool"]
        'foldseek'
        """
        return {
            "name": self.name,
            "directory": self.directory,
            "path": self.path,
            "tool": self.tool,
            "registered": self.registered,
            "is_gpu_encoded": self.is_gpu_encoded,
            "residue_fold": self.residue_fold,
            "index_entries": self.index_entries,
            "lookup_entries": self.lookup_entries,
            "files": self.files,
            "bytes": self.bytes,
            "completed_at": self.completed_at,
        }


class Database(ABC):
    """One registered database: a name, a directory of ffindex files, and a record.

    Abstract in exactly one place — the **External tool** it is searched with — because that
    is the only thing the two halves of the hierarchy disagree about.
    :class:`SequenceDatabase` brings MMseqs2 and :class:`StructureDatabase` brings Foldseek.

    **There is no ``__getitem__`` here.** A sequence database can hand back one entry because
    MMseqs2 supports exactly that (``mmseqs view --id-list``), and what comes back is a
    **Protein**. A structure database cannot: pdb100 is a search target, its entries are
    C-alpha traces keyed by assembly and chain, and coordinates come from
    :func:`protein.structure.fetch` instead. A shared retrieval verb would have had to return
    two unrelated types or raise on one side, so the verb lives where the tool supports it.

    Parameters
    ----------
    name : str, optional
        The registered name — a filesystem-safe slug. Defaults to :attr:`NAME`, which a
        concrete class such as :class:`~protein.db.swissprot.SwissProt` declares.
    source : str, optional
        The tool's own spelling of this database, for ``download`` — e.g.
        ``"UniProtKB/Swiss-Prot"``. Defaults to :attr:`SOURCE`.
    tool : protein.external.MmseqsLikeTool, optional
        The tool to drive. Defaults to this class's own; a test binds a
        :class:`~protein.external.RecordingTool` here.

    Attributes
    ----------
    TOOL_NAME : str
        The binary this database is searched with.
    KIND : str
        ``"sequence"`` or ``"structure"`` — what the record writes down, so a name adopted
        without a declaration can still be reopened as the right class.
    NAME : str or None
        The slug a concrete class registers under, when it has one.
    SOURCE : str or None
        The tool's spelling of it, when it has one.
    name : str
        The registered name.
    source : str or None
        The tool's spelling.

    Examples
    --------
    >>> from protein.db import SwissProt
    >>> SwissProt().name, SwissProt().source
    ('swissprot', 'UniProtKB/Swiss-Prot')
    """

    TOOL_NAME: ClassVar[str]
    KIND: ClassVar[str]
    NAME: ClassVar[str | None] = None
    SOURCE: ClassVar[str | None] = None

    def __init__(
        self,
        name: str | None = None,
        *,
        source: str | None = None,
        tool: MmseqsLikeTool | None = None,
    ) -> None:
        resolved = name if name is not None else self.NAME
        if resolved is None:
            raise ValueError(
                f"{type(self).__name__} has no default name, so one must be given: "
                f"{type(self).__name__}('uniref50'). A name is a filesystem-safe slug and "
                f"never the tool's spelling, which may hold a slash."
            )
        self.name = resolved
        self.source = source if source is not None else self.SOURCE
        self._tool = tool

    # -- where it is ---------------------------------------------------------

    @property
    def directory(self) -> Path:
        """The directory this name addresses, whether or not it exists.

        Returns
        -------
        pathlib.Path
            ``<LIULAB_DATA>/protein/db/<name>``.

        Examples
        --------
        >>> import os
        >>> from protein.db import SwissProt
        >>> os.environ["LIULAB_DATA"] = "/scratch/liulab"
        >>> SwissProt().directory
        PosixPath('/scratch/liulab/protein/db/swissprot')
        >>> del os.environ["LIULAB_DATA"]
        """
        return database_data_dir() / self.name

    @property
    def path(self) -> Path:
        """The ffindex prefix a tool is pointed at — see :func:`ffindex_prefix`.

        This one attribute is the whole of
        :class:`protein.search.mmseqs.SearchTarget`, so ``p.search(SwissProt())`` and
        ``p.search("swissprot")`` are the same call.

        Returns
        -------
        pathlib.Path
            ``<directory>/<prefix>``, which is not always ``<directory>/<name>``.

        Raises
        ------
        LookupError
            If the directory is absent or holds no ffindex database.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> SwissProt().path                                     # doctest: +SKIP
        PosixPath('/scratch/zhoulab/hanliu/protein/db/swissprot/swissprot')
        """
        return database_path(self.name)

    @property
    def tool(self) -> MmseqsLikeTool:
        """The **External tool** this database is searched with, built once and remembered.

        Returns
        -------
        protein.external.MmseqsLikeTool
            The tool given to the constructor, else this class's own.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> SwissProt().tool.name
        'mmseqs'
        """
        if self._tool is None:
            self._tool = self._new_tool()
        return self._tool

    @abstractmethod
    def _new_tool(self) -> MmseqsLikeTool:
        """Return the **External tool** this kind of database is searched with."""

    # -- whether it is finished ----------------------------------------------

    @property
    def record(self) -> CompletionRecord | None:
        """Return the **Completion marker** this database has, or ``None`` when it has none.

        Read from disk on every ask rather than cached: ``adopt`` and ``download`` write it,
        and an object that remembered the answer from before would keep saying no.

        Returns
        -------
        genome.store.completion.CompletionRecord or None
            What registration recorded — the files claimed, the tool version, the source.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> SwissProt().record                                   # doctest: +SKIP
        CompletionRecord(kind='database', name='swissprot', ...)
        """
        return completion.read_record(self.directory)

    @property
    def is_registered(self) -> bool:
        """Whether registration finished here.

        Returns
        -------
        bool
            Whether a record is present. A directory of files with no record is an
            interrupted run, which reads ``False`` here and raises from ``adopt``.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> SwissProt().is_registered                            # doctest: +SKIP
        True
        """
        return self.record is not None

    @property
    def is_gpu_encoded(self) -> bool:
        """Whether ``createdb --gpu 1`` built this database — see :func:`is_gpu_encoded`.

        Returns
        -------
        bool
            ``True`` when the ``.dbtype`` carries :data:`GPU_ENCODED_BIT`, and ``False``
            when it does not or cannot be read. ``foldseek databases PDB`` answers ``False``,
            so this is a Swiss-Prot problem rather than a **Database** problem.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> SwissProt().is_gpu_encoded                           # doctest: +SKIP
        True
        """
        try:
            prefix = self.path
        except LookupError:
            return False
        return is_gpu_encoded(prefix)

    def status(self) -> DatabaseStatus:
        """Report what is on disk here, reading no network and no sequence.

        Two entry counts are reported, not one, and each is named for the file it was
        counted from. They differ by five times for pdb100 — 324,204 searchable
        representatives in ``.index`` against 1,562,678 named chains in ``.lookup`` — and a
        report giving one number without saying which invites the wrong one to be quoted.

        Returns
        -------
        DatabaseStatus
            Everything a caller can learn offline, including whether the residues were
            folded and what that costs.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> SwissProt().status().is_gpu_encoded                  # doctest: +SKIP
        True
        """
        directory = self.directory
        record = completion.read_record(directory)
        try:
            prefix: Path | None = ffindex_prefix(directory)
        except LookupError:
            prefix = None

        folded = is_gpu_encoded(prefix) if prefix is not None else None
        if record is not None:
            files, size = len(record.files), sum(record.files.values())
        else:
            found = database_files(directory)
            files = len(found) or None
            size = sum(path.stat().st_size for path in found) if found else None

        return DatabaseStatus(
            name=self.name,
            directory=str(directory),
            tool=self.TOOL_NAME,
            path=str(prefix) if prefix is not None else None,
            registered=record is not None,
            is_gpu_encoded=folded,
            residue_fold=GPU_RESIDUE_FOLD if folded else None,
            index_entries=(
                _count_lines(prefix.parent / f"{prefix.name}.index") if prefix is not None else None
            ),
            lookup_entries=(
                _count_lines(prefix.parent / f"{prefix.name}.lookup")
                if prefix is not None
                else None
            ),
            files=files,
            bytes=size,
            completed_at=record.completed_at if record is not None else None,
        )

    # -- the two ways in -----------------------------------------------------

    @classmethod
    def adopt(
        cls,
        name: str,
        path: str | Path,
        *,
        force: bool = False,
        tool: MmseqsLikeTool | None = None,
    ) -> Self:
        """Point at a database already on disk under ``name``, and write a record for it.

        **The common case on a cluster**, where the gigabytes arrived by someone else's
        ``rsync`` and nothing is going to be downloaded again. Nothing is copied and nothing
        is rewritten: if ``path`` is not already ``<db>/<name>``, a symlink is made so the
        name addresses it, and the record is written beside the real files.

        ``adopt`` accepts a GPU-encoded database and a plain one alike, and records which it
        found. Refusing the first would refuse a database that is perfectly good for search —
        which is most of what these are for — and would close the reversal path ADR-0003
        deliberately leaves open.

        Parameters
        ----------
        name : str
            The registered name — a filesystem-safe slug.
        path : str or pathlib.Path
            The database's directory, or its ffindex prefix; either is accepted.
        force : bool, default False
            Write a fresh record even when one is already there. What to run after the files
            on disk legitimately changed.
        tool : protein.external.MmseqsLikeTool, optional
            The tool whose version the record notes. Defaults to this class's own.

        Returns
        -------
        Database
            The adopted database, registered.

        Raises
        ------
        LookupError
            If ``path`` is not a directory holding an ffindex database, or is not one.
        FileExistsError
            If ``<db>/<name>`` already holds something other than the database being adopted.
        genome.store.completion.RegistrationMismatchError
            If a record is already there and disagrees with the files. Re-run with ``force``
            once you know why.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> SwissProt.adopt("swissprot", "/scratch/db/swissprot")     # doctest: +SKIP
        SwissProt('swissprot')
        """
        database = cls(name, tool=tool)
        directory = _adopted_directory(Path(path).expanduser())
        target = database.directory
        if target.resolve() != directory.resolve():
            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    f"cannot adopt {directory} as {name!r}: {target} already exists. A name "
                    f"addresses one directory. Remove it, or adopt under another name."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(directory)

        existing = completion.read_record(target)
        if existing is not None and not force:
            differing = completion.disagreements(target, existing)
            if differing:
                raise completion.RegistrationMismatchError(
                    f"{target} disagrees with its record: {'; '.join(str(d) for d in differing)}"
                    f". Something changed these files after they were registered. Adopt it "
                    f"again with `protein db adopt {name} {directory} --force`."
                )
            return database
        database._write_record(details={"adopted_from": str(directory)})
        return database

    @classmethod
    def download(
        cls,
        name: str | None = None,
        *,
        source: str | None = None,
        force: bool = False,
        tool: MmseqsLikeTool | None = None,
    ) -> Self:
        """Delegate the download to the tool, then write a record for what it left.

        **We do not manage the download.** ``mmseqs databases`` and ``foldseek databases``
        fetch, unpack and build; this adds the record and nothing else. ADR-0003 records what
        that delegation costs for MMseqs2 and why it is accepted.

        The tool's temp directory is the database's own ``.work/``, which is `liulab-genome`'s
        convention and is removed only **after** the record is written — so a download killed
        half-way keeps what it fetched and a repeat does not pay for it twice. It is not
        :meth:`~protein.external.MmseqsLikeTool.scratch_dir`, which removes itself however
        the command ends and is right for a search rather than for an hour of downloading.

        Parameters
        ----------
        name : str, optional
            The registered name. Defaults to :attr:`NAME`.
        source : str, optional
            The tool's own spelling — ``"UniProtKB/Swiss-Prot"``, ``"PDB"``. Defaults to
            :attr:`SOURCE`.
        force : bool, default False
            Download again even when this name is already registered.
        tool : protein.external.MmseqsLikeTool, optional
            The tool to drive. Defaults to this class's own.

        Returns
        -------
        Database
            The downloaded database, registered.

        Raises
        ------
        ValueError
            If no ``source`` is known for this name, so there is nothing to ask the tool for.
        protein.external.ToolNotFoundError
            If the tool is not installed.
        RuntimeError
            If the tool exits non-zero.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> SwissProt.download()                                 # doctest: +SKIP
        SwissProt('swissprot')
        """
        database = cls(name, source=source, tool=tool)
        if database.is_registered and not force:
            return database
        if database.source is None:
            raise ValueError(
                f"no download source is known for {database.name!r}. Pass the tool's own "
                f"spelling, e.g. download({database.name!r}, source='UniRef50'), or adopt a "
                f"copy that is already on disk with `protein db adopt`."
            )

        directory = database.directory
        directory.mkdir(parents=True, exist_ok=True)
        work = completion.work_dir(directory)
        work.mkdir(parents=True, exist_ok=True)
        database.tool.databases(database.source, directory / database.name, work)
        database._write_record(source_url=database.source)
        completion.clear_work_dir(directory)
        return database

    # -- internals -----------------------------------------------------------

    def _write_record(
        self, *, source_url: str | None = None, details: dict[str, Any] | None = None
    ) -> CompletionRecord:
        """Claim every file in the directory and write the **Completion marker**."""
        directory = self.directory
        files = database_files(directory)
        if not files:
            raise LookupError(
                f"cannot register {self.name!r}: {directory} holds no files. Point `adopt` "
                f"at the directory the database is really in, or run `protein db download "
                f"{self.name}`."
            )
        prefix = ffindex_prefix(directory)
        record = completion.build_record(
            directory,
            kind=DATABASE_KIND,
            name=self.name,
            files=files,
            source_url=source_url,
            # Not `tools=`, which would reach a second copy of the tool through
            # liulab-genome and shell out past this object's own — including past the
            # RecordingTool a test bound here. Provenance never becomes a dependency, so a
            # tool that will not answer is recorded as having said nothing.
            details={
                "tool": self.TOOL_NAME,
                "tool_version": self._tool_version(),
                "kind": self.KIND,
                "prefix": prefix.name,
                "source": self.source,
                "gpu_encoded": is_gpu_encoded(prefix),
                **(details or {}),
            },
        )
        completion.write_record(directory, record)
        return record

    def _tool_version(self) -> str:
        """Return the tool's version line, or ``""`` when it is absent or will not say."""
        from protein.external import ToolNotFoundError

        try:
            return self.tool.version
        except ToolNotFoundError:
            return ""

    def __repr__(self) -> str:
        """Return ``ClassName('name')``.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> SwissProt()
        SwissProt('swissprot')
        """
        return f"{type(self).__name__}({self.name!r})"


def _adopted_directory(path: Path) -> Path:
    """Return the directory holding the database ``path`` names, accepting either spelling."""
    if path.is_dir():
        ffindex_prefix(path)
        return path
    if (path.parent / f"{path.name}.dbtype").is_file():
        return path.parent
    raise LookupError(
        f"{path} is neither a directory holding an ffindex database nor an ffindex prefix "
        f"with a .dbtype beside it. Point `adopt` at the directory `mmseqs databases` or "
        f"`foldseek databases` wrote, e.g. /scratch/zhoulab/hanliu/protein/db/swissprot."
    )


class SequenceDatabase(Database):
    """A **Database** of amino-acid sequences, searched with MMseqs2.

    Adds retrieval, because MMseqs2 supports it directly: ``mmseqs view --id-list`` prints
    one named entry, offline, with the source FASTA header byte-for-byte. See
    :meth:`__getitem__`.

    Examples
    --------
    >>> SequenceDatabase("uniref50").TOOL_NAME
    'mmseqs'
    >>> SequenceDatabase("uniref50").KIND
    'sequence'
    """

    TOOL_NAME: ClassVar[str] = "mmseqs"
    KIND: ClassVar[str] = "sequence"

    def _new_tool(self) -> MmseqsLikeTool:
        """Return :class:`~protein.external.Mmseqs`, deferred so importing costs nothing."""
        from protein.external import Mmseqs

        return Mmseqs()

    def key_for(self, name: str) -> str:
        r"""Return the numeric key ``name`` is stored under, from the database's ``.lookup``.

        **The keys are opaque numbers and have nothing to do with the name.** ``createdb``
        shuffles the database, so ``P12345`` is key ``415743`` in the real Swiss-Prot;
        ``.lookup`` is ``key \t name \t fileNumber`` and is the only map between them. It
        is read here rather than through ``mmseqs view --id-mode 1`` for two reasons: the
        header database has no ``.lookup`` of its own, so ``--idx-entry-type 2`` cannot take
        a name and the key is needed anyway; and **a missing name is a warning on stderr and
        exit 0**, so absence has to be detected rather than inferred from a status code.

        The file is *not* sorted by name on disk — measured on the real Swiss-Prot, whose
        first three rows are ``P83570``, ``P0DPR3``, ``P84761``. MMseqs2 sorts it in memory
        when it opens it. So this scans, which for Swiss-Prot's 9 MB is a few milliseconds
        against the quarter-second the ``view`` below costs.

        Parameters
        ----------
        name : str
            The name as ``.lookup`` spells it — a UniProt accession for Swiss-Prot.

        Returns
        -------
        str
            The numeric key.

        Raises
        ------
        KeyError
            If the database carries no such name.
        LookupError
            If the database is not registered, or has no ``.lookup``.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> SwissProt().key_for("P12345")                        # doctest: +SKIP
        '415743'
        """
        prefix = self.path
        lookup = prefix.parent / f"{prefix.name}.lookup"
        if not lookup.is_file():
            raise LookupError(
                f"{prefix} has no .lookup, so nothing maps a name to a key. This is a "
                f"database one entry cannot be pulled out of by name."
            )
        data = lookup.read_bytes()
        needle = b"\t" + name.encode("utf-8") + b"\t"
        at = data.find(needle)
        if at < 0:
            raise KeyError(
                f"{name!r} is not in {self.name}. Its .lookup carries the names this "
                f"database was built from, and this is not one of them."
            )
        return data[data.rfind(b"\n", 0, at) + 1 : at].decode("utf-8")

    def entry(self, name: str) -> tuple[str, str]:
        """Return one entry's ``(header, sequence)``, offline, from the local database.

        Two ``mmseqs view`` calls against the key :meth:`key_for` resolved: the header lives
        in the parallel ``_h`` database and the residues in the main one, and ``view`` prints
        one of them per call. Roughly a quarter of a second each on the real Swiss-Prot,
        almost all of it the process launch — **this is a per-entry door, not a bulk one**.

        Parameters
        ----------
        name : str
            The name the database keys on.

        Returns
        -------
        tuple of (str, str)
            The source FASTA header with its ``>`` stripped, and the residues as the tool
            decodes them.

        Raises
        ------
        KeyError
            If the database carries no such name.
        protein.external.ToolNotFoundError
            If the tool is not installed.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> SwissProt().entry("P12345")[0][:22]                  # doctest: +SKIP
        'sp|P12345|AATM_RABIT A'
        """
        key = self.key_for(name)
        prefix = self.path
        header = self.tool.view(prefix, [key], entry_type=_HEADER_ENTRY_TYPE).strip()
        sequence = self.tool.view(prefix, [key]).strip()
        return header, sequence

    def __contains__(self, name: str) -> bool:
        """Return whether this database carries ``name``, reading only its ``.lookup``.

        Parameters
        ----------
        name : str
            The name to look for.

        Returns
        -------
        bool
            Whether it is there. No subprocess runs.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> "P12345" in SwissProt()                              # doctest: +SKIP
        True
        """
        try:
            self.key_for(name)
        except KeyError:
            return False
        return True

    def __getitem__(self, name: str) -> Protein:
        """Return one entry as a :class:`~protein.core.Protein`.

        Parameters
        ----------
        name : str
            The name the database keys on.

        Returns
        -------
        protein.core.Protein
            Its residues, with the header's identifier as ``id`` and the rest as
            ``description``. :class:`~protein.db.swissprot.SwissProt` resolves the header
            further, into an accession, an entry name and annotation fields.

        Raises
        ------
        KeyError
            If the database carries no such name.

        Examples
        --------
        >>> from protein.db import SwissProt
        >>> SwissProt()["P12345"].name                           # doctest: +SKIP
        'AATM_RABIT'
        """
        header, sequence = self.entry(name)
        return self._to_protein(name, header, sequence)

    def _to_protein(self, name: str, header: str, sequence: str) -> Protein:
        """Build the **Protein** one entry becomes.

        A subclass that knows the header's conventions overrides this and fills more of it.
        """
        from protein.core import Protein
        from protein.io import fasta

        identifier, description = fasta.split_header(header)
        return Protein(
            sequence,
            id=identifier or name,
            description=description,
            metadata={"header": header, "database": self.name},
        )


class StructureDatabase(Database):
    """A **Database** of structures, searched with Foldseek.

    **A search target and nothing else, deliberately.** There is no ``__getitem__``: pdb100
    holds C-alpha only, 79% of its named chains are absent from the searchable index,
    residues are renumbered on the way out, and getting one chain back is ``createsubdb``
    plus ``convert2pdb`` writing files. Coordinates come from
    :func:`protein.structure.fetch` — the cache under ``<LIULAB_DATA>/protein/structures/``,
    with an RCSB fetch on a miss — so ``Structure("1UBQ")`` needs nothing from this class,
    and a lookup here would answer with a path out of a different tree.

    Examples
    --------
    >>> StructureDatabase("pdb").TOOL_NAME
    'foldseek'
    >>> hasattr(StructureDatabase("pdb"), "__getitem__")
    False
    """

    TOOL_NAME: ClassVar[str] = "foldseek"
    KIND: ClassVar[str] = "structure"

    def _new_tool(self) -> MmseqsLikeTool:
        """Return :class:`~protein.external.Foldseek`, deferred so importing costs nothing."""
        from protein.external import Foldseek

        return Foldseek()
