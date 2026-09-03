"""Sequence search with MMseqs2, and the hit table both **External tool**s answer with.

:func:`search` is the whole lane: one sequence, one **Database**, one
``mmseqs easy-search``, one :class:`~pandas.DataFrame`. It is what
:meth:`protein.search.mixin.SearchMixin.search` calls, and the query FASTA it writes lives
and dies inside :meth:`~protein.external.MmseqsLikeTool.scratch_dir`, so a search leaves
nothing behind.

**The hit table's grammar is here, not in each tool's module.** Foldseek vendors MMseqs2, so
the two write the same tab-separated columns chosen the same way — which is why
:class:`~protein.external.Foldseek` subclasses :class:`~protein.external.MmseqsLikeTool` and
why :mod:`protein.search.foldseek` reads its results with :func:`read_hits` from here rather
than with a second parser. :data:`COLUMN_DTYPES` types every column either tool can emit, so
a column is named and typed **once** and neither tool's order reaches a caller.

The two tools disagree about identity on purpose: MMseqs2 reports ``pident``, a percentage,
and Foldseek reports ``fident``, a fraction. Both are in :data:`COLUMN_DTYPES` under their
own names, and nothing here renames one to the other — a frame says which number it carries
by which column it has.

**Resolving a bare name is provisional.** ``search("swissprot")`` has to turn a name into a
path today, and the class that will own that — ``Database`` in ``protein.db`` — is not built
yet. :func:`database_path` is the one place the layout is spelled, so replacing it later is
one edit; see its docstring for the rule and what a ``Database`` must offer instead.

Examples
--------
>>> from protein.external import Foldseek, Mmseqs
>>> Mmseqs().format_columns[:3]
('query', 'target', 'pident')
>>> hit_dtypes(Mmseqs())["pident"], hit_dtypes(Foldseek())["fident"]
('float64', 'float64')
>>> empty_hits(Mmseqs()).shape
(0, 12)
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from protein.external import Mmseqs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from protein.external import MmseqsLikeTool

__all__ = [
    "COLUMN_DTYPES",
    "DATABASE_SUBDIR",
    "DEFAULT_QUERY_NAME",
    "SearchTarget",
    "database_path",
    "empty_hits",
    "hit_dtypes",
    "read_hits",
    "search",
    "search_flags",
]

#: Every column either **External tool** can be asked for, mapped to the dtype it is read
#: as. One table rather than one per tool, because the columns are one set: Foldseek vendors
#: MMseqs2 and adds two, and the identity column is the only name they disagree on.
#:
#: ``pident`` is a percentage and ``fident`` the same quantity as a fraction, so a caller
#: comparing two frames compares the column it has rather than a number that silently
#: changed scale. Both are ``float64``; nothing here converts one into the other.
COLUMN_DTYPES: Mapping[str, str] = MappingProxyType(
    {
        "query": "string",
        "target": "string",
        "pident": "float64",
        "fident": "float64",
        "alnlen": "int64",
        "mismatch": "int64",
        "gapopen": "int64",
        "qstart": "int64",
        "qend": "int64",
        "tstart": "int64",
        "tend": "int64",
        "evalue": "float64",
        "bits": "float64",
        "alntmscore": "float64",
        "lddt": "float64",
    }
)

#: This package's subdirectory of registered **Database**s, under
#: :func:`protein.store.protein_data_dir`. Spelled beside the code that reads it, which is
#: :func:`database_path` — and which ``protein.db`` takes over, since the layout is that
#: lane's to define once it exists.
DATABASE_SUBDIR = "db"

#: What the ``query`` column says when the searched **Protein** has no accession. A FASTA
#: record needs a header and MMseqs2 reports its first token, so the alternative is an empty
#: one — a hit table whose ``query`` column is blank rather than obviously a placeholder.
DEFAULT_QUERY_NAME = "query"


class SearchTarget(Protocol):
    """What a search needs of a **Database**: the path its **External tool** is pointed at.

    The whole contract, deliberately. ``protein.db``'s ``Database`` is not built yet, so this
    is what :func:`search` types its argument against and what the class must satisfy — one
    attribute, so satisfying it constrains nothing else about the class.

    :attr:`path` is the **ffindex prefix**, not the directory holding it: MMseqs2 and Foldseek
    take ``.../swissprot`` and find ``swissprot.index``, ``swissprot.dbtype`` and the rest
    beside it themselves.
    """

    @property
    def path(self) -> Path:
        """The ffindex prefix to search against."""
        ...


def hit_dtypes(tool: MmseqsLikeTool) -> dict[str, str]:
    """Return ``tool``'s result columns, in order, mapped to the dtype each is read as.

    Parameters
    ----------
    tool : protein.external.MmseqsLikeTool
        The tool whose ``--format-output`` the columns come from.

    Returns
    -------
    dict of str to str
        :attr:`~protein.external.MmseqsLikeTool.format_columns` against
        :data:`COLUMN_DTYPES`, in the tool's own order.

    Raises
    ------
    KeyError
        If the tool asks for a column :data:`COLUMN_DTYPES` does not type. That is the
        column table and the tool drifting apart, and it fails here rather than producing a
        frame whose dtype pandas guessed.

    Examples
    --------
    >>> from protein.external import Foldseek
    >>> list(hit_dtypes(Foldseek()))[-2:]
    ['alntmscore', 'lddt']
    """
    unknown = [name for name in tool.format_columns if name not in COLUMN_DTYPES]
    if unknown:
        raise KeyError(
            f"{tool.name} asks for {unknown}, which protein.search.mmseqs.COLUMN_DTYPES does "
            f"not type. Add each one there, so every column is named and typed in one place."
        )
    return {name: COLUMN_DTYPES[name] for name in tool.format_columns}


def empty_hits(tool: MmseqsLikeTool) -> pd.DataFrame:
    """Return a hit table of no rows, with the columns and dtypes a full one has.

    What a search that found nothing answers with, so a caller never has to branch on
    emptiness to learn the schema.

    Parameters
    ----------
    tool : protein.external.MmseqsLikeTool
        The tool whose columns the empty table carries.

    Returns
    -------
    pandas.DataFrame
        Zero rows, :func:`hit_dtypes`' columns and dtypes.

    Examples
    --------
    >>> from protein.external import Mmseqs
    >>> frame = empty_hits(Mmseqs())
    >>> frame.shape, frame.dtypes["evalue"]
    ((0, 12), dtype('float64'))
    """
    import pandas as pd

    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in hit_dtypes(tool).items()})


def read_hits(output: Path, tool: MmseqsLikeTool) -> pd.DataFrame:
    """Read what ``tool`` wrote to ``output`` as a named, typed hit table.

    The tab-separated file carries no header — the column names are the ones the tool was
    asked for, and this is the one place they are put back on. An output that is missing or
    empty is a search that found nothing, which :func:`empty_hits` answers.

    Parameters
    ----------
    output : pathlib.Path
        The tab-separated hits an ``easy-search`` or ``convertalis`` wrote.
    tool : protein.external.MmseqsLikeTool
        The tool that wrote it, whose ``--format-output`` named the columns.

    Returns
    -------
    pandas.DataFrame
        One row per hit, :func:`hit_dtypes`' columns in the tool's order.

    Examples
    --------
    >>> from pathlib import Path
    >>> from protein.external import Mmseqs
    >>> hits = read_hits(Path("tests/data/mmseqs_hits_p01308.tsv"), Mmseqs())  # doctest: +SKIP
    >>> hits.loc[0, "target"]                                                  # doctest: +SKIP
    'Q6YK33'
    """
    import pandas as pd

    dtypes = hit_dtypes(tool)
    if not output.is_file() or output.stat().st_size == 0:
        return empty_hits(tool)
    # Typed after the read rather than during it: `read_csv`'s own `dtype=` is annotated
    # `dict[Hashable, Dtype]`, and `dict` is invariant in its key, so no `dict[str, str]`
    # satisfies it. `astype` takes a mapping and is what the sibling package uses.
    return pd.read_csv(output, sep="\t", header=None, names=list(dtypes)).astype(dtypes)


def search_flags(
    *,
    sensitivity: float | None = None,
    evalue: float | None = None,
    max_seqs: int | None = None,
    threads: int | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """Turn the named search knobs into the arguments both tools spell the same way.

    Only what was asked for is passed: an omitted knob leaves the tool's own default, which
    differs between them and between versions and is not this package's to restate.

    Parameters
    ----------
    sensitivity : float, optional
        ``-s``. Lower is faster and finds less; MMseqs2 defaults to 5.7 and Foldseek to 9.5.
    evalue : float, optional
        ``-e``. Hits above it are not reported.
    max_seqs : int, optional
        ``--max-seqs``. How many targets per query pass the prefilter, which caps the hits.
    threads : int, optional
        ``--threads``. **Worth naming on a shared machine**: both tools default to every
        core, which on the lab's GPU node is 192 of them.
    extra : sequence of str, optional
        Further arguments, appended unread — anything neither this package nor the caller
        should have to wait for a release to pass.

    Returns
    -------
    list of str
        The arguments, in the order the parameters are listed, with ``extra`` last.

    Examples
    --------
    >>> search_flags(sensitivity=1.0, threads=4)
    ['-s', '1.0', '--threads', '4']
    >>> search_flags(extra=["--comp-bias-corr", "0"])
    ['--comp-bias-corr', '0']
    """
    named: list[tuple[str, float | int | None]] = [
        ("-s", sensitivity),
        ("-e", evalue),
        ("--max-seqs", max_seqs),
        ("--threads", threads),
    ]
    flags = [part for flag, value in named if value is not None for part in (flag, str(value))]
    return [*flags, *extra]


def database_path(database: SearchTarget | str) -> Path:
    """Return the ffindex prefix to search against, from a **Database** or a registered name.

    A :class:`SearchTarget` answers with its own :attr:`~SearchTarget.path`. A ``str`` is a
    registered name, and the layout is
    ``<protein_data_dir()>/<DATABASE_SUBDIR>/<name>/<ffindex prefix>``.

    **The prefix inside the directory is not always the name**, which is why it is looked up
    rather than assumed: measured on GPU71FM, ``db/swissprot/`` holds ``swissprot`` and
    ``db/pdb/`` holds ``pdb100``. The rule is the exact spelling when it is there, else the
    **shortest** stem with a ``.dbtype`` beside it — every derived ffindex sibling is that
    stem plus a suffix (``_h``, ``_ca``, ``_ss``, ``_clu``, ``_seq``), so the shortest is the
    database itself.

    **Provisional.** ``protein.db`` owns registration, and once it exists a name is resolved
    by asking it for a ``Database`` rather than by reading the layout here. Keeping the rule
    in one function is what makes that one edit.

    Parameters
    ----------
    database : SearchTarget or str
        A **Database** to take the path from, or the name of a registered one.

    Returns
    -------
    pathlib.Path
        The ffindex prefix. Nothing is created and no completion record is read.

    Raises
    ------
    LookupError
        If a name has no directory, or a directory holds no ffindex database. The message
        names the registered names there are.

    Examples
    --------
    >>> from pathlib import Path
    >>> class Registered:
    ...     path = Path("/data/protein/db/swissprot/swissprot")
    >>> database_path(Registered())
    PosixPath('/data/protein/db/swissprot/swissprot')
    >>> database_path("swissprot")                                   # doctest: +SKIP
    PosixPath('/scratch/zhoulab/hanliu/protein/db/swissprot/swissprot')
    """
    if not isinstance(database, str):
        return Path(database.path)

    # Deferred, and the module rather than the function: `protein.store` reaches
    # liulab-genome, and a test re-points the data root by patching that module.
    from protein import store

    root = store.protein_data_dir() / DATABASE_SUBDIR
    directory = root / database
    if not directory.is_dir():
        raise LookupError(f"{database!r} is not a registered database. {_registered(root)}")

    exact = directory / f"{database}.dbtype"
    if exact.is_file():
        return directory / database

    stems = sorted(
        (marker.with_suffix("") for marker in directory.glob("*.dbtype")),
        key=lambda stem: (len(stem.name), stem.name),
    )
    if not stems:
        raise LookupError(
            f"{directory} holds no ffindex database: nothing in it has a .dbtype file beside "
            f"it. {_registered(root)}"
        )
    return stems[0]


def search(
    sequence: str,
    database: SearchTarget | str,
    *,
    query_name: str = DEFAULT_QUERY_NAME,
    tool: MmseqsLikeTool | None = None,
    sensitivity: float | None = None,
    evalue: float | None = None,
    max_seqs: int | None = None,
    threads: int | None = None,
    extra: Sequence[str] = (),
) -> pd.DataFrame:
    """Search one amino-acid sequence against ``database`` and return the hits.

    One ``mmseqs easy-search``. The query FASTA and the tool's output both land in a
    :meth:`~protein.external.MmseqsLikeTool.scratch_dir`, which removes them however the
    search ends — a one-sequence query is not a file anyone wants left on a cluster
    filesystem, and the frame is the answer.

    Parameters
    ----------
    sequence : str
        The residues to search with, as ``str(protein.sequence)`` gives them.
    database : SearchTarget or str
        What to search against: a **Database**, or the name of a registered one.
    query_name : str, default "query"
        The FASTA header this query is written under, which is what the ``query`` column
        reports.
    tool : protein.external.MmseqsLikeTool, optional
        The tool to drive. Defaults to :class:`~protein.external.Mmseqs`; a test binds a
        stand-in here or patches :meth:`protein.external.ExternalTool.run`.
    sensitivity, evalue, max_seqs, threads : optional
        As :func:`search_flags`.
    extra : sequence of str, optional
        Further arguments, appended unread.

    Returns
    -------
    pandas.DataFrame
        The hits, :func:`hit_dtypes`' columns in MMseqs2's order — identity is ``pident``, a
        percentage. Empty with the same columns when nothing was found.

    Raises
    ------
    LookupError
        If ``database`` names nothing registered.
    protein.external.ToolNotFoundError
        If ``mmseqs`` is not installed.
    RuntimeError
        If the search exits non-zero. The message carries the tool's own output.

    Examples
    --------
    >>> hits = search("MKTAYIAKQRQISFVKSHFSRQ", "swissprot")   # doctest: +SKIP
    >>> hits.columns[:3].tolist()                              # doctest: +SKIP
    ['query', 'target', 'pident']
    """
    from protein.io import fasta

    tool = tool if tool is not None else Mmseqs()
    target = database_path(database)
    flags = search_flags(
        sensitivity=sensitivity,
        evalue=evalue,
        max_seqs=max_seqs,
        threads=threads,
        extra=extra,
    )
    with tool.scratch_dir("search") as work:
        query = work / "query.fasta"
        fasta.write_records(query, [(query_name, sequence)])
        output = work / "hits.tsv"
        tool.easy_search(query, target, output, extra=flags)
        return read_hits(output, tool)


def _registered(root: Path) -> str:
    """Name the registered databases under ``root``, for the end of a :class:`LookupError`."""
    names = (
        sorted(entry.name for entry in root.iterdir() if entry.is_dir()) if root.is_dir() else []
    )
    if not names:
        return f"Nothing is registered under {root}."
    return f"Registered under {root}: {', '.join(names)}."
