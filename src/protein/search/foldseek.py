"""Structural search with Foldseek — the same lane, over coordinates instead of residues.

Thin on purpose. Foldseek vendors MMseqs2, so everything except *what a query is* is already
:mod:`protein.search.mmseqs`': the database is resolved by
:func:`~protein.search.mmseqs.database_path`, the flags by
:func:`~protein.search.mmseqs.search_flags`, and the hits are read by
:func:`~protein.search.mmseqs.read_hits`. Writing a second parser here is how the two lanes
would come to disagree about a column.

**A query is a structure file that already exists**, so nothing is written into the scratch
directory — a ``Protein`` has no coordinates, which is why there is no ``foldseek_search()``
on it and why this module's caller is ``Structure`` and ``Chain`` rather than the search
mixin. The frame's identity column is ``fident``, a **fraction**, where MMseqs2's ``pident``
is a percentage, and Foldseek adds ``alntmscore`` and ``lddt``.

``q3di`` is not among them. It reads like the obvious third structural column and it does not
exist: it appears in no ``--format-output`` list on Foldseek 10-941cd33, and asking for it
fails the whole search with ``Format code q3di does not exist``.

Examples
--------
>>> from protein.external import Foldseek
>>> Foldseek().format_columns[2]
'fident'
>>> Foldseek().format_columns[-2:]
('alntmscore', 'lddt')
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from protein.external import Foldseek
from protein.search.mmseqs import database_path, read_hits, search_flags

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from protein.external import MmseqsLikeTool
    from protein.search.mmseqs import SearchTarget

__all__ = ["search"]


def search(
    structure: str | Path,
    database: SearchTarget | str,
    *,
    tool: MmseqsLikeTool | None = None,
    sensitivity: float | None = None,
    evalue: float | None = None,
    max_seqs: int | None = None,
    threads: int | None = None,
    extra: Sequence[str] = (),
) -> pd.DataFrame:
    """Search one structure file against ``database`` and return the hits.

    One ``foldseek easy-search``. **A multi-chain query is one invocation, not a loop**:
    Foldseek fans a structure out per chain itself and reports each in the ``query`` column,
    so a caller that has a whole structure passes the whole structure.

    Parameters
    ----------
    structure : str or pathlib.Path
        The query coordinates — mmCIF or PDB. It must already be on disk; Foldseek reads a
        file, and this lane writes none.
    database : protein.search.mmseqs.SearchTarget or str
        What to search against: a **Database**, or the name of a registered one.
    tool : protein.external.MmseqsLikeTool, optional
        The tool to drive. Defaults to :class:`~protein.external.Foldseek`.
    sensitivity, evalue, max_seqs, threads, extra : optional
        As :func:`protein.search.mmseqs.search_flags`.

    Returns
    -------
    pandas.DataFrame
        The hits, in Foldseek's column order — identity is ``fident``, a fraction, and
        ``alntmscore`` and ``lddt`` come last. Empty with the same columns when nothing was
        found.

    Raises
    ------
    LookupError
        If ``database`` names nothing registered.
    protein.external.ToolNotFoundError
        If ``foldseek`` is not installed.
    RuntimeError
        If the search exits non-zero. The message carries the tool's own output.

    Examples
    --------
    >>> hits = search("1ubq.cif", "pdb")            # doctest: +SKIP
    >>> hits.columns[:3].tolist()                   # doctest: +SKIP
    ['query', 'target', 'fident']
    """
    tool = tool if tool is not None else Foldseek()
    target = database_path(database)
    flags = search_flags(
        sensitivity=sensitivity,
        evalue=evalue,
        max_seqs=max_seqs,
        threads=threads,
        extra=extra,
    )
    with tool.scratch_dir("search") as work:
        output = work / "hits.tsv"
        tool.easy_search(Path(structure), target, output, extra=flags)
        return read_hits(output, tool)
