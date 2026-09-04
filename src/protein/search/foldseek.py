"""Structural search with Foldseek — the same lane, over coordinates instead of residues.

Thin on purpose. Everything except *what a query is* is shared: the database and the flags
come from :mod:`protein.search.target`, and the hits are parsed by
:func:`protein.search.mmseqs.read_hits`, because Foldseek vendors MMseqs2 and a second copy
of that parser could come to disagree about a column.

**A query is a structure file that already exists**, so this half of the lane writes nothing,
and its caller is ``Structure`` or ``Chain`` rather than the search mixin. The frame's
identity column is ``fident``, a **fraction**, where MMseqs2's ``pident`` is a percentage,
and Foldseek adds ``alntmscore`` and ``lddt``.

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
from protein.search.mmseqs import read_hits
from protein.search.target import database_path, search_flags

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from protein.external import MmseqsLikeTool
    from protein.search.target import SearchTarget

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
    Foldseek fans a structure out per chain itself and reports each in the ``query`` column.

    Parameters
    ----------
    structure : str or pathlib.Path
        The query coordinates — mmCIF or PDB. It must already be on disk.
    database : protein.search.target.SearchTarget or str
        What to search against: a **Database**, or the name of a registered one.
    tool : protein.external.MmseqsLikeTool, optional
        The tool to drive. Defaults to :class:`~protein.external.Foldseek`.
    sensitivity, evalue, max_seqs, threads, extra : optional
        As :func:`protein.search.target.search_flags`.

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
