"""Searching a **Database** with MMseqs2 and answering with an :class:`~protein.msa.msa.MSA`.

Four invocations, each of them a single one: ``createdb`` for the query, ``search``,
``result2msa``, ``unpackdb``. The chaining is here rather than in :mod:`protein.external`,
which owns the grammar and not the recipe.

The organism helpers live here because a search is what puts a header in front of a row.
:func:`with_organism_key` is applied to every row that comes back, and nothing else in this
package calls either of them.

Examples
--------
>>> from protein.msa import search
>>> search("MKTAYIAKQRQISFVKSHFSRQ", "uniref30")           # doctest: +SKIP
MSA(depth 1281, 22 match states)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from protein.external import Mmseqs
from protein.msa.msa import MSA
from protein.search.target import DEFAULT_QUERY_NAME, database_path, search_flags

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from protein.external import MmseqsLikeTool
    from protein.search.target import SearchTarget

__all__ = ["organism_id", "search", "with_organism_key"]

#: How a search's own header spells the organism, and what a folding tool reads instead.
#: UniProtKB writes ``OX=``, UniRef writes ``TaxID=``, and a header that has already been
#: through :func:`with_organism_key` carries ``key=``.
_ORGANISM_ID = re.compile(r"\b(?:OX|TaxID|key)=(\d+)\b")

#: What ``result2msa`` is asked to write. The A3M modes replace a hit's header with its
#: accession alone, which throws away the organism id that pairs the chains of a complex;
#: this one keeps the header whole. What it costs is the hit's own insertions, which this
#: writer drops — every row it writes is one column per query residue, which is a valid A3M
#: with no insert columns.
_MSA_FORMAT_MODE = 2

#: What the unpacked alignment is named. ``0`` is ``--unpack-name-mode``: one query, so one
#: file, named by its database key rather than through a ``.lookup`` the query database is
#: not obliged to have.
_A3M_SUFFIX = ".a3m"
_UNPACK_BY_KEY = 0


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
) -> MSA:
    """Search ``database`` with one sequence and return the alignment it found.

    Everything the run makes lives and dies inside a
    :meth:`~protein.external.MmseqsLikeTool.scratch_dir`, so it leaves nothing behind and
    **there is no output path**: an alignment is a value, like a hit table.
    :meth:`~protein.msa.msa.MSA.write` is how one is kept.

    The headers a hit arrives under are carried whole and gain ``key=<organism id>`` wherever
    they name one — see :func:`with_organism_key` for why that matters.

    Parameters
    ----------
    sequence : str
        The residues to search with.
    database : protein.search.target.SearchTarget or str
        What to search against: a **Database**, or the name of a registered one. Required;
        nothing is shipped or adopted behind it.
    query_name : str, default "query"
        The FASTA header the query is written under, and the header of row 0.
    tool : protein.external.MmseqsLikeTool, optional
        The tool to drive. Defaults to :class:`~protein.external.Mmseqs`.
    sensitivity, evalue, max_seqs, threads : optional
        As :func:`protein.search.target.search_flags`. They reach the ``search`` verb.
    extra : sequence of str, optional
        Further arguments for ``search``, appended unread.

    Returns
    -------
    MSA
        Query-anchored, row 0 the query. Depth 1 — the query alone — when the search found
        nothing; a thin alignment is not refused.

    Raises
    ------
    LookupError
        If ``database`` names nothing registered.
    protein.external.ToolNotFoundError
        If ``mmseqs`` is not installed.
    RuntimeError
        If any of the four invocations exits non-zero.
    protein.msa.msa.InvalidAlignmentError
        If what came back is not a query-anchored alignment.

    Examples
    --------
    >>> search("MKTAYIAKQRQISFVKSHFSRQ", "uniref30")           # doctest: +SKIP
    MSA(depth 1281, 22 match states)
    """
    from protein.io import fasta

    driver = tool if tool is not None else Mmseqs()
    target = database_path(database)
    flags = search_flags(
        sensitivity=sensitivity,
        evalue=evalue,
        max_seqs=max_seqs,
        threads=threads,
        extra=extra,
    )
    with driver.scratch_dir("msa") as work:
        query = work / "query.fasta"
        fasta.write_records(query, [(query_name, sequence)])
        query_db = driver.createdb([query], work / "querydb")
        result_db = driver.search(query_db, target, work / "result", extra=flags)
        msa_db = driver.result2msa(
            query_db, target, result_db, work / "msadb", format_mode=_MSA_FORMAT_MODE
        )
        unpacked = work / "unpacked"
        unpacked.mkdir()
        driver.unpackdb(msa_db, unpacked, suffix=_A3M_SUFFIX, name_mode=_UNPACK_BY_KEY)
        return _read_unpacked(unpacked, query_name, sequence)


def organism_id(header: str) -> int | None:
    """Return the NCBI organism id ``header`` names, or ``None`` when it names none.

    Three spellings, because the databases worth searching disagree: UniProtKB writes
    ``OX=``, UniRef writes ``TaxID=``, and a header already carrying ``key=`` answers with
    that.

    Parameters
    ----------
    header : str
        One FASTA header, as the search wrote it.

    Returns
    -------
    int or None
        The organism id.

    Examples
    --------
    >>> organism_id("sp|P01308|INS_HUMAN Insulin OS=Homo sapiens OX=9606 GN=INS")
    9606
    >>> organism_id("UniRef100_A0A0 Cluster: x n=2 Tax=Mus musculus TaxID=10090")
    10090
    >>> organism_id("101") is None
    True
    """
    found = _ORGANISM_ID.search(header)
    return int(found.group(1)) if found else None


def with_organism_key(header: str) -> str:
    """Return ``header`` with ``key=<organism id>`` on the end, where it names one.

    **A row without one is unpaired.** ESMFold2 pairs the chains of a complex by a
    ``key=<int>`` match over the FASTA header, and a chain whose rows carry no key folds
    block-diagonal with nothing raised — a wrong answer that looks like an answer. The
    original header stays in front byte-for-byte, so nothing it said is lost.

    Parameters
    ----------
    header : str
        One FASTA header, as the search wrote it.

    Returns
    -------
    str
        ``header`` unchanged when it names no organism, or already carries a key.

    Examples
    --------
    >>> with_organism_key("sp|P01315|INS_PIG Insulin OS=Sus scrofa OX=9823")
    'sp|P01315|INS_PIG Insulin OS=Sus scrofa OX=9823 key=9823'
    >>> with_organism_key("101 key=9606")
    '101 key=9606'
    >>> with_organism_key("101")
    '101'
    """
    if "key=" in header:
        return header
    found = organism_id(header)
    return header if found is None else f"{header} key={found}"


def _read_unpacked(directory: Path, query_name: str, sequence: str) -> MSA:
    """Return the one alignment ``unpackdb`` wrote, or the query alone when it wrote none."""
    from protein.io import a3m

    written = sorted(directory.glob(f"*{_A3M_SUFFIX}"))
    if not written:
        return MSA([(query_name, sequence)])
    comment, records = a3m.read_records(written[0])
    return MSA(((with_organism_key(header), row) for header, row in records), comment=comment)
