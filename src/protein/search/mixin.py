"""The mixin that gives :class:`~protein.core.Protein` its sequence search.

One method. MMseqs2 takes a sequence **directly**, which is the whole reason the method is
on ``Protein`` at all — the *direct support only* rule — and mmseqs holds nothing between
calls, which is why a subprocess stays a method where ESM-C's resident weights became the
``ESMC`` object instead.

Foldseek's structural search is **not** here. A ``Protein`` has no coordinates, so
``search()`` over structures belongs to ``Structure`` and ``Chain``, and it is
:mod:`protein.search.foldseek` they reach — the same lane, the same column parsing, a
different query.

**Nothing heavy is imported at module level.** ``protein.core`` imports this module, so
``import protein`` pays for whatever is up there: pandas at the top of this file would cost
every caller a second of import time to run a search none of them may run. The pandas import
is inside :meth:`SearchMixin.search`, ``Protein`` is named under
:data:`typing.TYPE_CHECKING` and reached with ``cast``, and a test walks the lane to hold
both.

Examples
--------
>>> from protein import Protein
>>> hits = Protein("MKTAYIAKQRQ", id="P12345").search("swissprot")   # doctest: +SKIP
>>> hits.columns[:3].tolist()                                        # doctest: +SKIP
['query', 'target', 'pident']
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pandas as pd

    from protein.core import Protein
    from protein.search.mmseqs import SearchTarget

__all__ = ["SearchMixin"]


class SearchMixin:
    """Sequence search over a **Database**, mixed into :class:`~protein.core.Protein`."""

    def search(self, database: SearchTarget | str, **kwargs: Any) -> pd.DataFrame:
        """Search this protein's sequence against ``database`` and return the hits.

        A thin entry point onto :func:`protein.search.mmseqs.search`, which is where the
        knobs are documented and where they stay spelled once. The query is written under
        this protein's accession, so the ``query`` column says which protein asked.

        Parameters
        ----------
        database : protein.search.mmseqs.SearchTarget or str
            What to search against: a **Database**, or the name of a registered one.
            ``p.search("swissprot")`` and ``p.search(SwissProt())`` are the same call.
        **kwargs : Any
            Forwarded to :func:`protein.search.mmseqs.search` — ``sensitivity``, ``evalue``,
            ``max_seqs``, ``threads``, ``extra`` and ``tool``. ``query_name`` is filled in
            from :attr:`~protein.core.Protein.id` unless it is given here.

        Returns
        -------
        pandas.DataFrame
            One row per hit, in MMseqs2's column order, with ``pident`` — a **percentage** —
            as the identity column. Empty with the same columns when nothing was found.

        Raises
        ------
        LookupError
            If ``database`` names nothing registered.
        protein.external.ToolNotFoundError
            If ``mmseqs`` is not installed.
        RuntimeError
            If the search exits non-zero.

        Examples
        --------
        >>> from protein import Protein
        >>> p = Protein("MKTAYIAKQRQISFVKSHFSRQ", id="P12345")
        >>> p.search("swissprot").loc[0, "target"]              # doctest: +SKIP
        'P12345'
        """
        from protein.search import mmseqs

        protein = cast("Protein", self)
        # `setdefault` rather than a parameter of its own: the accession is the right query
        # name for every call that does not say otherwise, and adding a knob here would put
        # one of `mmseqs.search`'s parameters in two signatures.
        kwargs.setdefault("query_name", protein.id or mmseqs.DEFAULT_QUERY_NAME)
        return mmseqs.search(str(protein.sequence), database, **kwargs)
