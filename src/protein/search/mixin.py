"""The mixin that gives :class:`~protein.core.Protein` its sequence search.

One method, because MMseqs2 takes a sequence **directly** and holds nothing between calls.
Foldseek's structural search is not here: a ``Protein`` has no coordinates, so ``search()``
over structures belongs to ``Structure`` and ``Chain``.

``protein.core`` imports this module, so **nothing heavy is imported at module level** —
pandas would cost every ``import protein`` a search none of them may run.

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

        A thin entry point onto :func:`protein.search.mmseqs.search`, where the knobs are
        documented.

        Parameters
        ----------
        database : protein.search.mmseqs.SearchTarget or str
            What to search against: a **Database**, or the name of a registered one.
        **kwargs : Any
            Forwarded to :func:`protein.search.mmseqs.search`. ``query_name`` is filled in
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
        kwargs.setdefault("query_name", protein.id or mmseqs.DEFAULT_QUERY_NAME)
        return mmseqs.search(str(protein.sequence), database, **kwargs)
