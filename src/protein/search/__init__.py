"""The search lane: sequence search with MMseqs2, structural search with Foldseek.

Two queries, one grammar. :mod:`protein.search.mmseqs` searches with a sequence and owns the
hit table both tools answer with; :mod:`protein.search.foldseek` searches with coordinates
and reads its results with the same parser, because Foldseek vendors MMseqs2 and a second
parser is how two lanes come to disagree about a column.

:class:`~protein.search.mixin.SearchMixin` puts the sequence half on
:class:`~protein.core.Protein`, whose sequence mmseqs takes directly. The structural half has
no mixin: it belongs to ``Structure`` and ``Chain``, which call
:func:`protein.search.foldseek.search` themselves.

:mod:`protein.search.cli` is what :mod:`protein.cli` mounts as ``protein search``.

Examples
--------
>>> from protein.search import SearchMixin, mmseqs
>>> SearchMixin.search.__name__
'search'
>>> mmseqs.DEFAULT_QUERY_NAME
'query'
"""

from __future__ import annotations

from protein.search.mixin import SearchMixin
from protein.search.mmseqs import SearchTarget

__all__ = ["SearchMixin", "SearchTarget"]
