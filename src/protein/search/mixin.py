"""The mixin that gives :class:`~protein.core.Protein` its sequence search.

**A stub.** It carries no method yet: the class exists so ``class Protein(SearchMixin)``
resolves, and the search lane fills it in. Deliberately empty rather than a ``search`` that
raises, because an unimplemented method on the public surface is a promise the package
cannot yet keep, and ``hasattr(p, "search")`` would answer wrongly.

What lands here is one method::

    def search(self, database: SequenceDatabase | str, **kwargs: Any) -> pd.DataFrame

driven by ``mmseqs easy-search`` through :mod:`protein.external`, taking either a
``SequenceDatabase`` or the name of a registered one, and returning a frame whose columns are
named in one place so mmseqs' column order reaches no caller. Write it the way
`liulab-genome`'s mixins are written: no ``__init__``, ``cast("Protein", self)`` under a
:data:`typing.TYPE_CHECKING` import, and every heavy import inside the method body — a mixin
module that imports pandas at module level costs every ``import protein`` the same.

Foldseek's structural search is **not** this mixin's. A ``Protein`` has no coordinates, so
``foldseek_search()`` lives on ``Structure`` and ``Chain`` instead — the *direct support
only* rule, settled in the map.
"""

from __future__ import annotations

__all__ = ["SearchMixin"]


class SearchMixin:
    """Sequence search over a **Database**, mixed into :class:`~protein.core.Protein`.

    Empty until the search lane lands; see this module's docstring for the shape of the one
    method that goes here.
    """
