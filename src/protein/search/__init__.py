"""The search lane: sequence search with MMseqs2, structural search with Foldseek.

Only :class:`~protein.search.mixin.SearchMixin` is here so far, and it is a stub — enough
that :class:`protein.core.Protein` has a base to resolve. The lane's own modules (``mmseqs``,
``foldseek``, ``cli``) arrive with the implementation; the ``cli`` one is what
:mod:`protein.cli` mounts as ``protein search``.
"""

from __future__ import annotations

from protein.search.mixin import SearchMixin

__all__ = ["SearchMixin"]
