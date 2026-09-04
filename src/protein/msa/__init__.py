"""The MSA lane: an alignment out of a **Database**, or out of sequences already in hand.

Four modules, each named for what drives it.
:class:`~protein.msa.msa.MSA` is the value both verbs answer with and the only shape an
alignment is held in here. :func:`~protein.msa.mmseqs.search` searches a **Database** and is
what :meth:`protein.core.Protein.msa` calls; :func:`~protein.msa.muscle.align` takes a set the
caller already holds. Two verbs, because the two jobs have different shapes.

:mod:`protein.msa.cli` is deliberately not re-exported: the root CLI mounts the sub-app by
module, the way it mounts every other lane's.

Examples
--------
>>> from protein import MSA
>>> MSA([("query", "MKTAY"), ("hit", "MKTaAY")], comment="#5")
MSA(depth 2, 5 match states)
>>> from protein.msa import align, search
>>> search.__module__, align.__module__
('protein.msa.mmseqs', 'protein.msa.muscle')
"""

from __future__ import annotations

from protein.msa.mmseqs import search
from protein.msa.msa import MSA
from protein.msa.muscle import align

__all__ = ["MSA", "align", "search"]
