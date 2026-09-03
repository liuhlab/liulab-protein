"""The structure lane: a **Structure** over the PDB, and the **Chain**s it holds.

Two classes and no mixin. :class:`~protein.structure.structure.Structure` is a peer of
:class:`~protein.core.Protein`, not a part of one: a protein is addressed by a UniProt
accession and a structure by a PDB id, the two are many-to-many in **both** directions
(47,348 entries reach more than one accession, 43,032 accessions reach more than one entry),
and a structure also holds nucleic acids and ligands that have no protein at all. SIFTS is
the join, it is chain-level, and it therefore lands on
:class:`~protein.structure.chain.Chain`.

``Protein.foldseek_search()`` is not here and is not anywhere: Foldseek takes a structure,
so the search lives where the coordinates are. *A method exists where the tool supports the
thing directly, never where the class would first have to acquire something else.*

Examples
--------
>>> from protein import Chain, Structure
>>> Structure("1UBQ")
Structure('1UBQ')
>>> Chain.__name__
'Chain'
"""

from __future__ import annotations

from protein.structure.chain import Chain
from protein.structure.structure import (
    CoordinatesNotDownloadedError,
    Structure,
    cached_path,
    fetch,
    structure_data_dir,
)

__all__ = [
    "Chain",
    "CoordinatesNotDownloadedError",
    "Structure",
    "cached_path",
    "fetch",
    "structure_data_dir",
]
