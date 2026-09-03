"""Reading and writing this package's file formats, one module per format.

A directory and never a flat ``io.py``, because a second format is already coming: FASTA is
here, and mmCIF and PDB arrive as ``structure.py`` beside it when **Structure** lands. Adding
one is adding a file, and nothing here moves.

The submodules are what this package exports, not their functions — ``fasta.read_records``
says which format it reads, where a bare ``read_records`` would not, and two formats both
wanting the name is a collision that never happens::

    from protein.io import fasta

    proteins = list(fasta.read_proteins("queries.fasta.gz"))

**The name ``io`` is this package's own.** Python 3 resolves ``import io`` absolutely, so a
module inside ``protein/`` that wants the standard library's ``io`` still gets it, and this
directory shadows nothing.
"""

from __future__ import annotations

from protein.io import fasta

__all__ = ["fasta"]
