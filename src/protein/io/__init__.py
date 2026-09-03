"""Reading and writing this package's file formats, one module per format.

A directory and never a flat ``io.py``, so adding a format is adding a file. The submodules
are what this package exports, not their functions — ``fasta.read_records`` says which format
it reads, where a bare ``read_records`` would not::

    from protein.io import fasta

    proteins = list(fasta.read_proteins("queries.fasta.gz"))

**The name ``io`` is this package's own.** Python 3 resolves ``import io`` absolutely, so a
module inside ``protein/`` that wants the standard library's ``io`` still gets it.
"""

from __future__ import annotations

from protein.io import fasta, structure

__all__ = ["fasta", "structure"]
