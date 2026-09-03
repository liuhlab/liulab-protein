"""The embedding lane: ESM-C's weights as an object, and the value it returns.

Two modules and no mixin. :class:`~protein.embed.esm.ESMC` holds the weights;
:class:`~protein.embed.embedding.Embedding` is what one call over them gives back.
:class:`protein.core.Protein` has no ``embed()`` at all — *resident state gets an object; a
subprocess does not*, which is the rule that keeps 1.33 GB of weights out of a method with
nowhere to put them.

Neither module's body imports torch. :mod:`protein.embed.embedding` never does at all, and
:mod:`protein.embed.esm` does it inside method bodies, so ``import protein`` stays cheap.
The ``cli`` module is what :mod:`protein.cli` mounts as ``protein esm``.
"""

from __future__ import annotations

from protein.embed.embedding import Embedding
from protein.embed.esm import CHECKPOINTS, ESMC, Embeddable

__all__ = ["CHECKPOINTS", "ESMC", "Embeddable", "Embedding"]
