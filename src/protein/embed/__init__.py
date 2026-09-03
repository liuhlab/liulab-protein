"""The embedding lane: ESM-C's weights as an object, and the value it returns.

Two modules and no mixin. :class:`~protein.embed.esm.ESMC` holds the weights;
:class:`~protein.embed.embedding.Embedding` is what one call over them gives back.
:class:`protein.core.Protein` has no ``embed()`` at all — *resident state gets an object; a
subprocess does not*.

Neither module's body imports torch, so ``import protein`` stays cheap.
"""

from __future__ import annotations

from protein.embed.embedding import Embedding
from protein.embed.esm import CHECKPOINTS, ESMC, Embeddable

__all__ = ["CHECKPOINTS", "ESMC", "Embeddable", "Embedding"]
