"""The embedding lane: ESM-C's weights as an object, and the value it returns.

No mixin. :class:`~protein.embed.esm.esmc.ESMC` holds the weights,
:class:`~protein.embed.embedding.Embedding` is what one call over them gives back,
:class:`~protein.embed.esm.sae.SAE` holds a sparse autoencoder's, and
:class:`~protein.embed.esm.sae.SaeActivation` is what one call over *those* gives back — a
peer of ``Embedding`` and not one of them. :class:`protein.core.Protein` has no ``embed()``
at all — *resident state gets an object; a subprocess does not*.

No module's body imports torch, so ``import protein`` stays cheap.
"""

from __future__ import annotations

from protein.embed.embedding import Embedding
from protein.embed.esm.esmc import CHECKPOINTS, ESMC, Embeddable
from protein.embed.esm.sae import SAE, SAE_CHECKPOINTS, SaeActivation

__all__ = [
    "CHECKPOINTS",
    "ESMC",
    "SAE",
    "SAE_CHECKPOINTS",
    "Embeddable",
    "Embedding",
    "SaeActivation",
]
