"""The embedding lane: ESM-C's weights as an object, and the value it returns.

No mixin. :class:`~protein.embed.esm.esmc.ESMC` holds the weights,
:class:`~protein.embed.embedding.Embedding` is what one call over them gives back, and
:class:`~protein.embed.esm.sae.SaeActivation` is what a sparse autoencoder over one of those
gives back — a peer of ``Embedding`` and not one of them. :class:`protein.core.Protein` has no
``embed()`` at all — *resident state gets an object; a subprocess does not*.

No module's body imports torch, so ``import protein`` stays cheap.
"""

from __future__ import annotations

from protein.embed.embedding import Embedding
from protein.embed.esm.esmc import CHECKPOINTS, ESMC, Embeddable
from protein.embed.esm.sae import SaeActivation

__all__ = ["CHECKPOINTS", "ESMC", "Embeddable", "Embedding", "SaeActivation"]
