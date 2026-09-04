"""The sparse autoencoder over an **Embedding**, and the value one call gives back.

**This module's body imports torch nowhere.** Only :class:`SAE`'s method bodies do, so
``import protein`` never pulls torch, and :class:`SaeActivation` reaches for it nowhere at
all: a frozen value object holds no weights and loads none, so everything it promises is
checkable without a GPU. Tests assert both.

:class:`SAE` is a class you construct and keep, beside :class:`~protein.embed.esm.esmc.ESMC`
and for the same reason — *resident state gets an object*. It takes a slug and **has no
default**, so a first pairing names both halves or neither.

**The identity check is total.** A wrong parent, a layer the checkpoint does not cover and a
wrong width all multiply fine and give plausible numbers, so :meth:`SAE.encode` refuses each
of them by name, reading recorded facts rather than tensors. The reconstruction loss it hands
back is the empirical backstop on whether the pairing made sense.

:class:`SaeActivation` is a **peer** of :class:`~protein.embed.embedding.Embedding` and not
one of them. An embedding is dense and ``d_model`` wide; an activation is sparse over a
codebook far wider than that, so ``d_model`` would be a lie here and ``.mean()`` would dilute
a feature's presence by protein length. That is ADR-0007.

The pair of ``(L, k)`` arrays is **lossless**: the top-k selection returns exactly ``k`` slots
per row whether or not all ``k`` are non-zero, so the sparse form is a representation rather
than an approximation. :meth:`SaeActivation.dense` materialises the full matrix on request and
nothing here holds it.

Examples
--------
>>> import numpy as np
>>> from protein.embed import SAE_CHECKPOINTS, SaeActivation
>>> SAE_CHECKPOINTS["300m-layer23-k64-cb16384"]
('biohub/ESMC-300M-sae-layer23-k64-codebook16384', '300m')
>>> activation = SaeActivation(
...     np.array([[2, 0], [1, 3]], dtype=np.uint8),
...     np.array([[0.5, 0.25], [1.0, 0.75]], dtype=np.float16),
...     np.array([0.1, 0.2], dtype=np.float32),
...     "P12345",
...     "300m",
...     23,
...     "300m-layer23-k64-cb16384",
...     4,
...     2,
... )
>>> activation.shape, len(activation)
((2, 4), 2)
>>> activation.dense().tolist()
[[0.25, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.75]]
>>> activation.max().tolist()
[0.25, 1.0, 0.5, 0.75]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from protein.embed.embedding import Embedding
from protein.embed.esm.esmc import CHECKPOINTS

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["SAE", "SAE_CHECKPOINTS", "SaeActivation"]

#: Every ESM-C SAE checkpoint this package knows, slug to ``(hf_id, parent)``. The parent is
#: the one fact the table adds: the checkpoint's own config carries the layers it covers, its
#: width, ``k`` and the codebook dimension, and :class:`SAE` cross-checks the width against
#: the parent's the way :class:`~protein.embed.esm.esmc.ESMC` cross-checks its own.
#:
#: Every entry is a **layer-specific, single-shard** repository, which is what bounds the
#: snapshot ``from_pretrained`` fetches. The all-layer and MLP families are orders of
#: magnitude larger, and leaving them out is also what lets :meth:`SAE.encode` assume hidden
#: states rather than MLP outputs.
SAE_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "300m-layer23-k64-cb16384": ("biohub/ESMC-300M-sae-layer23-k64-codebook16384", "300m"),
    "600m-layer27-k64-cb16384": ("biohub/ESMC-600M-sae-layer27-k64-codebook16384", "600m"),
    "6b-layer60-k64-cb16384": ("biohub/ESMC-6B-sae-layer60-k64-codebook16384", "6b"),
}


@dataclass(frozen=True, slots=True)
class SaeActivation:
    """One sequence's sparse features, and the facts that say what they can be compared with.

    Frozen and slotted, for the reason :class:`~protein.embed.embedding.Embedding` is: a
    measurement that can be edited in place is one nobody can trust. Built by an SAE's
    ``encode`` rather than by hand, and the dtypes below are checked at construction so the
    storage promise is a fact rather than a convention.

    Parameters
    ----------
    indices : numpy.ndarray
        ``(L, k)``, which codebook feature filled each of the ``k`` slots at each residue.
        In :meth:`index_dtype` of ``codebook_size``.
    values : numpy.ndarray
        ``(L, k)`` float16, how hard each of those features fired. The encoder is a ReLU
        followed by a top-k, so every value is non-negative.
    reconstruction_loss : numpy.ndarray
        ``(L,)`` float32, one per residue: how badly the decoder rebuilt that residue's
        embedding. The empirical backstop on whether the pairing made sense.
    source : str or None
        The UniProt accession or chain key the embedding came from, or ``None``.
    parent : str
        The backbone slug the embedding was taken from, e.g. ``"300m"``.
    layer : int
        Which of the backbone's hidden states, as a non-negative index.
    sae : str
        The SAE checkpoint slug, naming the full variant.
    codebook_size : int
        How many features the SAE has — the width of what :meth:`dense` materialises.
    k : int
        How many slots the top-k selection fills per residue.
    normalized : bool, default False
        Whether the checkpoint's own per-feature statistics were applied. Recorded because
        two otherwise identical runs differ by it.

    Raises
    ------
    ValueError
        If the arrays disagree with each other, with ``k``, or with the dtypes above.

    Examples
    --------
    >>> import numpy as np
    >>> SaeActivation(
    ...     np.zeros((3, 2), dtype=np.uint16),
    ...     np.zeros((3, 2), dtype=np.float16),
    ...     np.zeros(3, dtype=np.float32),
    ...     None,
    ...     "6b",
    ...     60,
    ...     "6b-layer60-k64-cb16384",
    ...     16384,
    ...     2,
    ... ).normalized
    False
    """

    indices: np.ndarray
    values: np.ndarray
    reconstruction_loss: np.ndarray
    source: str | None
    parent: str
    layer: int
    sae: str
    codebook_size: int
    k: int
    normalized: bool = False

    def __post_init__(self) -> None:
        """Hold the three arrays to the shapes and dtypes this class promises."""
        wanted = self.index_dtype(self.codebook_size)
        if self.indices.shape != self.values.shape or self.indices.ndim != 2:
            raise ValueError(
                f"indices {self.indices.shape} and values {self.values.shape} are one "
                f"(L, k) table each and must agree: a slot without its magnitude, or a "
                f"magnitude without its feature, is not an activation."
            )
        if self.indices.shape[1] != self.k:
            raise ValueError(
                f"k={self.k} but the arrays hold {self.indices.shape[1]} slots per residue. "
                f"Top-k fills exactly k slots per row, so the two cannot differ."
            )
        if self.reconstruction_loss.shape != (self.indices.shape[0],):
            raise ValueError(
                f"reconstruction_loss {self.reconstruction_loss.shape} is one number per "
                f"residue, so it is ({self.indices.shape[0]},) here."
            )
        for name, found, expected in (
            ("indices", self.indices.dtype, wanted),
            ("values", self.values.dtype, np.dtype(np.float16)),
            ("reconstruction_loss", self.reconstruction_loss.dtype, np.dtype(np.float32)),
        ):
            if found != expected:
                raise ValueError(
                    f"{name} is {found} where this class stores {expected}. The dtypes are "
                    f"what make a long protein's activations cheap to keep and to pass "
                    f"around; cast before constructing rather than after."
                )

    @staticmethod
    def index_dtype(codebook_size: int) -> np.dtype[Any]:
        """Return the smallest unsigned type that holds every index of that codebook.

        The one place this package spells the type a feature index is stored in, so the
        activations and the feature descriptions cannot disagree about it.

        Parameters
        ----------
        codebook_size : int
            How many features the SAE has. The largest index is one less.

        Returns
        -------
        numpy.dtype
            The narrowest unsigned integer type holding ``codebook_size - 1``.

        Raises
        ------
        ValueError
            If ``codebook_size`` is less than one, which names no index at all.

        Examples
        --------
        >>> SaeActivation.index_dtype(16384)
        dtype('uint16')
        >>> SaeActivation.index_dtype(65537)
        dtype('uint32')
        """
        if codebook_size < 1:
            raise ValueError(
                f"codebook_size={codebook_size} names no feature, so no type holds an index "
                f"into it. A codebook has at least one entry."
            )
        return np.min_scalar_type(codebook_size - 1)

    def __len__(self) -> int:
        """Return ``L``, the residue count.

        Examples
        --------
        >>> import numpy as np
        >>> len(
        ...     SaeActivation(
        ...         np.zeros((7, 2), dtype=np.uint8),
        ...         np.zeros((7, 2), dtype=np.float16),
        ...         np.zeros(7, dtype=np.float32),
        ...         None,
        ...         "300m",
        ...         23,
        ...         "300m-layer23-k64-cb16384",
        ...         4,
        ...         2,
        ...     )
        ... )
        7
        """
        return self.indices.shape[0]

    def __repr__(self) -> str:
        """Return e.g. ``SaeActivation('P12345', 2 x 4 top 2, sae-slug on 300m layer 23)``.

        The dataclass default would print three whole arrays.

        Examples
        --------
        >>> import numpy as np
        >>> SaeActivation(
        ...     np.zeros((2, 2), dtype=np.uint8),
        ...     np.zeros((2, 2), dtype=np.float16),
        ...     np.zeros(2, dtype=np.float32),
        ...     "P12345",
        ...     "300m",
        ...     23,
        ...     "300m-layer23-k64-cb16384",
        ...     4,
        ...     2,
        ... )
        SaeActivation('P12345', 2 x 4 top 2, 300m-layer23-k64-cb16384 on 300m layer 23)
        """
        return (
            f"{type(self).__name__}({self.source!r}, {len(self)} x {self.codebook_size} "
            f"top {self.k}, {self.sae} on {self.parent} layer {self.layer})"
        )

    @property
    def shape(self) -> tuple[int, int]:
        """``(L, codebook_size)`` — what :meth:`dense` materialises, not what is stored.

        The stored pair is ``(L, k)`` and ``indices.shape`` says so; this is the shape of
        the thing the pair represents.

        Examples
        --------
        >>> import numpy as np
        >>> SaeActivation(
        ...     np.zeros((33, 2), dtype=np.uint16),
        ...     np.zeros((33, 2), dtype=np.float16),
        ...     np.zeros(33, dtype=np.float32),
        ...     None,
        ...     "6b",
        ...     60,
        ...     "6b-layer60-k64-cb16384",
        ...     16384,
        ...     2,
        ... ).shape
        (33, 16384)
        """
        return len(self), self.codebook_size

    def max(self) -> np.ndarray:
        """Return the per-sequence vector, ``(codebook_size,)``: each feature's strongest hit.

        The maximum and not a mean, because a feature that fires hard at one residue is
        present in the protein, and averaging over length would report it as absent in a long
        one. A feature that never fired is ``0``, which is its true magnitude: the encoder is
        a ReLU, so nothing is negative.

        Computed without materialising :meth:`dense`.

        Returns
        -------
        numpy.ndarray
            ``(codebook_size,)`` float32 — a reduction someone will do arithmetic on, so it
            comes back wider than the stored float16.

        Examples
        --------
        >>> import numpy as np
        >>> SaeActivation(
        ...     np.array([[2, 0], [1, 3]], dtype=np.uint8),
        ...     np.array([[0.5, 0.25], [1.0, 0.75]], dtype=np.float16),
        ...     np.zeros(2, dtype=np.float32),
        ...     None,
        ...     "300m",
        ...     23,
        ...     "300m-layer23-k64-cb16384",
        ...     4,
        ...     2,
        ... ).max().tolist()
        [0.25, 1.0, 0.5, 0.75]
        """
        pooled = np.zeros(self.codebook_size, dtype=np.float32)
        np.maximum.at(pooled, self.indices.ravel(), self.values.astype(np.float32).ravel())
        return pooled

    def dense(self, dtype: npt.DTypeLike = np.float32) -> np.ndarray:
        """Materialise the full ``(L, codebook_size)`` matrix, and hand it over unheld.

        Nothing caches what comes back: the dense form is orders of magnitude larger than
        the pair that generated it, so memory is spent when a caller asks and not before.

        Parameters
        ----------
        dtype : numpy.typing.DTypeLike, default numpy.float32
            What to materialise into. ``numpy.float16`` gives back exactly the stored
            values at half the memory.

        Returns
        -------
        numpy.ndarray
            ``(L, codebook_size)``, zero everywhere no slot named.

        Examples
        --------
        >>> import numpy as np
        >>> activation = SaeActivation(
        ...     np.array([[2, 0], [1, 3]], dtype=np.uint8),
        ...     np.array([[0.5, 0.25], [1.0, 0.75]], dtype=np.float16),
        ...     np.zeros(2, dtype=np.float32),
        ...     None,
        ...     "300m",
        ...     23,
        ...     "300m-layer23-k64-cb16384",
        ...     4,
        ...     2,
        ... )
        >>> activation.dense().tolist()
        [[0.25, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.75]]
        >>> activation.dense(np.float16).dtype
        dtype('float16')
        """
        materialised = np.zeros(self.shape, dtype=dtype)
        rows = np.arange(len(self), dtype=np.intp)[:, None]
        materialised[rows, self.indices] = self.values
        return materialised


def _checked(
    embedding: object, *, sae: str, parent: str, layers: tuple[int, ...], d_model: int
) -> Embedding:
    """Return the embedding this SAE can read, or name what the two disagree about.

    A function rather than a method, so every refusal is tested in the gate with no weights
    anywhere: each one reads a fact the two objects recorded, never a tensor.
    """
    if not isinstance(embedding, Embedding):
        raise TypeError(
            f"encode() takes an Embedding, not {type(embedding).__name__}. An array carries "
            f"neither the checkpoint nor the layer it came from, so nothing about it could "
            f"be checked against this SAE."
        )
    if embedding.checkpoint != parent:
        raise ValueError(
            f"{sae} was trained on {parent} and this embedding came from "
            f"{embedding.checkpoint}. The features of one backbone name nothing in another, "
            f"and the multiply would succeed anyway."
        )
    if embedding.layer not in layers:
        covered = ", ".join(str(layer) for layer in layers)
        raise ValueError(
            f"{sae} covers {parent} layer {covered} and this embedding is layer "
            f"{embedding.layer}. A wrong layer multiplies fine and reconstructs badly, which "
            f"is the commonest silent mistake here."
        )
    width = embedding.array.shape[1]
    if width != d_model:
        raise ValueError(
            f"{sae} takes {d_model}-wide rows and this embedding is {width} wide, though "
            f"both name {parent}. A table is stale; fix it rather than trusting either "
            f"number."
        )
    return embedding


def _require_statistics(idf: Any, maximum: Any, *, sae: str) -> None:
    """Refuse ``normalize=True`` where the checkpoint ships no statistics to normalise by.

    Read off the buffers and never off a table: they default to ones, so a checkpoint
    shipping none would scale every feature by one and report success. One that gains them
    later needs nothing changed here.
    """
    if bool((idf == 1).all()) and bool((maximum == 1).all()):
        raise ValueError(
            f"{sae} ships no per-feature statistics: its idf and max buffers are all ones, "
            f"so normalize=True would scale every feature by one and call that normalised. "
            f"Encode without it, and read `normalized` on what comes back."
        )


class SAE:
    """One sparse autoencoder's weights, resident, plus :meth:`encode` over them.

    Parameters
    ----------
    checkpoint : str
        A key of :data:`SAE_CHECKPOINTS`, naming the full variant. **There is no default.**
        ``ESMC()`` defaults to its smallest checkpoint, so an ``SAE()`` defaulting to the 6B
        one would raise on every naive first pairing; naming it removes the trap rather than
        documenting it. An unknown slug fails here, by name, before anything is downloaded.
    device : str, optional
        Where the weights go, e.g. ``"cuda"``, ``"cuda:1"``, ``"cpu"``. Omitted, it resolves
        to ``"cuda"`` when torch can see a GPU and ``"cpu"`` otherwise; either way the answer
        is on :attr:`device`.
    token : str, optional
        A Hugging Face token. Unnecessary for the published checkpoints, which are ungated.

    Attributes
    ----------
    checkpoint : str
        The slug that was asked for.
    hf_id : str
        The Hugging Face repository the slug names.
    parent : str
        The backbone slug this was trained on — a key of
        :data:`protein.embed.esm.esmc.CHECKPOINTS`, and the one fact the table adds.
    device : str
        Where the weights actually are, resolved at construction.
    layers : tuple of int
        Which of the parent's hidden states this checkpoint covers, from its own config.
    d_model : int
        The row width it takes, from the config and cross-checked against the parent's.
    codebook_size : int
        How many features it has.
    k : int
        How many of them it keeps per residue.

    Raises
    ------
    ValueError
        If ``checkpoint`` is not a key of :data:`SAE_CHECKPOINTS`, or if its config's width
        disagrees with the width recorded for the parent.

    Examples
    --------
    >>> from protein.embed import SAE
    >>> SAE("6b")
    Traceback (most recent call last):
        ...
    ValueError: unknown SAE checkpoint '6b'. Known slugs: 300m-layer23-k64-cb16384, 600m-layer27-k64-cb16384, 6b-layer60-k64-cb16384.
    >>> SAE("300m-layer23-k64-cb16384")                       # doctest: +SKIP
    SAE('300m-layer23-k64-cb16384', cuda)
    """

    def __init__(
        self, checkpoint: str, *, device: str | None = None, token: str | None = None
    ) -> None:
        if checkpoint not in SAE_CHECKPOINTS:
            known = ", ".join(SAE_CHECKPOINTS)
            raise ValueError(f"unknown SAE checkpoint {checkpoint!r}. Known slugs: {known}.")
        # Inside the body, never at module level: this is what keeps `import protein` cheap.
        import torch  # pyright: ignore[reportMissingImports]
        from esm.models.esmc.sae import (  # pyright: ignore[reportMissingImports]
            EsmcSaeModel,
        )

        self.checkpoint = checkpoint
        self.hf_id, self.parent = SAE_CHECKPOINTS[checkpoint]
        # `is None` and not falsiness: `device=""` is a mistake worth surfacing.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # A single-shard repository, so this loads the one layer it ships and `encode` finds
        # it already there.
        self._model: Any = EsmcSaeModel.from_pretrained(
            self.hf_id, device=self.device, token=token
        ).eval()
        config = self._model.config
        self.layers: tuple[int, ...] = tuple(int(layer) for layer in config.available_layers)
        self.d_model = int(config.d_model)
        self.codebook_size = int(config.codebook_dim)
        self.k = int(config.k)
        width = CHECKPOINTS[self.parent][1]
        if self.d_model != width:
            raise ValueError(
                f"{self.hf_id} takes {self.d_model}-wide rows, and CHECKPOINTS records "
                f"d_model {width} for {self.parent!r}, which SAE_CHECKPOINTS names as slug "
                f"{checkpoint!r}'s parent. One of the two is wrong; fix it rather than "
                f"trusting either number."
            )

    def encode(self, embedding: Embedding, *, normalize: bool = False) -> SaeActivation:
        """Turn one embedding into the features that fired at each of its residues.

        Parameters
        ----------
        embedding : Embedding
            What :meth:`protein.embed.ESMC.embed` returned. An array is refused: it carries
            neither the checkpoint nor the layer this has to check itself against.
        normalize : bool, default False
            Scale the magnitudes by the checkpoint's own per-feature statistics. Off by
            default, so what comes back does not depend on which slug was loaded, and it
            **raises** where the checkpoint ships none rather than scaling by one.

        Returns
        -------
        SaeActivation
            ``(L, k)`` indices and values with one reconstruction loss per residue, carrying
            this SAE, its parent, the layer and whether normalisation happened. ``L`` is the
            residue count: BOS and EOS never reach this path.

        Raises
        ------
        TypeError
            If ``embedding`` is not an :class:`~protein.embed.embedding.Embedding`.
        ValueError
            If it came from another backbone, from a layer this checkpoint does not cover,
            or at another width — or if ``normalize`` is asked of a checkpoint that ships no
            statistics.

        Examples
        --------
        >>> from protein import ESMC, Protein
        >>> from protein.embed import SAE
        >>> model, sae = ESMC(), SAE("300m-layer23-k64-cb16384")   # doctest: +SKIP
        >>> embedding = model.embed(Protein("MKTAY", id="P12345"), layer=23)  # doctest: +SKIP
        >>> sae.encode(embedding).shape                            # doctest: +SKIP
        (5, 16384)
        """
        import torch  # pyright: ignore[reportMissingImports]

        checked = _checked(
            embedding,
            sae=self.checkpoint,
            parent=self.parent,
            layers=self.layers,
            d_model=self.d_model,
        )
        # The standalone layer and not the container's forward: it applies no token mask and
        # no flattening reshape, so `(L, d_model)` goes in and `(L, codebook)` comes out.
        layer: Any = self._model.layers[str(checked.layer)]
        rows = torch.as_tensor(checked.array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = layer(rows)
            magnitudes = output.feature_magnitudes
            if normalize:
                _require_statistics(layer.idf, layer.max, sae=self.checkpoint)
                magnitudes = magnitudes / layer.max * layer.idf
            # Dense out, so the k slots are read back off it. Lossless: at most k entries per
            # row are non-zero, so top-k finds every one of them.
            values, indices = torch.topk(magnitudes, self.k, dim=-1)
            loss = output.reconstruction_loss
        return SaeActivation(
            indices.to("cpu").numpy().astype(SaeActivation.index_dtype(self.codebook_size)),
            values.to("cpu", torch.float16).numpy(),
            loss.to("cpu", torch.float32).numpy(),
            checked.source,
            self.parent,
            checked.layer,
            self.checkpoint,
            self.codebook_size,
            self.k,
            normalize,
        )

    def __repr__(self) -> str:
        """Return e.g. ``SAE('300m-layer23-k64-cb16384', cuda)``.

        The slug and where the weights ended up, as :class:`~protein.embed.esm.esmc.ESMC`
        reports them.
        """
        return f"{type(self).__name__}({self.checkpoint!r}, {self.device})"
