"""The :class:`SaeActivation` value object — what one sparse-autoencoder call gives back.

**numpy only.** Nothing here imports torch or scipy: a frozen value object holds no weights
and loads none, so everything it promises is checkable without a GPU.

A **peer** of :class:`~protein.embed.embedding.Embedding` and not one of them. An embedding is
dense and ``d_model`` wide; an activation is sparse over a codebook far wider than that, so
``d_model`` would be a lie here and ``.mean()`` would dilute a feature's presence by protein
length. That is ADR-0007.

The pair of ``(L, k)`` arrays is **lossless**: the top-k selection returns exactly ``k`` slots
per row whether or not all ``k`` are non-zero, so the sparse form is a representation rather
than an approximation. :meth:`SaeActivation.dense` materialises the full matrix on request and
nothing here holds it.

Examples
--------
>>> import numpy as np
>>> from protein.embed import SaeActivation
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

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["SaeActivation"]


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
