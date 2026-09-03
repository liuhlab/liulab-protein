"""The :class:`Embedding` value object — an array, and what it came from.

**numpy only.** Nothing here imports torch; :mod:`protein.embed.esm` is where the weights
live.

An embedding is a value object rather than a bare array because an array found in a notebook
an hour later cannot say which checkpoint or which layer it came from, and arrays from
different ones are not comparable. So it travels with the three facts that make it one: which
sequence, which checkpoint, which layer. :meth:`Embedding.__array__` keeps it numpy-native,
so ``np.asarray(e)`` is the array and every numpy function takes it directly.

The array is **per residue**, ``(L, d_model)``, with BOS and EOS already stripped, so its
first axis is the sequence length. :meth:`Embedding.mean` is the per-sequence vector.

Examples
--------
>>> import numpy as np
>>> from protein.embed import Embedding
>>> e = Embedding(np.zeros((33, 960), dtype=np.float32), "P12345", "300m", 30)
>>> e
Embedding('P12345', 33 x 960, 300m layer 30)
>>> e.shape, len(e)
((33, 960), 33)
>>> np.asarray(e) is e.array
True
>>> e.mean().shape
(960,)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["Embedding"]


@dataclass(frozen=True, slots=True)
class Embedding:
    """One sequence's per-residue embedding, and the three facts that identify it.

    Frozen and slotted: a measurement that can be edited in place is one nobody can trust.
    Built by :meth:`protein.embed.ESMC.embed` rather than by hand.

    Parameters
    ----------
    array : numpy.ndarray
        ``(L, d_model)``, float32, on the CPU, **BOS and EOS stripped** — so ``L`` is the
        residue count of the sequence that went in.
    source : str or None
        The UniProt accession or chain key it came from, or ``None`` when the ``Protein`` had
        no id.
    checkpoint : str
        The slug, e.g. ``"300m"`` — a key of :data:`protein.embed.esm.CHECKPOINTS`, never an
        HF id.
    layer : int
        Which hidden state, as a **non-negative** index into the model's ``hidden_states``:
        ``0`` is the embedding-layer output and ``n_layers`` is the last hidden state.
        :meth:`protein.embed.ESMC.embed` normalises a negative ``layer=`` before it gets here.

    Examples
    --------
    >>> import numpy as np
    >>> e = Embedding(np.zeros((5, 4), dtype=np.float32), None, "300m", 30)
    >>> e.source is None
    True
    >>> e.array.dtype
    dtype('float32')
    """

    array: np.ndarray
    source: str | None
    checkpoint: str
    layer: int

    def __array__(self, dtype: npt.DTypeLike | None = None, copy: bool | None = None) -> np.ndarray:
        """Return the array itself, so ``np.asarray(embedding)`` needs no ``.array``.

        Parameters
        ----------
        dtype : numpy.typing.DTypeLike, optional
            What numpy asked to be given. Omitted, the stored float32 comes back unconverted.
        copy : bool, optional
            numpy 2's copy protocol.

        Returns
        -------
        numpy.ndarray
            :attr:`array`, converted only if ``dtype`` or ``copy`` asked for it.

        Examples
        --------
        >>> import numpy as np
        >>> e = Embedding(np.ones((2, 3), dtype=np.float32), None, "300m", 30)
        >>> np.asarray(e).shape
        (2, 3)
        >>> np.asarray(e, dtype=np.float64).dtype
        dtype('float64')
        """
        return np.array(self.array, dtype=dtype, copy=copy)

    def __len__(self) -> int:
        """Return ``L``, the residue count — **not** ``L + 2``.

        Examples
        --------
        >>> import numpy as np
        >>> len(Embedding(np.zeros((33, 960), dtype=np.float32), None, "300m", 30))
        33
        """
        return self.array.shape[0]

    def __repr__(self) -> str:
        """Return e.g. ``Embedding('P12345', 33 x 960, 300m layer 30)``.

        The dataclass default would print the whole array.

        Examples
        --------
        >>> import numpy as np
        >>> Embedding(np.zeros((5, 4), dtype=np.float32), "P12345", "300m", 12)
        Embedding('P12345', 5 x 4, 300m layer 12)
        """
        rows, columns = self.array.shape[0], self.array.shape[1]
        return (
            f"{type(self).__name__}({self.source!r}, {rows} x {columns}, "
            f"{self.checkpoint} layer {self.layer})"
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """``(L, d_model)`` — the array's shape, reached without unwrapping it.

        Examples
        --------
        >>> import numpy as np
        >>> Embedding(np.zeros((33, 960), dtype=np.float32), None, "300m", 30).shape
        (33, 960)
        """
        return self.array.shape

    def mean(self) -> np.ndarray:
        """Return the per-sequence vector, ``(d_model,)``: the mean over residues.

        No mask, because a padded position only exists in a batch and every row here is a
        residue, so the plain mean is exact.

        Returns
        -------
        numpy.ndarray
            ``(d_model,)``, float32.

        Examples
        --------
        >>> import numpy as np
        >>> e = Embedding(np.arange(6, dtype=np.float32).reshape(3, 2), None, "300m", 30)
        >>> e.mean()
        array([2., 3.], dtype=float32)
        """
        return self.array.mean(axis=0)

    def as_json(self) -> dict[str, Any]:
        """Return the provenance as JSON-ready values — everything but the numbers.

        What ``protein esm embed --json`` prints. The array is not in it; ``--out`` and a
        ``.npy`` file are for that.

        Returns
        -------
        dict
            ``source``, ``checkpoint``, ``layer`` and ``shape``.

        Examples
        --------
        >>> import numpy as np
        >>> Embedding(np.zeros((33, 960), dtype=np.float32), "P12345", "300m", 30).as_json()
        {'source': 'P12345', 'checkpoint': '300m', 'layer': 30, 'shape': [33, 960]}
        """
        return {
            "source": self.source,
            "checkpoint": self.checkpoint,
            "layer": self.layer,
            "shape": [int(size) for size in self.array.shape],
        }
