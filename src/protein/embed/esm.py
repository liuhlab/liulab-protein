"""ESM-C, as an object you construct and keep — the weights and the one call over them.

**This module's body imports torch nowhere.** Only method bodies do, so ``import protein``
never pulls torch; a test asserts it.

:class:`ESMC` is a class rather than a ``Protein.embed()`` method because ESM-C's weights are
resident across calls and a method has nowhere honest to put them — *resident state gets an
object; a subprocess does not*. Construction is therefore **eager**: holding the object
*means* the weights are loaded, which is why every docstring example that constructs one
carries ``# doctest: +SKIP``.

**Our** ``ESMC`` wraps ``esm.models.esmc.EsmcForMaskedLM``, not ``esm``'s own class of the
same name.

Examples
--------
>>> from protein import ESMC, Protein
>>> from protein.embed.esm import CHECKPOINTS
>>> CHECKPOINTS["300m"]
('biohub/ESMC-300M', 960)
>>> model = ESMC()                                       # doctest: +SKIP
>>> model.device, model.d_model, model.n_layers          # doctest: +SKIP
('cuda', 960, 30)
>>> model.embed(Protein("MKTAY", id="P12345"))           # doctest: +SKIP
Embedding('P12345', 5 x 960, 300m layer 30)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from biotite.sequence import ProteinSequence

from protein.embed.embedding import Embedding

if TYPE_CHECKING:
    from biotite.sequence import NucleotideSequence

__all__ = ["CHECKPOINTS", "ESMC", "Embeddable"]

#: Every ESM-C checkpoint this package knows, slug to ``(hf_id, d_model)``. :class:`ESMC`
#: takes the **slug**, so an unknown one fails by name; an arbitrary HF id is a string that
#: fails somewhere inside ``from_pretrained``, long after the download started.
CHECKPOINTS: dict[str, tuple[str, int]] = {
    "300m": ("biohub/ESMC-300M", 960),
    "600m": ("biohub/ESMC-600M", 1152),
    "6b": ("biohub/ESMC-6B", 2560),
}


def _layer_index(layer: int, n_layers: int, checkpoint: str) -> int:
    """Normalise ``layer`` onto ``hidden_states``, or say what the range was.

    ``hidden_states`` has ``n_layers + 1`` entries: element 0 is the embedding-layer output
    and element ``n_layers`` is the last hidden state. A function rather than a method, so
    the arithmetic is tested in the gate with no weights anywhere.
    """
    count = n_layers + 1
    index = layer + count if layer < 0 else layer
    if not 0 <= index < count:
        raise ValueError(
            f"layer={layer} is outside {checkpoint}'s range. It has {n_layers} transformer "
            f"layers, so layer runs 0 (the embedding-layer output) to {n_layers} (the last "
            f"hidden state), or -{count} to -1 counting back."
        )
    return index


@runtime_checkable
class Embeddable(Protocol):
    """What :meth:`ESMC.embed` needs from the thing it is handed.

    ``sequence`` is what the tokenizer reads. ``id`` is what lands in
    :attr:`protein.embed.Embedding.source`, and it is why a bare ``str`` or
    :class:`~biotite.sequence.ProteinSequence` is refused: neither could say afterwards what
    was embedded.

    ``sequence`` is typed as either of biotite's two because a ``Chain`` answers with either.
    :meth:`ESMC.embed` refuses the nucleotide one.
    """

    @property
    def sequence(self) -> ProteinSequence | NucleotideSequence:
        """The residues. ``str(...)`` of it is what reaches the tokenizer."""
        ...

    @property
    def id(self) -> str | None:
        """The UniProt accession or chain key, recorded as the embedding's ``source``."""
        ...


def _residues(item: Embeddable) -> str:
    """Return what the tokenizer reads, or refuse a sequence that is not protein.

    A function rather than a method, so the refusal is tested in the gate with no weights
    anywhere. A ``Chain`` whose ``kind`` is ``"nucleic"`` answers with a nucleotide sequence,
    and ``A``, ``C``, ``G``, ``T`` and ``N`` are all protein letters too, so an unguarded
    tokenizer would return a confident embedding of the wrong molecule.
    """
    sequence = item.sequence
    if not isinstance(sequence, ProteinSequence):
        raise TypeError(
            f"{item.id} carries a {type(sequence).__name__}, and ESM-C is a protein model. "
            f"A chain says which it is in `.kind` before you ask for its sequence."
        )
    return str(sequence)


class ESMC:
    """ESM-C's weights, resident, plus :meth:`embed` over them.

    Parameters
    ----------
    checkpoint : str, default "300m"
        A key of :data:`CHECKPOINTS`. Not an HF id and not a loaded model — an unknown slug
        fails here, by name, before anything is downloaded.
    device : str, optional
        Where the weights go, e.g. ``"cuda"``, ``"cuda:1"``, ``"cpu"``. Omitted, it resolves
        to ``"cuda"`` when torch can see a GPU and ``"cpu"`` otherwise; either way the answer
        is on :attr:`device`, so a run that quietly fell back to the CPU says so.
    token : str, optional
        A Hugging Face token. Unnecessary for the published checkpoints, which are ungated.

    Attributes
    ----------
    checkpoint : str
        The slug that was asked for.
    hf_id : str
        The Hugging Face repository the slug names.
    device : str
        Where the weights actually are, resolved at construction.
    d_model : int
        The embedding width, from :data:`CHECKPOINTS` and cross-checked against the loaded
        config, because a stale table is worse than no table.
    n_layers : int
        Transformer layers, from the loaded config. ``hidden_states`` is one longer than
        this: element 0 is the embedding-layer output.

    Raises
    ------
    ValueError
        If ``checkpoint`` is not a key of :data:`CHECKPOINTS`, or if the checkpoint's
        config disagrees with the width recorded there.

    Examples
    --------
    >>> from protein.embed import ESMC
    >>> ESMC("nonesuch")
    Traceback (most recent call last):
        ...
    ValueError: unknown checkpoint 'nonesuch'. Known slugs: 300m, 600m, 6b.
    >>> ESMC()                                            # doctest: +SKIP
    ESMC('300m', cuda)
    """

    def __init__(
        self, checkpoint: str = "300m", *, device: str | None = None, token: str | None = None
    ) -> None:
        if checkpoint not in CHECKPOINTS:
            known = ", ".join(CHECKPOINTS)
            raise ValueError(f"unknown checkpoint {checkpoint!r}. Known slugs: {known}.")
        # Inside the body, never at module level: this is what keeps `import protein` cheap.
        import torch  # pyright: ignore[reportMissingImports]
        from esm.models.esmc import (  # pyright: ignore[reportMissingImports]
            EsmcForMaskedLM,
            EsmcTokenizer,
        )

        self.checkpoint = checkpoint
        self.hf_id, self.d_model = CHECKPOINTS[checkpoint]
        # `is None` and not falsiness: `device=""` is a mistake worth surfacing.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # `from_pretrained` and not the bare `EsmcTokenizer()`, which uses a vocabulary
        # compiled into the library: a checkpoint shipping a different one would be
        # mis-tokenised in silence.
        self._tokenizer: Any = EsmcTokenizer.from_pretrained(self.hf_id, token=token)
        self._model: Any = EsmcForMaskedLM.from_pretrained(
            self.hf_id, device=self.device, token=token
        ).eval()
        config = self._model.config
        # `hidden_size`, never `d_model`: the published configs predate the field rename.
        if config.hidden_size != self.d_model:
            raise ValueError(
                f"{self.hf_id} reports hidden_size {config.hidden_size}, but CHECKPOINTS "
                f"records d_model {self.d_model} for slug {checkpoint!r}. The table is "
                f"stale; fix it rather than trusting either number."
            )
        self.n_layers: int = int(config.num_hidden_layers)

    def embed(self, item: Embeddable, *, layer: int = -1) -> Embedding:
        """Embed one protein or one chain, per residue.

        One at a time; batching is out of scope for v1.

        Parameters
        ----------
        item : Embeddable
            A :class:`protein.core.Protein` or a ``Chain``. A ``str`` or a bare
            :class:`~biotite.sequence.ProteinSequence` is refused: see :class:`Embeddable`.
        layer : int, default -1
            Which hidden state to return, indexed the way the model indexes
            ``hidden_states``: ``0`` is the embedding-layer output, ``n_layers`` is the last
            hidden state, and negatives count back from the end.

        Returns
        -------
        Embedding
            ``(L, d_model)``, float32, on the CPU, **BOS and EOS stripped** — so
            ``len(embedding) == len(item)``. Its ``layer`` is the normalised non-negative
            index, and its ``source`` is ``item.id``.

        Raises
        ------
        TypeError
            If ``item`` is not a :class:`Embeddable`, or if its ``sequence`` is a
            nucleotide sequence rather than a protein one.
        ValueError
            If ``layer`` is outside ``-(n_layers + 1) .. n_layers``.

        Examples
        --------
        >>> from protein import ESMC, Protein
        >>> model = ESMC()                                        # doctest: +SKIP
        >>> p = Protein("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", id="test")
        >>> model.embed(p).shape                                  # doctest: +SKIP
        (33, 960)
        >>> model.embed(p, layer=0).layer                         # doctest: +SKIP
        0
        """
        import torch  # pyright: ignore[reportMissingImports]

        if not isinstance(item, Embeddable):
            raise TypeError(
                f"embed() takes a Protein or a Chain, not {type(item).__name__}. A string "
                f"and a ProteinSequence carry no identity to record as the embedding's "
                f"source; wrap one in Protein(sequence, id=...) first."
            )
        index = _layer_index(layer, self.n_layers, self.checkpoint)
        batch = self._tokenizer([_residues(item)], return_tensors="pt")
        batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
        # `last_hidden_state` is `hidden_states[-1]`, so the default layer need not ask the
        # model to materialise every one.
        every_layer = index != self.n_layers
        with torch.no_grad():
            output = self._model(**batch, output_hidden_states=every_layer)
        tensor = output.hidden_states[index] if every_layer else output.last_hidden_state
        # `[0]` is the one sequence; `1:-1` drops BOS and EOS.
        array = tensor[0, 1:-1, :].detach().to("cpu", torch.float32).numpy()
        return Embedding(array=array, source=item.id, checkpoint=self.checkpoint, layer=index)

    def __repr__(self) -> str:
        """Return e.g. ``ESMC('300m', cuda)`` — the slug and where the weights ended up."""
        return f"{type(self).__name__}({self.checkpoint!r}, {self.device})"
