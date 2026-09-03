"""ESM-C, as an object you construct and keep — the weights and the one call over them.

**This module's body imports torch nowhere.** Only method bodies do, so ``from protein
import ESMC`` costs nothing and ``import protein`` never pulls a gigabyte of CUDA runtime
into a process that only wanted to read a FASTA. A test asserts it.

:class:`ESMC` is a class rather than a ``Protein.embed()`` method because ESM-C holds
1.33 GB of weights across calls and a method has nowhere honest to put them: it either
reloads per call or hides a module-level cache whose lifetime no caller can see. *Resident
state gets an object; a subprocess does not* — which is also why
:meth:`protein.search.SearchMixin.search` stays a method.

Construction is therefore **eager**: ``ESMC()`` imports torch, resolves the device and loads
the weights. That departs from `liulab-genome`'s ``Aligner``, which runs nothing at
construction, and it departs deliberately — the whole point here is that holding the object
*means* the weights are resident, and dropping it is how they stop being. Every docstring
example that constructs one carries ``# doctest: +SKIP``, and that marker covers only its
own line.

Note the name collision, which is deliberate and worth reading twice: **our** ``ESMC``
wraps ``esm.models.esmc.EsmcForMaskedLM``. It does **not** build on ``esm``'s own class of
the same name — that one is the deprecation shim in ``esm/models/esmc/compatibility.py``,
which warns and re-lays-out ``hidden_states``.

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

from protein.embed.embedding import Embedding

if TYPE_CHECKING:
    from biotite.sequence import ProteinSequence

__all__ = ["CHECKPOINTS", "ESMC", "Embeddable"]

#: Every ESM-C checkpoint this package knows, slug to ``(hf_id, d_model)``. A **slug** is
#: what :class:`ESMC` takes: it validates, it names itself in an error, and it carries
#: ``d_model`` here where reading it costs nothing — an arbitrary HF id is a string that
#: fails somewhere inside ``from_pretrained``, several gigabytes later.
#:
#: All three are MIT and ungated (the HF API reports ``gated: false`` on every one), so no
#: token is needed; :class:`ESMC` takes one anyway for whoever mirrors them privately.
CHECKPOINTS: dict[str, tuple[str, int]] = {
    "300m": ("biohub/ESMC-300M", 960),
    "600m": ("biohub/ESMC-600M", 1152),
    "6b": ("biohub/ESMC-6B", 2560),
}


def _layer_index(layer: int, n_layers: int, checkpoint: str) -> int:
    """Normalise ``layer`` onto ``hidden_states``, or say what the range was.

    ``hidden_states`` has ``n_layers + 1`` entries: element 0 is the embedding-layer output
    and element ``n_layers`` is the last hidden state. **Verified on ``biohub/ESMC-300M``**,
    where ``len(hidden_states)`` is 31 for 30 layers, ``hidden_states[0]`` equals
    ``esmc.embed(input_ids)`` and ``hidden_states[-1]`` equals ``last_hidden_state`` — both
    bit-identical under ``torch.equal``, which is what lets ``layer=-1`` read
    ``last_hidden_state`` and still be the same tensor ``layer=30`` names.

    A function rather than a method so the arithmetic — the one piece of :class:`ESMC` that
    is easy to get wrong — is tested in the gate, with no weights anywhere.
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

    Two members, and both are load-bearing. ``sequence`` is what the tokenizer reads.
    ``id`` is what lands in :attr:`protein.embed.Embedding.source`, and it is why a bare
    ``str`` or a bare :class:`~biotite.sequence.ProteinSequence` is refused: they carry no
    identity, so an embedding made from one could not say afterwards what it embedded.

    :class:`protein.core.Protein` satisfies this, and ``Chain`` is written to. Both are
    read-only here, so an implementation is free to make either a plain attribute or a
    property.
    """

    @property
    def sequence(self) -> ProteinSequence:
        """The residues. ``str(...)`` of it is what reaches the tokenizer."""
        ...

    @property
    def id(self) -> str | None:
        """The UniProt accession or chain key, recorded as the embedding's ``source``."""
        ...


class ESMC:
    """ESM-C's weights, resident, plus :meth:`embed` over them.

    Parameters
    ----------
    checkpoint : str, default "300m"
        A key of :data:`CHECKPOINTS`. Not an HF id and not a loaded model — an unknown slug
        fails here, by name, before anything is downloaded.
    device : str, optional
        Where the weights go, e.g. ``"cuda"``, ``"cuda:1"``, ``"cpu"``. Omitted, it resolves
        to ``"cuda"`` when torch can see a GPU and ``"cpu"`` otherwise, and either way the
        answer is on :attr:`device` — a run that quietly fell back to the CPU says so.
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
        The embedding width, from :data:`CHECKPOINTS` — so ``CHECKPOINTS["600m"][1]`` is a
        fact this package holds without downloading 2.3 GB to learn it. Cross-checked
        against the loaded config, because a stale table is worse than no table.
    n_layers : int
        Transformer layers, from the loaded config. ``hidden_states`` is one longer than
        this: element 0 is the embedding-layer output. This is what lets a bad ``layer=``
        name the range it fell outside.

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
        # Inside the body, never at module level: this import is the whole reason the rest
        # of the package can be imported on a laptop with no CUDA runtime installed.
        import torch  # pyright: ignore[reportMissingImports]
        from esm.models.esmc import (  # pyright: ignore[reportMissingImports]
            EsmcForMaskedLM,
            EsmcTokenizer,
        )

        self.checkpoint = checkpoint
        self.hf_id, self.d_model = CHECKPOINTS[checkpoint]
        # `is None` and not falsiness: `device=""` is a mistake worth surfacing, not a
        # request for the default. The resolved answer stays on `.device`, so a run that
        # quietly landed on the CPU says which one it was.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # `from_pretrained` and not the bare `EsmcTokenizer()`: the two give byte-identical
        # ids for ESMC-300M (measured), but the bare one uses a vocabulary compiled into the
        # library, so a checkpoint that ever ships a different one would be mis-tokenised in
        # silence. `tokenizer_config.json` naming the pre-rename class logs a warning here;
        # it is a `logger.warning` and not a `warnings.warn`, so the gate never sees it.
        self._tokenizer: Any = EsmcTokenizer.from_pretrained(self.hf_id, token=token)
        self._model: Any = EsmcForMaskedLM.from_pretrained(
            self.hf_id, device=self.device, token=token
        ).eval()
        config = self._model.config
        # `hidden_size`, never `d_model`: this checkpoint's config.json predates the field
        # rename, `hasattr(config, "d_model")` is False, and reading it raises a
        # FutureWarning naming both spellings.
        if config.hidden_size != self.d_model:
            raise ValueError(
                f"{self.hf_id} reports hidden_size {config.hidden_size}, but CHECKPOINTS "
                f"records d_model {self.d_model} for slug {checkpoint!r}. The table is "
                f"stale; fix it rather than trusting either number."
            )
        self.n_layers: int = int(config.num_hidden_layers)

    def embed(self, item: Embeddable, *, layer: int = -1) -> Embedding:
        """Embed one protein or one chain, per residue.

        One at a time. Batching is out of scope for v1, and accepting a list later breaks no
        call that exists.

        Parameters
        ----------
        item : Embeddable
            A :class:`protein.core.Protein` or a ``Chain`` — the two things in this package
            that hold a protein sequence *and* know what they are. A ``str`` or a bare
            :class:`~biotite.sequence.ProteinSequence` is refused: see :class:`Embeddable`.
        layer : int, default -1
            Which hidden state to return, indexed the way the model indexes
            ``hidden_states``: ``0`` is the embedding-layer output, ``n_layers`` is the last
            hidden state, and negatives count back from the end. The default ``-1`` reads
            ``last_hidden_state`` directly, so the common case never asks the model to
            materialise all 31 tensors; any other layer re-runs with
            ``output_hidden_states=True``.

        Returns
        -------
        Embedding
            ``(L, d_model)``, float32, on the CPU, **BOS and EOS stripped** — so
            ``len(embedding) == len(item)`` and not ``len(item) + 2``. Its ``layer`` is the
            normalised non-negative index, and its ``source`` is ``item.id``.

        Raises
        ------
        TypeError
            If ``item`` is not a :class:`Embeddable`.
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
        batch = self._tokenizer([str(item.sequence)], return_tensors="pt")
        batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
        # The last hidden state is `hidden_states[-1]` exactly — measured, bit-identical —
        # so asking for every layer to reach it would cost 31 tensors and buy nothing.
        every_layer = index != self.n_layers
        with torch.no_grad():
            output = self._model(**batch, output_hidden_states=every_layer)
        tensor = output.hidden_states[index] if every_layer else output.last_hidden_state
        # `[0]` is the one sequence; `1:-1` drops BOS and EOS. This slice is the whole point
        # of the wrapper: the model returns (1, L + 2, d_model) and callers want L rows.
        array = tensor[0, 1:-1, :].detach().to("cpu", torch.float32).numpy()
        return Embedding(array=array, source=item.id, checkpoint=self.checkpoint, layer=index)

    def __repr__(self) -> str:
        """Return e.g. ``ESMC('300m', cuda)`` — the slug and where the weights ended up."""
        return f"{type(self).__name__}({self.checkpoint!r}, {self.device})"
