"""ESMFold2, as an object you construct and keep — the weights and the one call over them.

**This module's body imports torch nowhere.** Only method bodies do, so ``import protein``
never pulls torch; a test asserts it.

:class:`ESMFold2` is a class rather than a ``Protein.fold()`` method because the weights are
resident across calls — *resident state gets an object; a subprocess does not*. Construction
is therefore **eager**: holding the object *means* the weights are loaded, which is why every
docstring example that constructs one carries ``# doctest: +SKIP``.

**What comes back is a** :class:`~protein.structure.Structure`, written into a directory the
caller named. A prediction is a structure like any other; what it has that a deposited entry
does not is provenance — the accessions it was folded from, and what the model said about its
own answer (ADR-0005, ADR-0006).

**Nothing here manages the card.** No length ceiling and no memory arithmetic: the limit
belongs to the hardware, and a fold that does not fit raises whatever CUDA raises. The one
guard is :func:`warn_about_esmc`, because ``load_esmc=False`` returns an mmCIF of the right
length holding a wrong structure and says nothing.

**The whole upstream schema is reachable.** Load-time arguments are named here and forwarded;
:meth:`ESMFold2.fold` forwards every other keyword to upstream's own ``fold``; and
:attr:`ESMFold2.model` is the loaded model, for anything neither covers.

**What goes in is plain Python.** :meth:`ESMFold2.fold` takes what
:class:`~protein.fold.FoldingRequest` takes and builds the request itself, so a batch is
data and a call site imports no class.

Examples
--------
>>> from protein.fold import ESMFold2
>>> from protein.fold.esmfold import CHECKPOINTS
>>> CHECKPOINTS["ESMFold2-Fast"]
'biohub/ESMFold2-Fast'
>>> model = ESMFold2()                                    # doctest: +SKIP
>>> complexed = [
...     {"kind": "protein", "sequence": "MKTAY", "accession": "P12345"},
...     {"kind": "dna", "sequence": "ACGT"},
... ]
>>> model.fold(complexed, "/scratch/folds")               # doctest: +SKIP
Structure('P12345')
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from protein.fold.predictions import (
    Confidence,
    pairwise_path,
    prediction_name,
    prediction_path,
    stored_prediction,
)
from protein.fold.request import FoldingRequest
from protein.structure import Structure

if TYPE_CHECKING:
    from protein.fold.request import RequestSpec
    from protein.msa import MSA

__all__ = [
    "CHECKPOINTS",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_KERNEL_BACKEND",
    "ESMFold2",
    "warn_about_esmc",
]

#: Every ESMFold2 checkpoint this package knows, slug to Hugging Face repository.
#: :class:`ESMFold2` takes the **slug**, so an unknown one fails by name; an arbitrary
#: repository is a string that fails inside ``from_pretrained``, long after the download
#: started.
CHECKPOINTS: dict[str, str] = {
    "ESMFold2-Fast": "biohub/ESMFold2-Fast",
    "ESMFold2": "biohub/ESMFold2",
}

#: What is loaded when the caller names no checkpoint: the fast one, which needs no
#: alignment and does not refuse one.
DEFAULT_CHECKPOINT = "ESMFold2-Fast"

#: Which kernels the trunk runs. The fused Triton path, which the ``esm`` environment is
#: built to compile and which falls back to the reference path on its own where it cannot.
DEFAULT_KERNEL_BACKEND = "fused"


def warn_about_esmc(load_esmc: bool) -> None:
    """Warn that ``load_esmc=False`` returns a wrong structure rather than a smaller one.

    The lane's one guard, and not a resource rule: without the language model the trunk
    still writes an mmCIF of the right length, holding coordinates that are wrong, and says
    nothing. A loud crash needs no guard; a silent wrong answer does. A function rather than
    a method, so the warning is tested in the gate with no weights anywhere.

    Parameters
    ----------
    load_esmc : bool
        Whether the language model is to be loaded.

    Warns
    -----
    UserWarning
        When it is not.

    Examples
    --------
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error")
    ...     warn_about_esmc(True)
    """
    if load_esmc:
        return
    warnings.warn(
        "load_esmc=False is not a memory option. The trunk folds without the language model "
        "it was trained against and writes an mmCIF of the right length holding a wrong "
        "structure, with no error and no mark in the file. Read the confidence before you "
        "trust one.",
        stacklevel=3,
    )


class ESMFold2:
    """ESMFold2's weights, resident, plus :meth:`fold` over them.

    Parameters
    ----------
    checkpoint : str, default "ESMFold2-Fast"
        A key of :data:`CHECKPOINTS`. Not a repository and not a loaded model — an unknown
        slug fails here, by name, before anything is downloaded.
    device : str, optional
        Where the weights go, e.g. ``"cuda"``, ``"cuda:1"``, ``"cpu"``. Omitted, it resolves
        to ``"cuda"`` when torch can see a GPU and ``"cpu"`` otherwise; either way the answer
        is on :attr:`device`, so a run that quietly fell back to the CPU says so.
    dtype : torch.dtype, optional
        What the trunk is cast to. Omitted, the checkpoint's own.
    load_esmc : bool, default True
        Load the language model the trunk conditions on. **Setting it ``False`` warns** —
        see :func:`warn_about_esmc`.
    esmc_precision : {"bf16", "fp32", "fp8"}, default "bf16"
        What the language model is cast to. ``"fp8"`` needs Transformer Engine.
    ccd_cache : str or pathlib.Path, optional
        Where the Chemical Component Dictionary is kept. Omitted, the Hugging Face cache.
    kernel_backend : str or None, default "fused"
        Which kernels the trunk runs: ``"fused"``, ``"cuequivariance"``, or ``None`` for the
        reference path. The fused kernels fall back to the reference path themselves where
        they cannot be compiled.
    **kwargs : Any
        Forwarded to upstream's ``from_pretrained`` — ``token``, ``revision``, ``cache_dir``,
        ``local_files_only``, ``force_download``, ``config``.

    Attributes
    ----------
    checkpoint : str
        The slug that was asked for.
    hf_id : str
        The Hugging Face repository the slug names.
    device : str
        Where the weights actually are, resolved at construction.
    model : Any
        The loaded ``EsmFold2Model``, for everything the arguments above do not name —
        ``set_chunk_size``, ``apply_torch_compile``, and the rest of upstream's surface.
    builder : Any
        The loaded ``ESMFold2InputBuilder``, which holds the parsed dictionary.

    Raises
    ------
    ValueError
        If ``checkpoint`` is not a key of :data:`CHECKPOINTS`.

    Warns
    -----
    UserWarning
        If ``load_esmc`` is ``False``.

    Examples
    --------
    >>> from protein.fold import ESMFold2
    >>> ESMFold2("esmfold2")
    Traceback (most recent call last):
        ...
    ValueError: unknown checkpoint 'esmfold2'. Known slugs: ESMFold2-Fast, ESMFold2.
    >>> ESMFold2()                                        # doctest: +SKIP
    ESMFold2('ESMFold2-Fast', cuda)
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: str | None = None,
        dtype: Any = None,
        load_esmc: bool = True,
        esmc_precision: str = "bf16",
        ccd_cache: str | Path | None = None,
        kernel_backend: str | None = DEFAULT_KERNEL_BACKEND,
        **kwargs: Any,
    ) -> None:
        if checkpoint not in CHECKPOINTS:
            known = ", ".join(CHECKPOINTS)
            raise ValueError(f"unknown checkpoint {checkpoint!r}. Known slugs: {known}.")
        warn_about_esmc(load_esmc)
        # Inside the body, never at module level: this is what keeps `import protein` cheap.
        import torch  # pyright: ignore[reportMissingImports]
        from esm.models.esmfold2 import (  # pyright: ignore[reportMissingImports]
            ESMFold2InputBuilder,
            EsmFold2Model,
        )

        self.checkpoint = checkpoint
        self.hf_id = CHECKPOINTS[checkpoint]
        # `is None` and not falsiness: `device=""` is a mistake worth surfacing.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model: Any = EsmFold2Model.from_pretrained(
            self.hf_id,
            load_esmc=load_esmc,
            esmc_precision=esmc_precision,
            device=self.device,
            dtype=dtype,
            **kwargs,
        ).eval()
        self.model.set_kernel_backend(kernel_backend)
        cache = None if ccd_cache is None else Path(ccd_cache)
        self.builder: Any = ESMFold2InputBuilder(ccd_cache=cache)

    def fold(
        self,
        request: RequestSpec,
        out: str | Path,
        *,
        name: str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Structure:
        """Fold ``request`` into the directory ``out``, and return what was written.

        Parameters
        ----------
        request : protein.fold.request.RequestSpec
            What to fold: a :class:`~protein.fold.FoldingRequest`, or anything it takes.
            The request is built here, so its checks are made here too.
        out : str or pathlib.Path
            The output directory. **Required and defaulting nowhere** — the lab **Data dir**
            holds reference and input data, never a user's outputs. It is created if it is
            not there.
        name : str, optional
            What to call the prediction. Omitted, it is derived — see
            :func:`protein.fold.predictions.prediction_name`.
        overwrite : bool, default False
            Fold and write over whatever is already under that name. Without it, a held name
            carrying this request's sequences comes back unfolded and one carrying anything
            else raises.
        **kwargs : Any
            Forwarded to upstream's ``fold`` — ``num_loops``, ``num_sampling_steps``,
            ``num_diffusion_samples``, ``seed``, ``noise_scale``, ``step_scale``,
            ``max_inference_sigma``, ``lm_mask_pct``, ``lm_dropout``, ``msa_max_depth``,
            ``msa_column_mask_rate``.

        Returns
        -------
        Structure
            The written prediction, carrying the accessions it was folded from and its
            :class:`~protein.fold.predictions.Confidence`. Per-residue confidence is the
            B-factor column of the file itself. Asking for several diffusion samples writes
            the one with the highest mean confidence. **A prediction already on disk comes
            back without a** ``Confidence`` — the scalars do not survive the file.

        Raises
        ------
        FileExistsError
            If ``out`` holds a prediction under this name whose residues are not this
            request's, and ``overwrite`` is not set.
        ValueError or TypeError
            If ``request`` is not a request and does not describe one — raised while it is
            built, so before the card is touched. See :class:`~protein.fold.FoldingRequest`.

        Examples
        --------
        >>> model.fold(request, "/scratch/folds", name="the mutant")   # doctest: +SKIP
        Structure('the mutant')
        """
        built = FoldingRequest(request)
        directory = Path(out)
        directory.mkdir(parents=True, exist_ok=True)
        called = prediction_name(built, name)
        path = prediction_path(directory, called)
        held = stored_prediction(path, built, overwrite=overwrite)
        if held is not None:
            return held

        answer = self.builder.fold(self.model, self._upstream(built), **kwargs)
        best = max(answer, key=_mean_plddt) if isinstance(answer, list) else answer
        path.write_text(best.complex.to_mmcif(), encoding="utf-8")
        return Structure(
            called,
            path=path,
            accessions=built.accessions,
            confidence=_confidence(best, directory, called),
        )

    def _upstream(self, request: FoldingRequest) -> Any:
        """Return the ``StructurePredictionInput`` this request is, one entry per chain."""
        from esm.models.esmfold2.types import (  # pyright: ignore[reportMissingImports]
            DNAInput,
            ProteinInput,
            RNAInput,
            StructurePredictionInput,
        )

        sequences: list[Any] = []
        for label, chain in zip(request.chain_ids, request.chains, strict=True):
            if chain.kind == "protein":
                sequences.append(
                    ProteinInput(id=label, sequence=chain.residues, msa=_alignment(chain.alignment))
                )
            elif chain.kind == "dna":
                sequences.append(DNAInput(id=label, sequence=chain.residues))
            else:
                sequences.append(RNAInput(id=label, sequence=chain.residues))
        return StructurePredictionInput(sequences=sequences)

    def __repr__(self) -> str:
        """Return e.g. ``ESMFold2('ESMFold2-Fast', cuda)`` — the slug and where it ended up."""
        return f"{type(self).__name__}({self.checkpoint!r}, {self.device})"


def _alignment(alignment: MSA | None) -> Any:
    """Return upstream's ``MSA`` for one chain, through A3M text and no temporary file.

    A protein chain always carries one — the depth-1 alignment where the caller gave none —
    so upstream's per-chain "no MSA provided" warning is never reached.
    """
    if alignment is None:
        return None
    import io

    from esm.utils import msa as upstream  # pyright: ignore[reportMissingImports]

    return upstream.MSA.from_a3m(io.StringIO(alignment.to_a3m()))


def _confidence(result: Any, directory: Path, name: str) -> Confidence:
    """Return what the model said, writing the pairwise matrix beside the coordinates."""
    pairwise = None
    if result.pae is not None:
        pairwise = pairwise_path(directory, name)
        np.save(pairwise, result.pae.numpy())
    return Confidence(
        plddt=_mean_plddt(result),
        ptm=None if result.ptm is None else float(result.ptm),
        iptm=None if result.iptm is None else float(result.iptm),
        pairwise_file=pairwise,
    )


def _mean_plddt(result: Any) -> float:
    """Return one diffusion sample's mean per-residue confidence."""
    return float(result.plddt.mean())
