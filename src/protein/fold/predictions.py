"""Where a prediction lands, what it is called, and when an existing one is reused.

**Not** :mod:`protein.store`. Nothing here looks at the lab **Data dir**, which holds
reference and input data and never a user's outputs: the directory is an argument and it has
no default.

**The name is a fact about the molecule and nothing else** — user-given, else the accession,
else a short stable hash of the sequences. Neither the checkpoint nor a sampler setting
enters it, because neither says what was folded. Fold a hundred proteins into one directory
and they are named after their accessions with no bookkeeping.

**A name already held is checked against the residues on disk**, not against any record of
what produced them. ``Protein("MQIFVKTLTG", id="P0CG48")`` is legal, so a mutant can arrive
carrying a reference accession and land on the reference's file: the same sequence is a cache
hit, a different one is an error, and ``overwrite=`` is how a caller says they meant it.
Reading the residues rather than a provenance record is what makes that refusal survive a
day later, since provenance does not survive the file (ADR-0005).

**The accepted edge**: settings are not in the path, so re-folding with a different seed hits
the cache. The escape is ``overwrite=`` or a distinct name.

Examples
--------
>>> from protein.fold import ChainRequest, FoldingRequest
>>> from protein.fold.predictions import prediction_name, prediction_path
>>> request = FoldingRequest([ChainRequest("protein", "MKTAY", accession="P12345")])
>>> prediction_name(request)
'P12345'
>>> prediction_name(request, "the mutant")
'the mutant'
>>> prediction_path("/scratch/folds", "P12345")
PosixPath('/scratch/folds/P12345.cif')
"""

from __future__ import annotations

from hashlib import blake2b
from pathlib import Path
from typing import TYPE_CHECKING

from protein.structure import Structure

if TYPE_CHECKING:
    from protein.fold.request import FoldingRequest

__all__ = [
    "PREDICTION_FORMAT",
    "prediction_name",
    "prediction_path",
    "stored_prediction",
]

#: What a prediction is written as. mmCIF, because it is the only format this package writes
#: and the only one that can spell a chain label of more than one character.
PREDICTION_FORMAT = "cif"

#: How many hex characters a derived name carries. Long enough that two sequences in one
#: directory do not collide, short enough to read off a listing.
_DIGEST_CHARACTERS = 16


def prediction_name(request: FoldingRequest, name: str | None = None) -> str:
    """Return what this request's prediction is called.

    Parameters
    ----------
    request : protein.fold.FoldingRequest
        What is being folded.
    name : str, optional
        What the caller called it. Given, it wins.

    Returns
    -------
    str
        ``name`` where there is one; else the one accession the request names, where it
        names exactly one; else a short hash of its chains' kinds and sequences. **No
        checkpoint and no setting is in it** — neither is a fact about the molecule.

    Examples
    --------
    >>> from protein.fold import ChainRequest, FoldingRequest
    >>> homodimer = FoldingRequest(
    ...     [ChainRequest("protein", "MKTAY", accession="P12345") for _ in range(2)]
    ... )
    >>> prediction_name(homodimer)
    'P12345'
    >>> prediction_name(FoldingRequest([ChainRequest("dna", "ACGT")]))
    'ebe73d9e841dfdaa'
    """
    if name is not None:
        return name
    accessions = {chain.accession for chain in request.chains if chain.accession is not None}
    if len(accessions) == 1:
        return accessions.pop()
    return _digest(request)


def prediction_path(directory: str | Path, name: str) -> Path:
    """Return where the prediction called ``name`` lives under ``directory``.

    Parameters
    ----------
    directory : str or pathlib.Path
        The output directory. **Required, and it defaults nowhere** — the **Data dir** holds
        reference and input data, never a user's outputs. Nothing is created by asking.
    name : str
        What the prediction is called, from :func:`prediction_name`.

    Returns
    -------
    pathlib.Path
        ``<directory>/<name>.cif``.

    Examples
    --------
    >>> prediction_path("/scratch/folds", "P12345")
    PosixPath('/scratch/folds/P12345.cif')
    """
    return Path(directory) / f"{name}.{PREDICTION_FORMAT}"


def stored_prediction(
    path: str | Path, request: FoldingRequest, *, overwrite: bool = False
) -> Structure | None:
    """Return the prediction at ``path`` when it is this request's, refusing another's.

    Parameters
    ----------
    path : str or pathlib.Path
        Where the prediction would be, from :func:`prediction_path`.
    request : protein.fold.FoldingRequest
        What is being folded, whose sequences the file's residues are weighed against.
    overwrite : bool, default False
        Say the caller meant it. Nothing is read and nothing is held: the answer is ``None``
        whatever is there.

    Returns
    -------
    Structure or None
        The prediction already on disk, whose chains hold this request's sequences — so the
        GPU is never spun up for it. ``None`` when there is nothing there, or when
        ``overwrite`` is set.

    Raises
    ------
    FileExistsError
        If ``path`` holds a structure whose residues are not this request's. The message
        names the chain that disagrees and what to do about it.

    Examples
    --------
    >>> stored_prediction("/scratch/folds/P12345.cif", request)   # doctest: +SKIP
    Structure('P12345')
    """
    file = Path(path)
    if overwrite or not file.is_file():
        return None
    held = Structure.from_file(file)
    stored = _residues(held)
    wanted = {
        label: str(chain.sequence)
        for label, chain in zip(request.chain_ids, request.chains, strict=True)
    }
    if stored != wanted:
        raise FileExistsError(
            f"{file} already holds {_disagreement(stored, wanted)}. Sequences are read back "
            f"off the residues, so a mutant carrying a reference accession never overwrites "
            f"or silently reuses the reference. Pass overwrite=True, or a distinct name."
        )
    return held


def _digest(request: FoldingRequest) -> str:
    """Return the short stable hash naming a request that names no single accession."""
    text = "\n".join(f"{chain.kind}:{chain.sequence}" for chain in request.chains)
    return blake2b(text.encode("utf-8"), digest_size=_DIGEST_CHARACTERS // 2).hexdigest()


def _residues(structure: Structure) -> dict[str, str]:
    """Return each polymer chain's observed sequence, keyed by label.

    Off the residues and never off a provenance record, which is what makes the refusal
    outlive the process that wrote the file.
    """
    return {
        chain.chain_id: str(chain.sequence) for chain in structure.chains if chain.kind != "other"
    }


def _disagreement(stored: dict[str, str], wanted: dict[str, str]) -> str:
    """Return the text naming how the file on disk differs from what is being folded."""
    if set(stored) != set(wanted):
        return f"chains {sorted(stored)} where this request folds {sorted(wanted)}"
    label = next(label for label in wanted if stored[label] != wanted[label])
    return (
        f"a different sequence in chain {label!r}: {len(stored[label])} residues on disk "
        f"against {len(wanted[label])} being folded"
        if len(stored[label]) != len(wanted[label])
        else f"a different sequence in chain {label!r} of the same length"
    )
