"""The folding lane: what goes in, and the model that turns it into coordinates.

:class:`~protein.fold.request.FoldingRequest` is the input a caller builds and drops;
:class:`~protein.fold.esmfold.ESMFold2` holds the weights, because *resident state gets an
object; a subprocess does not*. What comes back is a
:class:`~protein.structure.Structure` — a prediction is a structure like any other, carrying
provenance a deposited entry has no room for.

No module's body imports torch, so ``import protein`` stays cheap.
"""

from __future__ import annotations

from protein.fold.predictions import (
    PREDICTION_FORMAT,
    prediction_name,
    prediction_path,
    stored_prediction,
)
from protein.fold.request import POLYMERS, ChainRequest, FoldingRequest, PolymerKind

__all__ = [
    "POLYMERS",
    "PREDICTION_FORMAT",
    "ChainRequest",
    "FoldingRequest",
    "PolymerKind",
    "prediction_name",
    "prediction_path",
    "stored_prediction",
]
