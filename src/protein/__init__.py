"""liulab-protein: handling protein sequences, structures and the databases they live in."""

from importlib.metadata import version

from protein.core import Protein
from protein.embed import ESMC, Embedding
from protein.external import ToolNotFoundError
from protein.fold import ChainRequest, ESMFold2, FoldingRequest
from protein.msa import MSA
from protein.structure import Chain, Structure

#: From the installed distribution's metadata, which hatch-vcs fills from the newest git tag.
__version__ = version("liulab-protein")

__all__ = [
    "ESMC",
    "MSA",
    "Chain",
    "ChainRequest",
    "ESMFold2",
    "Embedding",
    "FoldingRequest",
    "Protein",
    "Structure",
    "ToolNotFoundError",
    "__version__",
]
