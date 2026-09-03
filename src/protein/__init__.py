"""liulab-protein: handling protein sequences, structures and the databases they live in."""

from importlib.metadata import version

from protein.core import Protein
from protein.embed import ESMC, Embedding
from protein.external import ToolNotFoundError

#: Read from the installed distribution's metadata, which hatch-vcs fills from the newest
#: git tag. No version string is written by hand anywhere in this repo.
__version__ = version("liulab-protein")

__all__ = ["ESMC", "Embedding", "Protein", "ToolNotFoundError", "__version__"]
