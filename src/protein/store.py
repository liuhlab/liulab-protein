"""Where this package's large local files live, filed under the lab's one **Data dir**.

There is **no second data root**. ``LIULAB_DATA`` is read by
:mod:`genome.store.data_dir` and by nothing here, so a machine that has told
``liulab-genome`` where the lab's reference data lives has told this package too. What is
added is one subdirectory name and the layout beneath it::

    <LIULAB_DATA>/protein/
    ├── db/<name>/          a registered database
    │   └── .completion.json
    ├── sifts/              the PDB<->UniProt map, a prepared set
    │   └── .completion.json
    ├── embed/<model>/      later
    └── .work/              the temp directories the external tools do not clean

Registration is `liulab-genome`'s and is not reimplemented: **a directory plus a completion
record is the registration; a name addresses a directory; nothing is persisted centrally.**
:mod:`genome.store.completion` writes and reads that record — ``build_record``,
``write_record`` (atomic), ``read_record``, ``check_registration`` — and a caller reaches it
through the module::

    from genome.store import completion

    record = completion.build_record(directory, kind="database", name=name, files=files)
    completion.write_record(directory, record)

**Import the module, never the function.** That is `liulab-genome`'s own rule and it is
load-bearing: a function bound into this namespace is a second reference that
``monkeypatch.setattr`` on the module would never reach, and the suite's offline guard is
spelled for the first. The exception classes below are the one exemption, because nothing
patches one — they are re-exported so a caller can name in an ``except`` what this package
hands it, rather than importing from a module the API reference is free to move.

Nothing here creates a directory. A path is an answer to *where would this go*, and the
write that needs it is what brings it into existence — except :func:`work_dir`'s root, which
:meth:`protein.external.MmseqsLikeTool.scratch_dir` makes because it is about to fill it.

Examples
--------
>>> import os
>>> from protein.store import protein_data_dir, work_dir
>>> os.environ["LIULAB_DATA"] = "/scratch/liulab"
>>> protein_data_dir()
PosixPath('/scratch/liulab/protein')
>>> work_dir()
PosixPath('/scratch/liulab/protein/.work')
>>> del os.environ["LIULAB_DATA"]
"""

from __future__ import annotations

from pathlib import Path

from genome.store import completion, data_dir
from genome.store.completion import (
    RegistrationError,
    RegistrationMismatchError,
    UnfinishedRegistrationError,
)

__all__ = [
    "PROTEIN_SUBDIR",
    "RegistrationError",
    "RegistrationMismatchError",
    "UnfinishedRegistrationError",
    "protein_data_dir",
    "work_dir",
]

#: This package's one directory under the **Data dir**. A sibling of the assembly tree and
#: of `liulab-genome`'s prepared sets, never a tenant of one: a protein database belongs to
#: no assembly.
PROTEIN_SUBDIR = "protein"


def protein_data_dir() -> Path:
    """Return this package's root under the lab **Data dir**.

    Returns
    -------
    pathlib.Path
        ``<LIULAB_DATA>/protein``. Nothing is created by asking.

    Examples
    --------
    >>> import os
    >>> os.environ["LIULAB_DATA"] = "/scratch/liulab"
    >>> protein_data_dir()
    PosixPath('/scratch/liulab/protein')
    >>> del os.environ["LIULAB_DATA"]
    """
    return data_dir.prepared_data_dir(PROTEIN_SUBDIR)


def work_dir() -> Path:
    """Return the disposable scratch area under this package's root.

    Both external tools take a working directory as a positional argument to their
    ``easy-*`` verbs and **neither removes it** — one database download left 2.7 GB of them
    behind. :meth:`protein.external.MmseqsLikeTool.scratch_dir` makes one directory per
    command in here and removes it however the command ends. It is under the data root
    rather than ``/tmp`` because these are the same gigabytes the outputs are, and a
    cluster node's ``/tmp`` is neither large enough nor on the same filesystem.

    Returns
    -------
    pathlib.Path
        ``<LIULAB_DATA>/protein/.work`` — the same ``.work`` name
        :mod:`genome.store.completion` gives every build's working area, spelled by that
        module and not again here.

    Examples
    --------
    >>> import os
    >>> os.environ["LIULAB_DATA"] = "/scratch/liulab"
    >>> work_dir()
    PosixPath('/scratch/liulab/protein/.work')
    >>> del os.environ["LIULAB_DATA"]
    """
    return completion.work_dir(protein_data_dir())
