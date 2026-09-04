"""The one place this package shells out to a native binary.

An **External tool** is a binary the package drives rather than reimplements: ``mmseqs``
searches sequences, ``foldseek`` searches structures, ``muscle`` aligns a set of them.
:class:`ExternalTool` says where one is, what version it is, and how to run it — including
only when what it would build is stale. :class:`InstalledTool` shells out for real;
:class:`RecordingTool` records the calls and runs nothing, which is what a test binds a
search to.

MUSCLE is the one this module never runs: biotite's ``Muscle5App`` takes a ``bin_path``, so
:class:`Muscle` answers where the binary is and biotite does the rest.

**Foldseek vendors MMseqs2**, so :class:`MmseqsLikeTool` owns the command grammar the two
share and the concrete tools name only what differs.

Nothing outside this module imports :mod:`subprocess`.

Examples
--------
>>> from protein.external import Mmseqs, RecordingTool
>>> tool = RecordingTool("mmseqs", version="18.8cc5c")
>>> tool.run(["easy-search", "q.fasta", "swissprot", "hits.tsv", "tmp"], capture=False)
''
>>> tool.version, tool.calls[0].args[0]
('18.8cc5c', 'easy-search')
>>> Mmseqs().format_output.split(",")[2]
'pident'
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

#: How a binary is asked for its version unless :data:`_INSTALLATIONS` says otherwise.
DEFAULT_VERSION_ARGS: tuple[str, ...] = ("--version",)


@dataclass(frozen=True)
class _Installation:
    """How one binary is installed, and how it is asked for its version."""

    package: str
    homepage: str | None = None
    version_args: tuple[str, ...] = DEFAULT_VERSION_ARGS


#: Where a binary differs from the default of ``pixi add <lowercased name>`` asked
#: ``--version``. No tool here accepts ``--version``: the two searchers answer ``version``
#: and MUSCLE answers ``-version``. Nothing outside this module spells a conda package name.
_INSTALLATIONS: dict[str, _Installation] = {
    "mmseqs": _Installation("mmseqs2", "https://github.com/soedinglab/MMseqs2", ("version",)),
    "foldseek": _Installation(
        "foldseek", "https://github.com/steineggerlab/foldseek", ("version",)
    ),
    "muscle": _Installation("muscle", "https://github.com/rcedgar/muscle", ("-version",)),
}

#: The **External tool**s :func:`doctor` checks. Every one of them is required — there is no
#: optional tier, so a tool named here is installed by ``pixi install`` like any other.
REQUIRED_TOOLS: tuple[str, ...] = ("mmseqs", "foldseek", "muscle")

#: What :func:`doctor` reports for a tool that runs but will not identify itself.
NO_VERSION_REPORTED = "installed; reports no version"

#: The BLAST-tab columns both tools report, without the identity column each spells its
#: own way.
_SHARED_COLUMNS: tuple[str, ...] = (
    "query",
    "target",
    "alnlen",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "evalue",
    "bits",
)


class ToolNotFoundError(RuntimeError):
    """Raised when an **External tool** cannot be located on ``PATH``.

    The message is the tool's :meth:`ExternalTool.install_instructions`, so the next action
    travels with the failure.

    Examples
    --------
    >>> from protein import ToolNotFoundError
    >>> try:
    ...     raise ToolNotFoundError("mmseqs is not on PATH. Install `mmseqs2`.")
    ... except ToolNotFoundError as missing:
    ...     print(missing)
    mmseqs is not on PATH. Install `mmseqs2`.
    """


@dataclass(frozen=True)
class ToolCall:
    """One invocation of an **External tool** — what it was asked to do, and where.

    Attributes
    ----------
    args : tuple of str
        The arguments after the executable itself.
    cwd : pathlib.Path or None
        The directory the tool ran in, or ``None`` for the caller's own.
    capture : bool
        Whether the tool's output was captured rather than inherited.

    Examples
    --------
    >>> ToolCall(("easy-search", "q.fasta"), None, capture=True).args[0]
    'easy-search'
    """

    args: tuple[str, ...]
    cwd: Path | None
    capture: bool


def is_fresh(output: Path, inputs: Sequence[Path]) -> bool:
    """Return whether ``output`` is fresh against ``inputs`` — the **Freshness** rule.

    Fresh means ``output`` exists, is non-empty, and is at least as new as every input — the
    same staleness rule ``make`` uses. Missing inputs are ignored.

    Parameters
    ----------
    output : pathlib.Path
        What a call would build.
    inputs : sequence of pathlib.Path
        What it would build ``output`` from.

    Returns
    -------
    bool
        Whether the call can be skipped.

    Examples
    --------
    >>> from pathlib import Path
    >>> is_fresh(Path("/tmp/definitely-not-here"), [])
    False
    """
    if not output.is_file() or output.stat().st_size == 0:
        return False
    out_mtime = output.stat().st_mtime
    return all(out_mtime >= inp.stat().st_mtime for inp in inputs if inp.is_file())


class ExternalTool(ABC):
    """One binary the package drives instead of reimplementing it.

    Four questions and nothing else: where the binary is (:attr:`path`), what version it is
    (:attr:`version`), how to run it (:meth:`run`), and how to run it only when what it
    would build is stale (:meth:`run_to`). All are answered lazily, so a caller that never
    runs a tool never needs it installed.

    Subclass it for an adapter — :class:`InstalledTool` or :class:`RecordingTool`, which
    between them are the seam — or for a command grammar, as :class:`MmseqsLikeTool` does.
    The freshness rule, the failure message and the install instructions live here so the
    adapters cannot drift.

    Parameters
    ----------
    name : str
        The executable's name, spelled as it is on ``PATH``.
    package : str, optional
        The conda package that installs it. Defaults to what :data:`_INSTALLATIONS`
        records, or the lowercased name.
    homepage : str, optional
        The tool's own documentation, quoted in :meth:`install_instructions`.

    Attributes
    ----------
    name : str
        The executable's name.
    package : str
        The conda package that installs it.
    homepage : str or None
        The tool's own documentation, when there is one.
    version_args : tuple of str
        What the binary is asked for its version.

    Examples
    --------
    >>> tool = RecordingTool("mmseqs", version="18.8cc5c")
    >>> tool.run(["createdb", "swissprot.fasta", "swissprot"])
    ''
    >>> tool.calls[0].args
    ('createdb', 'swissprot.fasta', 'swissprot')
    """

    def __init__(
        self, name: str, *, package: str | None = None, homepage: str | None = None
    ) -> None:
        known = _INSTALLATIONS.get(name)
        self.name = name
        self.package = package or (known.package if known else name.lower())
        self.homepage = homepage or (known.homepage if known else None)
        self.version_args = known.version_args if known else DEFAULT_VERSION_ARGS
        self._path: str | None = None
        self._version: str | None = None

    # -- what a caller asks of a tool ----------------------------------------

    @property
    def path(self) -> str:
        """Absolute path to the executable, located once and remembered.

        Returns
        -------
        str
            The path the tool will be run from.

        Raises
        ------
        ToolNotFoundError
            If the binary cannot be located. The message is :meth:`install_instructions`.

        Examples
        --------
        >>> RecordingTool("mmseqs").path
        '/fake/mmseqs'
        """
        if self._path is None:
            self._path = self._locate()
        return self._path

    @property
    def version(self) -> str:
        """The tool's version line, or ``""`` when it will not identify itself.

        Asked on first use and remembered, so holding a tool runs no subprocess. ``""``
        means *the tool ran and declined*, a different answer from a tool that is not
        there — that raises.

        Returns
        -------
        str
            The first non-empty line of the version output, stdout preferred over stderr,
            or ``""``.

        Raises
        ------
        ToolNotFoundError
            If the binary cannot be located.

        Examples
        --------
        >>> RecordingTool("foldseek", version="10.941cd33").version
        '10.941cd33'
        """
        if self._version is None:
            self._version = self._detect_version()
        return self._version

    def install_instructions(self) -> str:
        """Return the text to put in front of someone whose binary is missing.

        This is the message :class:`ToolNotFoundError` carries.

        Returns
        -------
        str
            Several lines: what is missing, what installs it, and what to check if it is
            installed already.

        Examples
        --------
        >>> print(RecordingTool("mmseqs").install_instructions())
        mmseqs is not installed. Add it to the project environment with:
            pixi add mmseqs2            # channels: conda-forge, bioconda
        Already installed? Activate the environment with `pixi shell`, or run via `pixi run`.
        See https://github.com/soedinglab/MMseqs2 for details.
        """
        lines = [
            f"{self.name} is not installed. Add it to the project environment with:",
            f"    pixi add {self.package}            # channels: conda-forge, bioconda",
            "Already installed? Activate the environment with `pixi shell`, or run via `pixi run`.",
        ]
        if self.homepage is not None:
            lines.append(f"See {self.homepage} for details.")
        return "\n".join(lines)

    def run(self, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True) -> str:
        """Run the tool with ``args`` and return what it wrote to stdout.

        Parameters
        ----------
        args : sequence of str
            The arguments after the executable.
        cwd : pathlib.Path, optional
            The directory to run in. Defaults to the caller's own — pass one when the tool
            drops files in its working directory and they belong beside its output.
        capture : bool, default True
            Whether to capture the tool's output or let it inherit this process's streams.
            Capturing puts the tool's own diagnostics into the error raised on failure;
            inheriting streams progress live, which is what a run of an hour wants.

        Returns
        -------
        str
            The captured stdout, or ``""`` when ``capture`` is false.

        Raises
        ------
        ToolNotFoundError
            If the binary cannot be located.
        RuntimeError
            If the tool exits non-zero. The message names the tool, its exit code and the
            arguments, and carries the tool's own output when it was captured.

        Examples
        --------
        >>> RecordingTool("mmseqs").run(["createdb", "swissprot.fasta", "swissprot"])
        ''
        """
        completed = self._execute(list(args), cwd=cwd, capture=capture)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip() if capture else ""
            explain = f": {detail}" if detail else "; see its output above for the error"
            raise RuntimeError(
                f"{self.name} failed (exit {completed.returncode}) for args {list(args)!r}{explain}"
            )
        return completed.stdout or ""

    def run_to(
        self,
        args: Sequence[str],
        *,
        output: Path,
        inputs: Sequence[Path],
        overwrite: bool = False,
    ) -> Path:
        """Run the tool to build ``output``, skipping the call when ``output`` is fresh.

        The cached primitive every step is built from: a fresh ``output`` costs a handful
        of ``stat`` calls rather than a second pass over a database.

        Running goes back out through :meth:`run` rather than around it to
        :meth:`_execute`, so one ``monkeypatch.setattr(ExternalTool, "run", ...)`` catches
        every invocation this package makes.

        Parameters
        ----------
        args : sequence of str
            The arguments after the executable, written so the tool produces ``output``.
        output : pathlib.Path
            What this call builds, and what its freshness is judged by.
        inputs : sequence of pathlib.Path
            What ``output`` is built from. ``output`` is stale once any of them is newer.
            Inputs that do not exist are ignored.
        overwrite : bool, default False
            Run regardless of freshness.

        Returns
        -------
        pathlib.Path
            ``output``, whether it was rebuilt or reused.

        Raises
        ------
        ToolNotFoundError
            If the binary cannot be located — and only when the tool is actually run, so a
            fresh output is served without the tool being installed at all.
        RuntimeError
            If the tool exits non-zero.

        Examples
        --------
        >>> from pathlib import Path
        >>> tool = RecordingTool("mmseqs")
        >>> tool.run_to(                                  # doctest: +SKIP
        ...     ["createdb", "swissprot.fasta", "swissprot"],
        ...     output=Path("swissprot"),
        ...     inputs=[Path("swissprot.fasta")],
        ... )
        PosixPath('swissprot')
        """
        if overwrite or not is_fresh(output, inputs):
            self.run(args)
        return output

    # -- the seam an adapter fills -------------------------------------------

    @abstractmethod
    def _locate(self) -> str:
        """Return the absolute path to the executable, or raise :class:`ToolNotFoundError`."""

    @abstractmethod
    def _execute(
        self, args: Sequence[str], *, cwd: Path | None, capture: bool
    ) -> subprocess.CompletedProcess[str]:
        """Run the executable with ``args`` and return what it did, however that is done."""

    def _detect_version(self) -> str:
        """Ask the tool for its version, answering ``""`` when it will not say."""
        try:
            completed = self._execute(list(self.version_args), cwd=None, capture=True)
        except (OSError, subprocess.SubprocessError):
            return ""
        if completed.returncode != 0:
            return ""
        text = (completed.stdout or completed.stderr or "").strip()
        return text.splitlines()[0] if text else ""


#: Every version line a binary has given in this process, keyed by the **located** path
#: rather than the tool name: two directories can each hold an ``mmseqs``. Only an answer a
#: binary gave lands here, so a tool installed midway through a process is still found.
_VERSIONS: dict[str, str] = {}


def clear_version_cache() -> None:
    """Forget every version learned so far, so the next ask reaches the binary again.

    :data:`_VERSIONS` lives as long as the process, which is right for a pipeline and wrong
    for a test suite or anything that swaps a binary out underneath itself.

    Examples
    --------
    >>> clear_version_cache()
    """
    _VERSIONS.clear()


class InstalledTool(ExternalTool):
    """The **External tool** as it is installed on this machine.

    Resolution is ``shutil.which``, then the ``bin/`` directory of the running interpreter:
    in a conda/pixi environment the native tools sit beside ``python``, so the second lookup
    still finds them when ``PATH`` lacks that directory.

    The version is remembered per *binary* rather than per object — see :data:`_VERSIONS`.

    Examples
    --------
    >>> InstalledTool("mmseqs").version                   # doctest: +SKIP
    '18.8cc5c'
    """

    def _locate(self) -> str:
        """Return the path ``shutil.which`` finds, else the interpreter's own ``bin/``."""
        path = shutil.which(self.name)
        if path is not None:
            return path

        sibling = Path(sys.executable).parent / self.name
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return str(sibling)

        raise ToolNotFoundError(self.install_instructions())

    def _detect_version(self) -> str:
        """Ask the binary at :attr:`path`, once per process — see :data:`_VERSIONS`.

        Membership, not truthiness: ``""`` is an answer, not *not asked yet*.
        """
        path = self.path
        if path not in _VERSIONS:
            _VERSIONS[path] = super()._detect_version()
        return _VERSIONS[path]

    def _execute(
        self, args: Sequence[str], *, cwd: Path | None, capture: bool
    ) -> subprocess.CompletedProcess[str]:
        """Shell out, letting a non-zero exit come back rather than raise."""
        return subprocess.run(
            [self.path, *args],
            cwd=cwd,
            capture_output=capture,
            text=True,
            check=False,
        )


class RecordingTool(ExternalTool):
    """An **External tool** that records what it was asked to do and runs nothing.

    The stand-in a test binds where a real binary would go, so a search runs end to end on
    a machine that has neither tool installed. It is a full :class:`ExternalTool`, not a
    patched-out method: only the execution is replaced.

    Parameters
    ----------
    name : str, default "tool"
        The executable's name, as :class:`ExternalTool` takes it.
    version : str, default "0.0-test"
        What :attr:`~ExternalTool.version` reports; ``""`` stands for a tool that runs
        but will not identify itself.
    path : str, optional
        What :attr:`~ExternalTool.path` reports. Defaults to ``/fake/<name>``.
    stdout : str, default ""
        What each captured run returns.
    on_run : callable, optional
        Called with each :class:`ToolCall` as it is made — how a test makes a stand-in
        leave behind the files a real run would have written.
    package, homepage : str, optional
        As :class:`ExternalTool`.

    Attributes
    ----------
    calls : list of ToolCall
        Every call made, in order.
    exit_code : int
        What the next run reports. Set it non-zero to make :meth:`~ExternalTool.run` fail
        as a real tool failing would.
    stdout : str
        What each captured run returns.
    on_run : callable or None
        Called with each :class:`ToolCall` as it is made.

    Examples
    --------
    >>> tool = RecordingTool("mmseqs")
    >>> tool.exit_code = 1
    >>> tool.run(["easy-search"])
    Traceback (most recent call last):
    RuntimeError: mmseqs failed (exit 1) for args ['easy-search']; see its output above for the error
    """

    def __init__(
        self,
        name: str = "tool",
        *,
        version: str = "0.0-test",
        path: str | None = None,
        stdout: str = "",
        on_run: Callable[[ToolCall], None] | None = None,
        package: str | None = None,
        homepage: str | None = None,
    ) -> None:
        super().__init__(name, package=package, homepage=homepage)
        self.calls: list[ToolCall] = []
        self.exit_code = 0
        self.stdout = stdout
        self.on_run = on_run
        self._reported_path = path or f"/fake/{name}"
        self._reported_version = version

    def _locate(self) -> str:
        """Return the stand-in path; nothing is looked up on ``PATH``."""
        return self._reported_path

    def _detect_version(self) -> str:
        """Return the version this stand-in was told to report, running nothing."""
        return self._reported_version

    def _execute(
        self, args: Sequence[str], *, cwd: Path | None, capture: bool
    ) -> subprocess.CompletedProcess[str]:
        """Record the call and hand back the canned result."""
        call = ToolCall(tuple(args), cwd, capture)
        self.calls.append(call)
        if self.on_run is not None:
            self.on_run(call)
        return subprocess.CompletedProcess(
            [self.path, *args],
            self.exit_code,
            stdout=self.stdout if capture else None,
            stderr="" if capture else None,
        )


class MmseqsLikeTool(InstalledTool):
    """The command grammar MMseqs2 and Foldseek share, owned once.

    Foldseek vendors MMseqs2, so the two spell the same verbs over the same ffindex
    database layout and both choose their result columns with ``--format-output``. What
    genuinely differs is declared per tool: :attr:`IDENTITY_COLUMN` — Foldseek's is a
    fraction, MMseqs2's a percentage — and :attr:`EXTRA_COLUMNS`.

    Every ``easy-*`` verb takes a temp directory and neither tool removes it, so
    :meth:`scratch_dir` owns that here rather than at each call site.

    Constructed with no arguments — a concrete tool knows its own binary name.

    Attributes
    ----------
    IDENTITY_COLUMN : str
        The identity column this tool's results carry.
    EXTRA_COLUMNS : tuple of str
        Columns this tool adds after the shared BLAST-tab set.

    Examples
    --------
    >>> Foldseek().format_columns[:3]
    ('query', 'target', 'fident')
    >>> Foldseek().format_columns[-2:]
    ('alntmscore', 'lddt')
    """

    #: Declared, never defaulted: a default here would impose one tool's convention on the
    #: other.
    IDENTITY_COLUMN: ClassVar[str]

    #: What this tool reports that the other cannot. Empty for a tool that adds nothing.
    EXTRA_COLUMNS: ClassVar[tuple[str, ...]] = ()

    @property
    def format_columns(self) -> tuple[str, ...]:
        """The result columns this tool is asked for, in order.

        Returns
        -------
        tuple of str
            The identity column in position 2 of the shared BLAST-tab set, and this tool's
            own columns after it.

        Examples
        --------
        >>> Mmseqs().format_columns[2]
        'pident'
        """
        query, target, *rest = _SHARED_COLUMNS
        return (query, target, self.IDENTITY_COLUMN, *rest, *self.EXTRA_COLUMNS)

    @property
    def format_output(self) -> str:
        """:attr:`format_columns` as the tool's ``--format-output`` argument.

        Returns
        -------
        str
            The column names joined by commas.

        Examples
        --------
        >>> Mmseqs().format_output.startswith("query,target,pident,")
        True
        """
        return ",".join(self.format_columns)

    @contextmanager
    def scratch_dir(self, purpose: str = "run") -> Iterator[Path]:
        """Yield a temp directory for one command, and remove it however the command ends.

        Every ``easy-*`` verb, ``createindex`` and ``cluster`` take a working directory as a
        positional argument, and **neither tool removes it**. It lands under the package's
        own data root rather than ``/tmp``: these are the same gigabytes the outputs are,
        and a cluster node's ``/tmp`` is neither large enough nor on the same filesystem.

        Parameters
        ----------
        purpose : str, default "run"
            Named in the directory, so one left behind by a killed process says which
            command made it.

        Yields
        ------
        pathlib.Path
            A fresh empty directory under ``<LIULAB_DATA>/protein/.work/``.

        Examples
        --------
        >>> with Mmseqs().scratch_dir("easy-search") as work:   # doctest: +SKIP
        ...     work.is_dir()
        True
        """
        # Deferred so `import protein` does not pay for liulab-genome, and the module
        # rather than the function so a monkeypatch of `store.work_dir` is seen here.
        from protein import store

        root = store.work_dir()
        root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(dir=root, prefix=f"{self.name}-{purpose}-"))
        try:
            yield scratch
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    # -- the shared verbs ----------------------------------------------------

    def createdb(
        self,
        inputs: Sequence[Path],
        database: Path,
        *,
        extra: Sequence[str] = (),
        overwrite: bool = False,
    ) -> Path:
        """Build a database from ``inputs`` — FASTA for MMseqs2, structures for Foldseek.

        Parameters
        ----------
        inputs : sequence of pathlib.Path
            The files or directories to read.
        database : pathlib.Path
            The database to write. Its siblings land beside it and are the tool's
            business, not this call's.
        extra : sequence of str, optional
            Further arguments, passed through unread.
        overwrite : bool, default False
            Build regardless of freshness.

        Returns
        -------
        pathlib.Path
            ``database``.

        Examples
        --------
        >>> from pathlib import Path
        >>> Mmseqs().createdb([Path("sp.fasta")], Path("sp"))       # doctest: +SKIP
        PosixPath('sp')
        """
        return self.run_to(
            ["createdb", *(str(path) for path in inputs), str(database), *extra],
            output=database,
            inputs=list(inputs),
            overwrite=overwrite,
        )

    def databases(
        self, source: str, database: Path, work: Path, *, extra: Sequence[str] = ()
    ) -> Path:
        """Fetch a published database by the tool's own name for it, and unpack it.

        ``mmseqs databases UniProtKB/Swiss-Prot``, ``foldseek databases PDB`` — **this
        package does not manage the download**. What that costs is ADR-0003.

        Unlike every other verb here, the output is **not captured**: this moves gigabytes,
        and the tool's own progress belongs on the terminal.

        ``work`` is passed in rather than taken from :meth:`scratch_dir`, which removes
        itself: an interrupted download has to be able to keep the archive it already
        fetched, so clearing ``work`` is the caller's job.

        Parameters
        ----------
        source : str
            The tool's own spelling, which is not the registered name: ``UniProtKB/Swiss-
            Prot`` carries a slash.
        database : pathlib.Path
            The ffindex prefix to write. Its siblings land beside it.
        work : pathlib.Path
            The tool's temp directory. It must exist and it is not removed here.
        extra : sequence of str, optional
            Further arguments, passed through unread.

        Returns
        -------
        pathlib.Path
            ``database``.

        Examples
        --------
        >>> from pathlib import Path
        >>> Mmseqs().databases(                                     # doctest: +SKIP
        ...     "UniProtKB/Swiss-Prot", Path("db/swissprot/swissprot"), Path("db/swissprot/.work")
        ... )
        PosixPath('db/swissprot/swissprot')
        """
        self.run(
            ["databases", source, str(database), str(work), *extra],
            capture=False,
        )
        return database

    def view(
        self,
        database: Path,
        ids: Sequence[str],
        *,
        entry_type: int | None = None,
        id_mode: int | None = None,
        extra: Sequence[str] = (),
    ) -> str:
        """Print named entries of ``database`` to stdout, and return what it printed.

        The retrieval verb, and the reason ``swissprot["P12345"]`` needs no network and no
        index build. **A name it cannot find is a warning on stderr and exit 0**, so an
        empty answer is how absence arrives here.

        Parameters
        ----------
        database : pathlib.Path
            The ffindex prefix.
        ids : sequence of str
            What to print, joined into one ``--id-list``: numeric keys by default, or the
            names in ``.lookup`` with ``id_mode=1``.
        entry_type : int, optional
            ``--idx-entry-type``. ``2`` is the parallel header database; the default is the
            sequence one. The header database has no ``.lookup``, so **it does not accept
            ``id_mode=1``** — a header is asked for by numeric key.
        id_mode : int, optional
            ``--id-mode``. ``1`` resolves each id through the database's ``.lookup``.
        extra : sequence of str, optional
            Further arguments, passed through unread.

        Returns
        -------
        str
            The tool's stdout, one entry after another.

        Examples
        --------
        >>> from pathlib import Path
        >>> Mmseqs().view(Path("swissprot"), ["415743"], entry_type=2)   # doctest: +SKIP
        'sp|P12345|AATM_RABIT Aspartate aminotransferase, mitochondrial ...'
        """
        args = ["view", str(database), "--id-list", ",".join(ids)]
        if entry_type is not None:
            args += ["--idx-entry-type", str(entry_type)]
        if id_mode is not None:
            args += ["--id-mode", str(id_mode)]
        return self.run([*args, *extra])

    def createindex(
        self, database: Path, *, extra: Sequence[str] = (), overwrite: bool = False
    ) -> Path:
        """Precompute ``database``'s search index, so a search does not build one per run.

        Parameters
        ----------
        database : pathlib.Path
            The database to index.
        extra : sequence of str, optional
            Further arguments, passed through unread.
        overwrite : bool, default False
            Index regardless of freshness.

        Returns
        -------
        pathlib.Path
            ``<database>.idx``, the file the index's freshness is judged by.

        Examples
        --------
        >>> from pathlib import Path
        >>> Mmseqs().createindex(Path("sp"))                        # doctest: +SKIP
        PosixPath('sp.idx')
        """
        index = Path(f"{database}.idx")
        with self.scratch_dir("createindex") as work:
            return self.run_to(
                ["createindex", str(database), str(work), *extra],
                output=index,
                inputs=[database],
                overwrite=overwrite,
            )

    def easy_search(
        self,
        query: Path,
        target: Path,
        output: Path,
        *,
        extra: Sequence[str] = (),
        overwrite: bool = False,
    ) -> Path:
        """Search ``query`` against ``target`` and write the hits to ``output``.

        The one verb both lanes are built on: ``query`` is a FASTA for MMseqs2 and a
        structure for Foldseek, ``target`` is either a registered database or a raw file the
        tool converts, and the columns are this tool's :attr:`format_columns`.

        Parameters
        ----------
        query : pathlib.Path
            What to search with.
        target : pathlib.Path
            What to search against.
        output : pathlib.Path
            The tab-separated hits.
        extra : sequence of str, optional
            Further arguments, passed through unread — search settings belong to the
            caller that knows what it is searching.
        overwrite : bool, default False
            Search regardless of freshness.

        Returns
        -------
        pathlib.Path
            ``output``.

        Examples
        --------
        >>> from pathlib import Path
        >>> Foldseek().easy_search(                                 # doctest: +SKIP
        ...     Path("1ubq.cif"), Path("pdb100"), Path("hits.tsv")
        ... )
        PosixPath('hits.tsv')
        """
        with self.scratch_dir("easy-search") as work:
            return self.run_to(
                [
                    "easy-search",
                    str(query),
                    str(target),
                    str(output),
                    str(work),
                    "--format-output",
                    self.format_output,
                    *extra,
                ],
                output=output,
                inputs=[query, target],
                overwrite=overwrite,
            )

    def convertalis(
        self,
        query_db: Path,
        target_db: Path,
        result_db: Path,
        output: Path,
        *,
        extra: Sequence[str] = (),
        overwrite: bool = False,
    ) -> Path:
        """Render an alignment database as the same columns :meth:`easy_search` writes.

        Parameters
        ----------
        query_db, target_db, result_db : pathlib.Path
            The three databases the alignment was made from and into.
        output : pathlib.Path
            The tab-separated hits.
        extra : sequence of str, optional
            Further arguments, passed through unread.
        overwrite : bool, default False
            Convert regardless of freshness.

        Returns
        -------
        pathlib.Path
            ``output``.

        Examples
        --------
        >>> from pathlib import Path
        >>> Mmseqs().convertalis(                                   # doctest: +SKIP
        ...     Path("q"), Path("sp"), Path("aln"), Path("hits.tsv")
        ... )
        PosixPath('hits.tsv')
        """
        return self.run_to(
            [
                "convertalis",
                str(query_db),
                str(target_db),
                str(result_db),
                str(output),
                "--format-output",
                self.format_output,
                *extra,
            ],
            output=output,
            inputs=[query_db, target_db, result_db],
            overwrite=overwrite,
        )

    def cluster(
        self,
        database: Path,
        clusters: Path,
        *,
        extra: Sequence[str] = (),
        overwrite: bool = False,
    ) -> Path:
        """Cluster ``database`` into ``clusters``.

        Parameters
        ----------
        database : pathlib.Path
            The database to cluster.
        clusters : pathlib.Path
            The cluster database to write.
        extra : sequence of str, optional
            Further arguments, passed through unread — the thresholds belong to the caller
            that knows what it is clustering.
        overwrite : bool, default False
            Cluster regardless of freshness.

        Returns
        -------
        pathlib.Path
            ``clusters``.

        Examples
        --------
        >>> from pathlib import Path
        >>> Mmseqs().cluster(Path("sp"), Path("sp_clu"))            # doctest: +SKIP
        PosixPath('sp_clu')
        """
        with self.scratch_dir("cluster") as work:
            return self.run_to(
                ["cluster", str(database), str(clusters), str(work), *extra],
                output=clusters,
                inputs=[database],
                overwrite=overwrite,
            )


class Mmseqs(MmseqsLikeTool):
    """MMseqs2 — the sequence half of the pair.

    Examples
    --------
    >>> Mmseqs().name, Mmseqs().package
    ('mmseqs', 'mmseqs2')
    """

    IDENTITY_COLUMN: ClassVar[str] = "pident"

    def __init__(self) -> None:
        super().__init__("mmseqs")


class Foldseek(MmseqsLikeTool):
    """Foldseek — the structure half, and the one that vendors the other.

    Its extra columns are what a structural hit is judged on, and MMseqs2 has no answer for
    either.

    Examples
    --------
    >>> Foldseek().EXTRA_COLUMNS
    ('alntmscore', 'lddt')
    """

    IDENTITY_COLUMN: ClassVar[str] = "fident"
    EXTRA_COLUMNS: ClassVar[tuple[str, ...]] = ("alntmscore", "lddt")

    def __init__(self) -> None:
        super().__init__("foldseek")


class Muscle(InstalledTool):
    """MUSCLE — the aligner, located here and driven by biotite.

    The one **External tool** this module does not run. ``Muscle5App.align()`` takes a
    ``bin_path`` and owns the temporary files, the arguments and the parsing, so what this
    class is for is :attr:`~ExternalTool.path` — and the version line :func:`doctor` reports
    beside the other two.

    Examples
    --------
    >>> Muscle().name, Muscle().package
    ('muscle', 'muscle')
    >>> Muscle().version_args
    ('-version',)
    """

    def __init__(self) -> None:
        super().__init__("muscle")


def doctor() -> dict[str, str]:
    """Verify every **External tool** the package needs and report what each one is.

    Returns
    -------
    dict of str to str
        Each name in :data:`REQUIRED_TOOLS` mapped to its version line, or to
        :data:`NO_VERSION_REPORTED` when the tool is there but will not identify itself.

    Raises
    ------
    ToolNotFoundError
        If any required tool is missing.

    Examples
    --------
    >>> sorted(doctor())                                  # doctest: +SKIP
    ['foldseek', 'mmseqs', 'muscle']
    """
    return {name: InstalledTool(name).version or NO_VERSION_REPORTED for name in REQUIRED_TOOLS}
