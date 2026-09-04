"""The read-back half of a **Prepared set**: its status, its cached read and its two commands.

:mod:`genome.store.prepared` owns the build — the fetch, the working area, the staged rename,
the digest and the **Completion marker** — and a set declares a URL, a directory and a reader
to it. What a caller does with the set afterwards is here.

**Here rather than upstream, because the two packages' sets are shaped differently.** Genome's
are parameterised families: each takes a species, a release or a tax group and builds a
directory per combination, so upstream answers *which of these are prepared* with a table.
This package's take no argument and address one fixed directory each, so a singleton status
and a two-command CLI fit them and would not fit upstream's.

**A set declares what differs and nothing else**: where its source is, what its status
reports, how one stored file is read back, and the words its commands print. The source is a
**callable**, not a built source, because a set's directory is read from the **Data dir** when
it is called — a process that re-points that root must get the new one, and a source built at
import would freeze the first.

**Not a base class.** Neither set has behaviour to override, and each keeps its public verbs
as module-level one-line delegates, so callers reach them through the module and one
``monkeypatch.setattr`` reaches every caller.

Examples
--------
>>> from pathlib import Path
>>> found = PreparedStatus(path=Path("/x.tsv.gz"), prepared=False, fields={"rows": None})
>>> found.as_json()
{'path': '/x.tsv.gz', 'prepared': False, 'rows': None, 'completed_at': None}
"""

from __future__ import annotations

import functools
import json as _json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import typer
from genome.store import completion, prepared

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    import pandas as pd

__all__ = ["PreparedSet", "PreparedStatus"]

#: What a failed call raises: a set that is not here or an interrupted run is a
#: ``RuntimeError``, a file a reader cannot slice a ``ValueError``, and a file that went away
#: under the read an ``OSError``.
_ERRORS = (ValueError, OSError, RuntimeError)


@dataclass(frozen=True)
class PreparedStatus:
    """What one **Prepared set** has on disk here, without touching the network.

    Attributes
    ----------
    path : pathlib.Path
        Where the stored file is, whether or not it exists.
    prepared : bool
        Whether the set is finished here — a marker beside a file that is present.
    fields : mapping of str to object
        What the set declared its status reports, in that order, every value ``None`` when
        nothing is prepared. Each is reached as an attribute too, so ``status().rows`` answers.
    completed_at : str or None
        When this machine finished preparing it, ISO-8601 in UTC.

    Examples
    --------
    >>> from pathlib import Path
    >>> found = PreparedStatus(path=Path("/x.tsv.gz"), prepared=False, fields={"rows": None})
    >>> found.rows is None
    True
    """

    path: Path
    prepared: bool
    fields: Mapping[str, Any]
    completed_at: str | None = None

    def __getattr__(self, name: str) -> Any:
        """Return one declared field, so a caller reads it off the status by its own name."""
        try:
            return self.__dict__["fields"][name]
        except KeyError:
            raise AttributeError(name) from None

    def as_json(self) -> dict[str, Any]:
        """Return this status as the mapping ``--json`` prints.

        Returns
        -------
        dict
            :attr:`path` as a string so the result is JSON, then :attr:`prepared`, then the
            declared fields in order, then :attr:`completed_at`.

        Examples
        --------
        >>> from pathlib import Path
        >>> PreparedStatus(path=Path("/x.tsv.gz"), prepared=False, fields={}).as_json()
        {'path': '/x.tsv.gz', 'prepared': False, 'completed_at': None}
        """
        return {
            "path": str(self.path),
            "prepared": self.prepared,
            **self.fields,
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True)
class PreparedSet:
    """One **Prepared set**'s read-back half, declared once at the set's own module scope.

    Attributes
    ----------
    source : callable
        Returns what the set declares — the URL, the directory, the reader, the error class.
        A callable, so the **Data dir** is read when it is called and never at import.
    status_fields : tuple of (str, str)
        What :meth:`status` reports, in the order ``--json`` prints it: the key the
        **Completion marker** recorded, then the key the JSON carries. Declared rather than
        read off a record, so an unprepared set prints the same keys as a prepared one.
    read_table : callable
        Reads one stored file into a frame. Called once per file; :meth:`clear_cache` is what
        makes it read again.
    app_help : str
        What :attr:`app` prints above its commands.
    prepare_help, status_help : str
        What each command prints above its options. Both are dedented the way a docstring
        is, and a line break inside one is a line break in the output.

    Examples
    --------
    >>> from protein import sifts
    >>> [command.name for command in sifts.app.registered_commands]
    ['prepare', 'status']
    """

    source: Callable[[], prepared.PreparedSource]
    status_fields: tuple[tuple[str, str], ...]
    read_table: Callable[[Path], pd.DataFrame]
    app_help: str
    prepare_help: str
    status_help: str

    def prepare(self, *, progressbar: bool = True) -> prepared.Prepared:
        """Fetch and store the set, or return the one already here.

        Parameters
        ----------
        progressbar : bool, default True
            Show the download's progress bar. Nothing is drawn when the set is already there.

        Returns
        -------
        genome.store.prepared.Prepared
            The stored file and its **Completion marker**.
        """
        result = prepared.prepare(self.source(), progressbar=progressbar)
        # The stored file may have just been replaced, and a cache outliving the bytes it read
        # would answer from a release that is gone.
        self.clear_cache()
        return result

    def status(self) -> PreparedStatus:
        """Report what is on disk here, reading the **Completion marker** and nothing else.

        Returns
        -------
        PreparedStatus
            What the marker recorded under :attr:`status_fields`, or every one of those
            fields ``None`` when nothing is prepared. A directory an interrupted run left
            behind reads as not prepared here; :meth:`prepare` is where that becomes an error.
        """
        source = self.source()
        record = completion.read_record(source.directory)
        if record is None or not source.path.exists():
            return PreparedStatus(
                path=source.path,
                prepared=False,
                fields=dict.fromkeys(json_key for _, json_key in self.status_fields),
            )
        return PreparedStatus(
            path=source.path,
            prepared=True,
            fields={
                json_key: record.details.get(record_key)
                for record_key, json_key in self.status_fields
            },
            completed_at=record.completed_at,
        )

    def read(self, path: Path) -> pd.DataFrame:
        """Return one stored file as a frame, read once and then held.

        Parameters
        ----------
        path : pathlib.Path
            The stored file, which exists. Whether it does is the caller's to say, in the
            words its own missing-set error is written in.

        Returns
        -------
        pandas.DataFrame
            What :attr:`read_table` made of it. **Do not mutate what comes back** — every
            caller shares it.
        """
        return _read_table(path, self.read_table)

    def clear_cache(self) -> None:
        """Forget every file read so far, so the next call re-reads from disk."""
        _read_table.cache_clear()

    @functools.cached_property
    def app(self) -> typer.Typer:
        """The set's two commands, ``prepare`` and ``status``, each taking ``--json``.

        Built once, and bound as a module attribute by the set that declared it, so the root
        CLI mounts this lane the way it mounts every other.
        """
        app = typer.Typer(help=self.app_help, no_args_is_help=True)

        @app.command("prepare", help=self.prepare_help)
        def prepare_command(
            json: bool = typer.Option(False, "--json", help="Emit JSON instead of plain text."),
        ) -> None:
            try:
                self.prepare(progressbar=not json)
            except _ERRORS as err:
                typer.echo(f"error: {err}", err=True)
                raise typer.Exit(code=1) from err
            _render(self.status(), json=json)

        @app.command("status", help=self.status_help)
        def status_command(
            json: bool = typer.Option(False, "--json", help="Emit JSON instead of plain text."),
        ) -> None:
            try:
                found = self.status()
            except _ERRORS as err:
                typer.echo(f"error: {err}", err=True)
                raise typer.Exit(code=1) from err
            _render(found, json=json)

        return app


@functools.cache
def _read_table(path: Path, read: Callable[[Path], pd.DataFrame]) -> pd.DataFrame:
    """Read one stored file, keyed by the file and by what read it.

    Keyed by the path rather than cached on a nullary call, so a process that re-points the
    **Data dir** reads the set it now names instead of the one it read first.
    """
    return read(path)


def _render(found: PreparedStatus, *, json: bool) -> None:
    """Print one status, as JSON or as one ``key: value`` line each."""
    if json:
        typer.echo(_json.dumps(found.as_json()))
        return
    for key, value in found.as_json().items():
        typer.echo(f"{key}: {value}")
