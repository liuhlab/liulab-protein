"""The SAE feature descriptions: what a feature index means, as a **Prepared set**.

A **SAE activation** carries feature indices and nothing that says what any of them is. The
publisher generates a label and a description for every feature in the codebook and serves
the whole set in one unauthenticated request, so it lands here the way **SIFTS** does —
through :mod:`genome.store.prepared`, which owns the fetch, the working area, the staged
rename, the digest and the **Completion marker**. What this module declares is a URL, a
directory name and a reader.

**One request for the whole codebook, never one per feature.** The publisher also serves a
richer record per feature; enumerating the set that way is one request per index, which
*bulk, not per-ID* forbids and the suite's network guard makes untestable.

**Descriptions are published for one checkpoint.** :func:`descriptions` takes an SAE slug and
raises for the others, so a caller analysing with a checkpoint that has none finds out by
name rather than through a join that answers nothing.

**Reached by its own function, and the caller does the join.** The frame comes back indexed
by feature index, in the dtype :meth:`protein.embed.SaeActivation.index_dtype` gives, so
``frame.loc[activation.indices[residue]]`` names the features that fired — and a frozen numpy
value object never touches the filesystem. The set is worth having on its own: the feature
vocabulary is browsable by someone who never runs the backbone.

**Nothing is pinned and nothing is checked for staleness.** The publisher calls this an alpha
interface and keeps no archive, so the digest of what was stored is recorded rather than
compared. Refresh is :attr:`~genome.store.prepared.PreparedSource.repair`: delete and rebuild.

Examples
--------
>>> from protein.embed.esm.features import app
>>> [command.name for command in app.registered_commands]
['prepare', 'status']
>>> COLUMNS
('feature_index', 'label', 'description')
"""

from __future__ import annotations

import functools
import json as _json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import typer
from genome.store import completion, prepared

from protein.embed.esm.sae import SaeActivation
from protein.store import protein_data_dir

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Mapping

__all__ = [
    "CODEBOOK_SIZE",
    "COLUMNS",
    "DESCRIBED_SAE",
    "DESCRIPTION",
    "ESM_SUBDIR",
    "FEATURES_KIND",
    "FEATURES_SUBDIR",
    "FEATURES_URL",
    "INDEX_DTYPE",
    "PREPARE_COMMAND",
    "READ_DTYPES",
    "ROWS_KEY",
    "STORED_NAME",
    "SaeFeaturesFormatError",
    "SaeFeaturesNotDownloadedError",
    "SaeFeaturesStatus",
    "app",
    "clear_cache",
    "descriptions",
    "features_data_dir",
    "prepare",
    "read_features",
    "source",
    "status",
]

#: Where the publisher's whole codebook is fetched from, in one request and with no token.
FEATURES_URL = "https://biohub.ai/esm/protein/api/v1alpha1/features"

#: The provider directory under :func:`protein.store.protein_data_dir`. The **Data dir**
#: groups by what produced the bytes, and these were produced by the ESM project.
ESM_SUBDIR = "esm"

#: This set's directory under :data:`ESM_SUBDIR`.
FEATURES_SUBDIR = "sae-features"

#: The file the set is read from: the publisher's three fields as a gzipped TSV.
STORED_NAME = "sae_features.tsv.gz"

#: What the **Completion marker** calls what it recorded.
FEATURES_KIND = "esm-sae-features"

#: The one SAE checkpoint these describe. The other published slugs have no descriptions at
#: all, so :func:`descriptions` raises for them rather than answering nothing.
DESCRIBED_SAE = "6b-layer60-k64-cb16384"

#: How many features that checkpoint's codebook holds, which bounds every index in the set
#: and so decides the type they are stored in.
CODEBOOK_SIZE = 16384

#: The type a feature index is stored and read back in — the same one
#: :attr:`protein.embed.SaeActivation.indices` uses, asked of the class rather than restated.
INDEX_DTYPE = SaeActivation.index_dtype(CODEBOOK_SIZE)

#: How an error names this set.
DESCRIPTION = "the ESM-C SAE feature descriptions"

#: The call that prepares it, quoted into every error a caller repairs by running it.
PREPARE_COMMAND = "protein esm features prepare"

#: The key the publisher's envelope carries its rows under.
ROWS_KEY = "data"

#: What is stored, and all of what is stored: the publisher's three fields, in the
#: publisher's own spelling, so a re-shaped record fails here rather than being read into the
#: wrong columns.
COLUMNS: tuple[str, ...] = ("feature_index", "label", "description")

#: What each stored column is read back as. Spelled out rather than inferred, so the index
#: comes back as the unsigned type it was written as. ``Hashable``-keyed because that is the
#: mapping ``read_csv`` takes.
READ_DTYPES: dict[Hashable, Any] = {
    COLUMNS[0]: INDEX_DTYPE,
    **dict.fromkeys(COLUMNS[1:], "string"),
}

#: What a failed call raises: a set that is not here or an interrupted run is a
#: ``RuntimeError``, a document this reader cannot parse a ``ValueError``, and a file that
#: went away under the read an ``OSError``.
_FEATURES_ERRORS = (ValueError, OSError, RuntimeError)


class SaeFeaturesNotDownloadedError(prepared.PreparedSetNotDownloadedError):
    """The descriptions are not prepared on this machine and could not be fetched.

    Examples
    --------
    >>> issubclass(SaeFeaturesNotDownloadedError, RuntimeError)
    True
    """


class SaeFeaturesFormatError(ValueError):
    """What arrived is not the document this reader knows how to read.

    Raised before anything is placed, so a response the publisher re-shaped never lands on
    disk to be read back as a finished set.

    Examples
    --------
    >>> issubclass(SaeFeaturesFormatError, ValueError)
    True
    """


def features_data_dir() -> Path:
    """Return this set's directory under the lab **Data dir**.

    Returns
    -------
    pathlib.Path
        ``<LIULAB_DATA>/protein/esm/sae-features``. Nothing is created by asking.

    Examples
    --------
    >>> import os
    >>> os.environ["LIULAB_DATA"] = "/scratch/liulab"
    >>> features_data_dir()
    PosixPath('/scratch/liulab/protein/esm/sae-features')
    >>> del os.environ["LIULAB_DATA"]
    """
    return protein_data_dir() / ESM_SUBDIR / FEATURES_SUBDIR


def source() -> prepared.PreparedSource:
    """Return what this **Prepared set** declares, and the whole of what it declares.

    A function rather than a constant because the directory is read from the environment at
    call time, so a process that re-points ``LIULAB_DATA`` gets the new root.

    Returns
    -------
    genome.store.prepared.PreparedSource
        The URL, the directory, the reader and the error class.
        :attr:`~genome.store.prepared.PreparedSource.checksum` is ``None``: the publisher
        keeps no archive, so the digest of what was stored is recorded instead.

    Examples
    --------
    >>> import os
    >>> os.environ["LIULAB_DATA"] = "/scratch/liulab"
    >>> source().path.name
    'sae_features.tsv.gz'
    >>> source().checksum is None
    True
    >>> del os.environ["LIULAB_DATA"]
    """
    return prepared.PreparedSource(
        url=FEATURES_URL,
        directory=features_data_dir(),
        stored_name=STORED_NAME,
        kind=FEATURES_KIND,
        name=DESCRIBED_SAE,
        prepare_command=PREPARE_COMMAND,
        description=DESCRIPTION,
        read=read_features,
        not_downloaded=SaeFeaturesNotDownloadedError,
        download_name="features.json",
        details={"publisher": "Biohub", "sae": DESCRIBED_SAE},
    )


def read_features(lines: Iterator[str], staged: Path, *, origin: str) -> Mapping[str, Any]:
    """Turn the publisher's response into the stored table, and say what was read.

    The whole of what this package adds to :func:`genome.store.prepared.prepare`: three
    fields per row, sorted by feature index, gzipped.

    Parameters
    ----------
    lines : iterator of str
        The response as it was served. It is one JSON document, so the lines are joined
        before anything is read out of them.
    staged : pathlib.Path
        Where to write the table, inside the working area.
    origin : str
        What the lines came out of, so a refusal names the file it refused.

    Returns
    -------
    mapping of str to object
        Which SAE the set describes, how wide its codebook is and how many rows were
        written. All of it lands in the **Completion marker**'s details.

    Raises
    ------
    SaeFeaturesFormatError
        If the document is not the envelope this reader knows, if a row is missing one of
        the three fields, if an index falls outside the codebook or repeats, or if the
        response carries no rows at all.

    Examples
    --------
    >>> from pathlib import Path
    >>> read_features(lines, Path("/tmp/f.tsv.gz"), origin="x")   # doctest: +SKIP
    {'sae': '6b-layer60-k64-cb16384', 'codebook_size': 16384, 'rows': 16384}
    """
    rows = _parse(lines, origin=origin)
    frame = pd.DataFrame({name: [row[name] for row in rows] for name in COLUMNS}).astype(
        {COLUMNS[0]: INDEX_DTYPE, **dict.fromkeys(COLUMNS[1:], "string")}
    )
    frame = frame.sort_values(COLUMNS[0], kind="stable", ignore_index=True)
    staged.parent.mkdir(parents=True, exist_ok=True)
    # `mtime=0`, so the same rows give byte-identical files and the recorded digest is a
    # fact about them rather than about the clock.
    frame.to_csv(staged, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    return {"sae": DESCRIBED_SAE, "codebook_size": CODEBOOK_SIZE, "rows": len(frame)}


def prepare(*, progressbar: bool = True) -> prepared.Prepared:
    """Fetch and store the descriptions, or return the ones already here.

    Parameters
    ----------
    progressbar : bool, default True
        Show the download's progress bar. Nothing is drawn when the set is already there.

    Returns
    -------
    genome.store.prepared.Prepared
        The stored table and its **Completion marker**.

    Raises
    ------
    SaeFeaturesNotDownloadedError
        If the bytes are not here and this machine could not fetch them.
    SaeFeaturesFormatError
        If what arrived is not the document this reader knows.
    genome.store.completion.RegistrationError
        If the directory holds an interrupted run, or disagrees with its marker.

    Examples
    --------
    >>> prepare(progressbar=False)                       # doctest: +SKIP
    Prepared(path=PosixPath('/scratch/liulab/protein/esm/sae-features/...'), ...)
    """
    result = prepared.prepare(source(), progressbar=progressbar)
    # The stored file may have just been replaced, and a cache outliving the bytes it read
    # would answer from a set that is gone.
    clear_cache()
    return result


@dataclass(frozen=True)
class SaeFeaturesStatus:
    """What :func:`status` found, without touching the network.

    Attributes
    ----------
    path : pathlib.Path
        Where the stored table is, whether or not it exists.
    prepared : bool
        Whether the set is finished here — a marker beside a file that is present.
    sae : str or None
        Which SAE checkpoint the descriptions belong to.
    codebook_size : int or None
        How many features that checkpoint has.
    rows : int or None
        How many descriptions were stored.
    completed_at : str or None
        When this machine finished preparing them, ISO-8601 in UTC.

    Examples
    --------
    >>> from pathlib import Path
    >>> SaeFeaturesStatus(path=Path("/tmp/x.tsv.gz"), prepared=False).rows is None
    True
    """

    path: Path
    prepared: bool
    sae: str | None = None
    codebook_size: int | None = None
    rows: int | None = None
    completed_at: str | None = None

    def as_json(self) -> dict[str, Any]:
        """Return this status as the mapping ``--json`` prints.

        Returns
        -------
        dict
            One key per attribute, with :attr:`path` as a string so the result is JSON.

        Examples
        --------
        >>> from pathlib import Path
        >>> SaeFeaturesStatus(path=Path("/tmp/x.tsv.gz"), prepared=False).as_json()["prepared"]
        False
        """
        return {
            "path": str(self.path),
            "prepared": self.prepared,
            "sae": self.sae,
            "codebook_size": self.codebook_size,
            "rows": self.rows,
            "completed_at": self.completed_at,
        }


def status() -> SaeFeaturesStatus:
    """Report what is on disk here, reading the marker and nothing else.

    **Offline, and it stays offline.** There is no archive to compare against and the lab's
    compute nodes have no network, so this says what is here and never whether the publisher
    has changed it.

    Returns
    -------
    SaeFeaturesStatus
        What the marker recorded, or a status with :attr:`~SaeFeaturesStatus.prepared`
        ``False`` when nothing is prepared. A directory an interrupted run left behind reads
        as not prepared here; :func:`prepare` is where that becomes an error.

    Examples
    --------
    >>> status()                                         # doctest: +SKIP
    SaeFeaturesStatus(path=PosixPath('...'), prepared=True, sae='6b-layer60-k64-cb16384', ...)
    """
    directory = features_data_dir()
    path = directory / STORED_NAME
    record = completion.read_record(directory)
    if record is None or not path.exists():
        return SaeFeaturesStatus(path=path, prepared=False)
    details = record.details
    return SaeFeaturesStatus(
        path=path,
        prepared=True,
        sae=details.get("sae"),
        codebook_size=details.get("codebook_size"),
        rows=details.get("rows"),
        completed_at=record.completed_at,
    )


def descriptions(sae: str) -> pd.DataFrame:
    """Return what every feature of ``sae``'s codebook is a description of.

    Indexed by feature index, so joining it against an activation is
    ``frame.loc[activation.indices[residue]]``. Read once per file and then held — **do not
    mutate what comes back**, since every caller shares it.

    Parameters
    ----------
    sae : str
        An SAE checkpoint slug. Only :data:`DESCRIBED_SAE` has published descriptions.

    Returns
    -------
    pandas.DataFrame
        ``label`` and ``description``, indexed by ``feature_index`` in
        :data:`INDEX_DTYPE` and sorted by it.

    Raises
    ------
    ValueError
        If ``sae`` is any other slug. The descriptions are hypotheses generated against one
        checkpoint's codebook and mean nothing against another's.
    SaeFeaturesNotDownloadedError
        If the set is not prepared here.

    Examples
    --------
    >>> descriptions("6b-layer60-k64-cb16384").loc[0, "label"]    # doctest: +SKIP
    'Nudix N-terminal substrate-binding loop'
    """
    if sae != DESCRIBED_SAE:
        raise ValueError(
            f"no feature descriptions are published for {sae!r}. They exist for "
            f"{DESCRIBED_SAE!r} alone, and a description generated against one codebook says "
            f"nothing about the features of another."
        )
    path = features_data_dir() / STORED_NAME
    if not path.exists():
        raise SaeFeaturesNotDownloadedError(
            f"{DESCRIPTION} are not prepared here: {path} does not exist. "
            f"{prepared.login_node_help(PREPARE_COMMAND)}"
        )
    return _read_table(path)


def clear_cache() -> None:
    """Forget every table read so far, so the next call re-reads from disk.

    Examples
    --------
    >>> clear_cache()
    """
    _read_table.cache_clear()


@functools.cache
def _read_table(path: Path) -> pd.DataFrame:
    """Read one stored table, keyed by the file it read.

    Keyed by the path rather than cached on a nullary call, so a process that re-points the
    **Data dir** reads the set it now names instead of the one it read first.

    ``keep_default_na=False`` so a label the publisher wrote is never read as a missing
    value; spelled this way and not as ``na_filter=False``, which the pyarrow engine
    silently ignores.
    """
    frame = pd.read_csv(
        path,
        sep="\t",
        engine="pyarrow",
        dtype=READ_DTYPES,
        keep_default_na=False,
        na_values=[],
    )
    return frame.set_index(COLUMNS[0])


def _parse(lines: Iterator[str], *, origin: str) -> list[dict[str, Any]]:
    """Read the publisher's envelope into rows, refusing anything else in this shape."""
    try:
        document = _json.loads("".join(lines))
    except ValueError as error:
        raise SaeFeaturesFormatError(
            f"{origin} is not JSON, and this set is served as one JSON document. Prepare it "
            f"again with `{source().repair}`."
        ) from error
    if not isinstance(document, dict) or not isinstance(document.get(ROWS_KEY), list):
        raise SaeFeaturesFormatError(
            f"{origin} is not an object carrying its rows under {ROWS_KEY!r}. The publisher "
            f"re-shaped the response; read anyway, every column would hold the wrong thing."
        )
    rows = [
        _row(raw, number=number, origin=origin) for number, raw in enumerate(document[ROWS_KEY])
    ]
    if not rows:
        raise SaeFeaturesFormatError(
            f"{origin} carries no descriptions, so every feature would come back unnamed. "
            f"That is not a release. Prepare the set again with `{source().repair}`."
        )
    seen = {row[COLUMNS[0]] for row in rows}
    if len(seen) != len(rows):
        raise SaeFeaturesFormatError(
            f"{origin} names a feature more than once, and a description is joined by index: "
            f"a repeated index would answer one feature with several rows."
        )
    return rows


def _row(raw: Any, *, number: int, origin: str) -> dict[str, Any]:
    """Read one record, and refuse one that is not the three fields this reader keeps."""
    if not isinstance(raw, dict) or not set(COLUMNS) <= set(raw):
        raise SaeFeaturesFormatError(
            f"{origin} record {number} does not carry {COLUMNS}. Every description is a "
            f"feature index, a label and the text; a record missing one names nothing."
        )
    index = raw[COLUMNS[0]]
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < CODEBOOK_SIZE:
        raise SaeFeaturesFormatError(
            f"{origin} record {number} is feature {index!r}, and this codebook runs 0 to "
            f"{CODEBOOK_SIZE - 1}. An index outside it does not fit the type the activations "
            f"store, so it would be read back as a different feature."
        )
    return {name: raw[name] for name in COLUMNS}


app = typer.Typer(
    help="The ESM-C SAE feature descriptions: prepare them, and say what is here.",
    no_args_is_help=True,
)


@app.command("prepare")
def prepare_command(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of plain text."),
) -> None:
    """Download the SAE feature descriptions and store them under the lab data dir.

    The one step in this lane that needs the network, so it runs on a login node. Already
    prepared, it fetches nothing and reports what is there.
    """
    try:
        prepare(progressbar=not json)
    except _FEATURES_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    _render(status(), json=json)


@app.command("status")
def status_command(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of plain text."),
) -> None:
    """Say whether the SAE feature descriptions are here, without touching the network.

    Nothing is checked against the publisher, which keeps no archive to compare with.
    """
    try:
        found = status()
    except _FEATURES_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    _render(found, json=json)


def _render(found: SaeFeaturesStatus, *, json: bool) -> None:
    """Print one status, as JSON or as one ``key: value`` line each."""
    if json:
        typer.echo(_json.dumps(found.as_json()))
        return
    for key, value in found.as_json().items():
        typer.echo(f"{key}: {value}")
