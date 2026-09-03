"""SIFTS: the PDB-UniProt map, and the one join between this package's two namespaces.

A **Protein** is addressed by a UniProt accession, a **Structure** by a PDB id, and SIFTS —
the EBI's re-curated chain-to-accession map — is the only join between them, many-to-many in
both directions. Every mmCIF carries its own ``_struct_ref_seq``, which holds the depositor's
reference frozen at deposition and disagrees; ``Chain.uniprot`` reads this table alone.

**A Prepared set, not a Database.** :mod:`genome.store.prepared` owns the fetch, the working
area, the staged rename, the digest and the **Completion marker**, and what this module
declares is a URL, a directory name and a reader. A **Database** is immutable ffindex files
searched by a subprocess, and SIFTS is none of those.

**Nothing is pinned and nothing is checked for staleness**, because the publisher keeps no
archive and overwrites the file in place, so a digest taken today would reject every release
after it. The reader records the release line it read instead, and :func:`status` prints it
back, offline. Refresh is :attr:`~genome.store.prepared.PreparedSource.repair`: delete and
rebuild.

**Two verbs, one per direction, over a cached table.** Reach them through the module —
``from protein import sifts``, then ``sifts.accessions_for(...)`` — never by importing the
name, so one ``monkeypatch.setattr`` reaches every caller. Their shapes follow the
cardinalities: a chain usually has one accession, so :func:`accessions_for` is a tuple; an
accession reaches many chains, so :func:`structures_for` is a frame.

**Both residue ranges come back verbatim and no offset is computed**, because for some
segments none is definable.

Examples
--------
>>> from protein.sifts import app
>>> [command.name for command in app.registered_commands]
['prepare', 'status']
>>> COLUMNS
('pdb', 'chain', 'accession', 'res_beg', 'res_end', 'sp_beg', 'sp_end')
"""

from __future__ import annotations

import functools
import json as _json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import typer
from genome.store import completion, prepared

from protein.store import protein_data_dir

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Mapping

__all__ = [
    "COLUMNS",
    "DESCRIPTION",
    "PREPARE_COMMAND",
    "READ_DTYPES",
    "SIFTS_KIND",
    "SIFTS_NAME",
    "SIFTS_SUBDIR",
    "SIFTS_URL",
    "SORT_COLUMNS",
    "SOURCE_COLUMNS",
    "STORED_NAME",
    "SiftsFormatError",
    "SiftsNotDownloadedError",
    "SiftsStatus",
    "accessions_for",
    "app",
    "clear_cache",
    "prepare",
    "read_sifts",
    "sifts_data_dir",
    "source",
    "status",
    "structures_for",
    "table",
]

#: Where the publisher's file is fetched from. The flat-file tree carries the current
#: release only, so this URL names no version and never will.
SIFTS_URL = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz"

#: This set's directory under :func:`protein.store.protein_data_dir`.
SIFTS_SUBDIR = "sifts"

#: The file the set is read from: the publisher's nine columns cut to seven, sorted and
#: gzipped.
STORED_NAME = "pdb_chain_uniprot.tsv.gz"

#: What the **Completion marker** calls what it recorded.
SIFTS_KIND = "sifts"

#: The name this set is addressed by in its own tree.
SIFTS_NAME = "pdb_chain_uniprot"

#: How an error names this set.
DESCRIPTION = "the SIFTS PDB-UniProt map"

#: The call that prepares it, quoted into every error a caller repairs by running it.
PREPARE_COMMAND = "protein sifts prepare"

#: The publisher's columns, in the publisher's order and spelling. Checked against the file's
#: own header line, so a re-shaped file fails here rather than being sliced into the wrong
#: columns.
SOURCE_COLUMNS: tuple[str, ...] = (
    "PDB",
    "CHAIN",
    "SP_PRIMARY",
    "RES_BEG",
    "RES_END",
    "PDB_BEG",
    "PDB_END",
    "SP_BEG",
    "SP_END",
)

#: What is stored, and all of what is stored. ``PDB_BEG``/``PDB_END`` — the author numbering
#: — are dropped: they are often blank, and an mmCIF's ``_pdbx_poly_seq_scheme`` gives the
#: same mapping per residue and with insertion codes.
COLUMNS: tuple[str, ...] = ("pdb", "chain", "accession", "res_beg", "res_end", "sp_beg", "sp_end")

#: The columns the stored frame is sorted by. Accession first, the direction
#: :func:`structures_for` takes.
SORT_COLUMNS: tuple[str, ...] = ("accession", "pdb", "chain")

#: What each stored column is read back as. Spelled out rather than inferred, so the residue
#: bounds come back as the ``int32`` they were written as. ``Hashable``-keyed because that is
#: the mapping ``read_csv`` takes.
READ_DTYPES: dict[Hashable, Any] = {
    **dict.fromkeys(COLUMNS[:3], "string"),
    **dict.fromkeys(COLUMNS[3:], "int32"),
}

#: The publisher's first line, the only thing in the file that says which release this is
#: and, with no archive to pin against, the whole reproducibility record.
_HEADER_RE = re.compile(
    r"^#\s*(?P<released>.+?)\s*\|\s*PDB:\s*(?P<pdb>\S+)\s*\|\s*UniProt:\s*(?P<uniprot>\S+)\s*$"
)

#: What a failed SIFTS call raises: a set that is not here or an interrupted run is a
#: ``RuntimeError``, a file this reader cannot slice a ``ValueError``, and a file that went
#: away under the read an ``OSError``.
_SIFTS_ERRORS = (ValueError, OSError, RuntimeError)


class SiftsNotDownloadedError(prepared.PreparedSetNotDownloadedError):
    """The SIFTS map is not prepared on this machine and could not be fetched.

    Distinct from an empty answer, and deliberately so: ``()`` from :func:`accessions_for`
    means the chain genuinely has no protein, and conflating the two would let a script map
    nothing and look like it worked.

    Examples
    --------
    >>> issubclass(SiftsNotDownloadedError, RuntimeError)
    True
    """


class SiftsFormatError(ValueError):
    """What arrived is not the file this reader knows how to slice.

    Raised before anything is placed, so a file the publisher re-shaped never lands on disk
    to be read back as a finished set.

    Examples
    --------
    >>> issubclass(SiftsFormatError, ValueError)
    True
    """


def sifts_data_dir() -> Path:
    """Return this set's directory under the lab **Data dir**.

    Returns
    -------
    pathlib.Path
        ``<LIULAB_DATA>/protein/sifts``. Nothing is created by asking.

    Examples
    --------
    >>> import os
    >>> os.environ["LIULAB_DATA"] = "/scratch/liulab"
    >>> sifts_data_dir()
    PosixPath('/scratch/liulab/protein/sifts')
    >>> del os.environ["LIULAB_DATA"]
    """
    return protein_data_dir() / SIFTS_SUBDIR


def source() -> prepared.PreparedSource:
    """Return what this **Prepared set** declares, and the whole of what it declares.

    A function rather than a constant because the directory is read from the environment
    at call time, so a process that re-points ``LIULAB_DATA`` gets the new root.

    Returns
    -------
    genome.store.prepared.PreparedSource
        The URL, the directory, the reader and the error class.
        :attr:`~genome.store.prepared.PreparedSource.checksum` is ``None``, because the
        publisher overwrites this file in place; the digest of what was stored is recorded
        instead.

    Examples
    --------
    >>> import os
    >>> os.environ["LIULAB_DATA"] = "/scratch/liulab"
    >>> source().path
    PosixPath('/scratch/liulab/protein/sifts/pdb_chain_uniprot.tsv.gz')
    >>> source().checksum is None
    True
    >>> del os.environ["LIULAB_DATA"]
    """
    return prepared.PreparedSource(
        url=SIFTS_URL,
        directory=sifts_data_dir(),
        stored_name=STORED_NAME,
        kind=SIFTS_KIND,
        name=SIFTS_NAME,
        prepare_command=PREPARE_COMMAND,
        description=DESCRIPTION,
        read=read_sifts,
        not_downloaded=SiftsNotDownloadedError,
        details={"publisher": "EBI PDBe"},
    )


def read_sifts(lines: Iterator[str], staged: Path, *, origin: str) -> Mapping[str, Any]:
    """Turn the publisher's lines into the stored slice, and say what was read.

    The whole of what this package adds to :func:`genome.store.prepared.prepare`: seven of
    the publisher's nine columns, sorted by :data:`SORT_COLUMNS`, and the release line for
    the marker to record.

    Parameters
    ----------
    lines : iterator of str
        The publisher's unpacked lines, line endings and all. They are mixed — CRLF and LF
        in one file — so every line is stripped of both.
    staged : pathlib.Path
        Where to write the slice, inside the working area.
    origin : str
        What the lines came out of, so a refusal names the file it refused.

    Returns
    -------
    mapping of str to object
        The release line verbatim, its date, the PDB and UniProt release numbers, and how
        many rows were written. All of it lands in the **Completion marker**'s details.

    Raises
    ------
    SiftsFormatError
        If the release line, the column header or a data row is not the shape this reader
        knows, or if the file carries no rows at all.

    Examples
    --------
    >>> from pathlib import Path
    >>> read_sifts(lines, Path("/tmp/sifts.tsv.gz"), origin="x")   # doctest: +SKIP
    {'sifts_header': '# 2026/08/30 - 13:24 | PDB: 35.26 | UniProt: 2026.03', ...}
    """
    header = _next_line(lines, origin=origin, expected="the release line")
    released, pdb_release, uniprot_release = _parse_header(header, origin=origin)
    _check_columns(_next_line(lines, origin=origin, expected="the column header"), origin=origin)

    rows: dict[str, list[Any]] = {name: [] for name in COLUMNS}
    for number, line in enumerate(lines, start=3):
        stripped = line.rstrip("\r\n")
        if not stripped:
            continue
        _add_row(rows, stripped, number=number, origin=origin)
    if not rows["pdb"]:
        raise SiftsFormatError(
            f"{origin} carries no mapping rows, so every query against it would answer "
            f"nothing. That is not a release. Prepare the set again with `{source().repair}`."
        )

    frame = pd.DataFrame(rows).astype(dict.fromkeys(COLUMNS[3:], "int32"))
    # Stable, so the several segments of one triple keep the publisher's order, which is the
    # order their residue ranges ascend in.
    frame = frame.sort_values(list(SORT_COLUMNS), kind="stable", ignore_index=True)
    staged.parent.mkdir(parents=True, exist_ok=True)
    # `mtime=0`, so the same rows give byte-identical files and the recorded digest is a fact
    # about them rather than about the clock.
    frame.to_csv(staged, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    return {
        "sifts_header": header,
        "sifts_released": released,
        "pdb_release": pdb_release,
        "uniprot_release": uniprot_release,
        "rows": len(frame),
    }


def prepare(*, progressbar: bool = True) -> prepared.Prepared:
    """Fetch and store the map, or return the one already here.

    Parameters
    ----------
    progressbar : bool, default True
        Show the download's progress bar. Nothing is drawn when the set is already there.

    Returns
    -------
    genome.store.prepared.Prepared
        The stored slice and its **Completion marker**.

    Raises
    ------
    SiftsNotDownloadedError
        If the bytes are not here and this machine could not fetch them.
    SiftsFormatError
        If what arrived is not the file this reader knows.
    genome.store.completion.RegistrationError
        If the directory holds an interrupted run, or disagrees with its marker.

    Examples
    --------
    >>> prepare(progressbar=False)                       # doctest: +SKIP
    Prepared(path=PosixPath('/scratch/liulab/protein/sifts/pdb_chain_uniprot.tsv.gz'), ...)
    """
    result = prepared.prepare(source(), progressbar=progressbar)
    # The stored file may have just been replaced, and a cache outliving the bytes it read
    # would answer from a release that is gone.
    clear_cache()
    return result


@dataclass(frozen=True)
class SiftsStatus:
    """What :func:`status` found, without touching the network.

    Attributes
    ----------
    path : pathlib.Path
        Where the stored slice is, whether or not it exists.
    prepared : bool
        Whether the set is finished here — a marker beside a file that is present.
    released : str or None
        When the publisher cut this release, as its own first line spells it.
    pdb_release : str or None
        The PDB release the map was built against.
    uniprot_release : str or None
        The UniProt release.
    rows : int or None
        How many mapping rows were stored.
    completed_at : str or None
        When this machine finished preparing it, ISO-8601 in UTC.

    Examples
    --------
    >>> from pathlib import Path
    >>> SiftsStatus(path=Path("/tmp/x.tsv.gz"), prepared=False).rows is None
    True
    """

    path: Path
    prepared: bool
    released: str | None = None
    pdb_release: str | None = None
    uniprot_release: str | None = None
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
        >>> SiftsStatus(path=Path("/tmp/x.tsv.gz"), prepared=False).as_json()["prepared"]
        False
        """
        return {
            "path": str(self.path),
            "prepared": self.prepared,
            "released": self.released,
            "pdb_release": self.pdb_release,
            "uniprot_release": self.uniprot_release,
            "rows": self.rows,
            "completed_at": self.completed_at,
        }


def status() -> SiftsStatus:
    """Report what is on disk here, reading the marker and nothing else.

    **Offline, and it stays offline.** There is no archive to compare against and the lab's
    compute nodes have no network, so this says which release is here and never whether a
    newer one exists.

    Returns
    -------
    SiftsStatus
        The recorded release, or a status with :attr:`~SiftsStatus.prepared` ``False`` when
        nothing is prepared. A directory an interrupted run left behind reads as not
        prepared here; :func:`prepare` is where that becomes an error.

    Examples
    --------
    >>> status()                                         # doctest: +SKIP
    SiftsStatus(path=PosixPath('...'), prepared=True, released='2026/08/30 - 13:24', ...)
    """
    directory = sifts_data_dir()
    path = directory / STORED_NAME
    record = completion.read_record(directory)
    if record is None or not path.exists():
        return SiftsStatus(path=path, prepared=False)
    details = record.details
    return SiftsStatus(
        path=path,
        prepared=True,
        released=details.get("sifts_released"),
        pdb_release=details.get("pdb_release"),
        uniprot_release=details.get("uniprot_release"),
        rows=details.get("rows"),
        completed_at=record.completed_at,
    )


def table() -> pd.DataFrame:
    """Return the whole map, read once per file and then held.

    **Do not mutate what comes back** — every caller shares it; the two verbs each hand back
    a slice of their own.

    Returns
    -------
    pandas.DataFrame
        :data:`COLUMNS`, sorted by :data:`SORT_COLUMNS`.

    Raises
    ------
    SiftsNotDownloadedError
        If the set is not prepared here.

    Examples
    --------
    >>> list(table().columns)                            # doctest: +SKIP
    ['pdb', 'chain', 'accession', 'res_beg', 'res_end', 'sp_beg', 'sp_end']
    """
    path = sifts_data_dir() / STORED_NAME
    if not path.exists():
        raise SiftsNotDownloadedError(
            f"{DESCRIPTION} is not prepared here: {path} does not exist. "
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


def structures_for(accession: str) -> pd.DataFrame:
    """Return every PDB chain segment SIFTS maps ``accession`` to.

    A frame and not a list of ids: an entry id alone does not say *which chain* to fetch
    coordinates for, and a well-studied accession reaches more chains than there is reason to
    build objects for.

    Parameters
    ----------
    accession : str
        A UniProt accession, in either case. SIFTS stores them upper-case, so that is what
        the input is folded to.

    Returns
    -------
    pandas.DataFrame
        :data:`COLUMNS`, one row per segment, in the stored order. Empty when SIFTS carries
        nothing for this accession, which is the true answer for a protein no structure has
        been solved for.

    Raises
    ------
    SiftsNotDownloadedError
        If the set is not prepared here.

    Examples
    --------
    >>> structures_for("P0CG48").iloc[0]["pdb"]          # doctest: +SKIP
    '11sy'
    """
    frame = table()
    wanted = accession.strip().upper()
    return frame.loc[frame["accession"] == wanted].reset_index(drop=True)


def accessions_for(pdb: str, chain: str) -> tuple[str, ...]:
    """Return the UniProt accessions SIFTS maps one PDB chain to.

    A tuple and never a scalar: a chain may carry more than one accession, and a surface that
    answered with the first would be wrong without saying so.

    Parameters
    ----------
    pdb : str
        A four-character PDB entry id, in either case. SIFTS stores them lower-case, so
        that is what the input is folded to.
    chain : str
        The author chain label — ``auth_asym_id``, which is what SIFTS keys on. **Not
        folded**: ``10eg`` carries both an ``A`` and an ``a``, so case is part of the name.

    Returns
    -------
    tuple of str
        The accessions, deduplicated and in accession order — the order :data:`SORT_COLUMNS`
        puts them in, not the publisher's row order, which is by residue range. ``()`` when
        this chain has no protein, and also for an id SIFTS does not carry at all, which is
        what a chain of a ``Structure.from_file`` that is not a PDB entry gets.

    Raises
    ------
    SiftsNotDownloadedError
        If the set is not prepared here.

    Examples
    --------
    >>> accessions_for("1UBQ", "A")                      # doctest: +SKIP
    ('P0CG48',)
    """
    frame = table()
    entry = pdb.strip().lower()
    label = chain.strip()
    rows = frame.loc[(frame["pdb"] == entry) & (frame["chain"] == label)]
    return tuple(str(value) for value in dict.fromkeys(rows["accession"]))


@functools.cache
def _read_table(path: Path) -> pd.DataFrame:
    """Read one stored slice, keyed by the file it read.

    Keyed by the path rather than cached on a nullary call, so a process that re-points the
    **Data dir** reads the set it now names instead of the one it read first.

    ``keep_default_na=False`` is load-bearing: **``NA`` is a real chain label**, which pandas'
    default missing-value list would read as a missing value and lose. Spelled this way and
    not as ``na_filter=False``, which the pyarrow engine silently ignores.
    """
    return pd.read_csv(
        path,
        sep="\t",
        engine="pyarrow",
        dtype=READ_DTYPES,
        keep_default_na=False,
        na_values=[],
    )


def _next_line(lines: Iterator[str], *, origin: str, expected: str) -> str:
    """Return the next line stripped of its ending, or say which one was missing."""
    line = next(lines, None)
    if line is None:
        raise SiftsFormatError(
            f"{origin} ends before {expected}. SIFTS publishes a release line, a column "
            f"header and then the rows; this file is truncated or is not that file."
        )
    return line.rstrip("\r\n")


def _parse_header(header: str, *, origin: str) -> tuple[str, str, str]:
    """Return the release date and the two release numbers from the publisher's first line.

    Examples
    --------
    >>> _parse_header("# 2026/08/30 - 13:24 | PDB: 35.26 | UniProt: 2026.03", origin="x")
    ('2026/08/30 - 13:24', '35.26', '2026.03')
    """
    match = _HEADER_RE.match(header)
    if match is None:
        raise SiftsFormatError(
            f"{origin} begins {header!r}, and SIFTS begins with a release line spelled "
            f"'# <date> - <time> | PDB: <release> | UniProt: <release>'. That line is the "
            f"only record of which release this is, so it is not read past."
        )
    return match["released"], match["pdb"], match["uniprot"]


def _check_columns(header: str, *, origin: str) -> None:
    """Hold the publisher's column header to the shape this reader slices by position."""
    found = tuple(header.split("\t"))
    if found != SOURCE_COLUMNS:
        raise SiftsFormatError(
            f"{origin} names the columns {found}, and this reader slices {SOURCE_COLUMNS} "
            f"by position. The publisher re-shaped the file; sliced anyway, every column "
            f"would hold the wrong thing."
        )


def _add_row(rows: dict[str, list[Any]], line: str, *, number: int, origin: str) -> None:
    """Append one mapping row, dropping the two author-numbering columns."""
    fields = line.split("\t")
    if len(fields) != len(SOURCE_COLUMNS):
        raise SiftsFormatError(
            f"{origin} line {number} holds {len(fields)} fields where SIFTS holds "
            f"{len(SOURCE_COLUMNS)}: {line!r}."
        )
    pdb, chain, accession, res_beg, res_end, _pdb_beg, _pdb_end, sp_beg, sp_end = fields
    rows["pdb"].append(pdb)
    rows["chain"].append(chain)
    rows["accession"].append(accession)
    for name, value in (
        ("res_beg", res_beg),
        ("res_end", res_end),
        ("sp_beg", sp_beg),
        ("sp_end", sp_end),
    ):
        rows[name].append(_as_int(value, column=name, number=number, origin=origin))


def _as_int(value: str, *, column: str, number: int, origin: str) -> int:
    """Read one residue bound, and refuse a row that does not hold an integer."""
    try:
        return int(value)
    except ValueError as error:
        raise SiftsFormatError(
            f"{origin} line {number} holds {value!r} in {column}, and the four residue "
            f"bounds are integers in every one of SIFTS's 1,033,045 rows. Only the two "
            f"author-numbering columns are ever blank, and those are dropped."
        ) from error


app = typer.Typer(
    help="The SIFTS PDB-UniProt map: prepare it, and say which release is here.",
    no_args_is_help=True,
)


@app.command("prepare")
def prepare_command(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of plain text."),
) -> None:
    """Download the SIFTS map and store it under the lab data dir.

    The one step in this lane that needs the network, so it runs on a login node. Already
    prepared, it fetches nothing and reports what is there.
    """
    try:
        prepare(progressbar=not json)
    except _SIFTS_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    _render(status(), json=json)


@app.command("status")
def status_command(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of plain text."),
) -> None:
    """Say which SIFTS release is prepared here, without touching the network.

    Nothing is checked against the publisher, which keeps no archive to compare with.
    """
    try:
        found = status()
    except _SIFTS_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    _render(found, json=json)


def _render(found: SiftsStatus, *, json: bool) -> None:
    """Print one status, as JSON or as one ``key: value`` line each."""
    if json:
        typer.echo(_json.dumps(found.as_json()))
        return
    for key, value in found.as_json().items():
        typer.echo(f"{key}: {value}")
