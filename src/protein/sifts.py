"""SIFTS: the PDB-UniProt map, and the one join between this package's two namespaces.

A **Protein** is addressed by a UniProt accession; a PDB id addresses a **Structure**.
Neither namespace knows the other, and SIFTS — the EBI's re-curated chain-to-accession
map, one 6.2 MB TSV republished weekly — is what joins them. It is many-to-many in both
directions: 1.00% of chains carry more than one accession, and one accession reaches a
median of 2 entries and a maximum of 3,668.

**The join reads SIFTS and never the structure file.** Every mmCIF carries its own
``_struct_ref_seq``, and it disagrees: ``1UBQ`` chain A is ``P62988`` residues 1-76 in the
file and ``P0CG48`` residues 609-684 here, because the file holds the depositor's reference
frozen at deposition and SIFTS holds PDBe's re-curated one. Mixing the two breaks the round
trip — ``Protein.structures`` yields ``1ubq``, whose chain A would then call itself
``P62988`` — so ``Chain.uniprot`` reads this table alone.

**A Prepared set, not a Database.** :mod:`genome.store.prepared` already runs this pipeline
three times, and what a fourth declares is a URL, a directory name and a reader. Everything
else — the fetch, the working area, the staged rename, the digest, the **Completion marker**
and the sentence that sends a caller to a login node — is that module's. This one is a peer
of :mod:`protein.store`, not a tenant of ``db/``: a **Database** is immutable ffindex files
searched by a subprocess, and SIFTS is none of those.

**The stored form is a gzipped TSV and not a parquet**, which #20 asked for and #23
repeated. Two measurements on GPU71FM against the real release forced the change.
:func:`genome.store.prepared.unpacked_digest` decodes every stored byte as UTF-8, so the
shared pipeline cannot digest a binary file at all. And the parquet's margin over a
seven-column sorted TSV is 98 ms against 212 ms, and 4.68 MB against **4.30 MB** — the TSV
is the smaller of the two. #20's table compared the parquet against reparsing the
publisher's nine-column source in pure Python (6.21 MB, 0.92 s), which is not this file.

**Nothing is pinned and nothing is checked for staleness.** There is no archive:
``ftp.ebi.ac.uk/pub/databases/msd/sifts/`` holds only the current release, overwritten in
place, so a digest pinned today would reject every release after it and a release not
downloaded while current is gone. The reader records what it read instead — the publisher's
own header line, ``PDB: 35.26 | UniProt: 2026.03`` — and :func:`status` prints it back,
offline. Refresh is :attr:`~genome.store.prepared.PreparedSource.repair`: delete and rebuild.

**Two verbs, one per direction, over a cached table.** Reach them through the module —
``from protein import sifts``, then ``sifts.accessions_for(...)`` — never by importing the
name, so one ``monkeypatch.setattr`` reaches every caller. Their shapes are asymmetric
because the cardinalities are: a chain has one accession 99% of the time, so
:func:`accessions_for` is a tuple; an accession reaches thousands of chains, so
:func:`structures_for` is a frame.

**Both ranges come back verbatim and no offset is computed.** 23,046 rows (2.2%) have
``res_end - res_beg != sp_end - sp_beg``, so a single integer shift is not always definable.

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

#: This set's directory under :func:`protein.store.protein_data_dir`. One release-free
#: directory, because there is nothing to keep a second one of.
SIFTS_SUBDIR = "sifts"

#: The file the set is read from: the publisher's nine columns cut to seven and sorted,
#: gzipped. 4.30 MB against the source's 6.21 MB, read whole in 212 ms.
STORED_NAME = "pdb_chain_uniprot.tsv.gz"

#: What the **Completion marker** calls what it recorded.
SIFTS_KIND = "sifts"

#: The name this set is addressed by in its own tree.
SIFTS_NAME = "pdb_chain_uniprot"

#: How an error names this set.
DESCRIPTION = "the SIFTS PDB-UniProt map"

#: The call that prepares it, quoted verbatim into every error a caller repairs by running
#: it.
PREPARE_COMMAND = "protein sifts prepare"

#: The publisher's columns, in the publisher's order and spelling. Checked against the
#: file's own second line, so a re-published file with a different shape fails loudly here
#: rather than being sliced into the wrong columns.
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
#: — are dropped: they are blank in 56% and 48% of rows, and the mmCIF's
#: ``_pdbx_poly_seq_scheme`` gives the same SEQRES-to-author mapping completely, per residue
#: and with insertion codes, for any structure a caller actually holds.
COLUMNS: tuple[str, ...] = ("pdb", "chain", "accession", "res_beg", "res_end", "sp_beg", "sp_end")

#: The columns the stored frame is sorted by. Accession first, because that is the direction
#: a whole-table scan would be slowest in and the one :func:`structures_for` takes.
SORT_COLUMNS: tuple[str, ...] = ("accession", "pdb", "chain")

#: What each stored column is read back as. Spelled out rather than inferred, so the four
#: residue bounds come back as the ``int32`` they were written as and the three names stay
#: strings. ``Hashable``-keyed because that is the mapping ``read_csv`` takes, and ``dict``
#: is invariant in both parameters.
READ_DTYPES: dict[Hashable, Any] = {
    **dict.fromkeys(COLUMNS[:3], "string"),
    **dict.fromkeys(COLUMNS[3:], "int32"),
}

#: The publisher's first line, e.g. ``# 2026/08/30 - 13:24 | PDB: 35.26 | UniProt: 2026.03``.
#: It is the only thing in the file that says which release this is, and with no archive to
#: pin against it is the whole reproducibility record.
_HEADER_RE = re.compile(
    r"^#\s*(?P<released>.+?)\s*\|\s*PDB:\s*(?P<pdb>\S+)\s*\|\s*UniProt:\s*(?P<uniprot>\S+)\s*$"
)

#: What a failed SIFTS call raises. A set that is not here, and a directory an interrupted
#: run left unfinished, are ``RuntimeError``s; a publisher's file this package cannot read is
#: a ``ValueError``; a file that went away under the read is an ``OSError``.
_SIFTS_ERRORS = (ValueError, OSError, RuntimeError)


class SiftsNotDownloadedError(prepared.PreparedSetNotDownloadedError):
    """The SIFTS map is not prepared on this machine and could not be fetched.

    Distinct from an empty answer, and deliberately so: ``()`` from
    :func:`accessions_for` means the chain genuinely has no protein — every nucleic-acid
    chain, every ligand chain, every entry SIFTS never curated — and conflating the two
    would let a script map nothing and look like it worked.

    Examples
    --------
    >>> issubclass(SiftsNotDownloadedError, RuntimeError)
    True
    """


class SiftsFormatError(ValueError):
    """What arrived is not the file this reader knows how to slice.

    A :class:`ValueError`: the bytes are a bad value. Raised while the lines go past and
    before anything is placed, so a file the publisher re-shaped never lands on disk to be
    read back as a finished set.

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
        :attr:`~genome.store.prepared.PreparedSource.checksum` is ``None``: the publisher
        overwrites this file weekly in place, so a pin would reject every release after the
        one it was taken from. The digest of what was stored is recorded instead.

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

    The whole of what this package adds to :func:`genome.store.prepared.prepare`. Seven of
    the publisher's nine columns are kept, the rows are sorted by :data:`SORT_COLUMNS`, and
    the first line — the only place the file names its own release — comes back as the
    marker's record of which SIFTS this is.

    Parameters
    ----------
    lines : iterator of str
        The publisher's unpacked lines, line endings and all. **The data rows end in CRLF
        and the first line does not**, so every line is stripped of both.
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
    # Stable, so the several segments one (pdb, chain, accession) triple carries stay in
    # the publisher's own order — which is the order their residue ranges ascend in.
    frame = frame.sort_values(list(SORT_COLUMNS), kind="stable", ignore_index=True)
    staged.parent.mkdir(parents=True, exist_ok=True)
    # `mtime=0`, so one release cut on two machines gives byte-identical files and the
    # digest the marker records is a fact about the rows rather than about the clock.
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

    One call to :func:`genome.store.prepared.prepare`, which is the point: the fetch, the
    working area, the staged rename, the digest and the marker are all that module's.

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
    # The file under the cache's key may have just been replaced, and a cache that outlives
    # the bytes it read is the one way this module can answer from a release that is gone.
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
        The PDB release the map was built against, e.g. ``"35.26"``.
    uniprot_release : str or None
        The UniProt release, e.g. ``"2026.03"``.
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

    **Offline, and it stays offline.** There is no archive of past releases to compare
    against, updates are rare, and the lab's compute nodes have no network — so this says
    which release is here and never whether a newer one exists.

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

    41 MB resident and 212 ms to read, so the first caller pays for the rest. **Do not
    mutate what comes back** — every caller shares it; the two verbs each hand back a slice
    of their own.

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

    A frame and not a list of ids, because an entry id alone does not say *which chain* to
    fetch coordinates for — and because ``P0DTD1`` reaches 6,181 chains across 3,668
    entries, which is not a number of objects to build.

    Parameters
    ----------
    accession : str
        A UniProt accession, in either case. SIFTS stores them upper-case, so that is what
        the input is folded to.

    Returns
    -------
    pandas.DataFrame
        :data:`COLUMNS`, one row per segment, in the stored order. Empty when SIFTS carries
        nothing for this accession — which is the true answer for a protein no structure
        has been solved for.

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

    A tuple and never a scalar: 9,941 chains — 1.00% — carry more than one accession, up to
    four (``8uqe`` B), and a surface that answered with the first would be wrong once in a
    hundred without saying so.

    Parameters
    ----------
    pdb : str
        A four-character PDB entry id, in either case. SIFTS stores them lower-case, so
        that is what the input is folded to.
    chain : str
        The author chain label — ``auth_asym_id``, which is what SIFTS keys on. **Not
        folded**: 62,551 rows carry a lower-case single-letter label, and ``10eg`` has both
        an ``A`` and an ``a``, so case is part of the name.

    Returns
    -------
    tuple of str
        The accessions, deduplicated and in accession order — the order
        :data:`SORT_COLUMNS` puts them in, not the publisher's row order, which is by
        residue range. ``()`` when this chain has no
        protein — every nucleic-acid chain, every ligand chain, every entry SIFTS never
        curated — and also for an id SIFTS does not carry at all, which is what a chain of a
        ``Structure.from_file`` that is not a PDB entry gets.

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
    **Data dir** — every test in this suite does — reads the set it now names instead of the
    one it read first.

    ``keep_default_na=False`` is load-bearing and not caution: **SIFTS carries 205 rows whose
    chain is labelled** ``NA`` — ``9on4`` has one, between its ``MA`` and its ``OA`` — and
    pandas' default missing-value list reads that name as a missing value and loses the
    chain. It is spelled this way and not as ``na_filter=False``, which does the same thing
    and which **the pyarrow engine silently ignores**: measured here, that spelling returned
    205 nulls and no ``NA``. The pyarrow engine is twice the C one, 209 ms against 410 ms.
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
    """Read one residue bound, which SIFTS populates in every row of these four columns."""
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

    Nothing is checked against the publisher: there is no archive to compare with, and the
    machines this runs on have no network.
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
