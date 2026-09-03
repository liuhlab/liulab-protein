"""FASTA, in two layers: biotite's records below, this package's proteins above.

The **record layer** is adopted whole. ``FastaFile.read_iter`` and ``FastaFile.write_iter``
parse and render, and a header survives byte-for-byte — every space, every ``OS=`` field,
exactly as the file spells it. :func:`read_records`, :func:`write_records` and
:func:`format_records` are thin over them, adding only gzip, which biotite has no reader for.

The **protein layer** is ours. :func:`read_proteins` turns each record into a
:class:`~protein.core.Protein`, whose sequence is built by
:func:`protein.seq.to_protein_sequence` and by nothing else.

**biotite's converters are never called.** ``fasta.get_sequence`` and ``fasta.get_sequences``
rewrite ``U`` to ``C`` and ``O`` to ``K`` without a word — a different residue, claimed
silently, in 285 Swiss-Prot entries. That is ADR-0002, and a test walks ``src/protein/`` to
enforce it.

**Two modules are called ``fasta``.** biotite's is ``biotite.sequence.io.fasta``; this one is
``protein.io.fasta``. The import below takes the one class this module needs rather than the
module, so no call site here can be read as the wrong one.

A header is split the plain FASTA way: the first whitespace-delimited token is the identifier
and the rest is free text. A UniProt header therefore lands whole — ``sp|P01308|INS_HUMAN``
is the identifier, pipes and all. Resolving that into an accession and an entry name is a
Swiss-Prot convention, not a FASTA one, and belongs to the **Database** that knows it.

Examples
--------
>>> print(format_records([("P12345 my protein", "MKTAY")]), end="")
>P12345 my protein
MKTAY
>>> split_header("sp|P01308|INS_HUMAN Insulin OS=Homo sapiens")
('sp|P01308|INS_HUMAN', 'Insulin OS=Homo sapiens')
>>> join_header("P12345", None)
'P12345'
"""

from __future__ import annotations

import gzip
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from biotite.sequence.io.fasta import FastaFile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import TextIO

    from protein.core import Protein

__all__ = [
    "DEFAULT_LINE_WIDTH",
    "format_records",
    "join_header",
    "read_proteins",
    "read_records",
    "split_header",
    "write_proteins",
    "write_records",
]

#: Residues per line when writing, which is biotite's own default. Named here so the two
#: writers and :meth:`protein.core.Protein.to_fasta` cannot drift apart.
DEFAULT_LINE_WIDTH = 80


def read_records(source: str | Path) -> Iterator[tuple[str, str]]:
    """Yield one ``(header, sequence)`` pair per record in the FASTA file at ``source``.

    The header is everything after ``>``, byte-for-byte, and the sequence is one unwrapped
    string. Neither is checked against any alphabet: this is the record layer, and the
    string-to-sequence step is :func:`read_proteins`'.

    Reading is lazy, so the file stays open until the iterator is exhausted or dropped.

    Parameters
    ----------
    source : str or pathlib.Path
        The FASTA file. A name ending ``.gz`` is decompressed.

    Yields
    ------
    tuple of (str, str)
        The header and the residues.

    Examples
    --------
    >>> next(read_records("tests/data/uniprot_p01308.fasta"))  # doctest: +SKIP
    ('sp|P01308|INS_HUMAN Insulin OS=Homo sapiens OX=9606 GN=INS PE=1 SV=1', 'MALW...')
    """
    with _open_text(Path(source), "rt") as handle:
        yield from FastaFile.read_iter(handle)


def write_records(
    destination: str | Path,
    records: Iterable[tuple[str, str]],
    *,
    line_width: int = DEFAULT_LINE_WIDTH,
) -> None:
    """Write ``records`` to the FASTA file at ``destination``, replacing what is there.

    Parameters
    ----------
    destination : str or pathlib.Path
        Where to write. A name ending ``.gz`` is compressed. The parent directory must
        exist; nothing here creates one.
    records : iterable of tuple of (str, str)
        ``(header, sequence)`` pairs. The header is written after ``>`` verbatim, so it must
        already be what the file should say.
    line_width : int, default 80
        Residues per line.
    """
    with _open_text(Path(destination), "wt") as handle:
        FastaFile.write_iter(handle, records, chars_per_line=line_width)


def format_records(
    records: Iterable[tuple[str, str]], *, line_width: int = DEFAULT_LINE_WIDTH
) -> str:
    """Render ``records`` as FASTA text, writing no file.

    Parameters
    ----------
    records : iterable of tuple of (str, str)
        ``(header, sequence)`` pairs, as :func:`write_records` takes.
    line_width : int, default 80
        Residues per line.

    Returns
    -------
    str
        The records, each newline-terminated.

    Examples
    --------
    >>> print(format_records([("a", "MKT"), ("b", "AY")]), end="")
    >a
    MKT
    >b
    AY
    """
    buffer = StringIO()
    FastaFile.write_iter(buffer, records, chars_per_line=line_width)
    return buffer.getvalue()


def read_proteins(source: str | Path) -> Iterator[Protein]:
    """Yield one :class:`~protein.core.Protein` per record in the FASTA file at ``source``.

    Each header is split by :func:`split_header` into the protein's ``id`` and
    ``description``, and each sequence goes through :func:`protein.seq.to_protein_sequence`,
    so a record whose residues are outside the alphabet raises here rather than downstream.

    Parameters
    ----------
    source : str or pathlib.Path
        The FASTA file. A name ending ``.gz`` is decompressed.

    Yields
    ------
    Protein
        One per record, in file order.

    Raises
    ------
    protein.seq.InvalidResidueError
        On the first record holding anything outside :data:`protein.seq.ALPHABET`. The error
        names the record's identifier.

    Warns
    -----
    protein.seq.ResidueCoercionWarning
        Once per record holding ``U``, ``O`` or ``J``.

    Examples
    --------
    >>> list(read_proteins("tests/data/uniprot_three.fasta"))  # doctest: +SKIP
    [Protein('sp|P01308|INS_HUMAN', 110 aa), ...]
    """
    from protein.core import Protein

    for header, sequence in read_records(source):
        identifier, description = split_header(header)
        yield Protein(sequence, id=identifier, description=description)


def write_proteins(
    destination: str | Path,
    proteins: Iterable[Protein],
    *,
    line_width: int = DEFAULT_LINE_WIDTH,
) -> None:
    """Write ``proteins`` to the FASTA file at ``destination``, replacing what is there.

    One record each, rendered by :meth:`protein.core.Protein.to_fasta`, so a protein written
    here and one written on its own say the same thing. Streaming: a generator is consumed
    one protein at a time.

    Parameters
    ----------
    destination : str or pathlib.Path
        Where to write. A name ending ``.gz`` is compressed.
    proteins : iterable of Protein
        What to write, in order.
    line_width : int, default 80
        Residues per line.
    """
    with _open_text(Path(destination), "wt") as handle:
        for protein in proteins:
            handle.write(protein.to_fasta(line_width=line_width))


def split_header(header: str) -> tuple[str | None, str | None]:
    """Split a FASTA header into its identifier and the free text after it.

    The plain FASTA convention and nothing more: the first whitespace-delimited token is the
    identifier, the rest is description. A UniProt header's ``sp|P01308|INS_HUMAN`` comes
    back whole — see this module's docstring for why it is not taken apart here.

    Parameters
    ----------
    header : str
        A header, without the leading ``>``, as :func:`read_records` yields it.

    Returns
    -------
    tuple of (str or None, str or None)
        The identifier and the description. Either is ``None`` when the header has no such
        part, so an empty header gives ``(None, None)``.

    Examples
    --------
    >>> split_header("P12345 the rest  of  it")
    ('P12345', 'the rest  of  it')
    >>> split_header("P12345")
    ('P12345', None)
    >>> split_header("")
    (None, None)
    """
    parts = header.split(maxsplit=1)
    identifier = parts[0] if parts else None
    description = parts[1] if len(parts) > 1 else None
    return identifier, description


def join_header(identifier: str | None, description: str | None) -> str:
    """Join an identifier and a description into one FASTA header.

    The inverse of :func:`split_header` for any header of that shape. It is not a byte-exact
    inverse of every header: a run of spaces between the two parts comes back as one, and
    leading whitespace is dropped.

    Parameters
    ----------
    identifier : str or None
        The first token of the header.
    description : str or None
        The free text after it.

    Returns
    -------
    str
        The two joined by one space, or whichever is present, or ``""`` when neither is.

    Examples
    --------
    >>> join_header("P12345", "Insulin")
    'P12345 Insulin'
    >>> join_header(None, None)
    ''
    """
    return " ".join(part for part in (identifier, description) if part)


def _open_text(path: Path, mode: Literal["rt", "wt"]) -> TextIO:
    """Open ``path`` as UTF-8 text, decompressing or compressing when it ends ``.gz``.

    biotite reads and writes any text handle, so gzip support is this one branch rather than
    a parallel set of functions.
    """
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")
