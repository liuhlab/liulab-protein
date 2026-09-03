"""FASTA in and out: the record layer, the protein layer, and the converters we never call."""

from __future__ import annotations

import ast
import gzip
from pathlib import Path

import pytest

from protein import Protein
from protein.io import fasta
from protein.seq import ResidueCoercionWarning

_DATA = Path(__file__).resolve().parent / "data"
_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "protein"

_INSULIN_HEADER = "sp|P01308|INS_HUMAN Insulin OS=Homo sapiens OX=9606 GN=INS PE=1 SV=1"
_INSULIN_LENGTH = 110

#: What ADR-0002 bans. Each substitutes a different residue without a word.
#:
#: The first three rewrite ``U`` to ``C`` and ``O`` to ``K``. ``convert_letter_3to1`` was
#: found by #15 and has the same defect from the other direction: it answers ``'C'`` for
#: ``"SEC"`` where ``biotite.structure.info.one_letter_code`` answers ``'U'``, and it
#: ``KeyError``s on every modified residue rather than saying so. ``Chain.sequence`` uses
#: ``one_letter_code`` and :func:`protein.seq.to_protein_sequence` instead.
_BANNED_CONVERTERS = frozenset(
    {"get_sequence", "get_sequences", "to_sequence", "convert_letter_3to1"}
)


# --- the record layer --------------------------------------------------------


def test_read_records_yields_the_header_and_the_unwrapped_sequence() -> None:
    header, sequence = next(iter(fasta.read_records(_DATA / "uniprot_p01308.fasta")))
    assert header == _INSULIN_HEADER
    assert len(sequence) == _INSULIN_LENGTH
    assert "\n" not in sequence


def test_read_records_reads_every_record_in_a_multi_record_file() -> None:
    headers = [header for header, _ in fasta.read_records(_DATA / "uniprot_three.fasta")]
    assert [header.split("|")[1] for header in headers] == ["P01308", "P69905", "P07203"]


def test_a_header_survives_byte_for_byte_including_a_run_of_spaces(tmp_path: Path) -> None:
    # The reason biotite's file layer is adopted rather than reimplemented: nothing in the
    # header is normalised, so `OS=` and friends arrive as the file spells them.
    odd = tmp_path / "odd.fasta"
    odd.write_text(">P12345   two  spaces  here\nMKTAY\n", encoding="utf-8")
    assert next(iter(fasta.read_records(odd)))[0] == "P12345   two  spaces  here"


def test_write_records_and_read_records_round_trip(tmp_path: Path) -> None:
    records = [("a one", "MKTAY"), ("b two", "QQRLIFAGK")]
    written = tmp_path / "pair.fasta"
    fasta.write_records(written, records)
    assert list(fasta.read_records(written)) == records


def test_format_records_returns_what_write_records_would_have_written(tmp_path: Path) -> None:
    records = [("a", "MKTAY"), ("b", "QQRLIFAGK")]
    written = tmp_path / "pair.fasta"
    fasta.write_records(written, records, line_width=4)
    assert fasta.format_records(records, line_width=4) == written.read_text(encoding="utf-8")


def test_a_gzipped_file_is_read_and_written_like_a_plain_one(tmp_path: Path) -> None:
    # biotite has no gzip FASTA reader, so this branch is ours; that it is one branch rather
    # than a parallel set of functions is what this test pins.
    records = [("a", "MKTAY")]
    compressed = tmp_path / "one.fasta.gz"
    fasta.write_records(compressed, records)
    assert gzip.decompress(compressed.read_bytes()).decode() == ">a\nMKTAY\n"
    assert list(fasta.read_records(compressed)) == records


# --- headers -----------------------------------------------------------------


def test_a_header_splits_at_the_first_whitespace_and_keeps_the_rest_intact() -> None:
    assert fasta.split_header("P12345 the  rest  of  it") == ("P12345", "the  rest  of  it")


def test_a_header_with_no_description_gives_none_for_it() -> None:
    assert fasta.split_header("P12345") == ("P12345", None)


def test_an_empty_header_gives_none_twice() -> None:
    assert fasta.split_header("") == (None, None)


def test_a_uniprot_header_is_not_taken_apart_here() -> None:
    # `sp|P01308|INS_HUMAN` is a Swiss-Prot convention, not a FASTA one, so the identifier
    # arrives whole and the Database that knows the convention resolves it.
    identifier, _ = fasta.split_header(_INSULIN_HEADER)
    assert identifier == "sp|P01308|INS_HUMAN"


def test_join_header_is_the_inverse_of_split_header_for_a_single_spaced_header() -> None:
    assert fasta.join_header(*fasta.split_header(_INSULIN_HEADER)) == _INSULIN_HEADER


def test_join_header_drops_the_part_that_is_absent() -> None:
    assert fasta.join_header("P12345", None) == "P12345"
    assert fasta.join_header(None, "just text") == "just text"
    assert fasta.join_header(None, None) == ""


# --- the protein layer -------------------------------------------------------


def test_read_proteins_builds_one_protein_per_record() -> None:
    with pytest.warns(ResidueCoercionWarning, match="P07203"):
        proteins = list(fasta.read_proteins(_DATA / "uniprot_three.fasta"))
    assert [p.length for p in proteins] == [110, 142, 203]
    assert proteins[0].id == "sp|P01308|INS_HUMAN"
    assert proteins[0].description == "Insulin OS=Homo sapiens OX=9606 GN=INS PE=1 SV=1"


def test_a_selenocysteine_in_the_file_is_folded_to_x_and_named_in_the_warning() -> None:
    # P07203 is glutathione peroxidase 1, one of the 285 Swiss-Prot entries carrying `U`.
    # ADR-0002 is what makes this `X` and not biotite's `C`.
    with pytest.warns(ResidueCoercionWarning, match="P07203"):
        proteins = list(fasta.read_proteins(_DATA / "uniprot_three.fasta"))
    assert "U" not in str(proteins[2].sequence)
    assert "X" in str(proteins[2].sequence)


def test_write_proteins_writes_what_each_protein_renders_itself_as(tmp_path: Path) -> None:
    proteins = [Protein("MKTAY", id="a"), Protein("QQRLIFAGK", id="b", description="two")]
    written = tmp_path / "pair.fasta"
    fasta.write_proteins(written, proteins)
    assert written.read_text(encoding="utf-8") == "".join(p.to_fasta() for p in proteins)


def test_write_proteins_then_read_proteins_round_trips(tmp_path: Path) -> None:
    proteins = [Protein("MKTAY", id="a", description="one"), Protein("QQRLIFAGK", id="b")]
    written = tmp_path / "pair.fasta"
    fasta.write_proteins(written, proteins)
    again = list(fasta.read_proteins(written))
    assert [(p.id, p.description, str(p.sequence)) for p in again] == [
        (p.id, p.description, str(p.sequence)) for p in proteins
    ]


# --- ADR-0002 ----------------------------------------------------------------


def test_no_module_in_this_package_reaches_for_biotites_converters() -> None:
    offenders = [
        f"{module.relative_to(_SOURCE_ROOT).as_posix()}:{line}: {name}"
        for module in sorted(_SOURCE_ROOT.rglob("*.py"))
        for name, line in _converter_references(module)
    ]
    assert offenders == [], (
        f"{offenders} name a biotite converter. Every name in {sorted(_BANNED_CONVERTERS)} "
        f"substitutes a residue silently (ADR-0002); the one string-to-sequence step in "
        f"this package is protein.seq.to_protein_sequence."
    )


def _converter_references(module: Path) -> list[tuple[str, int]]:
    """Return every ``(name, line)`` where ``module`` names a banned converter in code.

    Parsed rather than grepped, so a docstring or a comment naming one — this repo's own
    prose about the rule — is not an offence, and a call or an import is.
    """
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _BANNED_CONVERTERS:
            found.append((node.attr, node.lineno))
        elif isinstance(node, ast.Name) and node.id in _BANNED_CONVERTERS:
            found.append((node.id, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            found += [
                (alias.name, node.lineno)
                for alias in node.names
                if alias.name in _BANNED_CONVERTERS
            ]
    return found


def test_the_guard_would_catch_a_module_that_did_reach_for_one(tmp_path: Path) -> None:
    # A guard nobody has seen fire is a guard nobody knows works.
    tempted = tmp_path / "tempted.py"
    tempted.write_text(
        '"""A docstring naming get_sequence is not an offence."""\n'
        "from biotite.sequence.io.fasta import get_sequence\n"
        "\n"
        "value = get_sequence(file)\n",
        encoding="utf-8",
    )
    assert _converter_references(tempted) == [("get_sequence", 2), ("get_sequence", 4)]
