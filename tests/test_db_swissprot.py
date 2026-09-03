"""Tests for the Swiss-Prot database — the UniProt header grammar, and one entry out.

The headers here are real UniProt ones, byte-for-byte: `createdb` copies header bytes
through unchanged and `mmseqs view` hands them back, so what this parses is UniProt FASTA
and nothing MMseqs2-specific. `P12345` is the accession the UniProt documentation itself
uses for its example, and it is a real Swiss-Prot entry.

No binary and no database: every `view` rides on one `monkeypatch.setattr(ExternalTool,
"run", ...)`, and the `.lookup` file is three lines.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from protein.db import DECLARED, SwissProt, open_database, parse_uniprot_header
from protein.external import ExternalTool, InstalledTool

_FULL = (
    "sp|P12345|AATM_RABIT Aspartate aminotransferase, mitochondrial "
    "OS=Oryctolagus cuniculus OX=9986 GN=GOT2 PE=1 SV=2"
)


@pytest.fixture
def swissprot(liulab_data: Path) -> Path:
    """Write a Swiss-Prot-shaped database and return its ffindex prefix."""
    directory = liulab_data / "protein" / "db" / "swissprot"
    directory.mkdir(parents=True)
    (directory / "swissprot").write_text("data", encoding="utf-8")
    (directory / "swissprot.dbtype").write_bytes(b"\x00\x00\x08\x00")
    (directory / "swissprot.index").write_text("415743\t0\t5\n", encoding="utf-8")
    (directory / "swissprot.lookup").write_text(
        "7\tP0A031\t0\n415743\tP12345\t0\n9\tP83570\t0\n", encoding="utf-8"
    )
    return directory / "swissprot"


@pytest.fixture
def views(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Answer every `view` with the real P12345 header or its residues, running no binary."""
    calls: list[list[str]] = []

    def record(
        self: ExternalTool, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True
    ) -> str:
        calls.append(list(args))
        return f"{_FULL}\n" if "--idx-entry-type" in args else "MALLHSARVLSG\n"

    monkeypatch.setattr(ExternalTool, "run", record)
    monkeypatch.setattr(InstalledTool, "_detect_version", lambda self: "18.8cc5c")
    return calls


# --- the header grammar --------------------------------------------------------


def test_a_full_swissprot_header_resolves_into_every_field_it_names() -> None:
    header = parse_uniprot_header(_FULL)
    assert header.accession == "P12345"
    assert header.entry_name == "AATM_RABIT"
    assert header.entry_type == "sp"
    assert header.description == "Aspartate aminotransferase, mitochondrial"
    assert header.fields == {
        "organism": "Oryctolagus cuniculus",
        "taxon_id": 9986,
        "gene": "GOT2",
        "protein_existence": 1,
        "sequence_version": 2,
    }


def test_a_value_holding_spaces_survives_the_split() -> None:
    # `OS=Oryctolagus cuniculus` is two words, and only the two-letter keys are anchored.
    assert parse_uniprot_header(_FULL).fields["organism"] == "Oryctolagus cuniculus"


def test_the_three_numeric_fields_come_back_as_numbers() -> None:
    # `taxon_id` is what `genome.xref` would be joined on, so it is not left as text.
    fields = parse_uniprot_header(_FULL).fields
    assert isinstance(fields["taxon_id"], int)
    assert isinstance(fields["protein_existence"], int)
    assert isinstance(fields["sequence_version"], int)


def test_a_trembl_header_is_the_same_grammar_under_a_different_prefix() -> None:
    header = parse_uniprot_header("tr|A0A0A0|A0A0A0_HUMAN Uncharacterized OS=Homo sapiens")
    assert (header.entry_type, header.accession) == ("tr", "A0A0A0")


def test_a_header_that_is_not_this_grammar_is_left_whole_rather_than_sliced() -> None:
    # A UniRef or a locally built FASTA names things differently; guessing would put the
    # wrong text in the accession.
    header = parse_uniprot_header("UniRef50_P12345 Cluster: Aspartate aminotransferase n=2")
    assert header.accession is None
    assert header.entry_name is None
    assert header.identifier == "UniRef50_P12345"


def test_a_header_with_no_free_text_at_all_still_resolves_its_identifier() -> None:
    header = parse_uniprot_header("sp|P12345|AATM_RABIT")
    assert header.accession == "P12345"
    assert header.description is None
    assert header.fields == {}


def test_a_field_this_package_has_no_name_for_is_kept_under_its_own_two_letters() -> None:
    header = parse_uniprot_header("sp|P12345|AATM_RABIT Thing OS=Rabbit ZZ=something")
    assert header.fields["ZZ"] == "something"


def test_a_numeric_field_that_is_not_a_number_is_left_as_it_was_written() -> None:
    assert parse_uniprot_header("sp|P1|A_B Thing OX=unknown").fields["taxon_id"] == "unknown"


# --- one entry out --------------------------------------------------------------


def test_swissprot_is_addressed_by_accession_in_either_case(
    swissprot: Path, views: list[list[str]]
) -> None:
    assert SwissProt().key_for("  p12345 ") == "415743"


def test_one_entry_comes_back_as_a_protein_with_its_annotation_filled(
    swissprot: Path, views: list[list[str]]
) -> None:
    entry = SwissProt()["P12345"]
    assert entry.id == "P12345"
    assert entry.name == "AATM_RABIT"
    assert entry.description == "Aspartate aminotransferase, mitochondrial"
    assert str(entry.sequence) == "MALLHSARVLSG"
    assert entry.metadata["organism"] == "Oryctolagus cuniculus"
    assert entry.metadata["taxon_id"] == 9986
    assert entry.metadata["gene"] == "GOT2"


def test_the_header_is_kept_verbatim_beside_what_was_parsed_out_of_it(
    swissprot: Path, views: list[list[str]]
) -> None:
    assert SwissProt()["P12345"].metadata["header"] == _FULL


def test_an_accession_swissprot_does_not_carry_raises_rather_than_answering_empty(
    swissprot: Path, views: list[list[str]]
) -> None:
    # mmseqs warns on stderr and exits 0 for a name it cannot find, so absence is decided
    # from `.lookup` before any subprocess runs.
    with pytest.raises(KeyError, match="Q99999"):
        SwissProt()["Q99999"]
    assert views == []


def test_retrieval_does_not_warn_about_residues_it_did_not_fold(
    swissprot: Path, views: list[list[str]]
) -> None:
    # ADR-0003: nothing distinguishes a folded `D` from a real one, so there is no warning
    # here. `filterwarnings = ["error"]` makes this assertion the whole test.
    assert SwissProt()["P12345"].id == "P12345"


# --- the declaration ------------------------------------------------------------


def test_swissprot_is_the_declared_name_and_carries_the_tools_own_spelling() -> None:
    assert DECLARED["swissprot"].source == "UniProtKB/Swiss-Prot"
    assert SwissProt().source == "UniProtKB/Swiss-Prot"
    assert SwissProt().name == "swissprot"


def test_opening_the_declared_name_gives_the_class_that_reads_uniprot_headers() -> None:
    assert isinstance(open_database("swissprot"), SwissProt)
