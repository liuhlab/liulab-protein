"""Tests for protein.xref — the accession-to-gene hop, and what it declines to own.

Offline throughout. `genome.store.fetch.fetch_url` is monkeypatched through its module, so
genome's reader, marker, digest and indexes all run against the Alliance-shaped rows written
here. The curated row pins a digest over the publisher's whole file, which a handful of rows
cannot match, so those rows are pinned to their own digest — the check stays on, pointed at
what the fake fetch serves.

The claims worth holding are that a covered taxon answers with the species genome spells and
an uncovered one raises rather than answering nothing, that the answer is genome's own type
with its provenance intact, and that this module owns no set, no cache and no lifecycle.
"""

from __future__ import annotations

import gzip
import hashlib
import inspect
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from genome.store import fetch
from genome.xref import ResolvedStems, xref_table
from genome.xref import metadata as xref_metadata
from genome.xref.alliance import ALLIANCE_COLUMNS

from protein import xref
from protein.xref import TaxonNotCoveredError

#: One human and one worm gene, Alliance-shaped. Each carries the `ENSEMBL:` row that names
#: its **Gene id stem** and the `UniProtKB:` row that hangs off it; a gene with no `ENSEMBL:`
#: row reaches no hub and would appear nowhere in the slice.
_ROWS: tuple[str, ...] = (
    "HGNC:11998\tENSEMBL:ENSG00000141510\thttps://e\tgeneric\tNCBITaxon:9606",
    "HGNC:11998\tUniProtKB:P04637\thttps://u\tgeneric\tNCBITaxon:9606",
    "WB:WBGene00000001\tENSEMBL:WBGene00000001\thttps://e\tgeneric\tNCBITaxon:6239",
    "WB:WBGene00000001\tUniProtKB:G5EDP9\thttps://u\tgeneric\tNCBITaxon:6239",
)

#: An accession the served rows name no stem for.
_UNCARRIED = "Q9XYZ9"

#: A taxon the curated table has no row for.
_UNCOVERED = 7227


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Serve the rows above as the publisher's file, pinning the curated rows to them."""
    payload = "".join(f"{row}\n" for row in ["\t".join(ALLIANCE_COLUMNS), *_ROWS]).encode()
    publisher = tmp_path / "publisher" / "alliance.tsv.gz"
    publisher.parent.mkdir(parents=True)
    with publisher.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as out:
        out.write(payload)

    # md5 over the unpacked bytes, because that is what Alliance publishes and what the
    # pipeline therefore checks.
    digest = f"md5:{hashlib.md5(payload).hexdigest()}"
    pinned = tuple(replace(row, source_checksum=digest) for row in xref_table())
    monkeypatch.setattr(xref_metadata, "xref_table", lambda: pinned)

    def fake_fetch(url: str, dest_dir: Path, **kwargs: Any) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / (kwargs.get("fname") or publisher.name)
        shutil.copyfile(publisher, target)
        return target

    monkeypatch.setattr(fetch, "fetch_url", fake_fetch)


# --- the taxon lookup ---------------------------------------------------------


@pytest.mark.parametrize(
    ("taxon_id", "species"),
    [(9606, "Homo sapiens"), (10090, "Mus musculus"), (6239, "Caenorhabditis elegans")],
)
def test_a_covered_taxon_answers_with_the_species_the_curated_table_spells(
    taxon_id: int, species: str
) -> None:
    assert xref.species_for(taxon_id) == species


def test_an_uncovered_taxon_answers_none_so_a_caller_may_ask_before_it_leaps() -> None:
    assert xref.species_for(_UNCOVERED) is None


def test_the_lookup_reaches_no_set_and_so_needs_nothing_prepared() -> None:
    # Swiss-Prot is all of UniProt, so most taxa reach no set at all. Asking must therefore
    # be cheap and offline, which the shipped curated table makes it.
    assert xref.species_for(9606) == "Homo sapiens"


# --- the hop ------------------------------------------------------------------


def test_an_accession_answers_with_the_stem_the_release_names(served: None) -> None:
    assert xref.gene_stems_for(["P04637"], 9606).resolved == {"P04637": ("ENSG00000141510",)}


def test_the_taxon_selects_which_species_set_answers(served: None) -> None:
    assert xref.gene_stems_for(["G5EDP9"], 6239).resolved == {"G5EDP9": ("WBGene00000001",)}


def test_an_accession_the_release_names_no_stem_for_rides_back_unresolved(
    served: None,
) -> None:
    # Nothing is dropped: what a list holds and the release does not stays visible rather
    # than the answer coming back silently shorter.
    answer = xref.gene_stems_for(["P04637", _UNCARRIED], 9606)
    assert answer.unresolved == (_UNCARRIED,)
    assert _UNCARRIED not in answer.resolved


def test_the_answer_names_the_set_that_answered(served: None) -> None:
    answer = xref.gene_stems_for(["P04637"], 9606)
    assert (answer.species, answer.source, answer.namespace) == (
        "Homo sapiens",
        "alliance",
        "uniprot",
    )
    assert answer.release


def test_the_answer_is_genomes_own_type_rather_than_a_twin(served: None) -> None:
    assert xref.ResolvedStems is ResolvedStems
    assert isinstance(xref.gene_stems_for(["P04637"], 9606), ResolvedStems)


# --- the taxon that has no set ------------------------------------------------


def test_an_uncovered_taxon_raises_rather_than_answering_nothing() -> None:
    with pytest.raises(TaxonNotCoveredError) as caught:
        xref.gene_stems_for(["P04637"], _UNCOVERED)
    message = str(caught.value)
    assert str(_UNCOVERED) in message
    assert "9606" in message
    assert "Homo sapiens" in message


def test_the_uncovered_taxon_error_is_a_lookup_error() -> None:
    # Genome answers "this set carries no such namespace" with a LookupError too, so one
    # `except LookupError` covers a sweep over mixed input.
    assert issubclass(TaxonNotCoveredError, LookupError)


def test_an_uncovered_taxon_is_refused_before_anything_is_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("a taxon with no set must not reach the fetch step")

    monkeypatch.setattr(fetch, "fetch_url", refuse)
    with pytest.raises(TaxonNotCoveredError):
        xref.gene_stems_for(["P04637"], _UNCOVERED)


# --- what this module does not own --------------------------------------------


def test_the_surface_is_two_functions_an_error_and_genomes_answer_type() -> None:
    assert xref.__all__ == [
        "ResolvedStems",
        "TaxonNotCoveredError",
        "gene_stems_for",
        "species_for",
    ]


@pytest.mark.parametrize("absent", ["prepare", "status", "clear_cache", "app", "source"])
def test_the_module_owns_no_set_and_no_lifecycle(absent: str) -> None:
    # `sifts` wraps a publisher and so owns all of these; this wraps a package that already
    # does, and owning them again would put a second copy of one slice under a second root.
    assert not hasattr(xref, absent)


def test_the_hop_takes_the_accessions_and_the_taxon_and_nothing_else() -> None:
    # No `source` and no `release`: of genome's four sources only three carry `uniprot` at
    # all, so the choice is not a convenience argument.
    parameters = inspect.signature(xref.gene_stems_for).parameters
    assert list(parameters) == ["accessions", "taxon_id"]


def test_importing_protein_does_not_import_the_xref_module() -> None:
    probe = "import protein, sys; print('protein.xref' in sys.modules)"
    finished = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert finished.stdout.strip() == "False"
