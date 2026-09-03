"""Tests for `Chain` — what it is, what it reads as, whose it is, and how it searches.

The sharp one is `uniprot`: `1UBQ` chain A is `P62988` in the mmCIF sitting in `tests/data`
and `P0CG48` in SIFTS, so a chain that answered from the file it was parsed from would be
visibly wrong. It is run against the committed SIFTS slice, through the whole prepared-set
pipeline, so nothing about the answer is stubbed.
"""

from __future__ import annotations

import gzip
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from biotite.structure import AtomArray
from genome.store import fetch as genome_fetch

from protein import Structure, sifts
from protein.embed.esm import Embeddable
from protein.external import ExternalTool
from protein.io import structure as io
from protein.sifts import SiftsNotDownloadedError

_DATA = Path(__file__).resolve().parent / "data"
_UBQ = _DATA / "1ubq.cif.gz"
_BNA = _DATA / "1bna.cif.gz"
_SLICE = _DATA / "sifts_pdb_chain_uniprot_slice.tsv"

#: Ubiquitin, the 76 residues `1UBQ` was solved for.
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


@pytest.fixture
def ubq() -> Structure:
    """`1UBQ`, named as its entry so a chain key reads `1UBQ_A`."""
    return Structure.from_file(_UBQ, id="1UBQ")


@pytest.fixture
def bna() -> Structure:
    """`1BNA`, two chains of DNA."""
    return Structure.from_file(_BNA, id="1BNA")


@pytest.fixture
def prepared_sifts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run the prepared-set pipeline against the committed slice, offline.

    The same shape `tests/test_sifts.py` uses: the one call that would download is replaced
    through its module, and everything after it is the real code.
    """
    packed = tmp_path / "publisher" / "pdb_chain_uniprot.tsv.gz"
    packed.parent.mkdir(parents=True)
    with gzip.GzipFile(packed, "wb", mtime=0) as out:
        out.write(_SLICE.read_bytes())

    def fake_fetch(url: str, dest_dir: Path, **kwargs: Any) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / (kwargs.get("fname") or "pdb_chain_uniprot.tsv.gz")
        shutil.copyfile(packed, target)
        return target

    monkeypatch.setattr(genome_fetch, "fetch_url", fake_fetch)
    sifts.prepare(progressbar=False)


@dataclass
class _Runs:
    """Every call an `easy-search` made, and the query file each was handed."""

    calls: list[list[str]] = field(default_factory=list)
    queries: list[tuple[str, bytes]] = field(default_factory=list)


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> _Runs:
    """Replace `ExternalTool.run` and record the query file before it is cleaned up."""
    recorded = _Runs()

    def record(
        self: ExternalTool, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True
    ) -> str:
        recorded.calls.append(list(args))
        if args and args[0] == "easy-search":
            query = Path(args[1])
            recorded.queries.append((query.name, query.read_bytes()))
        return ""

    monkeypatch.setattr(ExternalTool, "run", record)
    return recorded


@pytest.fixture
def pdb_database(liulab_data: Path) -> Path:
    """A pdb100-shaped database registered under this test's own data root."""
    directory = liulab_data / "protein" / "db" / "pdb"
    directory.mkdir(parents=True)
    (directory / "pdb100.dbtype").write_bytes(b"\x00\x00\x00\x00")
    return directory / "pdb100"


# --- identity ------------------------------------------------------------------


def test_a_chain_is_keyed_the_way_sifts_and_foldseek_both_key_one(ubq: Structure) -> None:
    assert ubq["A"].id == "1UBQ_A"


def test_neither_half_of_the_key_is_folded() -> None:
    structure = Structure.from_file(_BNA, id="1bna")
    assert structure["B"].id == "1bna_B"


def test_a_chain_reports_its_atoms_and_says_what_it_is(ubq: Structure) -> None:
    assert repr(ubq["A"]) == "Chain('1UBQ_A', protein, 660 atoms)"


# --- atoms and kind ------------------------------------------------------------


def test_a_chains_atoms_are_every_atom_carrying_its_label(ubq: Structure) -> None:
    chain = ubq["A"]
    assert len(chain) == 660
    assert io.chain_ids(chain.atoms) == ("A",)


def test_the_atoms_are_taken_once_and_then_held(ubq: Structure) -> None:
    chain = ubq["A"]
    assert chain.atoms is chain.atoms


def test_a_protein_chain_says_so(ubq: Structure) -> None:
    assert ubq["A"].kind == "protein"


def test_a_dna_chain_says_so(bna: Structure) -> None:
    assert [chain.kind for chain in bna.chains] == ["nucleic", "nucleic"]


def test_the_kind_is_the_majority_and_not_a_demand_for_purity(bna: Structure) -> None:
    # `1BNA` chain A is 243 nucleotide atoms and 37 waters, because a chain carries what was
    # modelled against it. A rule wanting every atom to match would call this one `other`.
    chain = bna["A"]
    assert len(chain) == 280
    assert chain.kind == "nucleic"


def test_a_chain_of_neither_kind_is_other(ubq: Structure, tmp_path: Path) -> None:
    atoms = ubq.atoms
    waters = cast("AtomArray", atoms[atoms.res_name == "HOH"])
    only_water = tmp_path / "water.cif"
    io.write_atoms(only_water, waters)
    assert Structure.from_file(only_water)["A"].kind == "other"


# --- sequence ------------------------------------------------------------------


def test_the_sequence_is_the_residues_the_chain_was_solved_for(ubq: Structure) -> None:
    assert str(ubq["A"].sequence) == _UBIQUITIN


def test_the_waters_sharing_the_chain_label_are_not_in_the_sequence(ubq: Structure) -> None:
    # 1UBQ's 58 waters carry chain label `A` too, so a sequence built without the amino-acid
    # filter would be 58 residues too long.
    assert len(ubq["A"].sequence) == 76


def test_the_sequence_is_a_biotite_protein_sequence_and_not_a_string(ubq: Structure) -> None:
    sequence = ubq["A"].sequence
    assert sequence != _UBIQUITIN
    assert str(sequence) == _UBIQUITIN


def test_a_chain_that_is_not_protein_refuses_rather_than_answering(bna: Structure) -> None:
    # The `Embeddable` protocol knows nothing of `.kind`, so this refusal is what stops a
    # DNA chain reaching the tokenizer.
    with pytest.raises(ValueError, match="1BNA_A is nucleic, not protein"):
        _ = bna["A"].sequence


def test_the_refusal_names_the_attribute_that_would_have_said_so(bna: Structure) -> None:
    with pytest.raises(ValueError, match=r"\.kind"):
        _ = bna["A"].sequence


# --- the embedding protocol ----------------------------------------------------


def test_a_chain_is_embeddable(ubq: Structure) -> None:
    assert isinstance(ubq["A"], Embeddable)


def test_a_non_protein_chain_still_looks_embeddable_and_fails_when_asked(
    bna: Structure,
) -> None:
    # `isinstance` against a runtime-checkable protocol reads the class and never calls the
    # property, so the refusal has to come from the property itself.
    chain = bna["A"]
    assert isinstance(chain, Embeddable)
    with pytest.raises(ValueError, match="not protein"):
        _ = chain.sequence


# --- uniprot -------------------------------------------------------------------


def test_a_chain_answers_from_sifts_and_not_from_the_file_it_was_parsed_from(
    prepared_sifts: None,
) -> None:
    # The whole join in one assertion. `1UBQ` chain A is `P62988` in this very mmCIF's
    # `_struct_ref_seq` and `P0CG48` in SIFTS; the second is the answer.
    assert Structure.from_file(_UBQ, id="1UBQ")["A"].uniprot == ("P0CG48",)


def test_the_id_may_be_given_in_either_case(prepared_sifts: None) -> None:
    assert Structure.from_file(_UBQ, id="1ubq")["A"].uniprot == ("P0CG48",)


def test_a_chain_with_no_protein_answers_with_an_empty_tuple(prepared_sifts: None) -> None:
    # Not `None`: `()` means this chain has no protein, which is true of every DNA chain.
    assert Structure.from_file(_BNA, id="1BNA")["A"].uniprot == ()


def test_a_structure_sifts_does_not_carry_answers_with_an_empty_tuple_too(
    prepared_sifts: None,
) -> None:
    assert Structure.from_file(_UBQ, id="not-an-entry")["A"].uniprot == ()


def test_a_map_nobody_prepared_raises_rather_than_answering_nothing(ubq: Structure) -> None:
    # Distinct from `()`, and deliberately: one means nobody built the map here, the other
    # means this chain has no protein.
    with pytest.raises(SiftsNotDownloadedError, match="protein sifts prepare"):
        _ = ubq["A"].uniprot


def test_the_lane_reaches_sifts_through_the_module_so_one_patch_takes_it_offline(
    ubq: Structure, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sifts, "accessions_for", lambda pdb, chain: ("P00000",))
    assert ubq["A"].uniprot == ("P00000",)


def test_the_chain_label_is_passed_through_exactly_as_the_file_spells_it(
    bna: Structure, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, str]] = []

    def record(pdb: str, chain: str) -> tuple[str, ...]:
        seen.append((pdb, chain))
        return ()

    monkeypatch.setattr(sifts, "accessions_for", record)
    assert bna["B"].uniprot == ()
    assert seen == [("1BNA", "B")]


# --- search --------------------------------------------------------------------


def test_a_chain_writes_its_own_coordinates_under_its_own_key(
    ubq: Structure, pdb_database: Path, runs: _Runs
) -> None:
    # Foldseek reports a one-chain query under the file's stem, so the file's name is what
    # the `query` column will say.
    ubq["A"].search("pdb")
    name, _content = runs.queries[0]
    assert name == "1UBQ_A.cif"


def test_the_written_query_holds_that_chain_and_nothing_else(
    bna: Structure, pdb_database: Path, runs: _Runs, tmp_path: Path
) -> None:
    bna["B"].search("pdb")
    _name, content = runs.queries[0]
    assert io.read_atoms(_written(tmp_path, content)).array_length() == len(bna["B"])


def test_a_chain_label_of_more_than_one_character_survives_the_query(
    tmp_path: Path, pdb_database: Path, runs: _Runs
) -> None:
    # `foldseek convert2pdb` truncates such a label in silence and a PDB file cannot hold
    # one at all. The query is mmCIF, so the label survives in both the file's name -- which
    # is what Foldseek reports a one-chain query under -- and in the file.
    atoms = io.read_atoms(_UBQ).copy()
    atoms.chain_id = np.full(atoms.array_length(), "Q1", dtype="U4")
    relabelled = tmp_path / "1ubq.cif"
    io.write_atoms(relabelled, atoms)

    Structure.from_file(relabelled)["Q1"].search("pdb")
    name, content = runs.queries[0]
    assert name == "1ubq_Q1.cif"
    assert io.chain_ids(io.read_atoms(_written(tmp_path, content))) == ("Q1",)


def _written(tmp_path: Path, content: bytes) -> Path:
    """Put a recorded query file back on disk, since the real one has been cleaned up."""
    replayed = tmp_path / "replayed.cif"
    replayed.write_bytes(content)
    return replayed


def test_a_chain_search_forwards_the_knobs(ubq: Structure, pdb_database: Path, runs: _Runs) -> None:
    ubq["A"].search("pdb", sensitivity=1.0, threads=4)
    assert runs.calls[0][-4:] == ["-s", "1.0", "--threads", "4"]


def test_a_chain_search_leaves_nothing_behind(
    ubq: Structure, pdb_database: Path, runs: _Runs, liulab_data: Path
) -> None:
    ubq["A"].search("pdb")
    work = liulab_data / "protein" / ".work"
    assert list(work.iterdir()) == []
