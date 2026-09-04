"""Tests for `protein.fold.request` — the input, and every check upstream does not make.

All of it runs in the gate: a `FoldingRequest` is a value object, so there are no weights
anywhere here and no GPU. The one real fold lives in `tests/test_fold_model.py`.
"""

from __future__ import annotations

import ast
import inspect
import json
import warnings
from pathlib import Path

import pytest
from biotite.sequence import NucleotideSequence, ProteinSequence

from protein import MSA, Protein
from protein.fold import CHAIN_FIELDS, POLYMERS, ChainRequest, FoldingRequest
from protein.seq import InvalidResidueError, ResidueCoercionWarning

_SOURCE = Path(__file__).resolve().parents[1] / "src" / "protein" / "fold" / "request.py"

#: A query and two rows over it: one carries the `key=` field that pairs chains, one does not.
QUERY = "MKTAY"
_ROWS = [("query", QUERY), ("hit key=9606", "MKTaAY"), ("hit", "MK-AY")]


def alignment() -> MSA:
    """Return an alignment on QUERY whose rows mix paired and unpaired headers."""
    return MSA(_ROWS)


def test_a_request_holds_one_entry_per_chain_each_naming_its_kind_and_sequence() -> None:
    request = FoldingRequest(
        [ChainRequest("protein", QUERY), ChainRequest("dna", "ACGT"), ChainRequest("rna", "ACGU")]
    )
    assert len(request) == 3
    assert [chain.kind for chain in request.chains] == ["protein", "dna", "rna"]
    assert [str(chain.sequence) for chain in request.chains] == [QUERY, "ACGT", "ACGT"]


def test_the_three_kinds_are_the_ones_the_module_names() -> None:
    assert POLYMERS == ("protein", "dna", "rna")


def test_an_unknown_kind_fails_by_name_and_lists_the_three() -> None:
    with pytest.raises(ValueError, match=r"unknown chain kind 'nucleic'"):
        ChainRequest("nucleic", "ACGT")  # type: ignore[arg-type]


def test_a_protein_chain_holds_biotites_protein_sequence_and_a_nucleic_one_the_other() -> None:
    assert isinstance(ChainRequest("protein", QUERY).sequence, ProteinSequence)
    assert isinstance(ChainRequest("dna", "ACGT").sequence, NucleotideSequence)


def test_a_sequence_outside_its_kinds_alphabet_is_refused_at_construction() -> None:
    with pytest.raises(InvalidResidueError):
        ChainRequest("protein", "MK*TAY")
    with pytest.raises(InvalidResidueError) as refused:
        ChainRequest("dna", "ACGQ")
    assert refused.value.alphabet == "nucleic"


def test_an_rna_chain_is_stored_as_thymine_and_spelt_back_as_uracil() -> None:
    # biotite's alphabet has no U, and the model's RNA alphabet has no T. The fold is loud
    # in one direction and exact in the other.
    with pytest.warns(ResidueCoercionWarning):
        chain = ChainRequest("rna", "ACGU")
    assert str(chain.sequence) == "ACGT"
    assert chain.residues == "ACGU"


def test_a_dna_chain_is_spelt_to_the_tokenizer_as_it_is_stored() -> None:
    assert ChainRequest("dna", "ACGT").residues == "ACGT"


def test_a_protein_chain_accepts_an_alignment() -> None:
    chain = ChainRequest("protein", QUERY, alignment=alignment())
    assert chain.alignment is not None
    assert chain.alignment.depth == 3


@pytest.mark.parametrize("kind", ["dna", "rna"])
def test_a_nucleic_chain_refuses_an_alignment_rather_than_carrying_a_dropped_one(
    kind: str,
) -> None:
    # Upstream's DNAInput has no such field and RNAInput's is read by nothing, so an
    # alignment attached there is accepted, carried and dropped without a word.
    with pytest.raises(ValueError, match=r"takes no alignment"):
        ChainRequest(kind, "ACGT", alignment=alignment())  # type: ignore[arg-type]


def test_a_nucleic_chain_carries_no_alignment_at_all() -> None:
    assert ChainRequest("dna", "ACGT").alignment is None


def test_an_alignment_whose_query_row_is_not_the_chains_sequence_is_refused() -> None:
    with pytest.raises(ValueError, match=r"query row is not this chain's sequence"):
        ChainRequest("protein", "MKTAW", alignment=alignment())


def test_an_alignment_of_the_wrong_length_is_refused_before_upstream_pads_it() -> None:
    with pytest.raises(ValueError, match=r"5 match states and the chain has 6 residues"):
        ChainRequest("protein", "MKTAYQ", alignment=alignment())


def test_a_protein_chain_with_no_alignment_yields_the_depth_one_alignment_itself() -> None:
    # Which is what upstream builds one level down, after a per-chain `warnings.warn` that
    # `filterwarnings = ["error"]` would turn into an exception.
    chain = ChainRequest("protein", QUERY)
    assert chain.alignment is not None
    assert chain.alignment.depth == 1
    assert chain.alignment.query == QUERY


def test_the_derived_alignment_names_the_accession_when_there_is_one() -> None:
    named = ChainRequest("protein", QUERY, accession="P12345").alignment
    anonymous = ChainRequest("protein", QUERY).alignment
    assert named is not None
    assert anonymous is not None
    assert named.query_header == "P12345"
    assert anonymous.query_header == "query"


def test_building_a_request_with_no_alignment_warns_about_nothing() -> None:
    # The default path, and the reason this package builds the depth-1 alignment itself
    # rather than tolerating a new `filterwarnings` entry.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FoldingRequest([ChainRequest("protein", QUERY), ChainRequest("dna", "ACGT")])


def test_paired_and_unpaired_rows_are_one_field_and_their_headers_survive() -> None:
    # An unpaired row is one whose header carries no `key=`, so the two are told apart by
    # reading a header and never by which of two lists a row sits in.
    given = alignment()
    chain = ChainRequest("protein", QUERY, alignment=given)
    assert chain.alignment is not None
    assert chain.alignment.rows == given.rows
    assert chain.alignment.to_a3m() == given.to_a3m()


def test_a_chain_can_be_built_from_a_protein_and_keeps_its_accession() -> None:
    chain = ChainRequest.of(Protein(QUERY, id="P12345"))
    assert chain.kind == "protein"
    assert chain.accession == "P12345"
    assert str(chain.sequence) == QUERY


def test_chain_labels_are_derived_from_position_and_run_past_twenty_six() -> None:
    request = FoldingRequest([ChainRequest("protein", QUERY) for _ in range(28)])
    assert request.chain_ids[:3] == ("A", "B", "C")
    assert request.chain_ids[25:] == ("Z", "AA", "AB")


def test_the_accession_map_names_every_chain_and_answers_empty_for_the_rest() -> None:
    request = FoldingRequest(
        [ChainRequest("protein", QUERY, accession="P12345"), ChainRequest("dna", "ACGT")]
    )
    assert request.accessions == {"A": ("P12345",), "B": ()}


def test_a_request_holding_no_chains_is_refused() -> None:
    with pytest.raises(ValueError, match=r"holds at least one chain"):
        FoldingRequest([])


def test_a_request_has_no_output_path_field() -> None:
    # Where the answer is written is not an input, so one request folds to two destinations.
    request = FoldingRequest([ChainRequest("protein", QUERY)])
    assert not hasattr(request, "out")
    assert "out" not in inspect.signature(FoldingRequest.__init__).parameters
    with pytest.raises(TypeError):
        FoldingRequest([ChainRequest("protein", QUERY)], out="/tmp/somewhere")  # type: ignore[call-arg]


def test_a_request_is_built_from_mappings_so_a_caller_imports_no_chain_class() -> None:
    request = FoldingRequest(
        [
            {"kind": "protein", "sequence": QUERY, "accession": "P12345"},
            {"kind": "dna", "sequence": "ACGT"},
        ]
    )
    assert [chain.kind for chain in request.chains] == ["protein", "dna"]
    assert request.accessions == {"A": ("P12345",), "B": ()}


def test_the_fields_a_mapping_may_name_are_the_ones_a_chain_takes() -> None:
    # Read off `ChainRequest` and not written out, so a parameter added there is a field
    # the mapping form takes rather than one it refuses as unknown.
    assert tuple(inspect.signature(ChainRequest).parameters) == CHAIN_FIELDS
    assert CHAIN_FIELDS == ("kind", "sequence", "accession", "alignment")


def test_a_mapping_names_its_kind_because_nothing_here_guesses_one() -> None:
    # 'ACGT' is a valid protein sequence, so a guess is the silent wrong answer this lane
    # exists to refuse.
    with pytest.raises(ValueError, match=r"names its kind.*protein, dna, rna"):
        FoldingRequest([{"sequence": "ACGT"}])


def test_a_mapping_names_its_sequence() -> None:
    with pytest.raises(ValueError, match=r"names its sequence"):
        FoldingRequest([{"kind": "protein", "accession": "P12345"}])


def test_a_misspelt_field_is_named_rather_than_dropped() -> None:
    # Dropped, a typo'd accession loses the provenance a prediction is written with and
    # says nothing about it.
    with pytest.raises(ValueError, match=r"unknown chain field 'accesion'"):
        FoldingRequest([{"kind": "protein", "sequence": QUERY, "accesion": "P12345"}])


def test_a_mapping_carries_an_alignment_as_an_msa() -> None:
    request = FoldingRequest([{"kind": "protein", "sequence": QUERY, "alignment": alignment()}])
    held = request.chains[0].alignment
    assert held is not None
    assert held.depth == 3


def test_a_mapping_carries_an_alignment_as_the_path_to_one(tmp_path: Path) -> None:
    # The JSON-shaped spelling: a path is a string, and an alignment never is.
    path = alignment().write(tmp_path / "query.a3m")
    request = FoldingRequest([{"kind": "protein", "sequence": QUERY, "alignment": str(path)}])
    held = request.chains[0].alignment
    assert held is not None
    assert held.rows == alignment().rows


def test_the_checks_hold_whichever_way_the_chain_was_spelt() -> None:
    with pytest.raises(ValueError, match=r"takes no alignment"):
        FoldingRequest([{"kind": "dna", "sequence": "ACGT", "alignment": alignment()}])
    with pytest.raises(ValueError, match=r"unknown chain kind 'nucleic'"):
        FoldingRequest([{"kind": "nucleic", "sequence": "ACGT"}])


def test_one_chain_needs_no_list_around_it() -> None:
    assert len(FoldingRequest({"kind": "dna", "sequence": "ACGT"})) == 1
    assert len(FoldingRequest(QUERY)) == 1
    assert len(FoldingRequest(Protein(QUERY, id="P12345"))) == 1


def test_bare_residues_are_a_protein_chain_since_that_spelling_holds_no_kind() -> None:
    chain = FoldingRequest(QUERY).chains[0]
    assert chain.kind == "protein"
    assert str(chain.sequence) == QUERY
    assert chain.accession is None


def test_a_biotite_protein_sequence_is_one_chain_and_not_one_chain_per_residue() -> None:
    # biotite's sequences iterate into single residues, so the iterable branch would shred
    # one into five one-residue chains and fold that without a word.
    request = FoldingRequest(ProteinSequence(QUERY))
    assert len(request) == 1
    assert str(request.chains[0].sequence) == QUERY


def test_a_biotite_nucleotide_sequence_is_refused_since_it_says_neither_dna_nor_rna() -> None:
    with pytest.raises(ValueError, match=r"neither dna nor rna"):
        FoldingRequest(NucleotideSequence("ACGT"))


def test_a_protein_given_as_a_chain_keeps_its_accession() -> None:
    assert FoldingRequest([Protein(QUERY, id="P12345")]).accessions == {"A": ("P12345",)}


def test_the_two_spellings_mix_in_one_request() -> None:
    request = FoldingRequest(
        [
            ChainRequest("protein", QUERY, alignment=alignment()),
            {"kind": "dna", "sequence": "ACGT"},
            Protein(QUERY, id="P12345"),
            QUERY,
        ]
    )
    assert len(request) == 4
    assert request.chain_ids == ("A", "B", "C", "D")


def test_a_request_is_its_own_spec_so_coercing_one_twice_changes_nothing() -> None:
    # What lets `fold` widen without a second code path: coercion is idempotent.
    request = FoldingRequest([ChainRequest("protein", QUERY, accession="P12345")])
    assert FoldingRequest(request).chains == request.chains


def test_something_that_is_no_chain_says_what_a_chain_is() -> None:
    with pytest.raises(TypeError, match=r"a chain is a ChainRequest, a Protein"):
        FoldingRequest([42])  # type: ignore[list-item]


def test_a_batch_of_requests_is_prepared_from_json_and_nothing_else() -> None:
    # The point of the mapping form: a file of requests, no import, no call site per entry.
    batch = json.loads(
        """[
            [{"kind": "protein", "sequence": "MKTAY", "accession": "P12345"},
             {"kind": "dna", "sequence": "ACGT"}],
            [{"kind": "protein", "sequence": "MKTAW"}]
        ]"""
    )
    requests = [FoldingRequest(entry) for entry in batch]
    assert [len(request) for request in requests] == [2, 1]
    assert requests[0].accessions == {"A": ("P12345",), "B": ()}


def test_the_module_body_imports_neither_torch_nor_esm() -> None:
    # Against the syntax tree and not `sys.modules`, so this holds in an environment where
    # torch is absent and the import would have failed anyway.
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert not [name for name in imported if name.split(".")[0] in {"torch", "esm"}]
