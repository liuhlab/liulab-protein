"""`align(sequences, query=...)`: what reaches MUSCLE, and what comes back anchored.

biotite owns the subprocess here — `Muscle5App.align()` builds the temporary files and the
command line itself — so the seam a test can stand at is the app object. Every test below
puts a stand-in there and a `RecordingTool` behind `bin_path`, which means the binary is
never run and nothing is invented about what MUSCLE would have printed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import pytest
from biotite.application.muscle import Muscle5App
from biotite.sequence import ProteinSequence
from biotite.sequence.align import Alignment

from protein import MSA, msa
from protein.external import RecordingTool
from protein.msa import align
from protein.seq import ResidueCoercionWarning

#: Three of one length, so the stand-in's default answer is a symmetric alignment.
_HOMOLOGUES = {
    "P01308": "MKTAYIAK",
    "Q6YK33": "MKTAWIAK",
    "A0A0A0MRZ7": "MKTAYIAR",
}

#: One sequence a residue longer than the rest, and the rows MUSCLE would write for them.
_UNEVEN = {
    "P01308": "MKTAYIAK",
    "Q6YK33": "MKTAWIAKW",
    "A0A0A0MRZ7": "MKTAYIAR",
}
_UNEVEN_ALIGNED = ["MKTAYIAK-", "MKTAWIAKW", "MKTAYIAR-"]


@dataclass
class _Muscle:
    """What `align` handed biotite, and what biotite is told to hand back.

    Set `gapped` to the rows MUSCLE would have written; left empty, each sequence comes back
    aligned to itself.
    """

    gapped: list[str] = field(default_factory=list)
    sequences: list[ProteinSequence] = field(default_factory=list)
    bin_path: str = ""
    calls: int = 0


@pytest.fixture
def muscle(monkeypatch: pytest.MonkeyPatch) -> _Muscle:
    """Stand in for biotite's `Muscle5App` and record what `align` gave it."""
    recorded = _Muscle()

    class _App:
        @classmethod
        def align(cls, sequences: Iterable[ProteinSequence], bin_path: str = "muscle") -> Alignment:
            recorded.sequences = list(sequences)
            recorded.bin_path = bin_path
            recorded.calls += 1
            rows = recorded.gapped or [str(sequence) for sequence in recorded.sequences]
            return Alignment(recorded.sequences, Alignment.trace_from_strings(rows), None)

    monkeypatch.setattr(msa, "Muscle5App", _App)
    return recorded


@pytest.fixture
def tool() -> RecordingTool:
    """A stand-in binary, so `align` has a path to hand over and nothing to run."""
    return RecordingTool("muscle", version="muscle 5.3.linux64")


# --- what reaches MUSCLE -----------------------------------------------------


def test_align_drives_muscle_5_and_not_the_app_that_wraps_muscle_3() -> None:
    # MuscleApp is the version-3 interface and spells its arguments differently.
    assert msa.Muscle5App is Muscle5App


def test_align_hands_over_the_path_protein_external_located(
    muscle: _Muscle, tool: RecordingTool
) -> None:
    align(_HOMOLOGUES, query="P01308", tool=tool)
    assert muscle.bin_path == tool.path


def test_align_never_runs_the_binary_through_this_packages_own_boundary(
    muscle: _Muscle, tool: RecordingTool
) -> None:
    align(_HOMOLOGUES, query="P01308", tool=tool)
    assert tool.calls == []
    assert muscle.calls == 1


def test_the_sequences_reaching_muscle_are_biotites_own_type(
    muscle: _Muscle, tool: RecordingTool
) -> None:
    align(_HOMOLOGUES, query="P01308", tool=tool)
    assert [type(sequence) for sequence in muscle.sequences] == [ProteinSequence] * 3


def test_the_sequences_reaching_muscle_came_through_this_packages_one_door(
    muscle: _Muscle, tool: RecordingTool
) -> None:
    # ADR-0002: `to_protein_sequence` folds U to X and says so. The alignment is over those
    # same objects, so nothing is re-parsed from a string on the way back.
    with pytest.warns(ResidueCoercionWarning):
        align({"a": "MKUAYIAK", "b": "MKTAYIAK"}, query="a", tool=tool)
    assert str(muscle.sequences[0]) == "MKXAYIAK"


def test_the_sequences_reach_muscle_in_the_order_they_were_given(
    muscle: _Muscle, tool: RecordingTool
) -> None:
    align(_HOMOLOGUES, query="A0A0A0MRZ7", tool=tool)
    assert [str(sequence) for sequence in muscle.sequences] == list(_HOMOLOGUES.values())


# --- what comes back ---------------------------------------------------------


def test_align_answers_an_msa_and_not_a_biotite_alignment(
    muscle: _Muscle, tool: RecordingTool
) -> None:
    assert isinstance(align(_HOMOLOGUES, query="P01308", tool=tool), MSA)


def test_the_result_is_anchored_on_the_query_that_was_designated(
    muscle: _Muscle, tool: RecordingTool
) -> None:
    muscle.gapped = list(_UNEVEN_ALIGNED)
    aligned = align(_UNEVEN, query="Q6YK33", tool=tool)
    assert aligned.query_header == "Q6YK33"
    assert aligned.query == "MKTAWIAKW"


def test_a_symmetric_alignment_comes_back_compressed_into_a_query_anchored_one(
    muscle: _Muscle, tool: RecordingTool
) -> None:
    # The query's gap column stops being a column: the row that filled it carries an
    # insertion, and the row that did not loses the gap altogether.
    muscle.gapped = list(_UNEVEN_ALIGNED)
    aligned = align(_UNEVEN, query="P01308", tool=tool)
    assert aligned.query == "MKTAYIAK"
    assert dict(aligned.rows)["Q6YK33"] == "MKTAWIAKw"
    assert dict(aligned.rows)["A0A0A0MRZ7"] == "MKTAYIAR"


def test_the_headers_come_back_as_they_were_given(muscle: _Muscle, tool: RecordingTool) -> None:
    aligned = align(_HOMOLOGUES, query="Q6YK33", tool=tool)
    assert [header for header, _ in aligned.rows] == ["Q6YK33", "P01308", "A0A0A0MRZ7"]


def test_the_result_holds_every_sequence_that_was_given(
    muscle: _Muscle, tool: RecordingTool
) -> None:
    assert align(_HOMOLOGUES, query="P01308", tool=tool).depth == len(_HOMOLOGUES)


# --- what it takes -----------------------------------------------------------


def test_align_takes_the_pairs_the_fasta_reader_yields(
    muscle: _Muscle, tool: RecordingTool
) -> None:
    pairs: Sequence[tuple[str, str]] = list(_HOMOLOGUES.items())
    assert align(pairs, query="P01308", tool=tool).depth == 3


def test_align_takes_a_generator_of_pairs(muscle: _Muscle, tool: RecordingTool) -> None:
    pairs = ((header, row) for header, row in _HOMOLOGUES.items())
    assert align(pairs, query="P01308", tool=tool).query_header == "P01308"


# --- what it refuses ---------------------------------------------------------


def test_a_query_the_set_does_not_hold_raises_and_names_the_headers_there_are(
    muscle: _Muscle, tool: RecordingTool
) -> None:
    with pytest.raises(LookupError, match="P01308") as raised:
        align(_HOMOLOGUES, query="P99999", tool=tool)
    assert "P99999" in str(raised.value)
    assert muscle.calls == 0


def test_one_sequence_is_not_an_alignment_and_muscle_is_never_reached(
    muscle: _Muscle, tool: RecordingTool
) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        align({"P01308": "MKTAYIAK"}, query="P01308", tool=tool)
    assert muscle.calls == 0
