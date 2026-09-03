"""The model lane: one real ESM-C forward pass, on GPU71FM, in the `esm` environment.

    pixi run -e esm pytest -m model

`-m "not model"` is in `addopts`, so plain `pytest` deselects every test here and the gate
stays green on a machine with no weights. Selecting the lane by hand is the only way in.

**Nothing here skips.** Genome's `require_tools.sh` was written because a skip is green: a
lane that selects its tests, skips them all and exits 0 reports a pass having run nothing.
So a missing `torch`, a missing `esm` or a cold HF cache fails this lane loudly instead —
you asked for the model lane, and not running the model is not a smaller answer to that.

`HF_HUB_OFFLINE` is set below rather than left to chance. The suite's network guard is
autouse and cannot be opted out of, and hub clients raise straight through it rather than
falling back to the cache, so offline mode is what makes "the model lane embeds, it does not
download" true rather than merely intended. It is read once, at the hub client's import, so
it is set here at collection and not in a fixture.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from protein import ESMC, Protein
from protein.embed import CHECKPOINTS

os.environ.setdefault("HF_HUB_OFFLINE", "1")

pytestmark = pytest.mark.model

#: 33 residues, so the assertion the whole map is sharpest about — 33 rows and not 35 — is
#: readable in the failure message.
SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"

#: What ESMC-300M is, independent of what any object reports.
D_MODEL = 960
N_LAYERS = 30


_RECIPE = (
    "    export HF_HOME=/scratch/zhoulab/hanliu/protein/hf\n"
    "    pixi run -e esm pytest -m model\n\n"
    "Refusing to skip. A skipped model lane is a green run that embedded nothing, which is "
    "the failure this check exists to end."
)


def _require_the_esm_environment() -> None:
    """Fail, never skip, when this lane cannot do the thing it was selected to do."""
    missing = [name for name in ("torch", "esm") if importlib.util.find_spec(name) is None]
    if missing:
        pytest.fail(
            f"model lane: {', '.join(missing)} not importable. Both live in the `esm` "
            f"environment, which exists to hold them.\n\n{_RECIPE}"
        )
    from huggingface_hub import (  # pyright: ignore[reportMissingImports]
        try_to_load_from_cache,
    )

    # The weights, proved present rather than assumed: offline mode is on, so a cold cache
    # would otherwise fail deep inside `from_pretrained` naming neither HF_HOME nor the fix.
    if not isinstance(try_to_load_from_cache(CHECKPOINTS["300m"][0], "config.json"), str):
        pytest.fail(
            f"model lane: {CHECKPOINTS['300m'][0]} is not in the Hugging Face cache "
            f"this process can see. It is 1.3 GB and it is already on GPU71FM.\n\n{_RECIPE}"
        )


@pytest.fixture(scope="session")
def esmc() -> ESMC:
    """One ESMC-300M, loaded once for the whole lane — 1.33 GB is not per test."""
    _require_the_esm_environment()
    # biohub/ESMC-300M's config.json predates the ESM-C field rename, so loading it raises a
    # real FutureWarning naming both spellings (#19 measured it under `simplefilter("error")`).
    # Asserted rather than ignored: `filterwarnings = ["error"]` would fail this lane
    # otherwise, and pinning it here means an upstream fix shows up as a failing test rather
    # than as a stale entry in pyproject.toml that nobody removes.
    with pytest.warns(FutureWarning, match="pre-alignment ESMC field names"):
        model = ESMC()
    return model


def test_the_lane_reports_which_device_it_actually_ran_on(
    esmc: ESMC, capsys: pytest.CaptureFixture[str]
) -> None:
    # Reported, not asserted: #1 documents CPU as a supported fallback and #19 declined to
    # make GPU availability an environment assertion. But a run that took minutes because it
    # quietly landed on the CPU should say so in its own output.
    with capsys.disabled():
        print(f"\nESMC({esmc.checkpoint!r}) is on {esmc.device}")
    assert esmc.device in {"cpu", "cuda"} or esmc.device.startswith("cuda:")


def test_the_checkpoint_reports_the_width_and_depth_the_table_claims(esmc: ESMC) -> None:
    assert esmc.d_model == D_MODEL == CHECKPOINTS[esmc.checkpoint][1]
    assert esmc.n_layers == N_LAYERS


def test_one_real_embedding_has_one_row_per_residue_and_not_two_more(esmc: ESMC) -> None:
    # THE assertion. `last_hidden_state` is (1, L + 2, d_model) because ESM-C's tokenizer
    # adds BOS and EOS; the lane strips both, so 33 residues is 33 rows.
    protein = Protein(SEQUENCE, id="test")
    embedding = esmc.embed(protein)
    assert embedding.shape == (33, 960)
    assert len(embedding) == len(protein)


def test_the_returned_array_is_float32_on_the_cpu_whatever_the_model_ran_on(
    esmc: ESMC,
) -> None:
    embedding = esmc.embed(Protein(SEQUENCE, id="test"))
    assert isinstance(embedding.array, np.ndarray)
    assert embedding.array.dtype == np.float32
    assert np.isfinite(embedding.array).all()


def test_the_embedding_records_where_it_came_from(esmc: ESMC) -> None:
    embedding = esmc.embed(Protein(SEQUENCE, id="P12345"))
    assert embedding.source == "P12345"
    assert embedding.checkpoint == "300m"


def test_the_default_layer_is_the_last_hidden_state_under_its_own_number(esmc: ESMC) -> None:
    # `-1` does not survive onto the returned object: 30 is what a notebook reads a week
    # later, and 30 is an index into `hidden_states` that means something on its own.
    assert esmc.embed(Protein(SEQUENCE, id="test")).layer == N_LAYERS


def test_asking_for_the_last_layer_by_number_gives_the_same_array_as_the_default(
    esmc: ESMC,
) -> None:
    # The fast path is only sound because `hidden_states[-1]` IS `last_hidden_state`:
    # `layer=-1` reads the latter without asking for all 31 tensors, and `layer=30` reads
    # the former. If the two ever diverge, this fails rather than someone's analysis.
    protein = Protein(SEQUENCE, id="test")
    np.testing.assert_array_equal(
        esmc.embed(protein).array, esmc.embed(protein, layer=N_LAYERS).array
    )


def test_the_embedding_layer_is_index_zero_and_is_not_the_last_hidden_state(
    esmc: ESMC,
) -> None:
    # What #7 left unchecked and #13 was told to measure: `hidden_states[0]` is the
    # embedding-layer output, so it is a real answer for `layer=0` and a very different one
    # from the default.
    protein = Protein(SEQUENCE, id="test")
    first = esmc.embed(protein, layer=0)
    assert first.layer == 0
    assert first.shape == (33, 960)
    assert not np.allclose(first.array, esmc.embed(protein).array)


def test_a_negative_layer_counts_back_from_the_last_hidden_state(esmc: ESMC) -> None:
    protein = Protein(SEQUENCE, id="test")
    assert esmc.embed(protein, layer=-(N_LAYERS + 1)).layer == 0
    assert esmc.embed(protein, layer=-2).layer == N_LAYERS - 1


def test_a_layer_outside_the_range_is_refused_before_the_model_runs(esmc: ESMC) -> None:
    with pytest.raises(ValueError, match=r"30 transformer layers"):
        esmc.embed(Protein(SEQUENCE, id="test"), layer=N_LAYERS + 1)


def test_the_per_sequence_vector_is_one_row_per_model_dimension(esmc: ESMC) -> None:
    pooled = esmc.embed(Protein(SEQUENCE, id="test")).mean()
    assert pooled.shape == (D_MODEL,)


def test_a_bare_string_is_refused_because_it_carries_no_identity(esmc: ESMC) -> None:
    with pytest.raises(TypeError, match=r"takes a Protein or a Chain"):
        esmc.embed(SEQUENCE)  # type: ignore
