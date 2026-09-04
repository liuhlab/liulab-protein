"""The model lane: one real ESM-C forward pass and one real SAE over it, in `esm`.

    pixi run -e esm pytest -m model

`-m "not model"` is in `addopts`, so plain `pytest` deselects every test here and the gate
stays green on a machine with no weights. Selecting the lane by hand is the only way in.

**Nothing here skips.** A lane that selects its tests, skips them all and exits 0 reports a
pass having run nothing, so a missing `torch`, a missing `esm` or a cold HF cache fails
loudly instead.

`HF_HUB_OFFLINE` makes "the model lane embeds, it does not download" true rather than merely
intended: the suite's autouse network guard cannot be opted out of, and hub clients raise
straight through it rather than falling back to the cache. It is read once at the hub
client's import, so it is set here at collection and not in a fixture.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os

import numpy as np
import pytest

from protein import ESMC, Protein
from protein.embed import CHECKPOINTS, SAE, SAE_CHECKPOINTS, Embedding

os.environ.setdefault("HF_HUB_OFFLINE", "1")

pytestmark = pytest.mark.model

#: The lane's one query. Its length is what the row count is checked against.
SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"

#: What ESMC-300M is, independent of what any object reports.
D_MODEL = 960
N_LAYERS = 30


_RECIPE = (
    "    export HF_HOME=/path/to/hf-cache\n"
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

    # Offline mode is on, so a cold cache would otherwise fail deep inside `from_pretrained`
    # naming neither HF_HOME nor the fix.
    if not isinstance(try_to_load_from_cache(CHECKPOINTS["300m"][0], "config.json"), str):
        pytest.fail(
            f"model lane: {CHECKPOINTS['300m'][0]} is not in the Hugging Face cache "
            f"this process can see. Point HF_HOME at a cache holding it, or fetch it "
            f"there once: this lane itself never downloads.\n\n{_RECIPE}"
        )


@pytest.fixture(scope="session")
def esmc() -> ESMC:
    """One ESMC-300M, loaded once for the whole lane rather than per test."""
    _require_the_esm_environment()
    # The published config predates the ESM-C field rename, so loading it raises a real
    # FutureWarning. Asserted rather than ignored, so an upstream fix shows up as a failing
    # test rather than as a stale entry in pyproject.toml that nobody removes.
    with pytest.warns(FutureWarning, match="pre-alignment ESMC field names"):
        model = ESMC()
    return model


def test_the_lane_reports_which_device_it_actually_ran_on(
    esmc: ESMC, capsys: pytest.CaptureFixture[str]
) -> None:
    # Reported, not asserted: the CPU is a supported fallback. But a run that took minutes
    # because it quietly landed there should say so in its own output.
    with capsys.disabled():
        print(f"\nESMC({esmc.checkpoint!r}) is on {esmc.device}")
    assert esmc.device in {"cpu", "cuda"} or esmc.device.startswith("cuda:")


def test_the_checkpoint_reports_the_width_and_depth_the_table_claims(esmc: ESMC) -> None:
    assert esmc.d_model == D_MODEL == CHECKPOINTS[esmc.checkpoint][1]
    assert esmc.n_layers == N_LAYERS


def test_one_real_embedding_has_one_row_per_residue_and_not_two_more(esmc: ESMC) -> None:
    # THE assertion. ESM-C's tokenizer adds BOS and EOS; the lane strips both.
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
    # `-1` does not survive onto the returned object: an index into `hidden_states` means
    # something on its own a week later.
    assert esmc.embed(Protein(SEQUENCE, id="test")).layer == N_LAYERS


def test_asking_for_the_last_layer_by_number_gives_the_same_array_as_the_default(
    esmc: ESMC,
) -> None:
    # The fast path is only sound because `hidden_states[-1]` IS `last_hidden_state`: the
    # default reads the latter without materialising every layer. If the two ever diverge,
    # this fails rather than someone's analysis.
    protein = Protein(SEQUENCE, id="test")
    np.testing.assert_array_equal(
        esmc.embed(protein).array, esmc.embed(protein, layer=N_LAYERS).array
    )


def test_the_embedding_layer_is_index_zero_and_is_not_the_last_hidden_state(
    esmc: ESMC,
) -> None:
    # `hidden_states[0]` is the embedding-layer output, so it is a real answer for `layer=0`
    # and a very different one from the default.
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


# --- the sparse autoencoder over one of those embeddings ------------------------------

#: The SAE trained on the 300M backbone: the cheapest correct pairing there is.
SAE_SLUG = "300m-layer23-k64-cb16384"
SAE_LAYER = 23
CODEBOOK_SIZE = 16384
K = 64

#: The wrong layer worth testing: `embed` returns this one by default, so pairing it with a
#: layer-23 SAE is the mistake someone actually makes.
WRONG_LAYER = N_LAYERS


def _require_the_sae_checkpoint() -> None:
    """Fail, never skip, when the SAE weights this lane needs are not in the cache."""
    from huggingface_hub import (  # pyright: ignore[reportMissingImports]
        try_to_load_from_cache,
    )

    hf_id = SAE_CHECKPOINTS[SAE_SLUG][0]
    if not isinstance(try_to_load_from_cache(hf_id, "config.json"), str):
        pytest.fail(
            f"model lane: {hf_id} is not in the Hugging Face cache this process can see. "
            f"Point HF_HOME at a cache holding it, or fetch it there once: this lane itself "
            f"never downloads.\n\n{_RECIPE}"
        )


@pytest.fixture(scope="session")
def sae() -> SAE:
    """One SAE over ESMC-300M layer 23, loaded once for the whole lane."""
    _require_the_esm_environment()
    _require_the_sae_checkpoint()
    return SAE(SAE_SLUG)


@pytest.fixture(scope="session")
def embedding(esmc: ESMC) -> Embedding:
    """The layer that SAE was trained on, embedded once."""
    return esmc.embed(Protein(SEQUENCE, id="P12345"), layer=SAE_LAYER)


def test_the_sae_reports_what_its_own_config_says_it_is(sae: SAE) -> None:
    # Read off the checkpoint, not off the table: the table adds the parent and nothing else.
    assert sae.parent == "300m"
    assert sae.layers == (SAE_LAYER,)
    assert sae.d_model == D_MODEL
    assert sae.codebook_size == CODEBOOK_SIZE
    assert sae.k == K


def test_one_real_encode_fills_k_slots_per_residue_over_the_whole_codebook(
    sae: SAE, embedding: Embedding
) -> None:
    activation = sae.encode(embedding)
    assert activation.shape == (len(SEQUENCE), CODEBOOK_SIZE)
    assert activation.indices.shape == activation.values.shape == (len(SEQUENCE), K)
    assert activation.reconstruction_loss.shape == (len(SEQUENCE),)


def test_no_row_is_added_or_lost_because_bos_and_eos_never_enter_this_path(
    sae: SAE, embedding: Embedding
) -> None:
    # The standalone layer applies no token mask and no flattening reshape, so there is no
    # `L + 2` to strip here: one row in, one row out.
    assert len(sae.encode(embedding)) == len(embedding) == len(SEQUENCE)


def test_the_stored_dtypes_and_bounds_survive_a_real_forward_pass(
    sae: SAE, embedding: Embedding
) -> None:
    activation = sae.encode(embedding)
    assert activation.indices.dtype == np.dtype(np.uint16)
    assert activation.values.dtype == np.dtype(np.float16)
    assert int(activation.indices.max()) < CODEBOOK_SIZE
    # The encoder is a ReLU followed by a top-k, so nothing that comes back is negative.
    assert (activation.values >= 0).all()
    assert np.isfinite(activation.reconstruction_loss).all()


def test_the_activation_records_the_pairing_it_came_out_of(sae: SAE, embedding: Embedding) -> None:
    activation = sae.encode(embedding)
    assert activation.source == "P12345"
    assert activation.parent == "300m"
    assert activation.layer == SAE_LAYER
    assert activation.sae == SAE_SLUG
    assert activation.normalized is False


def test_asking_to_normalise_a_checkpoint_that_ships_no_statistics_raises(
    sae: SAE, embedding: Embedding
) -> None:
    # Off the loaded buffers, not off the table: these three checkpoints ship ones, so the
    # request would otherwise scale by one and report success.
    with pytest.raises(ValueError, match="ships no per-feature statistics"):
        sae.encode(embedding, normalize=True)


def test_the_reconstruction_loss_betrays_a_layer_the_identity_check_cannot(
    esmc: ESMC, sae: SAE, embedding: Embedding, capsys: pytest.CaptureFixture[str]
) -> None:
    # THE assertion, and why this lane exists. The mislabelled embedding passes every check
    # `encode` makes — its recorded facts say layer 23 — and only the reconstruction says
    # otherwise.
    mislabelled = dataclasses.replace(
        esmc.embed(Protein(SEQUENCE, id="P12345"), layer=WRONG_LAYER), layer=SAE_LAYER
    )
    right = float(sae.encode(embedding).reconstruction_loss.mean())
    wrong = float(sae.encode(mislabelled).reconstruction_loss.mean())
    with capsys.disabled():
        print(
            f"\nreconstruction loss: layer {SAE_LAYER} {right:.4f}, layer {WRONG_LAYER} {wrong:.4f}"
        )
    assert wrong > 5 * right
