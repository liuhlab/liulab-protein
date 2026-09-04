---
search:
  exclude: true
---

# What is ESMC-SAE, and what would it cost this package to support it?

Research note. Answers what the `biohub/` sparse autoencoders are, how they load, what they
return, and where — if anywhere — they belong in `src/protein/embed/`.

## Answer, shortest form

**An ESMC-SAE is a two-matrix TopK autoencoder over one layer's hidden states, and it is
already installable here: `esm >= 3.4` is pinned in `pyproject.toml` and ships
`EsmcSaeModel`.** The load-bearing surprise is that the entry point is **not**
`transformers.AutoModel` — the `esm` package never calls a `transformers` Auto class and
never registers `esmc_sae` with one, so the HuggingFace overview card's example is not
runnable as written. `allow_patterns` and `device` are `EsmcSaeModel.from_pretrained`'s own
kwargs.

**The SAE normalises each position independently, which makes an `Embedding` an exact
input.** `_zscore_normalize_representation` takes the mean and std over `d_model` per row, so
dropping BOS and EOS leaves every remaining row's value unchanged. Our `(L, d_model)` float32
CPU array is row-for-row the right thing to hand it, and `Embedding.layer` already uses the
same indexing as the SAE's `available_layers` — 0 is the embedding-layer output,
`n_layers` the last hidden state. A wrapper can therefore **check** the layer and the width
rather than trust the caller.

**It cannot check the checkpoint.** The `.safetensors` files carry five tensors and no
`__metadata__` block at all, and `config.json` holds only `d_model`, `codebook_dim`, `k` and
`available_layers`. Nothing in a published SAE names its parent model or says whether it was
trained on hidden states or on MLP outputs — only the repo id string does. Feed it the wrong
layer of the right-width model and you get silent garbage.

**An SAE activation is not an `Embedding`** and should not be made into one: it is
`codebook_size` wide, not `d_model` wide, it is sparse, and its pooling operator is `max`,
not `mean`. Because TopK guarantees exactly `k` non-zeros per row, the compact form is two
`(L, k)` numpy arrays — indices and values — which keeps `embed/`'s "numpy only, nothing
imports torch" rule intact.

**The interpretability payoff is 6B-only.** Agent-generated feature descriptions exist for
exactly one checkpoint, `ESMC-6B-sae-layer60-k64-codebook16384`, whose parent is 23.66 GiB of
fp32 weights. The 300M path is cheap but has no descriptions and cannot even use the
normalisation statistics — the SDK raises on `normalize_features=True` for any 300M SAE. That
asymmetry, not the code, is what decides whether this is worth building.

**One good piece of news for the bulk rule:** the 16,384 feature descriptions come back in a
single unauthenticated GET, 7,290,131 bytes. It is a prepared set, not a per-ID call.

## Method and provenance

| | |
| --- | --- |
| Read on | 2026-09-03, from the macOS laptop. Nothing was installed, no weights were downloaded, and nothing was run — this package is `platforms = ["linux-64"]` |
| Source read | `github.com/Biohub/esm` at `main` (`pushed_at` 2026-08-27), fetched as a tarball via `codeload`. `esm/models/esmc/sae.py`, `esm/models/esmc/model.py`, `esm/models/hub.py`, `esm/sdk/api.py`, `esm/models/esmc/__init__.py`, `README.md` |
| Cookbook read | `cookbook/snippets/sae.py`, `sae_example.py`, `sparse_utils.py`, and `cookbook/tutorials/esmc_sae_feature_interpretation.ipynb` |
| Model cards | `huggingface.co/biohub/ESMC-SAE-Overview/raw/main/README.md`, read raw rather than rendered |
| Checkpoint enumeration | `huggingface.co/api/models?author=biohub&limit=1000` — 149 repos, of which 98 match `sae` |
| File layouts and sizes | `huggingface.co/api/models/biohub/<id>?blobs=true`, plus `config.json` via `/raw/main/` |
| Tensor sets and dtypes | HTTP range request for the first 1024 bytes of each `.safetensors`, header parsed directly |
| Feature API | `biohub.ai/esm/protein/api/v1alpha1/features` and `/features/{idx}`, both probed live, unauthenticated |
| Packaging | `pypi.org/pypi/esm/json` |
| Paper | bioRxiv `10.64898/2026.06.03.729735`, full PDF; the `esm` README's Citations section; `arxiv.org/abs/2606.12209` and the arXiv API |

Every claim below is marked **VERIFIED** — read directly in the source, the file listing or a
live response — or **INFERRED**. No wall-clock timing is reported anywhere in this note,
because nothing was run; where a cost is given it is arithmetic over verified shapes, and it
says so.

## 1. What the models are

**VERIFIED.** Sparse autoencoders over ESM-C internal representations, trained to decompose a
dense hidden state into a much wider, mostly-zero feature vector.
([overview card](https://huggingface.co/biohub/ESMC-SAE-Overview))

The architecture is small enough to state completely, and `esm/models/esmc/sae.py`'s
`EsmcSaeLayer.forward` is the whole of it — **VERIFIED**
([sae.py](https://github.com/Biohub/esm/blob/main/esm/models/esmc/sae.py)):

1. z-score the input per position, over `d_model`;
2. subtract `b_dec`, project through `W_enc`, ReLU;
3. keep the top `k` values and zero the rest;
4. project back through `W_dec`, add `b_dec`, and report the per-position MSE.

Three families, three parent checkpoints. The card's own naming table and the org listing
agree — **VERIFIED**:

| Parent | `d_model` | `n_layers` | Layer-specific SAE trained at | Depth |
| --- | ---: | ---: | ---: | ---: |
| `ESMC-300M` | 960 | 30 | **23** | 76.7% |
| `ESMC-600M` | 1152 | 36 | **27** | 75.0% |
| `ESMC-6B` | 2560 | 80 | **60** | 75.0% |

**The claim that the single-layer variants are layer 23 / 27 / 60 is correct — VERIFIED**,
both from the card's naming table and from every layer-specific repo id in the org listing.
The card explains the choice: "we targeted a 75% depth after various analyses showed that
representations at this depth are often the most generalizable to a variety of downstream
tasks". The 300M figure is 76.7% rather than 75% because 30 layers does not divide evenly;
**INFERRED**, the card gives no per-model rationale.

The three families, from the card's "Model Naming" section — **VERIFIED**:

| Family | Trained on | Naming | Layers per repo |
| --- | --- | --- | --- |
| Hidden states, all layers | the hidden state at every layer | `{model}-sae-k64-codebook16384` | `n_layers + 1` |
| MLP outputs, all layers | the per-layer MLP output, before the residual | `{model}-sae-mlp-k64-codebook131072` | `n_layers + 1` |
| Layer-specific | the hidden state at one layer | `{model}-sae-layer{N}-k{k}-codebook{C}` | 1 |

`available_layers` runs `0 .. n_layers` inclusive for the all-layer repos — 31, 37 and 81
entries for 300M, 600M and 6B — **VERIFIED** from each `config.json`. **That is exactly the
indexing `Embedding.layer` and `protein.embed.esm._layer_index` already use**: element 0 is
the embedding-layer output, element `n_layers` the last hidden state. Confirmed against
`EsmcModel.forward`, which builds `layers_to_collect = list(range(num_hidden_layers + 1))`,
and against `EsmcOutput.hidden_states`, documented as "one entry per block input plus the
final post-LayerNorm output" — **VERIFIED**
([model.py](https://github.com/Biohub/esm/blob/main/esm/models/esmc/model.py)).

The card's naming rule is slightly narrower than reality. It gives the hidden-states family as
`{model}-sae-k64-codebook16384` for all three parents, but `biohub/ESMC-6B-sae-k64-codebook131072`
also exists and is not documented — **VERIFIED** from the org listing.

## 2. The full checkpoint list

**VERIFIED.** 97 SAE model repos plus the overview card, from
`https://huggingface.co/api/models?author=biohub&limit=1000` (149 repos total, 98 matching
`sae`). Every id below was returned by that call; none is guessed.

Licence is identical across all of them: **`mit` and `other`**, with
`license_link` pointing at
[`THIRD_PARTY_NOTICE.md`](https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md) —
**VERIFIED** from each repo's `cardData`.

Every repo holds `config.json` plus one `layer_{i}.safetensors` per available layer. Each
`.safetensors` holds exactly five F32 tensors and **no `__metadata__` block** — `W_enc`
`(d_model, codebook)`, `W_dec` `(codebook, d_model)`, `b_dec` `(d_model,)`, `idf`
`(codebook,)`, `max` `(codebook,)` — **VERIFIED** by parsing the header bytes of
`ESMC-6B-sae-layer60-k64-codebook16384` and `ESMC-300M-sae-layer23-k64-codebook65536`.

Shard size is therefore exactly `4 * (2 * d_model * codebook + d_model + 2 * codebook)` plus a
~390-byte header. Checked against the published byte counts for six repos; every one matches
to within the header — **VERIFIED**.

### Hidden states, all layers — 4 repos

| Repo id | `k` | `codebook_size` | `d_model` | Shards | Per shard | Repo total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `biohub/ESMC-300M-sae-k64-codebook16384` | 64 | 16384 | 960 | 31 | 120 MiB | 3.6 GiB |
| `biohub/ESMC-600M-sae-k64-codebook16384` | 64 | 16384 | 1152 | 37 | 144 MiB | 5.2 GiB |
| `biohub/ESMC-6B-sae-k64-codebook16384` | 64 | 16384 | 2560 | 81 | 320 MiB | **25.3 GiB** |
| `biohub/ESMC-6B-sae-k64-codebook131072` | 64 | 131072 | 2560 | 81 | 2561.0 MiB | **202.6 GiB** |

### MLP outputs, all layers — 3 repos

| Repo id | `k` | `codebook_size` | `d_model` | Shards | Per shard | Repo total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `biohub/ESMC-300M-sae-mlp-k64-codebook131072` | 64 | 131072 | 960 | 31 | 961 MiB | 29.1 GiB |
| `biohub/ESMC-600M-sae-mlp-k64-codebook131072` | 64 | 131072 | 1152 | 37 | 1153.0 MiB | 41.7 GiB |
| `biohub/ESMC-6B-sae-mlp-k64-codebook131072` | 64 | 131072 | 2560 | 81 | 2561 MiB | **202.6 GiB** |

### Layer-specific — 90 repos

A complete `3 × 6 × 5` cross product, with no gaps — **VERIFIED**, all 90 ids present:

```text
biohub/ESMC-300M-sae-layer23-k{16,32,64,128,256,512}-codebook{8192,16384,32768,65536,131072}
biohub/ESMC-600M-sae-layer27-k{16,32,64,128,256,512}-codebook{8192,16384,32768,65536,131072}
biohub/ESMC-6B-sae-layer60-k{16,32,64,128,256,512}-codebook{8192,16384,32768,65536,131072}
```

Each is a single shard. Sizes depend only on `d_model` and `codebook`, never on `k`:

| Parent | cb8192 | cb16384 | cb32768 | cb65536 | cb131072 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ESMC-300M` (960) | 60.1 MiB | 120.1 MiB | 240.3 MiB | 480.5 MiB | 961.0 MiB |
| `ESMC-600M` (1152) | 72.1 MiB | 144.1 MiB | 288.3 MiB | 576.5 MiB | 1153.0 MiB |
| `ESMC-6B` (2560) | 160.1 MiB | 320.1 MiB | 640.3 MiB | 1280.5 MiB | 2561.0 MiB |

Sizes marked with a measured repo above are **VERIFIED** from the file listing; the rest are
**INFERRED** by the shard-size formula, which matched every measured case exactly.

### Two ids that will 404

- **`biohub/esmc-6b-2024-12-sae-k64-codebook16384`** appears in `EsmcSaeModel`'s own docstring
  example. It is not in the org listing and the HF API returns 401 for it — **VERIFIED**. The
  docstring is stale.
- **The Biohub Platform names are not the HuggingFace ids.** The cookbook uses
  `esmc-600m-2024-12_k64_codebook16384_layer27` and the tutorial notebook uses
  `esmc-6b-2024-12-sae-layer60-k64-codebook16384` — neither is a HuggingFace repo id, and the
  two are not even spelled the same as each other — **VERIFIED**
  ([sae_example.py](https://github.com/Biohub/esm/blob/main/cookbook/snippets/sae_example.py),
  [notebook](https://github.com/Biohub/esm/blob/main/cookbook/tutorials/esmc_sae_feature_interpretation.ipynb)).

There are also `-hf` repos for the *backbones* — `biohub/ESMC-300M-hf`, `-600M-hf`, `-6B-hf` —
which are genuine `transformers`-native checkpoints (`model_type: esmc`, `hidden_size`,
`num_hidden_layers`, `transformers_version: 5.16.0.dev0`). **There is no `-hf` SAE repo** —
**VERIFIED**. The `transformers`-native path does not cover SAEs at all.

## 3. How you load and run one

**`transformers` is not required, and `AutoModel` is not the entry point — VERIFIED.**

The evidence is decisive. `esm`'s ESM-C classes descend from `HubPreTrainedModel`, which is a
plain `torch.nn.Module` and not a `transformers.PreTrainedModel` — **VERIFIED**
([hub.py](https://github.com/Biohub/esm/blob/main/esm/models/hub.py)). Grepping the whole
repository for `AutoModel`, `AutoConfig`, `register_for_auto_class` and `trust_remote_code`
returns no functional use: two comments in the ESMFold2 code mention `AutoConfig` in passing,
and that is all — **VERIFIED**. The only `transformers` import anywhere in the ESM-C lane is
`PreTrainedTokenizerFast`.

So:

- **Is `AutoModel` really the entry point?** No. `EsmcSaeModel.from_pretrained` is.
- **What accepts `allow_patterns`?** `EsmcSaeModel.from_pretrained`'s own signature, which
  forwards it to `huggingface_hub.snapshot_download` — **VERIFIED**, `sae.py`
  `_resolve_snapshot_dir`. It is not standard `transformers` API and `transformers` would
  reject it.
- **Does it need `trust_remote_code=True`?** No — it never goes through `transformers`, and
  the SAE `config.json` has no `auto_map`, so `AutoModel` could not resolve `model_type:
  esmc_sae` even if asked — **VERIFIED**.
- **Why does `allow_patterns` matter so much?** Because `from_pretrained` downloads the whole
  snapshot: "downloads the entire repo (every `layer_{i}.safetensors`) into the local cache
  but does **not** load any weights into memory" — **VERIFIED**, `EsmcSaeModel` docstring.
  Omit it on `ESMC-6B-sae-mlp-k64-codebook131072` and you pull 202.6 GiB.

The overview card's `AutoModel` example is a mis-transliteration of the repo README's, which
is correct. **This is the verbatim working minimal example**, from
[`README.md`](https://github.com/Biohub/esm/blob/main/README.md) — **VERIFIED**, quoted
unaltered but for the sequence, which is elided:

```python
import torch
from esm.models.esmc import EsmcForMaskedLM, EsmcSaeModel, EsmcTokenizer

sequence = "MGSNKSKPKDASQRRRSLEPAENVHGAGG..."

model = EsmcForMaskedLM.from_pretrained("biohub/ESMC-6B", device="cuda").eval()
tokenizer = EsmcTokenizer()
sae = EsmcSaeModel.from_pretrained(
    "biohub/ESMC-6B-sae-k64-codebook16384",
    allow_patterns=["config.json", "layer_30.safetensors", "layer_60.safetensors"],
    device=model.device,
)
sae.initialize_layers([30, 60])
model.add_sae_models([sae.layers["30"], sae.layers["60"]])

inputs = tokenizer(sequence, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model(**inputs)

output.sae_outputs["layer60"]  # sparse.coo tensor
print(output.sae_outputs["layer60"].shape)
```

Two details that matter for us. A repo shipping exactly one layer **auto-loads** it, so a
layer-specific checkpoint needs no `initialize_layers` and a bare `forward(x)` works —
**VERIFIED**, `from_pretrained`. And `esm` is on PyPI at **3.4.0** — **VERIFIED** — so the
card's "a PyPI release is coming soon" is stale, and `pyproject.toml`'s existing
`esm = ">=3.4,<4"` already installs all of this.

There is a second, remote path the cookbook prefers: `ESMCForgeInferenceClient` against
`https://biohub.ai` with an `ESM_API_KEY`, passing `SAEConfig` inside `LogitsConfig` —
**VERIFIED** ([api.py](https://github.com/Biohub/esm/blob/main/esm/sdk/api.py)). It is
per-sequence remote inference and is out of scope here.

## 4. The exact output type

The claim "sparse COO tensor of shape `(batch, seq_len, codebook_size)`" is **half right**.
The layout is correct; **the shape is wrong**.

| Question | Answer | Evidence |
| --- | --- | --- |
| `torch.sparse_coo_tensor`? | **Yes** — `features.to_sparse()`, whose default layout is COO | **VERIFIED**, `_get_sae_outputs` |
| Shape? | **2D `(n_non_pad_positions, codebook_size)`**, flattened across the batch — not 3D | **VERIFIED** |
| dtype? | **float32** | **VERIFIED** — every shipped tensor is F32 and nothing upcasts |
| Values? | the TopK activations | **VERIFIED** |
| Indices? | row = position, column = feature id | **VERIFIED** |
| BOS/EOS included? | **Yes** — you get `L + 2` rows | **VERIFIED** |

The shape correction is worth being precise about, because it is the one thing a caller would
get wrong. `EsmcSaeLayer.get_sae_output` does `layer_states[token_mask].view(-1, v_len)`,
where `token_mask` is the attention mask — non-pad, so BOS and EOS survive. The result is
flat. Three independent confirmations — **VERIFIED**:

- `sparse_utils.remove_indexes` documents its argument as "A sparse COO tensor of shape
  `(num_positions, num_features)`" and raises `ValueError` for `dim() != 2`.
- The tutorial notebook writes `features = output.sae_outputs[name].to_dense().numpy()` then
  `features = features[1:-1]  # Remove BOS/EOS tokens`, and prose above it: "the feature
  matrix returned by the SAE has two extra rows, one for BOS and one for EOS".
- `cookbook/snippets/sae.py` calls `remove_indexes(sae_tensor, {0, -1})` before pooling.

**Is there a decoder?** Yes — `W_dec` and `b_dec` ship in every checkpoint and
`forward` computes `reconstructed = feature_magnitudes @ W_dec + b_dec`. **But the API does
not expose the reconstruction**: `EsmcSaeOutput` carries `feature_magnitudes` and
`reconstruction_loss` only, and the reconstructed tensor is a local that is discarded —
**VERIFIED**, `sae.py`. Getting it back means either rerunning the matmul yourself or reaching
into `layer.W_dec`.

**Is reconstruction error reported?** Yes, per position:
`reconstruction_loss = (reconstructed - x).pow(2).mean(dim=-1)` — mean squared error over
`d_model`, against the **z-scored** input, not the raw hidden state — **VERIFIED**. Note it is
computed on the direct `EsmcSaeLayer`/`EsmcSaeModel` path but **dropped on the backbone path**:
`_get_sae_outputs` keeps `sae_out.feature_magnitudes` and discards the rest, so
`output.sae_outputs` never carries a loss — **VERIFIED**. So the error is computable per call
but **published nowhere**: not on the model cards, and not in the paper either, which gives
only relative curves (§8).

**Normalisation.** `idf` and `max` are registered buffers defaulting to ones, so the scaling
is a no-op for variants that ship no statistics — **VERIFIED**, `EsmcSaeLayer.__init__`. With
`normalize_sae=True` the backbone applies `(features / max) * idf`, matching the card's
"`(activation / max) * idf`". The card states only
`ESMC-6B-sae-layer60-k64-codebook16384` has accessible statistics, and the SDK enforces a
matching restriction from the other end: `SAEConfig` raises
`normalize_features=True is not supported for ESMC 300M SAE models` — **VERIFIED**.

**Two other guards.** `_validate_sae_inputs` asserts the input carries no mask token, because
"SAEs were trained on unmasked sequences"; and `add_sae_models` refuses two SAEs on the same
backbone layer — **VERIFIED**.

## 5. The layer coupling

This is the design question, and the answer is worse than "it errors".

**What breaks, by mistake:**

| Mistake | Result |
| --- | --- |
| Wrong `codebook_size` or wrong parent width | **Errors** — a shape mismatch in `x @ W_enc`. The three parents have distinct `d_model` (960 / 1152 / 2560), so cross-parent mixups are caught by luck, not by design |
| **Wrong layer, same parent** | **Silent garbage.** Every layer of one model has the same width, so the matmul succeeds and returns plausible-looking activations |
| Hidden-states SAE fed MLP outputs, or the reverse | **Silent garbage.** Same width, same shapes, no marker anywhere |
| Same-width sibling checkpoint (`ESMC-300M` vs `ESMC-300M-step500k`) | **Silent garbage.** Identical `d_model` |

**Is there metadata a wrapper could check?** Partially, and less than you would hope —
**VERIFIED** at both the config and the file level:

- `config.json` holds exactly `available_layers`, `codebook_dim`, `d_model`, `k`,
  `model_type` and `transformers_version`. **No parent model id. No training-target field.**
- The `.safetensors` files carry **no `__metadata__` block at all** — five tensors and
  nothing else. Confirmed by parsing the headers directly.

So a wrapper can check two things and only two: that the requested layer is in
`available_layers`, and that the input width equals `d_model`. It **cannot** verify that the
array came from the right checkpoint, or from a hidden state rather than an MLP output. The
only carrier of that information is the repo id string.

`esm` does enforce what it can. `initialize_layers` raises `KeyError` for a layer not in
`available_layers`; `forward` raises `KeyError` for a layer not loaded, and `RuntimeError`
when several are loaded and none is named — **VERIFIED**. And the **backbone path is safe by
construction**: `add_sae_models` keys each SAE by `f"layer{N}"` from `layer.params.layer`, and
`_get_sae_outputs` looks up `hidden_states[layer_to_idx[N]]` for exactly that `N`, so the
layer cannot be mismatched when the SAE is attached to the model — **VERIFIED**.

**The exposure is the standalone path** — `EsmcSaeModel.forward(x, layer=...)` or
`EsmcSaeLayer.forward(x)` — where `x` is any tensor of the right width. That is precisely the
path a wrapper taking an `Embedding` would use, which is why the wrapper would have to carry
the check itself.

## 6. Feature interpretation

**Both a bulk file and a per-feature API exist. The bulk one is the good news.**

| | |
| --- | --- |
| Bulk endpoint | `GET https://biohub.ai/esm/protein/api/v1alpha1/features` |
| Per-feature endpoint | `GET https://biohub.ai/esm/protein/api/v1alpha1/features/{feature_idx}` |
| Authentication | **None.** Both returned HTTP 200 with no key, no header, no account |
| Bulk response | `{"data": [...]}`, **16,384 rows**, indices `0 .. 16383` contiguous, **7,290,131 bytes** |
| Bulk row fields | `feature_index`, `label`, `description` |
| Covers | `ESMC-6B-sae-layer60-k64-codebook16384` **only** |
| Stability | the notebook labels it "This is an alpha API and likely to change!" |

All **VERIFIED** by live request on 2026-09-03.

**This matters for the bulk-not-per-ID rule.** The tutorial notebook's own pattern is the
forbidden one — a per-feature `requests.get` behind `@lru_cache(maxsize=16384)` — but that is
a choice, not a constraint. One GET returns the entire codebook. That makes the descriptions a
**prepared set** under `genome.store.prepared`, exactly like SIFTS: one download, one file, no
network inside tests.

The per-feature record is much richer than the bulk row, and includes the normalisation
statistics the model card says are reachable "through the feature-description API" —
**VERIFIED** by fetching feature 10425:

`feature_index`, `label`, `summary`, `description`, `activation_pattern`, `category`,
`exemplar_protein_families`, `uniref90_frequency`, `uniref90_idf`, `uniref90_max_activation`,
`top_100_uniref_ids`, `top_swissprot_activations`, `decoder_nearest_neighbors`, `threshold`.

Feature 10425's `label` is "P-loop NTPase switch module", `category` is "Ligand-binding site",
`uniref90_idf` is 3.065 and `uniref90_max_activation` is 17.375. So the full statistics are
reachable, but **only per feature** — the bulk endpoint returns three fields per row. Fetching
`uniref90_idf` for all 16,384 features would be 16,384 requests. **INFERRED**: no bulk variant
was found; `?fields=` was ignored, returning the full record unchanged.

The descriptions are hypotheses, and the notebook says so: "automatically generated hypotheses
based on large-scale activation patterns across many proteins... treated as suggestive
interpretations, not definitive annotations" — **VERIFIED**.

## 7. Cost

**No timing is reported here — nothing was run.** What follows is arithmetic over verified
shapes, plus the published weight sizes.

### Weights

| Path | Backbone (fp32) | SAE (fp32) | Total resident |
| --- | ---: | ---: | ---: |
| 300M + `layer23-k64-codebook16384` | 1.24 GiB | 120 MiB | **~1.36 GiB** |
| 300M + `layer23-k64-codebook65536` | 1.24 GiB | 481 MiB | ~1.71 GiB |
| 600M + `layer27-k64-codebook16384` | 2.14 GiB | 144 MiB | ~2.28 GiB |
| **6B + `layer60-k64-codebook16384`** | **23.66 GiB** | 320 MiB | **~23.97 GiB** |
| 6B + `layer60-k64-codebook65536` | 23.66 GiB | 1281 MiB | ~24.91 GiB |

Backbone sizes are **VERIFIED** from the HuggingFace file listing; `ESMC-300M` is
1,332,036,392 bytes, which is the 1.33 GB `AGENTS.md` already records.

**Is the 6B parent runnable on one GPU?** In fp32, weights alone are 23.66 GiB, so an 80 GB
H100 or A100 is comfortable and a 40 GB A100 is workable but tight once activations are added
— **INFERRED** from the weight size; not measured. Half precision would roughly halve it, but
the published `config.json` says `"dtype": "float32"` and nothing in `esm` downcasts by
default. On a shared academic host the 6B path means reserving a large-memory GPU for the
duration.

### The activation memory nobody warns you about

**The encoder materialises a dense `(L + 2, codebook_size)` float32 tensor before sparsifying
it** — `preactivations`, then `torch.zeros_like(preactivations).scatter(...)`, so **two** of
them plus the TopK buffers — **VERIFIED**, `sae.py`. Memory therefore scales with
`L × codebook_size` and **not** with `k`. Arithmetic for one dense copy:

| Sequence length | cb16384 | cb65536 | cb131072 |
| --- | ---: | ---: | ---: |
| 200 | 12.6 MiB | 50.5 MiB | 101.0 MiB |
| 500 | 31.4 MiB | 125.5 MiB | 251.0 MiB |
| 1000 | 62.6 MiB | 250.5 MiB | 501.0 MiB |
| 2000 | 125.1 MiB | 500.5 MiB | 1001.0 MiB |

At least double those for the `zeros_like` copy. A 2000-residue protein against a 131072
codebook wants ~2 GiB of transient activation on top of the weights — **INFERRED**, arithmetic
only. Batching multiplies it, which is a reason the cookbook's batch helper max-pools by
default "to save memory" — **VERIFIED**.

### Download

The realistic 300M path is a **120 MiB** single-file download on top of the 1.24 GiB backbone.
The trap is the all-layer repos, where forgetting `allow_patterns` fetches 25.3 GiB
(`ESMC-6B-sae-k64-codebook16384`) or 202.6 GiB (`ESMC-6B-sae-mlp-k64-codebook131072`) —
**VERIFIED** from the file listings.

### The asymmetry that decides it

The cheap path and the interpretable path are not the same path. 300M costs ~1.36 GiB and
gives you 16,384 unlabelled feature indices with no descriptions and no usable normalisation.
The descriptions exist only for the 6B SAE, at ~24 GiB. **VERIFIED** from the card and the SDK
restriction.

## 8. The paper

**There is one paper, and it is not an SAE paper — it is the whole ESMC / ESMFold2 / ESM Atlas
preprint, in which the SAEs are one section.** The arXiv item is a downstream application by
an unrelated group.

### The canonical citation

`https://biohub.ai/papers/esm_protein.pdf` is not a file. It is a vanity redirect —
301 → 302 → `https://www.biorxiv.org/content/10.64898/2026.06.03.729735` — **VERIFIED** by
following it.

> Candido, S., Hayes, T., Derry, A., Rao, R., Lin, Z., Verkuil, R., … Sercu, T., Rives, A.
> **"Language Modeling Materializes a World Model of Protein Biology."**
> bioRxiv 2026.06.03.729735 (2026). doi:`10.64898/2026.06.03.729735`

41 authors, first author Salvatore Candido, senior author Alexander Rives, institution
Biohub. Posted 4 June 2026, v1, CC-BY 4.0, 111 pages. **A preprint, not published**, and there
is no arXiv mirror — **VERIFIED** from the bioRxiv record and an arXiv title search returning
zero entries.

**Biohub itself names this as the SAE citation.** The `esm` README's Citations section carries
one BibTeX block under the heading `#### ESMC, SAEs, and ESMFold2` — **VERIFIED**, read at
`README.md` line 349. The entry is `@misc{candido2026language, ...}` with `note = {Preprint}`
and a bioRxiv URL but **no DOI field**, so a DOI has to be added by hand.

### What it says about the SAEs

**VERIFIED** from the full PDF:

- The studied SAE is ours: "the SAE with 2^14 features and 64 active per amino acid trained on
  representations after the 60th layer in the transformer (3/4th depth)" — exactly
  `ESMC-6B-sae-layer60-k64-codebook16384`. Appendix A.4.1 adds that "most analyses use layer 60
  of ESMC 6B".
- Training: TopK with an auxiliary dead-feature loss, 8B tokens from UniRef90 / MGnify / JGI,
  bf16-mixed, AdamW.
- The descriptions come from a **multi-agent GPT-5 pipeline** with a hypothesis–verification
  cycle over ~195k non-redundant SwissProt proteins, validated against held-out proteins per
  functional group.

**Two things not to cite.** First, **the paper reports no reconstruction-error number.** It
gives relative layerwise reconstruction-loss and perplexity-degradation curves (Figure S26)
with no absolute value — no explained variance, no FVU, no normalised MSE — **VERIFIED** by a
targeted search of the full text. Combined with §4, that means the reconstruction error is
*computable* (`EsmcSaeOutput.reconstruction_loss`) but nowhere *published*. Second, the paper's
released grid differs from its own table: Table S12 lists sparsity `K ∈ {8, 16, 32, 64, 128}`,
while HuggingFace ships `k ∈ {16, 32, 64, 128, 256, 512}` (§2). **VERIFIED** — the release is
not the table.

### The arXiv item is a downstream application

`arXiv:2606.12209` **resolves and is real** — **VERIFIED**, HTTP 200 and one result from the
arXiv API. The title given to me was truncated:

> Hu, Y., Cheng, W., Wang, J., Liu, Y. **"Interpretable enzyme function prediction via sparse
> autoencoder features of ESMC across the microbial protein universe."**
> arXiv:2606.12209 [q-bio.QM] (2026).

Submitted 10 June 2026, from Qilu University of Technology and Shandong First Medical
University — **no overlap with Biohub** — **VERIFIED** from the PDF title page. It *consumes*
`biohub/ESMC-6B` and `biohub/ESMC-6B-sae-layer60-k64-codebook16384` for EC-number prediction
and cites the Candido preprint as its reference [23] — where it misspells the first author's
initial and omits the DOI, so that reference should not be copied.

**Verdict: Candido et al. 2026 is the SAE paper; Hu et al. 2026 is a user of it.** No other
candidate exists — an arXiv search for `"sparse autoencoder" AND "ESMC"` returns exactly one
hit, which is Hu et al. itself, and Europe PMC returns none — **VERIFIED**.

## 9. Recommendation

Take the two standing rules literally and most of the design is forced.

**Resident state gets an object.** An SAE is 120 MiB to 2.5 GiB of weights held across calls.
That is the same argument that made `ESMC` a class, so an SAE is an object you construct and
keep — never a `Protein.sae_features()` method, and never a free function.

**Direct support only.** The SAE takes a `(n_positions, d_model)` tensor at one named layer.
It does not take a sequence. So nothing hangs off `Protein`: `Protein` would have to acquire
an embedding first, which is the exact thing the rule forbids.

That leaves the question of *what it takes* and *what it returns*.

### Is an SAE activation an `Embedding`?

**No, and forcing it would break two of `Embedding`'s three promises.** `Embedding` is
`(L, d_model)`, dense, and its `.mean()` is documented as "the per-sequence vector". An SAE
activation is `(L, codebook_size)` — 17× to 137× wider — is >99.5% zeros by construction, and
pools with `max`, not `mean`, which is what the cookbook does. Widening `Embedding` to cover
both would make `d_model` a lie and `.mean()` misleading.

**A frozen dataclass peer is the right shape**, and TopK gives it an unusually clean one.
Because exactly `k` entries are non-zero per row, the activation is two `(L, k)` numpy arrays
— int32 feature indices and float32 values — with no COO, no scipy, and **no torch**, which
keeps `embedding.py`'s "numpy only" rule intact across the whole lane. It carries more
provenance than `Embedding` does: source, parent checkpoint, layer, SAE repo id,
`codebook_size`, `k`, and whether normalisation was applied.

### Where does it hang?

| Option | Shape | Cost |
| --- | --- | --- |
| **A. `SAE` class taking an `Embedding`** | `SAE("300m-layer23-k64-codebook16384").encode(embedding) -> SaeActivation` | Composes with the `ESMC` you already have and needs no second forward pass. Can check `embedding.layer in available_layers` and width against `d_model`. **Cannot** check the checkpoint — §5 — so the check is partial and the error message must say so. Uses the standalone path, the one `esm` does not guard |
| **B. Hang it off `ESMC`** | `ESMC(...).add_sae(...)`, then `embed()` returns both | Mirrors upstream's `add_sae_models` and is **layer-safe by construction** (§5). But `ESMC` stops being "the weights and the one call over them" and grows a mode, and it forces a backbone forward even when the `Embedding` is already in hand |
| **C. Nothing in `embed/`; this note only** | — | Zero. Defensible while the only interpretable checkpoint needs a ~24 GiB backbone |
| **D. Descriptions only, as a prepared set** | `protein.store` + `genome.store.prepared`, one 7.3 MB GET | Small, independent of A/B, and the only piece that is unambiguously this package's kind of work |

**Option A is the one the facts favour**, for a reason worth stating precisely: **the SAE
z-scores each position independently over `d_model`, so an `Embedding` — BOS/EOS already
stripped — is numerically exact input, row for row.** Dropping those two rows cannot change
the others. Combined with `Embedding.layer` already sharing the SAE's indexing (§1), an
`Embedding` is not merely convertible into SAE input; it *is* SAE input, and the wrapper's
check is a comparison of integers it already holds.

Its honest weakness is §5: the wrapper can verify layer and width but not identity, so a
`SAE.encode` that accepts an `Embedding` from the wrong checkpoint will return confident
nonsense. If A is built, that limit belongs in the docstring and in a test, not in a comment.

**Option D is separable and cheap**, and should be judged on its own. One unauthenticated GET,
16,384 rows, no key, no per-ID call — it satisfies the bulk rule outright and mirrors SIFTS.
It is also the only part with a stability caveat: the endpoint is self-described as alpha.

**Nothing here needs a dependency change.** `esm >= 3.4` is already pinned in the `esm`
feature and already ships `EsmcSaeModel`. The whole lane would run in the existing `esm`
environment, under the existing `-m model` marker, and the gate would never touch it.

The question this note cannot answer is whether anyone wants 6B-scale interpretability badly
enough to reserve the GPU. If the answer is no, **C plus D** — this note, plus the description
table as a prepared set — is the proportionate outcome.
