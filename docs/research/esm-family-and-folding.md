---
search:
  exclude: true
---

# What is ESMFold2, and where does the ESM lineage stand for structure prediction?

Research note answering a request for "structure prediction with ESMFold2, note it supports
Protein, DNA and RNA". The brief that commissioned this note was sceptical that a model by that
name existed and sceptical that anything in the ESM family took nucleic acids. Both doubts are
wrong, and the note settles them against the shipped source rather than against prose.

## Answer, shortest form

**ESMFold2 is real, it is current, and it does take DNA and RNA.** It was released 2026-05-13 by
**Chan Zuckerberg Biohub** — which acquired EvolutionaryScale, announced 2025-11-06 — as
`biohub/ESMFold2` and `biohub/ESMFold2-Fast` on HuggingFace, **MIT, ungated, no API token**. Its
input type is literally
`StructurePredictionInput.sequences: Sequence[ProteinInput | RNAInput | DNAInput | LigandInput]`,
so one call folds a protein-DNA-ligand complex with CCD-coded modified residues. Nothing about the
request needs correcting.

**The pin this repo already carries — `esm = ">=3.4,<4"` — is exactly the release that ships it.**
`esm` 3.4.0 (2026-08-27) is the first PyPI release with `esm/models/esmfold2/`. No pin has to
move. One trap: 3.4.0 requires `transformers<5.0.0`, but the *model card's* example imports
`transformers.models.esmfold2`, which landed in **transformers 5.16**. Under this repo's own
resolve that import does not exist, so the working entry point is `esm.models.esmfold2` —
`EsmFold2Model.from_pretrained("biohub/ESMFold2")`, `ESMFold2InputBuilder().fold(...)`.

**The `esm` package is not embeddings-only and never was.** It ships three folding routes: ESMFold2
(all-atom, complexes), ESM3's generative structure track with a local encoder/decoder, and Forge
clients for the hosted service. Only the last needs `ESM_API_KEY`; ESMFold2 and ESMC run entirely
from local weights.

**The real cost is not the folding trunk, it is the language model under it.** ESMFold2-Fast is
189M parameters — but its `config.json` names `"esmc_id": "biohub/ESMC-6B"`, and ESMC-6B is
6.35B parameters and **25.41 GB of safetensors**. First run downloads about 27 GB.

**And the licensing changed completely.** The EvolutionaryScale-era Cambrian Non-Commercial
Licence — the thing that made ESM3 and ESM C 600M research-only, and kept ESMC-6B API-only — is
**gone from every current model card**, including the legacy checkpoints. Everything now reads MIT.
For an academic lab that is a strict improvement and removes the one clause that used to matter.

**ESMFold (v1) is effectively dead but not switched off.** `facebookresearch/esm` was archived
2024-08-01 with its last push 2024-02-07, yet `facebook/esmfold_v1` still pulls ~1.75M HuggingFace
downloads a month and `api.esmatlas.com` still folds. Use it for reproducing old results, not for
new ones.

## Method and provenance

| | |
| --- | --- |
| Executed on | this macOS laptop. The repo is `platforms = ["linux-64"]`, so **nothing was installed, imported or run against a GPU** |
| What made VERIFIED possible | `esm-3.4.0-py3-none-any.whl` is a **pure-Python wheel**, 2,678,321 bytes. Downloaded from PyPI and unpacked with `unzip`; every source claim below is read from that tree |
| Also queried | `pypi.org/pypi/esm/json` (and `/3.4.0/`, `/3.2.3/`), `huggingface.co/api/models/<id>` with `?blobs=true`, `api.github.com/repos/<id>`, and the raw `README.md` of each model card |
| Live endpoints probed | `api.esmatlas.com/foldSequence/v1/pdb/` (POST, ubiquitin), `esmatlas.com`, `biohub.ai/esm/protein/atlas`, the two `evolutionaryscale.ai` Cambrian policy pages |
| Read, not run | the ESMFold2 / ESMC model cards, `Biohub/esm` `README.md` and `LICENSE.md`, the preprint landing page |
| Date | 2026-09-03. Every version, download count and HTTP status is as of that day |

Claims are marked **VERIFIED** (read directly out of the unpacked wheel, or returned by an API
call made here) or **INFERRED** (read from a model card, README or paper without running it).
Arithmetic over verified numbers is called out as such — **no VRAM figure below was measured.**

## 1. ESMFold2

### It exists, and who owns it

**VERIFIED** — `api.github.com/repos/evolutionaryscale/esm` returns `full_name: Biohub/esm`. The
canonical repository was **renamed, not forked**: 2,941 stars, not archived, last push 2026-08-27.
The same redirect holds on HuggingFace: `EvolutionaryScale/esm3-sm-open-v1` resolves to
`biohub/esm3-sm-open-v1`, and `EvolutionaryScale/esmc-600m-2024-12` to `biohub/esmc-600m-2024-12`.

Two other repositories carry the name `esmfold2`. Neither is canonical and neither is a
reimplementation — **VERIFIED** from the GitHub API's `parent` and `source` fields:

| Repository | What it is | Stars | Created |
| --- | --- | --- | --- |
| [`Biohub/esm`](https://github.com/Biohub/esm) | **canonical**, not a fork | 2,941 | 2024-06-25 |
| [`atong01/esmfold2`](https://github.com/atong01/esmfold2) | fork, `parent` = `Biohub/esm` | 29 | 2026-05-27 |
| [`zhanglabxmu/esmfold2`](https://github.com/zhanglabxmu/esmfold2) | fork of the fork, `source` = `Biohub/esm` | 1 | 2026-05-27 |

Both forks' `README.md` is the upstream one verbatim. They are personal clones that happen to be
named after the model; ignore them.

The paper is Candido et al., *Language Modeling Materializes a World Model of Protein Biology*,
bioRxiv [`10.64898/2026.06.03.729735`](https://www.biorxiv.org/content/10.64898/2026.06.03.729735),
posted 2026-06-04 — **INFERRED** from the landing page and the citation block on the model card.
Alexander Rives is last author; the acknowledgements name "the Biohub AI Research team and prior
EvolutionaryScale team".

### The two variants

**VERIFIED** from the HuggingFace API, `?blobs=true`, queried 2026-09-03:

| | `biohub/ESMFold2` | `biohub/ESMFold2-Fast` |
| --- | --- | --- |
| Parameters (F32) | **234,822,979** | **188,819,011** |
| MSA conditioning | yes, optional | **no** — single sequence only |
| `model.safetensors` | 939,505,228 B | 755,416,924 B |
| `ccd.pkl` | 417,306,584 B | absent |
| Licence tag | `mit` | `mit` |
| Gated | **no** | **no** |
| Created | 2026-05-13 | 2026-05-13 |
| Downloads (30 d) | 265,341 | 257,586 |

The `ccd.pkl` is the PDB Chemical Component Dictionary — it is what makes CCD-coded ligands and
modified residues work, and it is why only the MSA-capable checkpoint carries one.

Neither number is the model's real size. **VERIFIED** from `ESMFold2-Fast/config.json`:

```json
{
  "esmc_id": "biohub/ESMC-6B",
  "lm_num_layers": 80,
  "lm_d_model": 2560,
  "folding_trunk": { "n_layers": 24, "n_heads": 8 },
  "msa_encoder": { "enabled": false }
}
```

So the trunk sits on a **frozen ESMC-6B**, and `EsmFold2Model.from_pretrained` loads it by default
— **VERIFIED**, the signature is `from_pretrained(..., load_esmc: bool = True, esmc_precision: str = "bf16", ...)`.

### It takes DNA, RNA and ligands — from the source, not the card

This is the load-bearing claim, so it is read out of the wheel. `esm/utils/structure/input_builder.py`,
**VERIFIED**:

```python
@dataclass
class StructurePredictionInput:
    sequences: Sequence[ProteinInput | RNAInput | DNAInput | LigandInput]
    pocket: PocketConditioning | None = None
    distogram_conditioning: list[DistogramConditioning] | None = None
    covalent_bonds: list[CovalentBond] | None = None
```

and the four member types, also **VERIFIED**:

| Type | Fields | Takes an MSA? |
| --- | --- | --- |
| `ProteinInput` | `id`, `sequence`, `modifications`, `msa` | **yes** |
| `RNAInput` | `id`, `sequence`, `modifications`, `msa` | **yes** |
| `DNAInput` | `id`, `sequence`, `modifications` | **no field** |
| `LigandInput` | `id`, `smiles`, `ccd` | n/a |
| `Modification` | `position` (zero-indexed), `ccd`, `smiles` | `smiles` is `TODO`, CCD only |

All ten names are re-exported from `esm.models.esmfold2.__all__` — **VERIFIED** by reading
`esm/models/esmfold2/__init__.py`.

The mixed-complex example, **verbatim from the model card** (**INFERRED** — I did not run it): HhaI
DNA methyltransferase, its cognate DNA carrying a trapped 5-fluoro-2′-deoxycytidine as CCD `C36`,
and the SAH cofactor, i.e. PDB `1MHT`.

```python
from esm.models.esmfold2 import (
    DNAInput, ESMFold2InputBuilder, LigandInput,
    Modification, ProteinInput, StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()

spi = StructurePredictionInput(
    sequences=[
        ProteinInput(id="A", sequence=HHAI_SEQ),
        DNAInput(id="B", sequence="GATAGCGCTATC",
                 modifications=[Modification(position=5, ccd="C36")]),
        DNAInput(id="C", sequence="TGATAGCGCTATC",
                 modifications=[Modification(position=6, ccd="C36")]),
        LigandInput(id="L", ccd=["SAH"]),
    ]
)

result = ESMFold2InputBuilder().fold(
    model, spi, num_loops=3, num_sampling_steps=50, num_diffusion_samples=1, seed=0
)
with open("1mht_pred.cif", "w") as f:
    f.write(result.complex.to_mmcif())
```

**That import line is wrong for this repo** — see §3. Substitute
`from esm.models.esmfold2 import EsmFold2Model`.

The card's own sentence, quoted because it is the one the request paraphrased: *"Unlike ESMFold,
ESMFold2 is able to predict structures for all biomolecules, including small molecules, DNA, RNA,
and modified amino acids."*

### Single sequence versus MSA

Both modes exist on the MSA-capable checkpoint; `ESMFold2-Fast` has `msa_encoder.enabled: false`
and cannot use one. The card claims single-sequence mode buys "an order of magnitude speedup"
(**INFERRED**), and that inference-time compute — the `num_loops` / `num_sampling_steps` /
`num_diffusion_samples` knobs above — "can dramatically improve performance ... especially across
antibody-antigen complexes".

Benchmark claim, **INFERRED** from the card and unverified here: ESMFold2 "meets or exceeds
performance by AlphaFold3 on antibody-antigen complex prediction, protein-protein complex
prediction and Runs N' Poses benchmarks", evaluated on FoldBench. Training data is PDB + AFDB
with a **September 2021 cutoff** — the same cutoff for both variants, so anything deposited since
is a genuine holdout.

### What it costs to run

**No number here was measured.** The download figures are **VERIFIED** file sizes; the resident
figures are arithmetic over verified parameter counts and are **INFERRED**.

| | Bytes | Note |
| --- | --- | --- |
| `biohub/ESMC-6B`, six safetensors shards | **25,408,297,673** (25.41 GB) | **VERIFIED**, summed from the API listing |
| `biohub/ESMFold2` weights + CCD | 1,356,811,812 | **VERIFIED** |
| First-run download for ESMFold2 | **≈ 26.8 GB** | **INFERRED**, the sum of the two |
| ESMC-6B resident at the default `bf16` | ≈ 12.7 GB | **INFERRED**, 6,352,005,184 × 2 bytes |
| Folding trunk resident | ≈ 0.94 GB | **INFERRED**, F32 |
| Weights alone, default settings | **≈ 13.6 GB** | **INFERRED**. Activations, MSA and long complexes are on top |

So a 24 GB card is a plausible floor for a short single chain and an 80 GB card is the safe
answer for complexes, MSA mode or high inference-time compute — **INFERRED, and worth measuring on
GPU71FM before anyone commits to it.** `set_esmc_precision` accepts `"bf16"`, `"fp32"` and
`"fp8"`; fp8 "requires H100 + TransformerEngine ≥ 2.x" (**VERIFIED**, quoted from its docstring).
`set_chunk_size` exists on the trunk and the confidence head for trading speed against memory.
ESMC's context window is 2048 tokens (**INFERRED** from the ESMC card).

Note where those 25 GB land: `esm/models/hub.py` calls `huggingface_hub.snapshot_download`
(**VERIFIED**), so the weights go to the HuggingFace cache, **not** to `<LIULAB_DATA>/protein/`.

## 2. The family lineup

Two eras, and the boundary is the archive date of `facebookresearch/esm`.

| Model | Owner, year | What it does | Folds? | Polymers | MSA | Licence now | Weights |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ESM-1b** | Meta, 2020 | masked LM, 650M | no | protein only | no | MIT | `facebook/esm1b_t33_650M_UR50S` |
| **ESM-2** | Meta, 2022 | masked LM, 8M–15B | no | protein only | no | MIT | `facebook/esm2_t33_650M_UR50D` and siblings |
| **ESMFold** | Meta, 2022 | ESM-2 + folding head | **yes**, backbone + side chains | **protein only** | **no** | MIT | `facebook/esmfold_v1` |
| **ESM3** | EvolutionaryScale, 2024 | generative over sequence / structure / function | yes, via the structure track | protein only | no | **MIT now** | `biohub/esm3-sm-open-v1` (1.4B); larger sizes API-only |
| **ESM C** | 2024, re-released 2026 | masked LM, 300M / 600M / **6B** | no | protein only | no | **MIT now** | `biohub/ESMC-300M` / `-600M` / `-6B` |
| **ESMFold2** | Biohub, 2026 | all-atom complexes on frozen ESMC-6B | **yes** | **protein, DNA, RNA, ligands, modified residues** | **optional** | MIT | `biohub/ESMFold2`, `biohub/ESMFold2-Fast` |

**Nothing before ESMFold2 accepts a nucleic acid.** ESM-1b, ESM-2, ESMFold, ESM3 and ESM C are all
protein-sequence models. The request's "supports Protein, DNA and RNA" is true of exactly one model
in the family, and it is the one it named.

Parameter counts, **VERIFIED** from the HuggingFace API: ESMC-300M is 332,997,184; ESMC-600M is
575,036,992; ESMC-6B is 6,352,005,184; `facebook/esm2_t33_650M_UR50D` is 652,358,616.

### The licence change is the headline for an academic lab

Historically (**INFERRED**, and both policy pages still return HTTP 200 at
[evolutionaryscale.ai](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement)):
ESM3-open and ESM C 600M weights sat under the **Cambrian Non-Commercial License** — no commercial
activity, no serving outputs as a service, "Built with ESM" attribution — ESM C 300M under the
permissive Cambrian Open License, and ESM3 7B/98B plus ESMC-6B were **API-only, no weights at all**.

That is over. **VERIFIED**, 2026-09-03: the string `Cambrian` appears **zero** times in the current
`README.md` of `biohub/ESMC-6B`, `biohub/ESMFold2`, `biohub/esm3-sm-open-v1` **or** the legacy
`biohub/esmc-600m-2024-12`. Every one carries `license: [mit]` or `license: [mit, other]`, with
`license_link` pointing at
[`THIRD_PARTY_NOTICE.md`](https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md) — which is
a table of **dependency** licences (flash-attn, PyTorch, xformers…), not a restriction on the
weights. `Biohub/esm/LICENSE.md` is plain MIT, "Copyright 2026 Chan Zuckerberg Biohub, Inc."
`biohub/esm3-sm-open-v1`'s card body says outright: *"This repository is under a MIT license."*
None of the four repositories is gated.

The only remaining restriction is behavioural, and it applies to the hosted service rather than the
weights: the card forbids "any use that is prohibited by the Acceptable Use Policy", and says the
Biohub Platform "implement[s] guardrails that detect and restrict the use of keywords and sequences
corresponding to controlled pathogens and toxins" (**INFERRED**, quoted from the card). Local
inference has no such filter.

**Consequence: ESMC-6B — 6.35 billion parameters, formerly paywalled behind Forge — is now a
25.41 GB MIT download that this lab can put on GPU71FM.**

### ESMFold v1 is unmaintained and still answering

**VERIFIED** from the GitHub API: `facebookresearch/esm` has `archived: true`, last push
2024-02-07, MIT, 4,170 stars. The README carries no deprecation or redirection notice — it was
simply frozen.

And yet, **VERIFIED** by POSTing ubiquitin to `https://api.esmatlas.com/foldSequence/v1/pdb/` from
this laptop today:

```text
HEADER                                            18-OCT-22
TITLE     ESMFOLD V1 PREDICTION FOR INPUT
```

A real PDB comes back. `facebook/esmfold_v1` drew **1,752,389** HuggingFace downloads in the last
30 days and `facebook/esm2_t33_650M_UR50D` **1,396,064** — versus 6,657 for ESM-1b and 1,921 for
ESM-2 15B. So ESMFold and ESM-2 650M are the two survivors of the Meta era by a wide margin, and
everything else in it is effectively retired.

Treat ESMFold v1 as **frozen, not removed**: fine for reproducing a published result, wrong for new
work, and carrying no maintainer if it breaks.

## 3. What `esm` 3.4.0 actually ships

All **VERIFIED** by unpacking the wheel. The package layout:

```text
esm/
├── __init__.py        one line: __version__ = "3.4.0"
├── pretrained.py      ESM3 encoders/decoders; the ESMC_*_202412 loaders, all "Deprecated"
├── data, layers, tokenization, utils, widgets
├── models/
│   ├── esm3.py        class ESM3(nn.Module, ESM3InferenceClient)
│   ├── esmc/          EsmcModel, EsmcForMaskedLM, EsmcTokenizer, EsmcSae*
│   ├── esmfold2/      EsmFold2Model, ESMFold2InputBuilder, the four input types
│   ├── vqvae.py, function_decoder.py, hub.py
└── sdk/               client(), esmc_client(), esmfold2_client(), the Forge clients
```

### It is not embeddings-only

The existing note [What biotite and its peers already cover of v1's surface](biotite-coverage-of-v1.md)
treats `esm` as the thing `embed/` sits on. That is still true, but it undersells the package —
**three separate folding routes ship in the pin this repo already has**:

| Route | Entry point | Local? |
| --- | --- | --- |
| **ESMFold2, all-atom complexes** | `EsmFold2Model.from_pretrained` → `ESMFold2InputBuilder().fold(...)` | **yes** |
| ESMFold2, single protein chain | `EsmFold2Model.infer_protein(seq)`, `.infer_protein_as_pdb(seq)`, `.output_to_pdb(out)` | **yes** |
| ESM3 generative structure track | `ESM3.from_pretrained(...).generate(protein, GenerationConfig(...))`, plus `ESM3_structure_encoder_v0` / `ESM3_structure_decoder_v0` in `pretrained.py` | **yes**, `esm3_sm_open_v1` only |
| Hosted | `esmfold2_client()`, `client()`, `esmc_client()` | **no**, needs a token |

`esm/sdk/api.py` also defines `FoldingConfig`, `FoldMaxAccuracyConfig` and **`InverseFoldingConfig`**
— so inverse folding is in the API surface too, though against the Forge clients.

### What needs a token, and what does not

**VERIFIED** from `esm/sdk/__init__.py`. All three factory functions default to
`url="https://biohub.ai"` and `token=os.environ.get("ESM_API_KEY", "")`, and
`esm/sdk/base_forge_client.py` raises with *"Please provide a token to connect to Forge/Biohub
Platform via token=YOUR_API_TOKEN_HERE"* when it is empty:

```python
def esmfold2_client(model="esmfold2-fast-2026-05", url="https://biohub.ai",
                    token=os.environ.get("ESM_API_KEY", ""), request_timeout=None): ...
```

Note `forge.evolutionaryscale.ai` has become `biohub.ai` throughout.

**Nothing in `esm.models` requires a token.** ESMFold2, ESMFold2-Fast, all three ESMC sizes, the
SAEs and `esm3-sm-open-v1` are ungated HuggingFace repositories that `snapshot_download` pulls
anonymously. The token buys hosted inference, guardrails and batch execution — not access to
weights.

### Three facts that bite this repo specifically

**The transformers pin makes the model card's example unusable here.** **VERIFIED** from the wheel
metadata (`Requires-Dist: transformers<5.0.0,>=4.57.6`) and from the docstring of
`esm/models/esmfold2/hf_adapter.py`, quoted:

> The port landed in `transformers` 5.16.0.dev0 and this repo pins 4.x, so nothing upstream is
> imported at module scope; `upstream_available` gates the two entry points that need it.

The card's `from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model` therefore
raises `ImportError` under `esm` 3.4.0's own resolve. `EsmFold2HFAdapter` exists to bridge the two
when transformers ≥ 5.16 *is* present, and its error message says so. Use `esm.models.esmfold2`.
The sibling note's candidate table lists transformers 5.16.1 as latest — correct, but not
installable alongside `esm` 3.4.0.

**`import esm` still does not import torch.** **VERIFIED** — `esm/__init__.py` is one line,
`__version__ = "3.4.0"`, with no other statement. `CLAUDE.md`'s rule that "`import protein` must
never import torch" survives contact with this package as long as the heavy names stay in method
bodies.

**The torch bound in `pyproject.toml` is exactly right and not a coincidence.** **VERIFIED** —
`esm` 3.4.0 declares `torch<2.12.0,>=2.11.0`, which is what the `esm` feature already mirrors as
`pytorch = ">=2.11,<2.12"` from conda-forge.

Two smaller ones. `esm` 3.4.0 declares `Requires-Python: >=3.12`, but **3.2.3 declared
`<3.13,>=3.12`** — so `>=3.4` is load-bearing for a Python 3.13 repo, not merely a freshness
preference (**VERIFIED** from both PyPI metadata records). And `esm` depends on `biotite>=1.0.0`,
which this repo already carries, plus `rdkit`, `biopython`, `pydssp`, `boto3` and
`cuequivariance-torch` (marked Linux x86_64 only) — the ligand and DSSP machinery that ESMFold2
needs.

## 4. If you want single-sequence folding from this lineage today

**Use ESMFold2-Fast.** It is the maintained, single-sequence, no-MSA path, it is MIT, it is
ungated, and it runs locally. `EsmFold2Model.from_pretrained("biohub/ESMFold2-Fast")` then
`infer_protein_as_pdb(seq)` is the two-line version; `ESMFold2InputBuilder().fold(...)` with a
`StructurePredictionInput` is the version that gets complexes, ligands and nucleic acids.

Ranked, with the reason:

| Option | Verdict |
| --- | --- |
| **ESMFold2-Fast** | **The answer.** Maintained, MIT, local, single-sequence by construction, and the only ESM model that folds anything but protein |
| ESMFold2 (MSA-capable) | Same weights family; take it when you have an MSA or need the CCD dictionary. Costs another 600 MB and an MSA pipeline this repo does not have |
| ESM3-open structure track | Local and MIT, but it is a *generative* model whose folding is a side effect of the structure track. Only the 1.4B size has weights. Reach for it for design, not for folding |
| ESMFold v1 | Frozen 2024-08-01. Reproduction only |
| Biohub Platform API | Needs `ESM_API_KEY`, sends sequences off-site, and applies pathogen guardrails. Nothing this repo needs, given the weights are free |

Is it competitive? **INFERRED, and unverified here** — the card claims parity with or better than
AlphaFold3 on FoldBench antibody-antigen and protein-protein complexes and on Runs N' Poses, and
the README claims it "surpasses other models in DockQ pass-rate". Those are the authors' own
numbers on the authors' own chosen benchmark. Treat them as a reason to try it, not as a
measurement. The honest independent statement is narrower: **it is the only actively maintained
folding model in the ESM lineage, and the only one that takes DNA and RNA.**

## Open items for this repo

- **No pin changes.** `esm = ">=3.4,<4"` and `pytorch = ">=2.11,<2.12"` are already correct for
  ESMFold2. Nothing in `pyproject.toml` needs to move to adopt it.
- **VRAM is unmeasured.** The ≈13.6 GB weights-resident figure is arithmetic. Somebody should run
  ESMFold2-Fast on GPU71FM against `1UBQ` and record what it actually takes, because the answer
  decides whether folding is a routine call or a scheduled job.
- **25.41 GB of ESMC-6B lands in the HuggingFace cache, not `<LIULAB_DATA>/protein/`.** That is a
  bulk artefact by any reading of the map's "bulk, not per-ID" rule, and `store.py` currently has
  no opinion about it. Decide whether `HF_HOME` gets pointed at the data root.
- **`Direct support only` has a genuinely awkward case here.** ESMFold2 takes *sequence* and
  returns *structure*, so by the rule the method belongs on `Protein`. But its real input is a
  mixed complex of protein, DNA, RNA and ligand chains, and this package has no nucleic-acid
  sequence type and no complex type at all — `Structure` and `Chain` hold coordinates, not
  sequences to fold. Adopting the full input surface means inventing vocabulary; adopting only
  `infer_protein` means `Protein.fold()` quietly discards the capability the request asked for.
  That is a design question for an issue, and this note deliberately does not settle it.
- **The sibling note's `embed/` section is now understated.** It says "nothing stands between us
  and the `esm` SDK", which is still true, but the package it describes as an embeddings dependency
  also ships three folding routes and an inverse-folding config.
- **The model cards say "Please install `esm` from GitHub (a PyPI release is coming soon)".** That
  sentence is stale: 3.4.0 shipped to PyPI on 2026-08-27 with `esm/models/esmfold2/` in it —
  **VERIFIED** by unpacking the wheel. Do not follow the card's `pip install esm@git+...` line.
