---
search:
  exclude: true
---

# Which structure predictor takes protein, DNA and RNA together, and what would it cost to drive one?

Research note for the request to add structure prediction and to make "supports protein, DNA and
RNA" a fact the design reflects. That request rules out the ESM family and points at the
AlphaFold3-class complex predictors. This note settles which of them exist on 2026-09-03, what
each one literally eats, and which one survives contact with pixi, `platforms = ["linux-64"]`
and Python 3.13.

## Answer, shortest form

**Pick OpenFold3. It is the only one of the five whose upstream already installs the way this
repo does — its own `pixi.toml` declares `channels = ["conda-forge", "bioconda"]`, takes
`pytorch-gpu` and `hmmer`/`hhsuite` from those channels, and installs only `openfold3` itself
from PyPI. That is the `esm` feature's pattern, written by someone else, already tested on
linux-64 and A100.** Code and weights are both Apache-2.0, the weights come off an unauthenticated
AWS Open Data bucket, `requires-python` is `>=3.10` with no ceiling, and the classifiers name
3.13 and 3.14. It also already holds biotite, which is the type layer this package holds.

**Two disqualify themselves on facts you can check without installing anything.** Boltz declares
`requires-python = ">=3.10,<3.13"` and `numpy>=1.26,<2.0` — it cannot enter any environment of
this repo as written, and its last release was 2025-09-08. AlphaFold 3's weights are not open:
they are non-commercial-only under a bespoke terms of use whose restrictions **follow the output**,
so every structure this package produced would carry them.

**The expensive half is already solved here.** Every one of these tools accepts a precomputed A3M
per chain, and this package already drives `mmseqs`. So the choice is not "adopt a 630 GB
jackhmmer pipeline or call a public server" — it is "keep producing the MSA locally, hand over a
path". That is the strongest argument for adding prediction at all, and it is independent of
which model wins.

**The design cost is one type this package does not have.** These models take an *assembly* —
protein plus DNA plus RNA plus ligand in one job. "Direct support only" therefore forbids
`Protein.fold()`: a `Protein` is not what the tool takes. The method belongs on a new peer, not
on an existing one.

## Method and provenance

| | |
| --- | --- |
| Executed on | this macOS laptop, **read-only**. Nothing installed, nothing resolved — the repo is `platforms = ["linux-64"]` |
| How | `curl` against `raw.githubusercontent.com`, `pypi.org/pypi/<name>/json`, `api.anaconda.org/package/<channel>/<name>`, `huggingface.co/api/models/...`, and the release Atom feeds. All on **2026-09-03** |
| Source read | the literal `LICENSE`, `pyproject.toml`, `setup.py`, `requirements.txt`, `pixi.toml`, `docs/` and `examples/` of each repo at `main` |
| Repo side read | `src/protein/external.py`, `pyproject.toml`, `.vale.ini`, `.markdownlint-cli2.yaml` in this working tree |
| Not read | any weights file, any Docker image, any paper PDF. No tool was run |

Every claim is marked **VERIFIED** (the literal file, header or API response was read) or
**INFERRED** (deduced from what was read). Where a subagent's relay and a file disagreed, the file
won — two relayed claims were wrong and are corrected in place below.

**One correction worth stating up front.** AlphaFold 3's weights are commonly described as
CC-BY-NC-SA and gated behind an application form. Both are wrong as of today: the licence is a
bespoke *AlphaFold 3 Model Parameters Terms of Use*, and the README publishes a direct download
URL that answers `HTTP 200`, `x-goog-stored-content-length: 1020545840`,
`last-modified: Thu, 04 Jun 2026`. **VERIFIED** by `HEAD`.

## The field on 2026-09-03

Release dates from each repo's Atom feed; last-commit dates from the commits API. **VERIFIED.**

| Tool | Latest release | Last commit | Code | Weights | conda-forge / bioconda |
| --- | --- | --- | --- | --- | --- |
| **AlphaFold 3** | v3.0.4, 2026-07-28 | 2026-08-19 | Apache-2.0 | bespoke, **non-commercial** | neither |
| **Boltz** | v2.2.1, **2025-09-08** | 2026-05-29 | MIT | **MIT** | neither |
| **Chai-1** | v0.6.1, **2025-03-18** | 2026-06-30 | Apache-2.0 | **Apache-2.0** | neither |
| **Protenix** | v2.0.0, 2026-04-07 | 2026-08-01 | Apache-2.0 | **Apache-2.0** | neither |
| **OpenFold3** | v0.5.0, **2026-08-21** | active | Apache-2.0 | **Apache-2.0** | neither |

`colabfold` 1.5.5 (bioconda, `noarch`, 2024-04-18) is the **only** hit anywhere near this space on
either channel. `boltz`, `chai_lab`, `chai-lab`, `protenix`, `openfold3` and `alphafold3` all
return `"could not be found"` from both `api.anaconda.org/package/conda-forge/…` and
`.../bioconda/…`. **VERIFIED** — I ran all fourteen queries.

So "conda-forge first, PyPI when necessary" resolves to *PyPI, necessarily*, for every candidate.
The question is not which one is a conda package — none is — but which one's **other** dependencies
come from conda-forge cleanly. Only OpenFold3 has answered that question upstream.

## 1. AlphaFold 3

Repo: [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3).

### Entities

`protein`, `rna`, `dna`, `ligand`. **There is no ion type** — "Ions are treated as ligands, e.g. a
magnesium ion would simply be a ligand with `ccdCodes: ["MG"]`". Ligands come three ways: CCD codes
("CCD from 2022-09-28 is used"), a SMILES string, or a user-supplied CCD mmCIF. Modifications are
CCD codes at 1-based positions, spelled `ptmType`/`ptmPosition` on protein and
`modificationType`/`basePosition` on DNA and RNA. **VERIFIED**,
[docs/input.md](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md).

Covalent bonds go in a top-level `bondedAtomPairs`, addressed by `[entityId, residueId, atomName]`.
Two limits are load-bearing: a SMILES ligand **cannot** be bonded ("there is no atom name that
could be used to define the bond"), and "Defining covalent bonds between or within polymer entities
is not currently supported". **VERIFIED.**

### Input

A custom JSON, one job per file, `dialect` and `version` both required. **VERIFIED**; note the
document contradicts itself — the `version` field description says "must be set to 1 or 2" while
the `## Versions` section two lines later documents 1 through 4.

```json
{
  "name": "Job name goes here",
  "modelSeeds": [1, 2],
  "sequences": [
    {"protein": {...}},
    {"rna": {...}},
    {"dna": {...}},
    {"ligand": {...}}
  ],
  "bondedAtomPairs": [...],
  "userCCD": "...",
  "userCCDPath": "...",
  "dialect": "alphafold3",
  "version": 4
}
```

A real, runnable protein-plus-DNA job, verbatim from
[`examples/tetr_dimer_dna.json`](https://github.com/google-deepmind/alphafold3/blob/main/examples/tetr_dimer_dna.json)
— **VERIFIED**, fetched at `HTTP 200`, sequence elided here only for width:

```json
{
  "name": "TetR_dimer_tetO_DNA",
  "sequences": [
    {
      "protein": {
        "id": ["A", "B"],
        "sequence": "MARLNRESVIDAALELLNETGIDGLTTRKLAQKLGIEQPTLYWHVKNKRALLDALAVEILARHHDY…",
        "description": "Tetracycline repressor protein class D, E. coli (UniProt P0ACT4, 218 aa)"
      }
    },
    {
      "dna": {
        "id": "C",
        "sequence": "TACTCTATCATTGATAGAGT",
        "description": "tetO2 operator, forward strand (19 bp palindromic core + 1 bp flank, 20 nt)"
      }
    },
    {
      "dna": {
        "id": "D",
        "sequence": "ACTCTATCAATGATAGAGTA",
        "description": "tetO2 operator, reverse complement strand (20 nt)"
      }
    }
  ],
  "modelSeeds": [42],
  "dialect": "alphafold3",
  "version": 4
}
```

Note the shape a builder has to produce: **`sequences` is a list of single-key dicts**, the key
naming the type. A double-stranded operator is two `dna` entries, not one — the same convention
every tool here uses.

DNA modification, verbatim from `examples/methylated_dna.json` — **VERIFIED**:

```json
{
  "dna": {
    "id": "A",
    "sequence": "AGATCGATCGATCGATCGAT",
    "modifications": [
      {"modificationType": "5CM", "basePosition": 5},
      {"modificationType": "5CM", "basePosition": 9}
    ]
  }
}
```

### MSA

Optional, and the empty-string rule is the trap. **VERIFIED**, quoted:

- RNA `unpairedMsa` unset or `null` → "AlphaFold 3 will build MSA for this RNA chain
  automatically."
- Set to `""` → "AlphaFold 3 won't build the MSA for this RNA chain … equivalent to running
  MSA-free for this RNA chain."
- For protein, "both `unpairedMsa` and `pairedMsa` have to either be *both* set (i.e. non-`null`),
  or both unset". The normal custom-MSA call is `unpairedMsa` = your A3M, `pairedMsa` = `""`.
  Both `""` is fully MSA-free.

**DNA takes no MSA at all** — the `dna` entity has no `unpairedMsa` field. **VERIFIED** by reading
the whole DNA section. Templates are protein-only: "Structural templates can be specified only for
protein chains."

The built-in pipeline is Jackhmmer/Nhmmer over BFD small, MGnify, PDB, PDB seqres, UniProt,
UniRef90, **NT, Rfam and RNACentral** — "The total download size for the full databases is around
252 GB and the total size when unzipped is 630 GB". **VERIFIED**,
[docs/installation.md](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md).
There is no MSA server to point anywhere; the alternative to 630 GB is supplying A3M yourself.

### Licence

Two documents, and they diverge. **Code**: the `LICENSE` file is the literal Apache License 2.0 —
**VERIFIED**. **Weights**: `WEIGHTS_TERMS_OF_USE.md`, last modified 2024-11-09, **VERIFIED**,
quoted:

> The AlphaFold 3 model parameters and output are **only** available for non-commercial use by,
> or on behalf of, non-commercial organizations (*i.e.*, universities, non-profit organizations
> and research institutes, educational, journalism and government bodies).

> You **must not** use nor allow others to use: … AlphaFold 3 model parameters or output in
> connection with **any commercial activities, including research** **on behalf of commercial
> organizations;** or … AlphaFold 3 output to **train machine learning models** … similar to
> AlphaFold 3.

> You ***must not* publish or share AlphaFold 3 model parameters**, except sharing these within
> your organization in accordance with these Terms.

Output is separately governed by `OUTPUT_TERMS_OF_USE.md` and may be published, but only with
notice that continued use is subject to those terms. **That is the disqualifying property for a
library**: a package that shells out to AF3 hands its callers files whose licence they did not
choose and probably will not read.

Access is *not* an application form any more. `README.md` says "You can download the AlphaFold 3
model parameters from `https://storage.googleapis.com/alphafold3/af3.bin.zst`" and that URL
answers `HTTP 200` unauthenticated. **VERIFIED.** The gate is the terms, not the download.

### Install and run

Docker is the documented path; the container "requires that the host machine has CUDA 12.6
installed". `pyproject.toml` says `requires-python = ">=3.12"` and pins `jax==0.10.2`. There is no
packaged Python API — `run_alphafold.py` is an absl-flags script. **VERIFIED.**

Timings, verbatim from
[docs/performance.md](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md)
— **VERIFIED**. AF3 counts **tokens**, not residues; a 500-residue monomer is ~500 tokens, so it
sits below the smallest row:

| Num Tokens | 1 A100 80 GB | 1 H100 80 GB |
| --- | --- | --- |
| 1024 | 62 s | 34 s |
| 2048 | 275 s | 144 s |
| 5120 | 2547 s | 1416 s |

Supported hardware is exactly "1 NVIDIA A100 (80 GB)" or "1 NVIDIA H100 (80 GB)"; 40 GB needs
unified memory and a source edit. The data pipeline wants "at least 64 GB of RAM" and fast disk.

### Output

mmCIF only — "We do not provide the output in the PDB format". Five samples per seed by default,
in `seed-<seed>_sample-<n>/` directories. `summary_confidences.json` carries `ptm`, `iptm`,
`fraction_disordered`, `has_clash`, `ranking_score` and the chain-level arrays; the full
`confidences.json` adds `pae`, `atom_plddts` and `contact_probs`. **VERIFIED**,
[docs/output.md](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md).

## 2. Boltz-1 and Boltz-2

Repo: [jwohlwend/boltz](https://github.com/jwohlwend/boltz).

### Entities

`protein`, `dna`, `rna`, `ligand` (`smiles` or `ccd`, mutually exclusive). Modified residues via
`modifications: [{position, ccd}]`. Covalent bonds, pocket and contact conditioning via
`constraints`. `cyclic: true` on a polymer. Boltz-2 adds binding affinity via
`properties: [{affinity: {binder: CHAIN_ID}}]`. Ions are ligands with an ion CCD code — **INFERRED**,
the docs never name an ion type. **VERIFIED** otherwise from
[docs/prediction.md](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md).

### Input

YAML, and the schema block verbatim — **VERIFIED**:

```yaml
sequences:
    - ENTITY_TYPE:
        id: CHAIN_ID
        sequence: SEQUENCE      # only for protein, dna, rna
        smiles: 'SMILES'        # only for ligand, exclusive with ccd
        ccd: CCD                # only for ligand, exclusive with smiles
        msa: MSA_PATH           # only for protein
        modifications:
          - position: RES_IDX   # index of residue, starting from 1
            ccd: CCD            # CCD code of the modified residue
        cyclic: false
constraints:
    - bond:
        atom1: [CHAIN_ID, RES_IDX, ATOM_NAME]
        atom2: [CHAIN_ID, RES_IDX, ATOM_NAME]
    - pocket:
        binder: CHAIN_ID
        contacts: [[CHAIN_ID, RES_IDX/ATOM_NAME], [CHAIN_ID, RES_IDX/ATOM_NAME]]
        max_distance: DIST_ANGSTROM
        force: false
templates:
    - cif: CIF_PATH
properties:
    - affinity:
        binder: CHAIN_ID
```

There is also a deprecated FASTA form whose header is `>CHAIN_ID|ENTITY_TYPE|MSA_PATH`, with
`>A|protein|empty` meaning single-sequence. It cannot express modifications, bonds, pockets or
affinity. **VERIFIED.**

Note the structural difference from AF3: Boltz's `id` is a scalar or list on the entity, and
`ENTITY_TYPE` is the mapping key, so a builder emits the same single-key-dict shape but in YAML
and with different field names. Nothing translates mechanically between the two.

### MSA

"By default, an `msa` must be provided." The escape hatches, all **VERIFIED**:

- `--use_msa_server` calls a server whose `--msa_server_url` **defaults to
  `https://api.colabfold.com`** — and that flag exists precisely so it can point elsewhere.
- `msa: <path>` to a precomputed `.a3m`, or a CSV with `sequence` and `key` columns when chains
  must be paired.
- `msa: empty` forces single-sequence, "not recommended, as it reduces accuracy".
- `--msa_server_username`/`--msa_server_password`, or `--api_key_header`/`--api_key_value`, for a
  server behind auth.

**DNA and RNA take no MSA**: the schema comment is `msa: MSA_PATH  # only for protein`. Templates
are a separate top-level `templates:` key taking a CIF or PDB path.

That `--msa_server_url` is worth naming for all five tools at once: the ColabFold API is
[soedinglab/MMseqs2-App](https://github.com/soedinglab/MMseqs2-App), whose README says it "runs
either: on your server through docker-compose, where it can make your sequence, profile or
structure databases easily accessible over the web". So "point it at a local database" is a
supported deployment, not a hack. It is **GPL-3.0-only** — **VERIFIED** from its `LICENSE`, which
matters if anything ever vendored it rather than talking to it.

### Licence

The best of the five. `LICENSE` is MIT, "Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro
Passaro" — **VERIFIED**. The README says it twice: "All the code and weights are provided under
MIT license, making them freely available for both academic and commercial uses" and "Our model
and code are released under MIT License". No form, no gate. **VERIFIED.**

### Install and run — and why it is disqualified

From `pyproject.toml` on `main`, **VERIFIED**, and identical on PyPI:

```toml
requires-python = ">=3.10,<3.13"
dependencies = [
    "torch>=2.2",
    "numpy>=1.26,<2.0",
    ...
]
```

**Both bounds collide with this repo head-on.** `requires-python` excludes 3.13, which
`[tool.pixi.dependencies] python = "3.13.*"` pins and `conformance` holds every other declared
Python version to. And `numpy<2.0` contradicts `numpy = ">=2"` in `[project] dependencies`. A
separate feature with `no-default-feature = true` could carry its own Python — pixi documents
`no-default-feature` as "Whether to include the default feature in that environment"
([pixi manifest reference](https://pixi.sh/latest/reference/pixi_manifest/), **VERIFIED**) — but
that means a *second* Python in the workspace purely for this one lane. That is the customization
the restraint rule exists to refuse.

Add the maintenance signal: last release 2025-09-08, last commit 2026-05-29, and the README's
Training and Evaluation sections both still read "⚠️ **Coming soon: updated evaluation code for
Boltz-2!**". **VERIFIED.**

It is CLI-only (`[project.scripts] boltz = "boltz.main:cli"`, a `click` group), and it states no
VRAM or wall-clock figures anywhere — **VERIFIED** as an absence, by grep.

### Output

`.cif` by default, `.pdb` via `--output_format`. `confidence_*.json` carries `confidence_score`,
`ptm`, `iptm`, `ligand_iptm`, `protein_iptm`, `complex_plddt`, `complex_iplddt`, `complex_pde`,
`chains_ptm`, `pair_chains_iptm`; `pae`/`pde`/`plddt` come as `.npz`. `--diffusion_samples`
controls the number of ranked models. Boltz-2 adds `affinity_*.json` with `affinity_pred_value`
(log10 IC50 in µM) and `affinity_probability_binary`. **VERIFIED.**

## 3. Chai-1 — and where Chai went

Repo: [chaidiscovery/chai-lab](https://github.com/chaidiscovery/chai-lab).

### Entities and input

A typed FASTA. Header grammar is `entity_type|name=...` over `protein`, `ligand`, `rna`, `dna`,
`glycan`. Verbatim from `examples/predict_structure.py` — **VERIFIED**:

```text
>protein|name=example-of-short-protein
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>ligand|name=example-ligand-as-smiles
CCCCCCCCCCCCCC(=O)O
```

Modified residues are inline CCD codes in parentheses — `RKDES(MSE)EES`. Glycans get their own
bond syntax, `NAG(4-1 NAG(4-1 BMA(3-1 MAN)(6-1 MAN)))`. Restraints and covalent bonds go in a
separate CSV — **VERIFIED**, verbatim from `examples/restraints/contact.restraints`:

```text
chainA,res_idxA,chainB,res_idxB,connection_type,confidence,min_distance_angstrom,max_distance_angstrom,comment,restraint_id
A,C387,B,Y101,contact,1.0,0.0,5.5,protein-heavy,restraint_1
C,I32,A,S483,contact,1.0,0.0,5.5,protein-light,restraint_2
```

Alone among the five it ships a real, documented Python API — **VERIFIED**, the literal signature
from `chai_lab/chai1.py`:

```python
def run_inference(
    fasta_file: Path,
    *,
    output_dir: Path,
    use_esm_embeddings: bool = True,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_directory: Path | None = None,
    constraint_path: Path | None = None,
    use_templates_server: bool = False,
    template_hits_path: Path | None = None,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    num_diffn_samples: int = 5,
    seed: int | None = None,
    device: str | None = None,
) -> StructureCandidates:
```

### MSA, licence, install

MSAs are genuinely optional — "the model generates five sample predictions, and uses embeddings
without MSAs or templates" by default, and `examples/msas/README.md` opens "While Chai-1 performs
very well in 'single-sequence mode'". Custom MSAs are `.aligned.pqt` parquet files in
`msa_directory`; there is no local-database search, only `msa_server_url`. **VERIFIED.**

Licence is the pleasant surprise and corrects a common belief. `README.md`, verbatim: "Chai-1 is
released under an Apache 2.0 License (**both code and model weights**), which means it can be used
for both academic and commerical purposes, including for drug discovery." **VERIFIED** — and the
`LICENSE` file is plain Apache 2.0 with no rider.

`requires-python = ">=3.10"` with no ceiling, and `requirements.in` has `numpy>=1.21` with no
ceiling — so **Chai-1 is the only other candidate that fits Python 3.13 and numpy 2**. Its torch
line reads `torch>=2.3.1        # 2.2 is broken, latest-patch versions 2.3.1 - 2.7.1 are confirmed
to work correctly`. **VERIFIED.**

Output is mmCIF (`candidates.cif_paths`) plus per-sample `scores.model_idx_<N>.npz`. The README's
prose still claims "a list of PDB files"; the code returns CIF. **VERIFIED** — prose lagging code.

### Why it is not the pick

`pip install chai_lab==0.6.1` is the documented install and 0.6.1 shipped **2025-03-18** — eighteen
months stale relative to a `main` that moved as recently as 2026-06-30. **VERIFIED** from PyPI and
the commits API.

More decisively, Chai's own work has left the open repo. `chai-lab` covers Chai-1 only; the
`chaidiscovery` org has no Chai-2 or Chai-3 repo. Chai-2 has a preprint
(bioRxiv 10.1101/2025.07.05.663018, cited in chai-lab's own README) but no weights, and Chai-3
appears only inside partnership announcements on the company's news page — access is by commercial
agreement, not download. **Chai-2 and Chai-3 are disqualified outright: there is nothing to
install.** **INFERRED** for Chai-3's capabilities; **VERIFIED** that no repo or weights exist.

## 4. Protenix

Repo: [bytedance/Protenix](https://github.com/bytedance/Protenix). The strongest runner-up, and
the most complete entity model of the five.

### Entities

Five types, not four: `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand` and — uniquely —
**`ion`**. The doc opens by naming exactly what it relaxes relative to the AlphaFold Server format
— **VERIFIED**, verbatim:

> 1. There are no restrictions on the types of ligands, ions, and modifications … 2. Users can
> specify bonds between entities … 3. It supports inputting ligands in the form of SMILES strings
> or molecular structure files. 4. Ligands composed of multiple CCD codes can be treated as a
> single entity … "NAG-NAG". 5. The "glycans" field is no longer supported.

A ligand is a CCD code prefixed `CCD_`, a raw SMILES string, or `FILE_<path>` to a PDB/SDF/MOL/MOL2
carrying a 3D conformer. An ion is a bare CCD code with **no** prefix. Polymer-to-polymer covalent
bonds are unsupported except head-to-tail cyclic peptides and disulfides. **VERIFIED**,
[docs/infer_json_format.md](https://github.com/bytedance/Protenix/blob/main/docs/infer_json_format.md).

### Input

A JSON **list** of jobs — that top-level list is the shape difference from AF3. Verbatim from
`examples/example.json`, PDB 7r6r, protein plus two DNA strands — **VERIFIED**:

```json
[{
    "sequences": [
        {
            "proteinChain": {
                "sequence": "MGSSHHHHHHSSGLVPRGSHMSGKIQHKAVVPAPSRIPLTLSEIEDLRRKGFNQTEIAELYGVTRQAVSWHKKTYGGRLTT…",
                "count": 1,
                "id": ["A"],
                "msa": {
                    "precomputed_msa_dir": "./examples/7r6r/msa/1",
                    "pairing_db": "uniref100"
                }
            }
        },
        {"dnaSequence": {"sequence": "TTTCGGTGGCTGTCAAGCGGG", "count": 1, "id": ["B"]}},
        {"dnaSequence": {"sequence": "CCCGCTTGACAGCCACCGAAA", "count": 1, "id": ["D"]}}
    ],
    "name": "7r6r"
}]
```

That `msa` dict is the **deprecated** spelling; the current fields are per-chain
`pairedMsaPath`, `unpairedMsaPath` and `templatesPath` — plain paths, which is the friendliest
interface of the five for a package that already produces A3M. `count` plus optional `id` replaces
AF3's list-valued `id`. Ligands and ions:

```json
{"ligand": {"ligand": "CCD_ATP", "count": 1}},
{"ligand": {"ligand": "FILE_your_file_path/atp.sdf", "count": 1}},
{"ion":    {"ion": "MG", "count": 2}}
```

### MSA

Everything is optional and everything is toggleable: `--use_msa`, `--use_template`, `--use_rna_msa`.
It ships `protenix msa`, `protenix mt` and `protenix prep` CLI subcommands that run the search and
**write the paths back into your JSON**. The default remote is ByteDance's own MMseqs2-protocol
host, and `docs/colabfold_compatible_msa.md` documents running a ColabFold-compatible search
against local databases instead. RNA MSA is a first-class `unpairedMsaPath` on `rnaSequence`, and
templates take `.a3m` or `.hhr`. **VERIFIED** for the JSON fields and the CLI subcommands;
**INFERRED** for the local-database workflow, which I read about but did not run.

### Licence

`README.md`, verbatim: "The Protenix project including both code and model parameters is released
under the Apache 2.0 License. It is free for both academic research and commercial use." The
`LICENSE` file is Apache 2.0. **VERIFIED.** No gate.

### Install and run — and why it loses

`setup.py` says `python_requires=">=3.11"`, so 3.13 is allowed. The problem is `requirements.txt`,
**VERIFIED** verbatim in part:

```text
torch==2.7.1
torchvision==0.22.1
cuequivariance-ops-torch-cu12==0.8.0
rdkit==2025.9.3
biotite==1.4.0
deepspeed==0.17.5
triton==3.3.1
numpy==2.4.1
```

**Twenty-plus exact `==` pins**, including torch, numpy, triton and deepspeed. Every one is a
solve constraint this repo would have to absorb or wall off. `biotite==1.4.0` happens to match
what this repo already resolves (see
[the biotite note](biotite-coverage-of-v1.md#why-the-lock-holds-biotite-at-140)) — pleasant, and
pure luck. It is CLI-only: `protenix pred -i input.json -o ./output`.

To its credit it is the only tool of the five that publishes a sizing table, **VERIFIED**:

| N_token | N_atom | Peak Mem (GB) | Latency (s) |
| --- | --- | --- | --- |
| 500 | 5,000 | 6.1 | 17 |
| 1,000 | 10,000 | 18.2 | 59 |
| 2,000 | 20,000 | 66.6 | 226 |
| 4,000 | 40,000 | 78.1 | 1,424 |

So a ~500-residue monomer is **~6 GB and ~17 s**, and a protein-DNA complex of a few hundred
residues plus twenty base pairs stays in the same row. That is the only primary-source answer to
the sizing question in this note, and it is worth carrying across to the others as an order of
magnitude — **INFERRED** for anyone but Protenix.

Output is `<name>/<seed>/<name>_<seed>_sample_N.cif` plus a summary JSON carrying `plddt`, `gpde`,
`ptm`, `iptm`, `chain_ptm`, `chain_pair_iptm`, `has_clash`, `disorder` and `ranking_score`.
**VERIFIED.**

## 5. OpenFold3 — the pick

**Repo correction: it is [aqlaboratory/openfold-3](https://github.com/aqlaboratory/openfold-3),
hyphenated.** `aqlaboratory/openfold3` is a 404 and `aqlaboratory/openfold` is still the
AlphaFold2 reproduction — **VERIFIED**, I probed all three.

### Why it wins, in one file

`pixi.toml` at the repo root. **VERIFIED**, verbatim:

```toml
[workspace]
name = "openfold3"
channels = ["conda-forge", "bioconda"]
platforms = [
    "linux-64",
    "linux-aarch64",
    { name = "linux-64-cuda12", platform = "linux-64", cuda = "12.0" },
    …
]
```

Channels identical to this repo's. Rich CUDA platform selectors — the same construct this repo
already uses in `platforms = [{ platform = "linux-64", cuda = "12" }, "linux-64"]`. Torch comes
from conda-forge (`[feature.pytorch-conda-cuda.dependencies] pytorch-gpu = "*"`), the MSA binaries
come from bioconda (`[feature.not-in-pypi.target.linux.dependencies] hmmer = "*"`,
`hhsuite = "*"`), and **only `openfold3` itself comes from PyPI**. The header table records
"A100 cuda12 … works" on linux-64 for both the conda and PyPI variants.

Its dependency floors are all open — **VERIFIED** from `pyproject.toml`:

```toml
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
dependencies = ["numpy", "scipy", "pandas", "torch", "ml-collections",
                "pytorch-lightning >=2.1", "biotite", "rdkit<2026", …]

[project.scripts]
run_openfold = "openfold3.run_openfold:cli"
setup_openfold = "openfold3.setup_openfold:main"
```

Classifiers name Python 3.13 **and** 3.14 explicitly. `numpy` and `torch` are bare. It holds
biotite, same as this package. PyPI `openfold3` is at **0.5.0, 2026-08-21** with nine releases
since 2025-10-28 — **VERIFIED**.

One interaction to expect: `pixi.toml` asks for `biotite = ">=1.6"`, and every conda-forge biotite
since 1.5.0 carries a `numpy <2.4` run export. A fold environment would therefore land on numpy
2.3.x while `default` stays on 2.5.x. That is exactly what a separate solve group is for, as
`esm` already demonstrates. **INFERRED**, from the measured cap in
[the biotite note](biotite-coverage-of-v1.md#why-the-lock-holds-biotite-at-140).

### Entities and input

Its own schema, and it is **not** AF3's: the top level is a `queries` **dict** keyed by job name,
and each chain declares its own `molecule_type`. Verbatim from `docs/source/input_format_reference.md`,
the complete protein + DNA + RNA + two-ligand example — **VERIFIED** (the doc's own copy carries
trailing commas and is therefore not strictly valid JSON; the shipped `examples/` files are clean):

```json
{
    "queries": {
        "query_1": {
            "chains": [
                {"molecule_type": "protein", "chain_ids": "A", "sequence": "PVLSCGEWQCL",
                 "use_msas": true, "use_main_msas": true, "use_paired_msas": true},
                {"molecule_type": "protein", "chain_ids": "B", "sequence": "RPACQLWWSRGNWERINQLWW",
                 "use_msas": true},
                {"molecule_type": "dna", "chain_ids": "C", "sequence": "GACCTCT"},
                {"molecule_type": "rna", "chain_ids": "E", "sequence": "AGCU", "use_msas": true},
                {"molecule_type": "ligand", "chain_ids": "Z",
                 "smiles": "CC(=O)OC1C[NH+]2CCC1CC2"},
                {"molecule_type": "ligand", "chain_ids": "I", "ccd_codes": ["NAG"]}
            ]
        }
    }
}
```

A real shipped file, `examples/example_inference_inputs/query_dna_ptm.json`, **VERIFIED** at
`HTTP 200`, showing how modifications are spelled — a dict from 1-based index to CCD code, not a
list of records:

```json
{
    "queries": {
        "ptm-DNA": {
            "chains": [
                {
                    "molecule_type": "dna",
                    "chain_ids": "A",
                    "sequence": "ATUCGTATTCGAT",
                    "non_canonical_residues": {"3": "PSU", "4": "5MC"}
                }
            ]
        }
    }
}
```

Note for `seq.py`: the protein `sequence` field is documented as "supporting standard residues,
X (unknown), and **U (selenocysteine)**". **VERIFIED.** That is one symbol more than biotite's
alphabet carries, and it lands on the same question
[issue #8](https://github.com/liuhlab/liulab-protein/issues/8) is holding.

**The honest gap: there is no covalent-bond field.** Grepping the whole input reference for "bond"
returns one hit, in the prose phrase "non-covalently bound ligands". **VERIFIED** as an absence.
There is also no `ion` molecule type — presumably a ligand CCD code, but the docs do not say.
`pocket_constraint` and per-chain `cyclic` exist; `bondedAtomPairs` has no counterpart. The README
lists "Full parity on all modalities with AlphaFold3" under **Upcoming**, and the project calls
itself `OpenFold3-preview`. **VERIFIED.**

### MSA

`use_msas` defaults to `true` per chain and can be turned off, with the docs' own caveat that
MSA-free "is {discouraged} if the goal is to obtain the highest-accuracy structures". Precomputed
MSAs go in `main_msa_file_paths` / `paired_msa_file_paths` as paths or directories, `.a3m` or
`.sto`. **RNA takes MSAs** — `use_msas` and `main_msa_file_paths` are both documented on the `rna`
chain, unlike Boltz and AF3's DNA. **DNA takes none**, matching AF3. Templates are protein-only,
supplied as alignment files or bare `template_cif_paths` that get aligned with Kalign. The
`--use-msa-server` path targets a ColabFold server whose URL is a `runner.yml` setting, so a local
MMseqs2-App works. **VERIFIED** from
[docs/source/input_format_reference.md](https://github.com/aqlaboratory/openfold-3/blob/main/docs/source/input_format_reference.md);
**INFERRED** for the private-server deployment, which I read but did not run.

### Licence and weights

Apache-2.0 for the code (`pyproject.toml`, `LICENSE`) and Apache-2.0 for the weights. README,
verbatim: "our repository is freely available for academic and commercial use under the Apache 2.0
license". The HuggingFace model card carries `license: apache-2.0`. **VERIFIED.**

The download needs no credentials at all — `docs/source/Installation.md`, verbatim:

```bash
aws s3 cp s3://openfold3-data/openfold3-parameters/of3-ob-2025-06-30-174k.pt  <dst_path> --no-sign-request
```

`HEAD` on that object returns `HTTP/1.1 200 OK`, `Last-Modified: Fri, 24 Jul 2026`. **VERIFIED.**
`setup_openfold` does it for you into `~/.openfold3`, and it also seeds the CCD **through biotite's
own `get_ccd`** — the same library this package already holds.

The default parameters since 0.5.0 are **OpenBind-0** (2026-08-21, OpenBind Consortium, June 2025
training cutoff). Its licence is stated in the announcement rather than in a file I fetched, so:
Apache-2.0 for the checkpoint is **VERIFIED** from the model card's `license` tag and the README;
the consortium blog's wording is **INFERRED**.

### Install, run, output

`pip install openfold3`, or `pixi run -e openfold3-base setup_openfold`. Environments offered are
`openfold3-base` (no GPU), `openfold3-cuda12`, `openfold3-cuda13`, `-pypi` variants of each, and
`openfold3-rocm7`. Docker images exist too. Stated requirement: "minimum of CUDA 12.1 and 32GB of
memory… Most of our testing has been performed on A100s with 40GB". **No VRAM or wall-clock table
is published** — **VERIFIED** as an absence; use Protenix's row as the order of magnitude.

CLI only: `run_openfold predict --query_json=… [--use-msa-server] [--num-diffusion-samples=5]
[--num-model-seeds=1]`. No importable prediction API is documented — **VERIFIED** as an absence in
the docs, not as proof none exists.

Output is `.cif` by default, `.pdb` optional with pLDDT in the B-factor column. Per-atom
`*_confidences.json` gives `plddt`, `pae`, `pde`; `*_confidences_aggregated.json` gives
`avg_plddt`, `gpde`, `ptm`, `iptm`, `disorder`, `has_clash`, `sample_ranking_score`, `chain_ptm`,
`chain_pair_iptm`. Default is 1 seed × 5 diffusion samples. **VERIFIED.**

## 6. NVIDIA BioNeMo changes nothing

Two products share the name, and neither helps a pixi project. **VERIFIED** except where noted.

- **BioNeMo Framework / Recipes** (`NVIDIA-BioNeMo/bionemo-framework`, Apache-2.0, v3.0.0
  2026-06-24) ships `amplify`, `codonfm`, `esm2`, `geneformer`, `llama3`, `mixtral`, `qwen` —
  **no structure predictor at all**. pip or NGC container; not on conda.
- **BioNeMo NIM microservices** is where the predictors live — AlphaFold2, AlphaFold2-Multimer,
  OpenFold2, OpenFold3 and Boltz-2 all have NIMs. They ship **only** as prebuilt Docker/NGC
  containers under NVIDIA's AI Foundation Models EULA. Self-hosting is free for Developer Program
  members on up to two nodes or 16 GPUs for development and research; production wants NVIDIA AI
  Enterprise. **INFERRED** — relayed from NVIDIA's own docs and blog, which I did not fetch
  myself.

So BioNeMo would replace "PyPI package inside a pixi feature" with "Docker container behind an
EULA and an API key" on a shared academic host. Strictly worse on every axis this repo cares
about, and it adds a licence that the current one does not carry.

## 7. What is actually new since May 2026

The gap this note was asked to fill. **VERIFIED** unless marked.

- **OpenFold3 0.4.2 → 0.5.0** (2026-06-29 through 2026-08-21), with OpenBind-0 becoming the default
  checkpoint. This is the significant one and it is two weeks old.
- **AlphaFold 3 v3.0.3 (2026-06-09) and v3.0.4 (2026-07-28)**, weights re-uploaded 2026-06-04. No
  licence change — the terms still say "Last Modified: 2024-11-09".
- **Protenix v2.0.0** landed 2026-04-07 (just inside the old cutoff) and development continued to
  2026-08-01.
- **Chai-3 exists and is closed.** It appears only in partnership announcements from 2026-06-04
  onward. No repo, no weights, no preprint found. **INFERRED** for anything beyond its existence.
- **No Boltz-3.** No tag after v2.2.1; no announcement found.
- **No open Chai-2.**
- HelixFold3 remains PaddlePaddle-only with a licence its own tracker disputes; NeuralPLexer3's
  weights are non-commercial; no 2026 successor to RoseTTAFold All-Atom was found. All three are
  **INFERRED** and none changes the ranking.

## Comparison, for this repo specifically

Ranked for: pixi, conda-forge first, `platforms = ["linux-64"]`, Python 3.13, numpy ≥ 2, one shared
academic GPU, and a package that already produces MSAs with `mmseqs`.

| | **OpenFold3** | **Protenix** | **Chai-1** | **Boltz-2** | **AlphaFold 3** |
| --- | --- | --- | --- | --- | --- |
| Protein / DNA / RNA | yes | yes | yes | yes | yes |
| Ligand SMILES / CCD | yes / yes | yes / yes / file | yes / — | yes / yes | yes / yes / userCCD |
| Explicit ion type | no | **yes** | no | no | no |
| Modified residues | yes | yes | yes | yes | yes |
| Covalent bonds | **no** | yes | yes | yes | yes |
| Input | JSON, `queries` dict | JSON, list of jobs | typed FASTA + CSV | YAML | JSON, one job |
| MSA required | no | no | **no, by design** | yes by default | no |
| Accepts a precomputed A3M | yes | yes | parquet only | yes | yes |
| Local MSA server | yes | yes | yes | yes | **no** — 630 GB pipeline |
| RNA MSA | yes | yes | — | no | yes |
| Code licence | Apache-2.0 | Apache-2.0 | Apache-2.0 | MIT | Apache-2.0 |
| **Weights licence** | **Apache-2.0** | **Apache-2.0** | **Apache-2.0** | **MIT** | **non-commercial, binds output** |
| Weights gated | no | no | no | no | terms, not a form |
| `requires-python` | `>=3.10`, names 3.13 | `>=3.11` | `>=3.10` | **`<3.13`** | `>=3.12` |
| numpy | bare | `==2.4.1` | `>=1.21` | **`<2.0`** | n/a (JAX) |
| conda-forge / bioconda | no | no | no | no | no |
| **Upstream pixi manifest** | **yes, same channels** | no | no | no | no |
| Latest release | **2026-08-21** | 2026-04-07 | 2025-03-18 | 2025-09-08 | 2026-07-28 |
| Python API | no | no | **yes** | no | no |
| Output | mmCIF / PDB | mmCIF | mmCIF | mmCIF / PDB | mmCIF only |
| **Verdict** | **pick** | strong runner-up | fallback | **disqualified** | **disqualified** |

**Disqualified, and why:**

- **Boltz** — install, not licence. `requires-python = ">=3.10,<3.13"` and `numpy>=1.26,<2.0` both
  contradict this repo's floors, so it needs a second Python in the workspace to exist at all. Its
  MIT weights are the most permissive of the five, which is why it is worth revisiting *if and only
  if* those two bounds move.
- **AlphaFold 3** — licence, and it is not close. The weights are non-commercial-only and the
  restrictions ride along with the output, so a library that shells out to it silently attaches
  terms to files its callers produce. Add Docker-first install, no MSA server, and 630 GB of
  databases with no way to substitute the `mmseqs` this repo already runs.
- **Chai-2, Chai-3** — nothing to install. Platform access by commercial agreement.
- **BioNeMo** — Docker and an EULA in place of a package.

**Why OpenFold3 over Protenix**, since both are Apache-2.0 on code and weights, both are actively
developed, and Protenix has the better entity model:

1. **Protenix pins twenty-plus dependencies with `==`**, including `torch==2.7.1`, `numpy==2.4.1`,
   `triton==3.3.1` and `deepspeed==0.17.5`. OpenFold3 pins none of them and has already proved the
   solve on conda-forge for CUDA 12 and 13.
2. **OpenFold3 ships the pixi manifest.** Nobody has to invent the environment; the upstream one
   is a reference for the feature this repo would add, on the same two channels.
3. It is two weeks old versus five months, and it holds biotite.

And why *not* OpenFold3, stated plainly so the trade is visible: it is self-described as a
**preview**, "Full parity on all modalities with AlphaFold3" is still listed as upcoming, and it
has **no covalent-bond input at all**. If a covalent ligand or a polymer-spanning bond is a real
requirement, Protenix is the answer and the exact pins are the price.

## What driving one would cost this package

Five costs, each grounded in a file in this working tree.

**1. The subprocess seam nearly fits, and the install message does not.**
`external.py` sets `REQUIRED_TOOLS = ("mmseqs", "foldseek")`, and `InstalledTool` locates a binary
on `PATH`. `run_openfold` *is* on `PATH` inside a pixi environment that has the package, so
`InstalledTool("run_openfold")` works unchanged. What breaks is one string:
`install_instructions()` emits `pixi add {self.package}            # channels: conda-forge,
bioconda`, and no channel carries this. `_Installation` already carries a `package` and a
`homepage`; it would need to carry the *kind* of install too, or the message lies. That is a small,
generalizable change to a message that is currently true for exactly two tools.

**2. It is a subprocess, so it does not get an object.** The map's rule is "resident state gets an
object; a subprocess does not" — `ESMC()` holds weights across calls, `mmseqs` holds nothing.
`run_openfold` loads its checkpoint per invocation and exits, so by that rule prediction is a
**method**, not a class. This is the one place where a folding tool looks tempting to model as
`ESMC` and should not be.

**3. "Direct support only" forbids `Protein.fold()`.** These models take an assembly: protein plus
DNA plus RNA plus ligand, in one job, with per-chain MSA paths. A `Protein` would first have to
acquire the other chains, which is precisely the hidden acquisition the rule exists to prevent —
the same reason `search()` lives on `Structure` and `Chain` and not on `Protein`. The input is its
own thing and wants its own type; the tables above are, read another way, a specification for what
that type has to be able to say.

**4. It creates a third relation between the two namespaces, and it is not SIFTS.**
`CLAUDE.md` is emphatic that a `Protein` and a `Structure` are joined by SIFTS alone. Prediction
*manufactures* a `Structure` from `Protein`s — a `Structure` with no PDB id, which is the thing a
PDB id is defined to address. Whether a predicted model is a `Structure` at all, and what addresses
it if so, is a design question this note raises and declines to settle.

**5. The gate constrains the lane in three known ways.** `src/` is collected, so `import protein`
must not import torch — imports stay in method bodies, as `embed/` already does. The markers are a
boolean partition and the gate is `-m 'not model'`, so a prediction test reuses the `model` marker
rather than adding a third. And `--doctest-modules` runs every `Examples` block, so anything that
invokes the binary needs `# doctest: +SKIP` on its own line.

Against those five, the thing that is **already paid for**: every candidate accepts a precomputed
A3M per chain, this package already drives `mmseqs`, and `run_to`'s make-style freshness rule
already exists to avoid rebuilding an alignment that is current. Nothing here needs the ColabFold
public server, and nothing here needs 630 GB.

## Open items

- **What the input type is.** Not `Protein.fold()`. The five schemas agree on the shape — a list of
  typed chains, each with an id, a sequence, optional per-chain MSA paths and optional
  modifications — and disagree on every field name. That agreement is what a type should capture.
- **Whether a predicted model is a `Structure`.** It has chains and coordinates and no PDB id.
- **Whether the covalent-bond gap disqualifies OpenFold3 for this lab.** If it does, the note's
  answer is Protenix and the exact pins are the cost.
- **Boltz is worth re-checking when its two bounds move.** MIT weights remain the most permissive
  of the five; only `requires-python <3.13` and `numpy<2.0` keep it out, and both are one upstream
  commit away.
- **The biotite floor.** OpenFold3 asks `>=1.6`; conda-forge biotite at or above 1.5.0 caps
  `numpy <2.4`. A separate solve group absorbs it, exactly as `esm` does — worth confirming on
  GPU71FM before anyone writes the feature.
