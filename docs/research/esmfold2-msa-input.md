---
search:
  exclude: true
---

# What type does ESMFold2's `ProteinInput.msa` field take?

Resolves issue #49. The ESM research note recorded `ProteinInput(id, sequence, modifications, msa)`
as verified field *names* with no types. This note pins the type, the parser, the silent failure
modes, and what a call site in this package would actually have to hand upstream.

## Answer, shortest form

**It is a class, not a string and not a path.** `msa: MSAInput = None`, where
`MSAInput: TypeAlias = Union[MSA, None]` and `MSA` is `esm.utils.msa.MSA` — a frozen dataclass
holding `entries: list[FastaEntry]` plus an optional `deletions` array. Identical on `ProteinInput`
and `RNAInput`; `DNAInput` has no `msa` field. The default is `None`.

**The builder parses nothing.** `ESMFold2InputBuilder` never sees A3M text or a filesystem path.
Parsing is the caller's job, done by `MSA` classmethods before construction —
`MSA.from_a3m(path, remove_insertions=..., max_sequences=...)`, `MSA.from_sequences(list[str])`,
`from_stockholm`, `from_bytes`. `from_a3m` takes `PathOrBuffer`, so **an in-memory `io.StringIO` of
A3M text works with no temp file**.

**Almost nothing is validated, and what goes wrong goes wrong silently.** Nothing checks that row 0
equals the query. Nothing checks that a row is query-length. Rows longer than row 0 are truncated
mid-row; rows shorter are gap-filled; an MSA shorter than the chain has its last column repeated
across the tail. Only `from_a3m` enforces anything — that all rows agree on match-column count — and
`from_sequences` bypasses even that.

**One field, not AF3's two.** Pairing is internal and driven by a `key=<int>` token in each FASTA
header: rows sharing a `key` across chains land in the same MSA row, `key=-1` or no `key` means
unpaired. A complex is one `ProteinInput` per chain, each carrying its own `MSA`. The caller
annotates headers; it does not build the layout.

**"ESMFold2-Fast ignores the MSA" is true remotely and false locally.** On Forge it warns and
ignores. Running the local weights, `msa_encoder.enabled: false` only removes the pair-stream
conditioning — the `profile` and `deletion_mean` input features are still computed from the MSA
rows, because `model.py` never reads `disable_msa_features`. No error either way.

**And folding with no MSA emits a `UserWarning` per protein chain.** Under this repo's
`filterwarnings = ["error"]` that is an exception, not a log line.

## Method and provenance

| | |
| --- | --- |
| Executed on | this macOS laptop. **Nothing installed, imported, resolved or run**; no GPU |
| Primary source | `esm-3.4.0-py3-none-any.whl`, 2,678,321 bytes, sha256 `847bf3cf…c55cdbef` — matching the PyPI digest. Downloaded and unpacked with `unzip`; every **VERIFIED** source claim is read from that tree |
| Version chosen | `pypi.org/pypi/esm/json` lists 3.4.0 as the newest release and the only one inside the repo's `esm >=3.4,<4` pin |
| Cross-checked | six MSA-path files fetched from `raw.githubusercontent.com/Biohub/esm/v3.4.0` are **byte-identical** to the wheel: `paired_msa.py`, `input_builder.py`, `msa.py`, `processor.py`, `prepare_input.py`, `constants/models.py` |
| Also queried | `huggingface.co/biohub/ESMFold2` and `…/ESMFold2-Fast` `config.json` and `README.md`; `api.github.com/repos/Biohub/esm/tags`; the first-party `cookbook/tutorials/esmfold2.ipynb` and its `g3l5_chainA.a3m` / `g3l5_chainB.a3m` at tag `v3.4.0` |
| Date | 2026-09-03 |

**VERIFIED** = read out of the unpacked wheel, or returned by an API call made here.
**INFERRED** = read from a model card, notebook or README without running it.

## 1. The annotation and the default

**VERIFIED.** `esm/utils/structure/input_builder.py`, lines 6–36:

```python
from esm.utils.msa import MSA

# fmt: off
MSAInput: TypeAlias = Union[
    MSA,
    None,
]
# fmt: on
```

```python
@dataclass
class ProteinInput:
    id: str | list[str]
    sequence: str
    modifications: list[Modification] | None = None
    msa: MSAInput = None


@dataclass
class RNAInput:
    id: str | list[str]
    sequence: str
    modifications: list[Modification] | None = None
    msa: MSAInput = None


@dataclass
class DNAInput:
    id: str | list[str]
    sequence: str
    modifications: list[Modification] | None = None
```

Byte-identical alias duplicated at `esm/sdk/forge.py:61`. `esm/models/esmfold2/types.py` re-exports
`MSA`, `ProteinInput`, `RNAInput` for the ESMFold2 namespace; the definitions live only in
`input_builder.py`.

`MSA` is `esm/utils/msa/msa.py:62`:

```python
@dataclass(frozen=True)
class MSA(SequentialDataclass):
    entries: list[FastaEntry]
    deletions: np.ndarray | None = dataclasses.field(default=None, compare=False)
```

with `FastaEntry = NamedTuple("FastaEntry", [("header", str), ("sequence", str)])`
(`esm/utils/parsing.py:6`).

**The annotation is the whole story at construction.** **VERIFIED** — neither `ProteinInput`,
`RNAInput` nor `MSA` defines `__post_init__`; the only one in either file is on `FastMSA`
(`msa.py:475`), and `SequentialDataclass` adds slicing helpers, not validation. So
`ProteinInput(id="A", sequence="M", msa="ACDE")` constructs without complaint.

Exactly one place issues a clean type error, and it is not on the local folding path —
`serialize_structure_prediction_input`, `input_builder.py:138-141`:

```python
        elif isinstance(seq_input.msa, MSA):
            chain_data["msa"] = seq_input.msa.state_dict(json_serializable=True)
        else:
            error_msg = f"MSA must be None or MSA. Got {seq_input.msa} instead."
            raise AttributeError(error_msg)
```

That fires only when serializing for Forge. The deserializer refuses a string explicitly
(`input_builder.py:217-218`): `raise ValueError(f"Unexpected MSA string value: {msa_blk!r}")`. Pass a
`str` to the *local* builder and you instead get an `AttributeError` on `.depth` deep inside
`construct_paired_msa`.

`RNAInput.msa` is **dead on the ESMFold2 path** — see §4.

## 2. Who parses it

**VERIFIED — the builder parses nothing.** `ESMFold2InputBuilder.prepare_input`
(`esm/models/esmfold2/processor.py:189`) takes a `StructurePredictionInput` whose `msa` is already an
`MSA` object. There is no A3M reader, no path handling, no tokeniser in `esm/models/esmfold2/` that
accepts text.

Parsing is the caller's, via `MSA` classmethods (`esm/utils/msa/msa.py`):

| Constructor | Signature | Notes |
| --- | --- | --- |
| `MSA.from_a3m` | `(path: PathOrBuffer, remove_insertions: bool = False, max_sequences: int \| None = None)` | The canonical one. Computes and stores per-row `deletions` |
| `MSA.from_sequences` | `(sequences: list[str], remove_insertions: bool = False)` | **Sets every header to `""`** — kills pairing (§4) |
| `MSA.from_stockholm` | `(path, remove_insertions: bool = True, max_sequences=None)` | Strips query-gap columns |
| `MSA.from_bytes` / `from_sequence_bytes` / `from_state_dict` | — | Wire and storage formats |

`PathOrBuffer = T.Union[PathLike, io.StringIO]` (`esm/utils/system.py:7`), and `read_sequences`
duck-types it (`esm/utils/parsing.py:39-56`): it tries `open(path)`, catches `TypeError`, and falls
back to treating the argument as a text buffer. **A `StringIO` of A3M text is a first-class
argument.** `.gz` is auto-detected by name; no other extension is checked. `parse_fasta` skips blank
lines and lines starting with `#`.

**The path from field to tensors, VERIFIED, in order:**

1. `ESMFold2InputBuilder.prepare_input` → `clean_esmfold2_input` (`processor.py:87`) — splits
   chainbreak sequences and column-slices a single MSA per chain (§4).
2. → `prepare_esmfold2_input` (`prepare_input.py`), which warns on a missing MSA at lines 723-726.
3. → `compute_msa_features` (`prepare_input.py:1131`) — collects `asym_id -> MSA`, substituting
   `MSA.from_sequences([item.sequence])` for a protein with none.
4. → `construct_paired_msa` (`paired_msa.py:95`) — pairing, capping, block-diagonal layout.
5. → `msa_to_res_type_and_deletions` (`paired_msa.py:42`) — the only place A3M *convention* is
   interpreted, mapping letters through `protein_letter_to_res_type()` and accumulating insertion
   runs into deletion counts.
6. → `torch.from_numpy`, returning `{"msa", "deletion_value", "has_deletion", "deletion_mean",
   "msa_attention_mask"}` (`prepare_input.py:1213-1218`). `deletion_value` is
   `(np.pi / 2) * torch.arctan(del_data / 3)`.
7. → `EsmFold2Model.forward`, where the rows become the `profile` feature (`model.py:1002-1011`) and,
   if the encoder exists, pair conditioning through `MSAEncoder` (`model.py:1096-1116`).

## 3. Constraints beyond the A3M format

### Nothing requires row 0 to equal the query

**VERIFIED — no such check exists anywhere in the tree.** Row 0 is *used* as the query
(`paired_msa.py:56-57`) but never compared against `ProteinInput.sequence`:

```python
    query = msa.entries[0].sequence
    L = sum(1 for ch in query if not is_a3m_insertion(ch))
```

The first-party cookbook treats it as the caller's responsibility, printing rather than asserting
(**INFERRED**, `cookbook/tutorials/esmfold2.ipynb` cell 29):

```python
# Sanity check — first MSA seq should match the query
print("First MSA seq matches query:", msa.sequences[0] == ubiquitin_sequence)
```

and cell 31: *"The query sequence should always be first."* AF3's hard equality requirement has no
counterpart here.

### Ragged rows: checked at parse, silently mangled after

**VERIFIED.** `MSA.from_a3m` is the only enforcement, and it compares rows to each other, not to the
query (`msa.py:102-106`):

```python
            if deletion_rows and len(deletion_row) != len(deletion_rows[0]):
                raise ValueError(
                    "A3M match-column count mismatch. "
                    f"Expected: {len(deletion_rows[0])}, Received: {len(deletion_row)}"
                )
```

`MSA.from_sequences` performs no check at all. Past that gate everything degrades in silence
(`paired_msa.py:63-81`):

- a row **longer** than `L` is cut mid-row — `if col >= L: break`;
- a row **shorter** than `L` leaves the remaining columns at `MSA_GAP_TOKEN_ID`, initialised by
  `np.full`.

And if the MSA is shorter than the chain it describes, `paired_msa.py:238-240` repeats its last
column across every remaining token:

```python
        # Clamp residue indices to the MSA's column range. Modified-residue
        # tokens that exceed the query length fall back to the last column.
        cols = np.minimum(token_res_in_chain, Lc - 1)
```

None of these raise. **A misaligned MSA produces a confident wrong fold, not an error.**

### Caps, truncation, reordering

**VERIFIED.** Four separate limits, all silent:

| Limit | Value | Where | Behaviour |
| --- | --- | --- | --- |
| `max_pairs` | 8192 | `paired_msa.py:102` | stops adding taxonomy-paired rows |
| `max_total` | 16384 | `paired_msa.py:103` | stops adding unpaired rows |
| `max_seqs` | 16384 | `paired_msa.py:104`, applied as `pairing = pairing[:max_seqs]` at line 216 | hard row cut |
| `msa_max_depth` | 1024 | `processor.py` `fold(..., msa_max_depth: int \| None = 1024, ...)`; also `EsmFold2MsaEncoderConfig.max_depth = 1024` (`config.py:217`) | **random** per-loop subsample |

`ESMFOLD2_MAX_MSA_SEQS = 16384` (`esm/utils/constants/models.py:16`) is the same number surfaced to
Forge users as a warning. On the local path there is no warning — rows past 16384 are dropped.

The 1024 subsample is `maybe_subsample_msa` (`layers.py:182-211`), which keeps row 0 and draws a
fresh `torch.randperm` of the rest **on every loop**, so a fold is not deterministic in its MSA rows
unless seeded. Alongside it `maybe_apply_msa_column_masking` masks `msa_column_mask_rate = 0.1` of
columns once per fold.

**Row order is not file order** when headers carry repeated `key=` values: taxonomies are emitted
sorted by how many distinct chains they touch (`paired_msa.py:162-164`), and unpaired rows follow
after. For a single chain whose headers carry no `key=` at all, order is preserved.

**No deduplication exists anywhere.** Identical rows are kept and counted toward every cap.

## 4. Paired vs unpaired — one field, pairing done internally

**VERIFIED.** One field. There is no second field for paired alignments, and no AF3-style
paired/unpaired split. `esm/models/esmfold2/paired_msa.py:1-7` states the contract:

```python
"""Taxonomy-paired MSA construction for ESMFold2 inference.

Taxonomy IDs are read from FASTA headers as ``key=N`` tokens. Rows
where any chain has ``key=-1`` (or no ``key=`` at all) are treated as
unpaired and assigned to that chain's block-diagonal section after
the paired rows.
"""
```

The whole mechanism is a regex over the header (`paired_msa.py:21`, `35-39`):

```python
_KEY_RE = re.compile(r"key=(-?\d+)")

def _taxonomy_from_header(header: str) -> int:
    if not header:
        return -1
    m = _KEY_RE.search(header)
    return int(m.group(1)) if m else -1
```

So **pairing is done internally, from data the caller puts in the headers.** A third feature tensor,
`is_paired`, is broadcast per row and chain and returned alongside the residues and deletions.

A multi-chain complex is **one `ProteinInput` per chain, each with its own `MSA`** — **INFERRED**
from cookbook cell 32-33, which folds PDB 7YTU as two `ProteinInput`s with
`MSA.from_a3m("g3l5_chainA.a3m", remove_insertions=True)` and the B-chain equivalent, and states:
*"To make your own MSAs: run a paired search (e.g. ColabFold/MMseqs2), then rewrite each hit's
`OX=<organism_id>` tag to `key=<organism_id>`."*

The shipped example files are worth seeing, because they are simpler than the prose suggests —
**VERIFIED**, `g3l5_chainA.a3m` and `g3l5_chainB.a3m` at `v3.4.0`:

```text
>key=0
GPYYPTNKLQAAVMETDRENAIIRQRNDEIPTRTLDTAIFTDASTVASAQIHLYYNSNIGKIIMSLNGKKHTFNLYDDNDIRTLLPILLLSK
>key=1
--YYPTNKLQAAVMETDRENSIIRQRNDEIPTRTLDTAIFTDASTVASAQIHLYYNSNIGKIIMSLNGKKHTFNLYDDNDIRTLLPILLLSK
```

The header is *only* the key — no accession, no description. 41 distinct headers across 100 rows in
chain A and 97 in chain B, all 41 shared between the two files, `key=-1` among them. Repeated keys
within a chain are cycled (`seq_idxs[i % len(seq_idxs)]`, `paired_msa.py:184`).

Two consequences a caller must know:

- **`MSA.from_sequences` cannot express pairing.** It writes `FastaEntry("", seq)` for every row
  (`msa.py:189-194`), and `_taxonomy_from_header("")` returns `-1`. Rows are unpaired by
  construction.
- **`RNAInput.msa` is inert.** `compute_msa_features` reads `item.msa` only for proteins
  (`prepare_input.py:1161-1167`):

  ```python
              if isinstance(item, ProteinInput):
                  msa = item.msa
                  if msa is None:
                      msa = MSA.from_sequences([item.sequence])
                  chain_msas[chain.asym_id] = msa
              else:
                  chain_msas[chain.asym_id] = None
  ```

  and the letter mapping is `protein_letter_to_res_type()` regardless. An `MSA` on an `RNAInput` is
  accepted, serialised, and then dropped without a warning.

Two structural cases beyond one-MSA-per-chain, both **VERIFIED**:

- **One `ProteinInput` with `id=["A","B"]`** broadcasts the same MSA to every chain it names
  (`prepare_input.py:1157-1165`).
- **A chainbreak sequence** (`"AAA|BBB"` or `"AAA:BBB"`) in one `ProteinInput` carries **one MSA for
  the concatenation**, sliced by raw column index (`processor.py:147-152`):

  ```python
                          chain_msa = item.msa.select_positions(  # type: ignore
                              np.arange(chain_start, chain_end)
                          )
  ```

  with `pos += len(chain) + 1` (line 133). The alignment must therefore contain **one column per
  separator character**, and must be insertion-free for column indices to mean anything.

## 5. What happens with no MSA, and what Fast does

### No MSA: a warning, then a depth-1 alignment

**VERIFIED**, `prepare_input.py:723-726`:

```python
                if item.msa is None:
                    warnings.warn(
                        f"No MSA provided for {item.id}, using single sequence mode"
                    )
```

A `UserWarning` per protein chain, unconditional, on the ordinary single-sequence path. **Under this
repo's `filterwarnings = ["error"]` that is an exception**, so any test or doctest that folds without
an MSA needs a targeted entry in `pyproject.toml`.

The chain then gets `MSA.from_sequences([item.sequence])` — a depth-1 alignment of the query — so
`profile` becomes the query's own one-hot and `deletion_mean` is zero. Nothing is left unset.

### ESMFold2-Fast neither rejects nor errors, and only partly ignores

**VERIFIED** from the HuggingFace `config.json` of each checkpoint, fetched 2026-09-03:

| | `biohub/ESMFold2` | `biohub/ESMFold2-Fast` |
| --- | --- | --- |
| `msa_encoder.enabled` | `true` | `false` |
| `msa_encoder_overwrite` | `true` | `true` |
| `disable_msa_features` | `false` | `false` |
| `type` | `release` | `release` |

Neither ships `max_depth` or `column_mask_rate`, so the dataclass defaults (1024, 0.1) apply.

`enabled: false` means the module is never built (`model.py:617-620`) and the pair-conditioning
branch is skipped (`model.py:1096`): `if self.msa_encoder is not None and msa is not None:`.

But the `profile` feature is upstream of that branch and is gated on the MSA alone
(`model.py:1002-1011`):

```python
        if msa is not None:
            msa_oh_profile = F.one_hot(msa.long(), num_classes=NUM_RES_TYPES).float()
            if msa_attention_mask is not None:
                mask_f = msa_attention_mask.float().unsqueeze(-1)
                msa_oh_profile = msa_oh_profile * mask_f
                valid_seq_count = msa_attention_mask.float().sum(dim=1).clamp(min=1)
                profile = msa_oh_profile.sum(dim=1) / valid_seq_count.unsqueeze(-1)
            else:
                profile = msa_oh_profile.mean(dim=1)
        else:
            profile = res_type_oh
```

The `disable_msa_features` switch that would zero `profile` and `deletion_mean` is read **only** in
`experimental.py:946,958` — **VERIFIED**, `grep -rn disable_msa_features` returns no hit in
`model.py`. Both released checkpoints are `type: release`, so they run `model.py`.

**So locally: ESMFold2-Fast plus an MSA runs, does not warn, drops the pair conditioning, and still
lets the alignment shape `profile` and `deletion_mean`.** That contradicts the remote path's own
wording. On Forge the SDK warns explicitly (`forge.py:393-399`, and `163-168` for the single-sequence
`fold`):

```python
                warnings.warn(
                    f"Model '{model_name}' was not trained with MSA and will ignore "
                    "any MSA provided in ProteinInput. Remove the MSA to suppress this warning.",
                    UserWarning,
                    stacklevel=4,
                )
```

The model card agrees in prose — *"The ESMFold2-Fast variant is an inference optimized single-sequence
structure prediction model and is not MSA conditioned"* (**INFERRED**). Whether the server-side model
is a different build, or whether the warning simply overstates, cannot be settled from the wheel.
Passing an MSA to Fast is pointless either way; it is just not free of effect locally.

One path does hard-reject: `fold_max_accuracy`, **VERIFIED** `esm/sdk/validation.py:20-23`:

```python
        if isinstance(seq, ProteinInput) and seq.msa is not None:
            raise ValueError(
                "fold_max_accuracy generates its own MSA and cannot fold against a supplied one."
            )
```

## 6. Non-obvious caller obligations

**All VERIFIED unless marked.**

- **No id matching.** Headers are read for `key=` and nothing else. Nothing ties an MSA header to
  `ProteinInput.id`, and no header format is required — the shipped examples are bare `>key=7`.
- **No file extension requirement.** Only `.gz` is special-cased, by name, in `read_sequences`. An
  `.a3m` suffix is convention.
- **Use `from_a3m`, not `from_sequences`, whenever pairing matters** — `from_sequences` blanks every
  header (§4).
- **`remove_insertions=True` is safe and is what the cookbook uses.** `from_a3m` stores the deletion
  counts before stripping (`msa.py:99-113`) and `msa_to_res_type_and_deletions` prefers the stored
  array over recomputing (`paired_msa.py:83-86`). The stored-shape check there is a bare `assert`, so
  `python -O` removes it.
- **`MSA.to_a3m` is a writer, not a serialiser.** `def to_a3m(self, path: PathOrBuffer) -> None`
  (`msa.py:115`) — it takes a destination and returns nothing. Do not confuse it with a
  `to_a3m() -> str` on a class of our own.
- **An empty `MSA` is a landmine.** `MSA(entries=[])` constructs, then `seqlen` raises `IndexError`
  on `self.entries[0]`. `construct_paired_msa` guards `m.depth == 0` but nothing else does.
- **`MSA.hhfilter()` shells out** to an `hhfilter` binary via `run_subprocess_with_errorcheck`
  (`esm/utils/msa/filter_sequences.py`). It is not a wheel dependency. `greedy_select` and
  `select_diverse_sequences` are pure Python and numpy.
- **Round-tripping through the Forge wire format loses pairing.** `state_dict` emits `sequences`,
  `deletions` and `headers` (`msa.py:204-211`), but the deserialiser reconstructs with
  `MSA.from_sequences(msa_blk["sequences"])` (`input_builder.py:219`) — **headers and deletions are
  dropped**, and with the headers goes every `key=`. The shipped test asserts only that sequences
  survive (`esm/utils/structure/input_builder_test.py:68-79`). This looks like an upstream bug.

## What this decides for this package

**The folding call site hands upstream an `esm.utils.msa.MSA` object.** Not a string, not a path, not
rows. Neither `to_a3m() -> str` nor `write(path) -> Path` is passed directly, so the #37 decision to
offer both is not settled *for* us by upstream — but it does pick a winner:

```python
msa = esm.utils.msa.MSA.from_a3m(io.StringIO(our_msa.to_a3m()), remove_insertions=True)
ProteinInput(id="A", sequence=protein.sequence, msa=msa)
```

**A string beats a path**, because `from_a3m` accepts a buffer and the temp file a path would force
is pure overhead — and because `LIULAB_DATA holds no outputs` makes a written path a required
argument we would otherwise have to invent. The a3m text must carry **real headers**, since dropping
them costs cross-chain pairing; that rules out anything shaped like `MSA.from_sequences(rows)`.

Two constraints propagate into whatever we build:

- **Our a3m writer must preserve headers verbatim**, so a caller can rewrite `OX=` to `key=`. That is
  an argument for the `io/a3m.py` record layer the MSA-formats note already proposed, over rows of
  bare sequence.
- **We should assert what upstream does not.** Row 0 equals the query, and every row is
  query-length once insertions are stripped. Upstream fails silently on both; a check on our side is
  cheap and is the difference between an error and a wrong structure.

## Open items

- **`RNAInput.msa` is accepted and discarded** by the ESMFold2 path (§4). Whether the Forge server
  honours it is not answerable from the wheel. Worth an upstream issue.
- **The wire round-trip drops headers and deletions** (§6), so a serialised complex loses its
  pairing. Also worth an upstream issue, and a reason to prefer the local builder over Forge for
  paired complexes.
- **Whether ESMFold2-Fast's server build differs from the local one** (§5) would be settled by
  folding the same input twice with and without an MSA on a GPU and comparing coordinates — the one
  question here that needs hardware.
- **The `assert` in `msa_to_res_type_and_deletions`** is the only guard on stored-deletion shape and
  vanishes under `python -O`.
