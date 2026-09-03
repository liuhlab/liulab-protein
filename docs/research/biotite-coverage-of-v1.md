---
search:
  exclude: true
---

# What biotite and its peers already cover of v1's surface

Research note for [issue #21](https://github.com/liuhlab/liulab-protein/issues/21). Walks
[issue #1](https://github.com/liuhlab/liulab-protein/issues/1)'s module layout and records what
a maintained package already does. The stance is settled by the map's Notes and is not reopened
here: our classes **hold** biotite's types and never subclass them.

## Answer, shortest form

**biotite's `ProteinSequence` alphabet is 24 symbols — `ACDEFGHIKLMNPQRSTVWYBZX*`. It has no
`U`, no `O` and no `J`, and never has.** `ProteinSequence("MKTU")` raises
`AlphabetError: Symbol 'U' is not in the alphabet`, which names one symbol, carries no position
and carries no attributes. Worse than a hard failure: at three I/O boundaries biotite *silently
repairs* the input instead — `U` becomes `C`, `O` becomes `K` — so a selenoprotein loses its
selenocysteine without an exception and, for `O`, without even a warning.

That is the whole of the answer to the load-bearing question, and it settles
[#8](https://github.com/liuhlab/liulab-protein/issues/8): `seq.py`'s alphabet and its
`InvalidResidueError` are **ours to build**. Nothing about that contradicts the map — validation
is a pure function over a string, and it runs *before* a `ProteinSequence` is constructed.

Elsewhere biotite covers more than #1 assumed. `FastaFile` keeps the full UniProt header
verbatim and streams Swiss-Prot in under a second. Both `fetch()` functions already implement the
look-local-first rule of #9. `filter_amino_acids` / `filter_nucleotides` make `Chain.kind` a
one-liner. And biotite ships a native 3Di encoder, which foldseek's own database format does not
give us.

Three places have nothing to adopt: **SIFTS** (no maintained package parses the flat file),
**`external.py`** (biotite wraps neither MMseqs2 nor Foldseek, and neither has ever been asked
for), and **`embed/`** (nothing stands between us and the `esm` SDK that still installs).

## Method and provenance

| | |
| --- | --- |
| Executed on | `GPU71FM`, the default pixi environment of `~/pkg/liulab-protein`, biotite **1.4.0**, Python 3.13 |
| Also executed | the `esm` environment, `esm` **3.4.0** — by file-path import, because `import esm` is broken there (already filed under [#5](https://github.com/liuhlab/liulab-protein/issues/5)) |
| Data measured | Swiss-Prot `uniprot_sprot.fasta.gz`, downloaded to GPU71FM 2026-09-03, `Last-Modified: 2026-06-10`, 93,706,469 bytes; PDB entries `4HHB`, `1GP1`, `2OR1`, `1AA6`, `1L2Y` fetched live from RCSB |
| Source read | the installed biotite 1.4.0 tree on GPU71FM, plus `github.com/biotite-dev/biotite` at tag `v1.7.1` |
| Packaging facts | `pypi.org/pypi/<name>/json` and `api.anaconda.org/package/<channel>/<name>`, queried 2026-09-03 |

Every claim below is marked **VERIFIED** (measured by running it) or **INFERRED** (read from
source or documentation without running it). Where the two disagree, the measurement wins.

**GPU71FM reached every network resource this note needed** — UniProt's FTP, RCSB's REST
download service, and EBI's SIFTS flat-file directory. No host but GPU71FM was used.

## Coverage table

Module names follow #1's layout, amended by [#9](https://github.com/liuhlab/liulab-protein/issues/9)
(`StructureMixin` dissolves; `Structure` and `Chain` become peers of `Protein`).

| Module | What already exists | Verdict | Why |
| --- | --- | --- | --- |
| **`seq.py`** — alphabet | `ProteinSequence.alphabet`, a 24-symbol `LetterAlphabet` | **Build ours** | No `U`, `O` or `J`. Covers `B`, `Z`, `X` and `*` only. Unchanged since 2017 and unchanged in 1.7.1. |
| **`seq.py`** — validation | `AlphabetError` | **Build ours** | Message names the *first* offender only, no position, no attributes. #1's `InvalidResidueError` with `.offenders` has no equivalent. A full scan is a one-line comprehension over `alphabet.get_symbols()`. |
| **`seq.py`** — the type `Protein.sequence` holds | `ProteinSequence` | **Hold it, and decide #8 first** | It cannot represent 285 Swiss-Prot entries. Extending it needs a subclass *and* a `get_alphabet()` override, and breaks `get_molecular_weight` and every bundled substitution matrix. |
| **`io/fasta.py`** — read/write, headers | `FastaFile`, `read_iter`, `write_iter` | **Adopt** | Header survives byte-for-byte, `OS=`/`OX=`/`GN=`/`PE=`/`SV=` included. 575,503 entries in 0.9 s at 0.05 GB peak RSS. |
| **`io/fasta.py`** — string to sequence | `fasta.get_sequence` / `get_sequences` | **Build ours** | Tries `NucleotideSequence` first, so `"MKTU"` comes back as *DNA*. Then `U`→`C` with a warning, `O`→`K` with none, `J` → `ValueError`. |
| **`io/fasta.py`** — gzip | nothing | **Build ours, three lines** | `TextFile.read` has no gzip and no extension sniffing; a `.gz` path raises `UnicodeDecodeError`. `gzip.open(p, "rt")` passed as a file object works. |
| **`store.py`** — fetch on miss | `rcsb.fetch`, `uniprot.fetch`, `afdb.fetch` | **Adopt the fetch; build the root** | Each already skips the download when the target file exists and is non-empty — exactly #9's rule. None owns a cache root; `target_path` is the caller's. |
| **`store.py`** — the data root | `genome.store.data_dir` / `completion` | **Adopt** (sibling package) | Settled by #1. Nothing third-party involved. |
| **`db/`** — Swiss-Prot retrieval | nothing in biotite | **Build ours** | Settled by [#3](https://github.com/liuhlab/liulab-protein/issues/3): `mmseqs view --id-list --id-mode 1`, or a direct ffindex read. |
| **`db/`** — PDB | `pdbx.CIFFile` + `pdbx.get_structure` | **Adopt** | Settled by #9. `Structure` holds a path and parses lazily. |
| **`Chain.kind`** | `filter_amino_acids`, `filter_nucleotides` | **Adopt** | One call each, both CCD-backed, both return atom masks. Correct on a protein-DNA complex. |
| **`Chain.atoms`** | `get_chains`, `chain_iter` | **Build a thin accessor over `chain_id`** | Both are per chain **segment**, not per chain id: `get_chains(4HHB)` returns twelve entries for four chains. `atoms[atoms.chain_id == cid]` is what a caller means. |
| **`Chain.sequence`** | `to_sequence` | **Build ours** | Raises `BadStructureError` on every entry tested, even with `allow_hetero=True`, because waters get their own chain segment. And it silently maps `U`→`C`, `O`→`K`. |
| **SIFTS ([#20](https://github.com/liuhlab/liulab-protein/issues/20))** | nothing maintained, anywhere | **Build ours** | No package on PyPI, conda-forge or bioconda parses `pdb_chain_uniprot.tsv`. The file is 6,211,584 bytes and GPU71FM reaches it. |
| **`external.py`** | `biotite.application` — 9 tools, none of them ours | **Build ours** (port from `liulab-genome`) | No MMseqs2 wrapper, no Foldseek wrapper, and `mmseqs` returns zero hits across biotite's whole tracker. `pymmseqs` covers half the requirement and vendors a second `mmseqs` binary. |
| **`search/`** — 3Di | `biotite.structure.alphabet.to_3di` | **Note it, do not use it in v1** | A native 3Di encoder, no Foldseek needed. It encodes; it does not search. Relevant to #1's deferred ProstT5 lane. |
| **`embed/`** | `esm` 3.4.0 direct | **Build ours, thin** | `bio-embeddings` is dead and pins `<3.10`; `fair-esm` is archived and predates ESM-C. `transformers` is a dependency of `esm`, not an alternative to it. |
| **`core.py`, `cli.py`** | — | **Build ours** | This package's vocabulary. Typer and `liulab-genome`'s conventions, as #1 specifies. |

## Candidate table

Queried 2026-09-03. The repo is `platforms = ["linux-64"]`, Python 3.13, pixi-only, channels
`conda-forge` then `bioconda` — so the last two columns are what decides whether a row is even
available to us. **VERIFIED** against the PyPI and anaconda.org APIs.

| Package | Latest | Date | Python 3.13 | conda-forge linux-64 | Licence | Relevant to |
| --- | --- | --- | --- | --- | --- | --- |
| **biotite** | **1.7.1** | 2026-06-22 | yes, cp313 wheels; `requires-python >=3.12` | yes — `py313h0aaa388_0` | BSD-3-Clause | everything. **We run 1.4.0**; see below |
| gemmi | 0.7.5 | 2026-03-02 | yes, cp313 wheels | yes — `py313hbaa079c_0` | MPL-2.0 | structure I/O, partial SIFTS |
| biopython | 1.88 | 2026-08-06 | yes, cp313 wheels + classifier | yes — `py313h07c4f96_0` | Biopython Licence Agreement | FASTA fallback |
| MDAnalysis | 2.10.0 | 2025-10-17 | yes, cp313 wheels + classifier | yes — `py313h08cd8bf_0` | LGPL-3.0-or-later | none found |
| **ProDy** | 2.6.1 | 2025-08-19 | **no** — cp312 wheel only | **no py313 build at any version**; no osx-arm64 ever | MIT | ruled out by the toolchain |
| pyfastx | 2.3.1 | 2026-06-10 | yes, cp313 wheels + classifier | **bioconda only** — `py313hfeada96_0` | MIT | large-FASTA random access |
| pymmseqs | 1.2.0 | 2026-08-11 | yes, classifier through 3.14 | **bioconda only**, and `noarch` | MIT | `external.py` — see the caveat |
| esm | 3.4.0 | 2026-08-27 | yes; `requires-python >=3.12` | **neither channel** — PyPI only | MIT (Chan Zuckerberg Biohub) | `embed/` — already our dependency |
| transformers | 5.16.1 | 2026-08-26 | yes, pure Python | yes, `noarch` | Apache-2.0 | pulled in by `esm` |
| fair-esm | 2.0.0 | 2022-11-01 | untested; upstream **archived** 2023 | yes, `noarch` | MIT | superseded by `esm` |
| bio-embeddings | 0.2.2 | 2021-09-06 | **no** — pins `<3.10` | `noarch`, but the recipe dropped the ceiling | MIT | ruled out |
| atomium | 1.0.11 | 2021-11-28 | untested; repo idle since 2022 | yes, `noarch` | MIT | none found |
| PDBeCif | 1.5 | 2021-03-10 | untested; repo idle since 2021 | **bioconda only**, `noarch` | disputed — PyPI says GPLv3, repo says Apache-2.0 | none found |
| pdbe-sifts | 1.0 | 2026-04-30 | classifiers stop at 3.12 | **neither channel** | Apache-2.0 | SIFTS — but see below |

Three rows are worth reading twice.

**ProDy cannot be installed here.** No cp313 wheel on PyPI and no `py313` build on conda-forge at
any version; the "Rebuild for python 3.13" feedstock pull request has been open since 2025-08-07.
Do not propose it for any module. **VERIFIED** against the anaconda.org file listing.

**pyfastx and pymmseqs are bioconda-only**, which is fine — this repo already takes `mmseqs2` and
`foldseek` from bioconda — but neither has a conda-forge feedstock, so neither gets that channel's
rebuild guarantees.

**We run biotite 1.4.0 and the newest is 1.7.1 — and that is not drift.** See
[Why the lock holds biotite at 1.4.0](#why-the-lock-holds-biotite-at-140): a conda-forge numpy
cap forces it. Either way **the alphabet is identical in 1.7.1** (**VERIFIED** by reading
`seqtypes.py` at tag `v1.7.1`), so no upgrade rescues `U`, `O` or `J`.

### Why the lock holds biotite at 1.4.0

Not a pin and not an oversight. `pyproject.toml` asks for `biotite = ">=1"` and `numpy = ">=2"`,
both open. The conflict is downstream:

| | numpy constraint in the conda package | Built |
| --- | --- | --- |
| biotite 1.4.0 | `numpy >=1.23,<3` | 2025-09-22 |
| biotite 1.5.0 – **1.7.1** | **`numpy >=1.26,<2.4`** | 2026-01-03 – 2026-06-24 |

**VERIFIED** against the anaconda.org file listing. numpy 2.4 reached conda-forge 2026-05-16 and
2.5.0 on 2026-06-22; biotite 1.7.1 was built two days later against the 2026-06-19 conda-forge
pinning, which still pinned numpy 2.3 — so its run export caps at `<2.4`. No biotite has been
rebuilt since, and the feedstock has **no open pull request**. With `numpy = ">=2"` the solver
takes numpy 2.5.2, and 1.4.0 is then the newest biotite that fits.

The cap is **conda-forge's, not biotite's**: upstream declares only `numpy>=1.25`, in 1.4.0 and
1.7.1 alike — **VERIFIED** against the PyPI metadata for both.

The upgrade does solve, at a price. Re-locking the real manifest with `biotite = ">=1.7"` in a
throwaway worktree on GPU71FM gives biotite **1.7.1** and **numpy 2.3.5** — **VERIFIED**. Nothing
else moves: pandas 2.3.3, pyarrow 25.0.0, scipy 1.18.0, `mmseqs2` and `foldseek` are unchanged.

So it is a straight trade — newest biotite **or** newest numpy, not both — and nothing in this
note needs either. Worth revisiting when the feedstock rebuilds; not worth forcing now.

---

## 1. The alphabet

### What is in it

**VERIFIED** on GPU71FM, biotite 1.4.0:

```pycon
>>> biotite.sequence.ProteinSequence.alphabet.get_symbols()
'ACDEFGHIKLMNPQRSTVWYBZX*'   # 24 symbols
```

| Symbol | In the alphabet? |
| --- | --- |
| the 20 standard | yes |
| `B` (Asx), `Z` (Glx), `X` (any) | **yes** |
| `*` (stop) | **yes** |
| `U` (selenocysteine) | **no** |
| `O` (pyrrolysine) | **no** |
| `J` (Leu/Ile) | **no** |
| `-` (gap) | no |

So #1's `AMBIGUOUS = X B Z J U O` overlaps biotite's alphabet on three of six. It also assumes
`*` is rejected; biotite accepts it. And biotite silently uppercases, so `ProteinSequence("mkt")`
returns `MKT` — **VERIFIED**.

The class docstring says so outright, and says the same in 1.7.1:

> The `Alphabet` of this `Sequence` class does not support selenocysteine. Please convert
> selenocysteine (`U`) into cysteine (`C`) or use a custom `Sequence` class, if the
> differentiation is necessary.

Pyrrolysine is not mentioned anywhere in the class. The `O`→`K` rule was added as an unannounced
rider on the selenocysteine work (biotite [PR #246](https://github.com/biotite-dev/biotite/pull/246),
merged 2020-11-10) — **INFERRED** from the tracker.

The maintainer's reason, on [issue #232](https://github.com/biotite-dev/biotite/issues/232):

> One problem is that Selenocysteine (`U`) is currently not recognized by the amino acid
> alphabet. **To fix this, `U` needs also to be added to substitution matrices.**

and, closing it:

> If selenocysteine is explicitly required, a custom `Sequence` class needs to be created by the
> user.

That is a deliberate, standing position, not an oversight. Searching biotite's tracker for
`pyrrolysine` returns **zero** hits — **INFERRED**.

### What construction does with a symbol outside it

**VERIFIED**:

```pycon
>>> ProteinSequence("MKTU")
biotite.sequence.AlphabetError: Symbol 'U' is not in the alphabet
>>> ProteinSequence("M-KT1 x?")
biotite.sequence.AlphabetError: Symbol '-' is not in the alphabet
```

| Question #1 asks | Answer |
| --- | --- |
| Which exception? | `AlphabetError`, which subclasses `Exception` — **not** `ValueError` |
| Does it carry the offending symbols? | Only in the message string, and only the **first** one |
| Does it carry their positions? | **No** |
| Any attributes to read? | **No.** `e.args` is a 1-tuple holding that one string |

So #1's `InvalidResidueError(ValueError)` carrying `.offenders` and `.name` has no counterpart, and
a caller repairing input **would** have to parse a message — which is precisely what #1 says it
must not do. `seq.py` builds its own. The full scan biotite will not give you is one line:

```python
SYMBOLS = frozenset(ProteinSequence.alphabet.get_symbols())
[(i, c) for i, c in enumerate(seq.upper()) if c not in SYMBOLS]
# 'M-KT1 x?U' -> [(1, '-'), (4, '1'), (5, ' '), (7, '?'), (8, 'U')]
```

**VERIFIED.** Note that `seq.py` will use its own 26-letter set here, not biotite's 24.

### Can the alphabet be extended?

Two paths, both measured, both with a cost.

**Path A — reassign the class attribute.** `ProteinSequence.alphabet = LetterAlphabet(...)`
does let construction accept `U`, `O` and `J`. It also breaks the library for the whole process:

| After the reassignment | Result |
| --- | --- |
| `ProteinSequence("MKTUOJBZX*")` | **works** — codes `[10 8 16 24 25 26 20 21 22 23]` |
| `get_molecular_weight()` | `IndexError: index 24 is out of bounds for axis 0 with size 24` |
| `align_optimal(..., SubstitutionMatrix.std_protein_matrix())` | `ValueError: The sequences' alphabets do not fit the matrix` |

**VERIFIED.** It is a monkeypatch of a third-party global. Do not.

**Path B — subclass.** Setting `alphabet` on a subclass is **not enough**: `ProteinSequence`
hardcodes

```python
def get_alphabet(self):
    return ProteinSequence.alphabet
```

so the subclass must override `get_alphabet()` too. With both, it works — **VERIFIED**:

| Subclass with `alphabet` + `get_alphabet()` | Result |
| --- | --- |
| `Ext("MKTUOJ")` | **works**, and `isinstance(x, ProteinSequence)` is `True` |
| `remove_stops()` | works |
| `get_molecular_weight()` | `IndexError` — `_mol_weight_average` is a 24-entry array |
| any bundled substitution matrix | `ValueError` |
| `convert_letter_1to3("U")` | `KeyError` — `_dict_1to3` has 24 entries |

The two breakages are the maintainer's stated reason for the 24-letter alphabet, so they are not
bugs to route around. Whether they matter here is a scoping question for #8: v1 aligns with
MMseqs2, not with `biotite.sequence.align`, and does not weigh proteins.

Note also that the extended alphabet passes `EXT.extends(ProteinSequence.alphabet)` — biotite's
own `AlphabetMapper` would recode between them — **VERIFIED**.

### The dangerous part: silent repair, at three boundaries

The alphabet failing loudly is survivable. What is not is that biotite's convenience layers
rewrite the input rather than fail. **VERIFIED**, all of it:

```python
# biotite/sequence/io/fasta/convert.py
def _process_protein_sequence(x):
    """Replace selenocysteine with cysteine and pyrrolysine with lysine."""
    return x.upper().replace("U", "C").replace("O", "K")
```

```python
# biotite/structure/sequence.py, inside to_sequence()
one_letter_symbols[one_letter_symbols == "U"] = "C"
one_letter_symbols[one_letter_symbols == "O"] = "K"
```

| Boundary | `U` | `O` | `J` |
| --- | --- | --- | --- |
| `ProteinSequence(...)` directly | `AlphabetError` | `AlphabetError` | `AlphabetError` |
| `fasta.get_sequence(..., seq_type=ProteinSequence)` | → `C`, **`UserWarning`** | → `K`, **no signal at all** | `AlphabetError` |
| `fasta.get_sequence(...)` with no `seq_type` | see §2 — worse | → `K`, silent | `ValueError` |
| `structure.to_sequence(...)` | → `C`, **no signal at all** | → `K`, **no signal at all** | n/a |

Measured end to end on a real entry: **PDB `1AA6`** (formate dehydrogenase H) carries a `SEC`
residue at chain-A index 139. `to_sequence` returns `'C'` there and emits nothing —
**VERIFIED**. biotite *knows* the right letter: `biotite.structure.info.one_letter_code("SEC")`
returns `'U'` and `("PYL")` returns `'O'` — **VERIFIED**. The alphabet is the bottleneck, not the
data.

The repo sets `filterwarnings = ["error"]`, so the one warning that does fire becomes a test
failure rather than a silent corruption. That is the gate working, but it only covers `U`
through the FASTA path. **`O` and the whole structure path are silent.**

### How much does this cost, in entries?

Counted over the full Swiss-Prot release of 2026-06-10 — 575,503 entries, 208,906,902 residues.
**VERIFIED** on GPU71FM.

| Symbol | Entries containing it | Share | Residue occurrences |
| --- | --- | --- | --- |
| `X` | 2,286 | 0.397% | 8,183 |
| **`U`** | **256** | **0.0445%** | 331 |
| `B` | 113 | 0.0196% | 276 |
| `Z` | 87 | 0.0151% | 249 |
| **`O`** | **29** | **0.0050%** | 29 |
| **`J`** | **0** | **0%** | **0** |

The set of distinct symbols across all of Swiss-Prot is `ABCDEFGHIKLMNOPQRSTUVWXYZ` — every
letter **except `J`**.

So: 285 entries (0.0495%) cannot round-trip through biotite, and `J` has no support in Swiss-Prot
at all.

### And what the ESM-C tokenizer actually accepts

Issue #1 justifies all six symbols with "UniProt and the ESM tokenizers accept them". Half of
that is wrong. `EsmcTokenizer`'s vocabulary is 33 tokens — **VERIFIED** by loading
`esm/models/esmc/tokenizer.py` by file path in the `esm` environment (`import esm` itself is
broken there, filed under #5):

| Symbol | ESM-C token id |
| --- | --- |
| `X` | 24 |
| `B` | 25 |
| **`U`** | **26** |
| `Z` | 27 |
| **`O`** | **28** |
| `.` | 29 |
| `-` | 30 |
| `\|` | 31 |
| **`J`** | **absent → `<unk>` (3)** |
| `*` | **absent → `<unk>` (3)** |

So ESM-C carries five of #1's six. **`J` is not in the vocabulary and never reaches the model as
itself**, and neither does `*`. Meanwhile `-` and `.` *are* — the tokenizer would happily eat a
gapped alignment row. #1's rationale for rejecting gaps ("a stray `*` or `-` reaching a tokenizer
fails far from its cause") is therefore about our policy, not about the tokenizer failing.

`J` has no evidence behind it from either source #1 cites: zero occurrences in Swiss-Prot, and
`<unk>` at ESM-C. It is still correct IUPAC and still what makes `ALPHABET` every ASCII letter,
which is the property #1 leans on. That is a call for #8, not for this note.

### What #8 has to decide

The finding, stated as a constraint rather than a design:

- `seq.py`'s alphabet, `outside_alphabet`, `offending_positions` and `InvalidResidueError` are
  **ours**, with no adoptable part. Nothing in the map conflicts — these are pure functions over
  a `str`.
- Whether `Protein.sequence` can stay a plain `ProteinSequence` depends on whether 285 Swiss-Prot
  entries may lose a residue. If they may not, the choices are a `ProteinSequence` subclass with
  an overridden `get_alphabet()` (giving up molecular weight and biotite's aligners), or holding
  the raw `str` and building a `ProteinSequence` only where biotite is actually called.
- Either way, **never call `fasta.get_sequence` or `structure.to_sequence` on data whose fidelity
  matters** without knowing they rewrite it.
- [#3](https://github.com/liuhlab/liulab-protein/issues/3) already found that a
  `mmseqs databases` Swiss-Prot maps `U`,`O`→`X`, `B`→`D`, `Z`→`E`, `J`→`L`. So a `Protein` read
  from that database is *already* lossy before biotite sees it. The two losses are independent
  and they are not the same loss.

## 2. `io/fasta.py`

### Headers survive verbatim — adopt

**VERIFIED.** `FastaFile` is a `MutableMapping` whose keys are the header lines minus the leading
`>`, with nothing else stripped:

```pycon
>>> list(f.keys())[0]
'sp|P12345|AATM_RABIT Aspartate aminotransferase, mitochondrial OS=Oryctolagus cuniculus OX=9986 GN=GOT2 PE=1 SV=2'
```

Pipes, spaces, commas, `OS=`, `OX=`, `GN=`, `PE=`, `SV=` — all present, byte for byte. This is
what [#3](https://github.com/liuhlab/liulab-protein/issues/3) needs for `Protein.metadata`, and
it comes free. biotite does **no** header parsing at all; splitting the fields is ours.

Write round-trips it too: `out[HDR] = "MKTAYIAKQRQ"` then `.write(path)` produces the identical
header line. Wrapping is `chars_per_line=80` by default — **VERIFIED**.

One trap: `_entries` is keyed by header, so **two entries with the same header silently
collide**, and `len(file)` under-counts. biotite's own `write_iter` docstring admits it does not
check for ambiguity — **INFERRED** from source.

### Type conversion — build ours

`fasta.get_sequence` tries `NucleotideSequence` **before** `ProteinSequence`. On a peptide the
result is not a warning, it is a wrong type with wrong letters. **VERIFIED**:

| FASTA content | `get_sequence()` returns | Warning |
| --- | --- | --- |
| `MKTU` | **`NucleotideSequence("MKTT")`** | none |
| `AGCTAGCT` | **`NucleotideSequence("AGCTAGCT")`** | none |
| `MKTO` | `ProteinSequence("MKTK")` | none |
| `MKTBZX` | `ProteinSequence("MKTBZX")` | none |
| `MKT*` | `ProteinSequence("MKT*")` | none |
| `MKTJ` | `ValueError: FASTA data cannot be converted either to 'NucleotideSequence' nor to 'ProteinSequence'` | — |
| `MKT-` | same `ValueError` | — |

`MKTU` becoming DNA is the sharp one: `M`, `K` and `T` are all IUPAC *nucleotide* ambiguity codes,
`U`→`T` is the nucleotide preprocessor, and the guess succeeds. A selenoprotein peptide comes back
as a DNA sequence with nothing said. Passing `seq_type=ProteinSequence` defuses the misdetection
but re-arms the `U`→`C` rewrite. Neither branch is safe for us; `io.py` reads the raw string and
converts under our own rules.

### gzip — build ours, three lines

**VERIFIED**:

| Call | Result |
| --- | --- |
| `FastaFile.read("x.fasta.gz")` | `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b` |
| `FastaFile.read(gzip.open(p, "rt"))` | **works** |

`TextFile.read` calls builtin `open()` on any path-like, with no decompression and no extension
sniffing; a text-mode file object passes its `is_text` check. `gzip.open(p, "rb")` does not.

### Does it scale? Yes — measured

Full Swiss-Prot, 288,037,659 bytes uncompressed, on GPU71FM — **VERIFIED**:

| Call | Entries | Time | Peak RSS |
| --- | --- | --- | --- |
| `FastaFile.read_iter(path)` | 575,503 | **0.9 s** | **0.05 GB** |
| `FastaFile.read(path)` | 575,503 | 1.8 s | 0.87 GB |
| `read_iter` over `gzip.open(p, "rt")` | 575,503 | 2.4 s | — |

`read_iter` yields raw `(header, sequence_string)` tuples and **bypasses `convert.py` entirely**,
so it does no type guessing and no `U`→`C` rewriting. It is both the fast path and the safe one.

### Where `pyfastx` would come in — and why not yet

`pyfastx` 2.3.1 (2026-06-10, MIT, bioconda linux-64 `py313`) gives what biotite does not: random
access by index or name into a gzipped FASTA, via a persistent `.fxi` index and bgzf. That is a
real gap, but not one v1 has. #3 settled that `swissprot["P12345"]` resolves through
`mmseqs view` / the ffindex `.lookup`, not through a FASTA; and #1's FASTA surface is
`Protein.from_fasta` / `to_fasta` over user files. **Verdict: adopt biotite, and name `pyfastx`
as the answer if a large-FASTA random-access requirement ever appears.** Taking it now buys
nothing and costs a bioconda-only pin.

## 3. Fetching

### Both `fetch()` functions already do the local-first rule

**VERIFIED** by execution, with the repo's own network guard installed
(`requests.sessions.Session.request` and `socket.socket.connect` patched, as `tests/_guards.py`
does):

| Call | Result |
| --- | --- |
| `rcsb.fetch("1L2Y", "cif", d)` with nothing on disk | `RuntimeError: BLOCKED BY GUARD` — it *did* try the network |
| the same call with a non-empty `1L2Y.cif` already in `d` | returns the path, **no network touched** |
| the same with a **zero-byte** `9XXX.cif` in `d` | `BLOCKED BY GUARD` — it re-fetches |

The single line that does it, identical in both modules:

```python
if file is None or not isfile(file) or getsize(file) == 0 or overwrite:
```

So #9's "look local first, fetch on miss" is already implemented, per ID, and the zero-byte
re-fetch guards a truncated previous run. What biotite does **not** own is the "cache in one
place" half: `target_path` is the caller's argument and there is no package-level root. That
stays `store.py`'s job — pass `protein_data_dir() / ...` and biotite does the rest.

Two more things worth knowing:

- **The network guard catches biotite.** Both fetches go through `requests.get`, which routes
  through `Session.request`. So a test that accidentally reaches RCSB fails loudly rather than
  hanging — **VERIFIED**.
- `format="cif"` and `format="mmcif"` are synonyms upstream but write **different filenames**, so
  the cache does not dedupe them. Pick one spelling and keep it — **INFERRED** from source.

### Formats

| Function | Accepted `format=` |
| --- | --- |
| `rcsb.fetch(pdb_ids, format, target_path=None, overwrite=False, verbose=False)` | `pdb`, `pdbx`, `cif`, `mmcif`, `bcif`, `fasta` |
| `uniprot.fetch(ids, format, target_path=None, overwrite=False, verbose=False)` | `fasta`, `gff`, `txt`, `xml`, `rdf`, `tab` |

`rcsb` also exposes the RCSB search API — `search`, `count`, and the `Query` hierarchy
(`BasicQuery`, `FieldQuery`, `SequenceQuery`, `StructureQuery`, `MotifQuery`, plus `Sorting` and
the `Grouping` classes). `uniprot` exposes `search` and `SimpleQuery` but **no `count`** and no
pagination — its `number` argument defaults to 500 and is passed straight through.
`biotite.database.afdb` exists too, for AlphaFold models. **VERIFIED** by listing the modules;
signatures **VERIFIED** by `inspect.signature`.

### The constraint the map puts on all of this

Both are **per-ID network calls**, and the map's Notes say "bulk, not per-ID", with
`tests/_guards.py` making a per-ID call untestable by construction. That does not rule them out:
issue #9 chose "local cache first, then RCSB on demand" for coordinates, which is exactly
one `rcsb.fetch` per cache miss. It does rule them out as a way to *populate* anything.
`uniprot.fetch` in particular has no bulk mode at all; Swiss-Prot arrives as a database, not as
575,503 REST calls.

## 4. `Chain`

How much of #9's `Chain` is a one-line biotite call? **`.kind` yes; `.atoms` and `.sequence` no.**
Measured on `4HHB` (four protein chains, four hemes, waters), `1GP1` (selenoprotein), `2OR1`
(two DNA chains plus two protein chains).

### `.kind` — adopt

```python
struc.filter_amino_acids(atoms)   # ndarray[bool] over atoms
struc.filter_nucleotides(atoms)
```

**VERIFIED** on `2OR1`: chains `A` and `B` give `nuc=405`, `nuc=409` and `aa=0`; chains `L` and
`R` give `aa=484` each and `nuc=0`. Both are backed by the PDB Chemical Component Dictionary, so
modified residues (`MSE`, `SEP`, `TPO`) pass `filter_amino_acids`. `filter_canonical_amino_acids`
uses a hardcoded 22-name list — the 20 plus `SEC` and `PYL`. Note the asymmetry that creates: a
`SEC` residue passes the *canonical* filter and is then flattened to `C` by `to_sequence`.

### `.atoms` — `get_chains` and `chain_iter` are not what a caller means

**VERIFIED**:

```pycon
>>> struc.get_chains(atoms_4hhb)
array(['A','B','C','D','A','B','C','D','A','B','C','D'], dtype='<U4')
>>> struc.get_chain_count(atoms_4hhb)
12
```

Twelve, for an entry with four chains. `get_chains` is literally
`array.chain_id[get_chain_starts(array)]` — chain ids at every **segment** boundary, not distinct
ids. 4HHB's protein, heme and water records each open a new segment per chain letter.
`get_chain_starts` also splits when `res_id` *decreases*, so numbering restarts split a chain too.

`chain_iter` yields those same twelve slices. So neither is the enumeration `Structure` wants.
The one-liner that is:

```python
ids   = list(dict.fromkeys(struc.get_chains(atoms)))   # order-preserving unique
chain = atoms[atoms.chain_id == cid]
```

Cheap, but ours to write, and worth a comment saying why `chain_iter` is not used.

### `.sequence` — `to_sequence` fails on real entries

**VERIFIED**, and this is the surprise:

| Call | Result |
| --- | --- |
| `to_sequence(atoms_4hhb)` | `BadStructureError: Chain A contains neither amino acids nor nucleotides` |
| `to_sequence(atoms_4hhb, allow_hetero=True)` | **the same error** |
| `to_sequence(atoms_1gp1, allow_hetero=True)` | the same |
| `to_sequence(atoms_2or1, allow_hetero=True)` | the same |

`allow_hetero` does not help, because the failure is not about hetero residues: the water block
is its own chain segment, it contains no amino acids and no nucleotides, and that is a hard
error. Filter first and it works:

```python
poly = atoms[struc.filter_amino_acids(atoms) | struc.filter_nucleotides(atoms)]
seqs, chain_starts = struc.to_sequence(poly, allow_hetero=True)
```

**VERIFIED** — 4HHB then gives four `ProteinSequence`s of 141/146/141/146, and 2OR1 gives two
`NucleotideSequence`s of 20 and two `ProteinSequence`s of 63, correctly typed. Note that
`to_sequence` returns a **tuple** `(sequences, chain_start_indices)`, that peptide-vs-nucleic is
decided by majority vote over residue names with ties going to nucleic, and that it applies the
silent `U`→`C` / `O`→`K` rewrite from §1.

**Verdict: `Chain.sequence` is ours.** It is three lines over `to_sequence`, but the three lines
are load-bearing and the residue-level fidelity question rides on them.

## 5. SIFTS

**Nothing maintained parses `pdb_chain_uniprot.tsv`. #20 is ours to build.**

- **biotite has no SIFTS support of any kind** — `grep -ril sifts` over the installed 1.4.0 tree
  returns nothing. **VERIFIED** on GPU71FM.
- **No conda package named `sifts` exists on any channel.** `api.anaconda.org/search?name=sifts`
  returns literally `[]`, as do `proteofav`, `pdb-profiling`, `prointvar` and `pdbe-sifts`. The
  endpoint works — `?name=gemmi` returns two results. **VERIFIED**.
- The PyPI package named **`sifts` is not structural biology** — it is "Simple full-text search
  library with SQL backend". **VERIFIED**.

The near misses, and why each is one:

| Candidate | Status | Reads the TSV? |
| --- | --- | --- |
| **ProteoFAV** | last code commit 2018-02-21 | No — parses per-entry SIFTS **XML** with `lxml` |
| **pdb-profiling** | last commit 2023-03-30 | No — calls the PDBe REST API |
| **ProIntVar** | not on PyPI; repo idle | No |
| **SIFTSParse** | GitHub only, pushed 2016 | Yes — and abandoned ten years ago |
| **gemmi** | active | No TSV. It reads the `_pdbx_sifts*` mmCIF categories from PDBe "updated" files. Its own docs: *"Gemmi has limited support for both DBREF and SIFTS annotations. The API is undocumented yet and may change."* |
| **Biopython** | active | No SIFTS module in `Bio/PDB/` |
| **ProDy** | active | No SIFTS anywhere — and unusable here anyway (no Python 3.13) |
| **pdbe-sifts 1.0** | the official EMBL-EBI package, 2026-04-30 | No — it *runs the SIFTS pipeline locally* (MMseqs2 + FASTA36 into DuckDB) to **produce** mappings. Not on conda-forge or bioconda; classifiers stop at 3.12; deps include `ete4` and `scikit-learn` |

Everything else that touches the file is an in-repo script (EVcouplings, CosMIS, pypath), not a
library.

That is the right answer for us anyway. **VERIFIED from GPU71FM:**
`https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz` returns
HTTP 200, `Content-Length: 6211584`, `Last-Modified: 2026-08-30`. Six megabytes of gzipped TSV,
one bulk download, parsed with `pandas.read_csv` — exactly the map's "bulk, not per-ID". Adopting
a dependency to read a four-column TSV would be the customization the restraint rule warns about.

## 6. `external.py` and the search lanes

### biotite wraps neither tool, and never will by accident

`biotite.application` in 1.4.0 wraps exactly nine tools — **VERIFIED** by listing the package:

`autodock`, `blast`, `clustalo`, `dssp`, `mafft`, `muscle`, `sra`, `tantan`, `viennarna`, plus
the `Application` / `LocalApp` / `WebApp` / `MSAApp` base classes.

**No MMseqs2. No Foldseek.** The set is identical at tag `v1.7.1`. Searching biotite's tracker for
`mmseqs` returns **zero** hits — it has never been requested. `foldseek` returns two, both about
something else: biotite **reimplemented the 3Di alphabet natively** rather than shelling out
(PR #665, shipped in v1.1.0 as `biotite.structure.alphabet.to_3di`). **INFERRED** from the
tracker; the encoder itself is **VERIFIED** — `to_3di` on 1L2Y returns
`I3DSequence('dqqvvcvvcpnvvnvdhgdd')`.

Worth recording for the deferred ProstT5 lane: we can get a 3Di string from coordinates with no
subprocess and no extra dependency.

### Is any wrapper worth taking over #1's own seam? No

The only live candidate is **`pymmseqs`** — 1.2.0, 2026-08-11, MIT, 47 stars, PyPI classifiers
through 3.14, bioconda `noarch`. Four reasons it does not displace `external.py`:

1. **It covers half the requirement.** `REQUIRED_TOOLS = ("mmseqs", "foldseek")`. There is **no
   Foldseek wrapper on PyPI at all** — `foldseek`, `pyfoldseek` and `foldseek-python` are all
   404, and anaconda.org knows only the bioconda binary. **VERIFIED.** Adopting `pymmseqs` would
   split the subprocess boundary in two, which is the one thing #1's seam exists to prevent.
2. **Its PyPI wheels vendor their own `mmseqs` binary** — `pymmseqs/bin/mmseqs`, 18 MB unpacked.
   **VERIFIED** by unzipping the manylinux wheel. This repo already pins bioconda `mmseqs2`
   `18-8cc5c`; adopting it means two binaries at possibly different versions.
3. **Its bioconda build is `noarch` and declares no `mmseqs2` dependency** — deps are `numpy`,
   `pandas`, `python >=3.10`, `pyyaml`. **VERIFIED** against the anaconda.org file list. So on
   conda it neither ships nor requires the binary it wraps.
4. **The seam is not the wrapping.** `external.py` carries `RecordingTool`, the `run_calls`
   fixture, the make-style freshness rule in `run_to`, the path-keyed version cache and
   `doctor()`. #1 is explicit that `run_to` must call `self.run` so one
   `monkeypatch.setattr(ExternalTool, "run", ...)` catches every invocation. Every test in the
   package rests on that. No third-party wrapper offers it, and adopting one would mean building
   it again on the outside.

Add that Foldseek vendors MMseqs2 — same CLI grammar, same database format, same
`--format-output` — which is what makes #1's `MmseqsLikeTool` base worth having and what a
single-tool wrapper cannot express.

**Verdict: port the seam from `liulab-genome`, exactly as #1 says.** `pymmseqs` is worth
remembering only if someone later wants MMseqs2 in-process rather than as a subprocess, and that
would be a different design.

## 7. `embed/`

**Nothing stands between us and the `esm` SDK.**

| Candidate | Verdict |
| --- | --- |
| **`bio-embeddings` 0.2.2** | Dead. Released 2021-09-06, repo idle since 2022-08-04, and it pins `>=3.7.1,<3.10` — it **cannot install on Python 3.13**. The conda recipe quietly dropped the ceiling, which makes the conda package worse than the wheel, not better. |
| **`fair-esm` 2.0.0** | Released 2022-11-01. Upstream `facebookresearch/esm` is **archived**, last commit 2023-06-27. Predates ESM-C entirely. |
| **`transformers` 5.16.1** | Not an alternative — it is what `esm` 3.4.0 is *built on*. `EsmcForMaskedLM` and `EsmcTokenizer` subclass `transformers` types. Adopting it is a consequence of adopting `esm`, not a choice. |
| **`esm` 3.4.0** | PyPI only — **on neither conda-forge nor bioconda**, so it stays a PyPI dependency in the `esm` feature, as the lock already has it. MIT, Chan Zuckerberg Biohub, `requires-python >=3.12`. |

So `embed/` is a thin lane over `esm` as #1 specifies. The two things v1 must write itself —
stripping BOS/EOS and the masked-mean pooling — have no library to take them from; #1 already
says there is no local pooling helper.

One thing this note adds to that lane: the tokenizer's vocabulary is fixed and small (33 tokens,
§1), so `embed()` should validate against **our** alphabet before tokenizing. `J` and `*` become
`<unk>` silently, which is exactly the "fails far from its cause" failure `seq.py` exists to
catch.

## Open items

- **#8 must choose what `Protein.sequence` holds.** This note gives the measurement (285
  Swiss-Prot entries, 0.0495%) and the two mechanisms (subclass with `get_alphabet()`, or hold a
  `str`), and declines to choose.
- **Whether `J` stays in `ALPHABET`.** Zero occurrences in Swiss-Prot, `<unk>` at ESM-C, `L` after
  a `mmseqs databases` build. It survives only on the IUPAC argument and on making `ALPHABET`
  every ASCII letter.
- **Upgrading biotite 1.4.0 → 1.7.1 costs numpy 2.5.2 → 2.3.5**, because every biotite build
  since 1.5.0 carries conda-forge's `numpy <2.4` cap and none has been rebuilt. Nothing here
  needs either side of that trade — the alphabet is identical — so the question is whether v1
  would rather wait for the feedstock or cap numpy itself.
- **`filterwarnings = ["error"]` and biotite.** The `U` conversion in `fasta.convert` raises a
  `UserWarning` that will become a test error. That is the desired behaviour, but it needs a
  targeted entry with a comment rather than the blanket ignore #1 forbids — and it does **not**
  cover `O`, nor the structure path, both of which are silent.
