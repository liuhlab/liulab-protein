---
search:
  exclude: true
---

# What would it take to support multiple sequence alignments in this package?

Two questions, deliberately kept apart: which MSA **file formats** this package would have to read
and write, and how it would **generate** an MSA at all. They have different answers and different
costs, and only one of them is cheap.

## Answer, shortest form

**A3M is the only format that matters here, and biotite cannot supply it.** Every biotite feature
that would make MSA support pleasant landed *after* the version this repo's lock can install:
`get_a3m_alignments` in **1.6.0**, `Alignment.from_strings` in **1.6.0**, Clustal I/O in **1.7.0**.
The repo runs **1.4.0**, where `Alignment` has no string constructor at all and `trace_from_strings`
is a pure-Python double loop over every cell. Buying any of it means taking the `numpy <2.4`
downgrade [the sibling note](biotite-coverage-of-v1.md#why-the-lock-holds-biotite-at-140) already
priced.

**And biotite's alignment readers are converters, so ADR-0002 bans them anyway.**
`fasta.get_alignment`, `fasta.get_a3m_alignments` and `clustal.get_alignment` all funnel through
`ProteinSequence(s.upper().replace("U", "C").replace("O", "K"))` — the same silent rewrite the ADR
was written about. **The guard test does not name them**: `_BANNED_CONVERTERS` is
`{get_sequence, get_sequences, to_sequence, convert_letter_3to1}`, so adopting `get_a3m_alignments`
today would pass `pixi run check` while corrupting selenoproteins. The hole is exactly where MSA
support would land.

**There is no `mmseqs easy-msa`.** Generation is three verbs — a search with `-a`, then
`result2msa --msa-format-mode 5`, then `unpackdb` — of which this package already owns one. Nothing
to thinly wrap, but nothing expensive either: no new binary, because `mmseqs` is already required.

**The whole format problem is one bit per column: which columns are match states.** A3M stores that
bit in the letter case and drops insert-column padding entirely; aligned FASTA does not store it at
all. Every conversion question below is a restatement of that.

So the smallest honest support is **`io/a3m.py` at the record layer plus two verbs on
`MmseqsLikeTool`** — no new class, no new binary, no biotite bump. Aligning a set of proteins you
already have is a *different tool* (MAFFT, FAMSA) and a third required binary, and is worth
declining until someone asks.

## Method and provenance

| | |
| --- | --- |
| Read directly (Part 2) | the `biotite-dev/biotite` tree at tag **v1.7.1**, downloaded and grepped whole, plus `sequence/align/alignment.py` and `sequence/io/fasta/convert.py` at tags v1.4.0, v1.5.0, v1.6.0 and v1.7.0 |
| Read directly (Part 4) | this repo: `src/protein/{external,seq}.py`, `src/protein/io/{fasta,structure}.py`, `src/protein/search/{mmseqs,mixin}.py`, `tests/test_io_fasta.py`, `tests/test_external.py`, `pyproject.toml`, `.vale.ini` |
| Packaging facts | `api.anaconda.org/package/<channel>/<name>` and `api.anaconda.org/dist/...`, queried 2026-09-03 |
| Delegated (Part 1) | the HH-suite wiki and `scripts/reformat.pl`, `scripts/a3m.py`, `src/hh*.cpp`; the HMMER 3.4 User's Guide; Easel's `esl_msafile_*.c` and `esl_alphabet.c`; the UCSC SAM A2M description; Sonnhammer's Stockholm spec; ClustalW 2.1 and Clustal Omega source; EMBOSS 6.6.0; the MODELLER manual; Felsenstein's PHYLIP docs; the NEXUS 1997 spec; the UCSC MAF spec |
| Delegated (Part 3) | MMseqs2 `src/commons/Parameters.cpp`, `src/util/{result2msa,unpackdb}.cpp`, `src/alignment/MultipleAlignment.cpp`, `src/MMseqsBase.cpp` at tag `18-8cc5c` and master; the MMseqs2 user guide; ColabFold `colabfold/mmseqs/search.py`, `setup_databases.sh`, `MsaServer/`; AlphaFold 2 `data/parsers.py` and `data/tools/`; AlphaFold 3 `docs/input.md`; Boltz and Chai-1 docs |

Every claim is marked **VERIFIED** — the primary source was read and can be quoted — or **INFERRED**.
Parts 1 and 3 were verified by delegated agents that quoted the source text back; Parts 2 and 4 were
read here. **Nothing was executed**: this is a macOS laptop and the repo is `platforms =
["linux-64"]`, so no binary ran and no environment was solved. Where the delegates *measured*
something (round-tripping real files through `reformat.pl`) it is marked **MEASURED**.

Knowledge cutoff is May 2026 and today is 2026-09-03, so post-cutoff items are flagged where they
appear.

## 1. The formats

### The information model, stated once

Everything below follows from this. **VERIFIED** — three independent implementations converged on
the same structure: Easel's `esl_msafile_a2m.c`, AlphaFold's `parsers.py`, and biotite's
`get_a3m_alignments`.

An A3M or A2M alignment carries **two** things:

1. a **match-state matrix**, `N` rows by `M` columns of {residue, deletion}, where `M` is the same
   for every row; and
2. per row, an **insertion list** — for each of the `M+1` junctions between match states, a possibly
   empty run of inserted residues, understood to be **unaligned** to any other row's run.

An **aligned FASTA** carries only one thing: an `N × L` character matrix with no distinguished
subset of columns. That asymmetry is the entire conversion problem.

AlphaFold calls (2) the "deletion matrix" and stores one integer per match position
(`parsers.py:183-193`). biotite models the whole object as `N-1` **pairwise** query-to-target
alignments rather than one N-row MSA, which is the same thing said a third way — **VERIFIED**,
`get_a3m_alignments` docstring: *"The i-th alignment is an alignment of the first sequence in the
file (the query) to the i+1-th sequence in the file."*

### A3M

**VERIFIED**, HH-suite User Guide (the wiki `Home.md` *is* the guide; there is no separate "File
formats" page): *"The A3M format is a condensed version of A2M format. It is obtained by omitting
all `.` symbols from A2M format. Hence residues emitted by Match states of the HMM are in upper
case, residues emitted by Insert states are in lower case and deletions are written `-`."*

| Character | Meaning |
| --- | --- |
| `A`-`Z` | residue in a **match** column |
| `-` | **deletion** in a match column — the row spans it and has no residue |
| `a`-`z` | residue **inserted** relative to the match-state consensus |
| `.` | **never present** |

**Rows have unequal length.** Only the match-state count is constant — `reformat.pl:698` skips its
equal-length assertion for `a3m` and `ufas` alone, and `a3m.py`'s `check_match_states` raises
*"Sequence with diverging number of match states"*. **VERIFIED.**

**The first row is the query and carries no lowercase.** **VERIFIED** — the guide: *"The query
sequence is the first sequence that does not start with a special name"*, and `-M first`: *"exactly
those columns of the MSAs which contain a residue in the query sequence will be assigned to Match /
Delete states."* A query cannot insert relative to itself. biotite states it as a hard precondition;
so does AlphaFold 3. **This invariant is what makes A3M self-describing**, and §1's round-trip
results show what breaks without it.

Produced by `hhblits -oa3m`, `hhconsensus`, `reformat.pl`, and MMseqs2 `result2msa
--msa-format-mode 5|6`. Consumed by every `hh*` tool, AlphaFold 2 and 3, ColabFold, Boltz, and
Chai-1 after conversion. **VERIFIED.**

**Two different `#` lines, and the brief conflated them.** **VERIFIED:**

- HH-suite's `#` line is a **name and description** line, used by `hhmake` to name the HMM
  (`hhalignment.cpp:366`). `reformat.pl:805` preserves it for `a3m` output only.
- The `#123,456<TAB>1,1` form is **ColabFold's**, not HH-suite's — `colabfold/input.py:81-82`
  writes `"#" + ",".join(chain lengths) + "\t" + ",".join(copy numbers)`. Downstream parsers skip
  any line starting `#`.

Annotation rides in specially-named FASTA records at the top — `>ss_dssp`, `>ss_pred`, `>ss_conf`,
`>aa_dssp`, `>aa_pred`, `><query>_consensus` — with alphabets validated by `a3m.py`
(`VALID_SS_STATES = set("ECH")`). There is no markup channel. **VERIFIED.**

One trap worth naming: **`hhconsensus` uses case for a second, unrelated purpose.**
`hhalignment.cpp:2260-2276` emits the consensus residue uppercase when its weight exceeds 0.6,
lowercase above 0.4, else `x`. That case is a *conservation* signal, not a match/insert signal.
**VERIFIED.**

### A2M

**VERIFIED**, verbatim from the UCSC SAM description (the canonical URL now 301-redirects; a mirror
was read):

> Uppercase characters and "-" represent alignment columns, and there must be exactly the same
> number of alignment columns in each sequence. Lowercase characters (and spaces or ".") represent
> insertion positions between alignment columns or at the ends of the sequence. White space
> (including line breaks) and periods are ignored. The spaces or periods in the multiple alignments
> are only for human readability, and may be omitted.

**The dots are optional by UCSC's own spec.** "Dotless A2M" is legal A2M — and HH-suite's A3M is
exactly that, a renamed dotless A2M. The guide concedes it: *"A3M, though very practical and
space-efficient, is not a standard format, and the name A3M is our personal invention."* HMMER uses
the same term independently (User's Guide p.221): *"the aligned sequences in a 'dotless' A2M file do
not necessarily all have the same number of characters."* **VERIFIED.**

**The `O` trap, and why it matters to `seq.py`.** UCSC reads `O` as a **free-insertion module**, not
as pyrrolysine. HMMER's guide, p.222: *"A2M format alignments must not contain pyrrolysine residues,
lest they be read as FIMs. For this reason, Easel converts 'O' residues to 'X' when it writes an
amino acid alignment in A2M format."* A2M also cannot carry `*` or `~`; Easel writes them as gaps.
**VERIFIED.** `protein.seq` already folds `U`, `O` and `J` to `X` — for A2M that fold is *required*,
not merely convenient, which is a point ADR-0002 could gain if A2M I/O ever lands.

**One divergence between the two toolchains. VERIFIED:** the HH-suite C++ tools accept dotless A2M
(all of `hhblits.cpp:216`, `hhsearch.cpp:79`, `hhmake.cpp:89`, `hhfilter.cpp:58`, `hhalign.cpp:68`,
`hhconsensus.cpp:82` carry the identical help string *"'.' = gaps aligned to inserts (may be
omitted)"*), but `reformat.pl` does **not** — its dot-insertion block at line 481 is guarded
`if ($informat eq "a3m" …)`. A dotless A2M declared as `a2m` to `reformat.pl` is mis-parsed. This is
the *"some conversion programs misinterpret dotless a2m files"* the UCSC page warned about, still
true twenty-odd years later.

### Aligned FASTA

Plain `-` padding, all rows equal length. **Case is not meaningful** — three independent primary
confirmations, all **VERIFIED**:

1. `reformat.pl -h`: *"fas: aligned fasta; lower and upper case equivalent, '.' and '-'
   equivalent"*.
2. `reformat.pl:463` uppercases the whole alignment on reading anything that is not a3m or a2m.
3. Easel's amino alphabet calls `esl_alphabet_SetCaseInsensitive` (`esl_alphabet.c:277`).

Produced by MUSCLE v5 (its **only** alignment output), MAFFT (its default), Clustal Omega
`--outfmt=fa`, `esl-reformat afa`, and MMseqs2 `--msa-format-mode 2`, which is MMseqs2's default.
Metadata is the FASTA description line and nothing else. **VERIFIED.**

### Stockholm

**VERIFIED**, HMMER User's Guide p.217: the first line must be `# STOCKHOLM 1.x`; blocks are
separated by blank lines with the same sequences in the same order in every block; `//` terminates.
Easel matches `//` as a *prefix*, not an exact line (`esl_msafile_stockholm.c:294`). MMseqs2's
`convertmsa` is stricter than the spec and demands the exact string `# STOCKHOLM 1.0`
(`convertmsa.cpp:40`).

| Magic | Grammar | Scope |
| --- | --- | --- |
| `#=GF <tag> <text>` | free text | per-file |
| `#=GS <seqname> <tag> <text>` | free text | per-sequence |
| `#=GC <tag> <chars>` | exactly one character per column | per-column |
| `#=GR <seqname> <tag> <chars>` | exactly one character per column | per-residue |

**There is no single authoritative feature vocabulary — there are three lists and they disagree.**
**VERIFIED:**

- What **Easel's parser special-cases** (`esl_msafile_stockholm.c`): `#=GF` `ID AC DE AU GA NC TC`;
  `#=GS` `WT AC DE`; `#=GC` `SS_cons SA_cons PP_cons RF MM`; `#=GR` `SS SA PP`. Eighteen tags;
  everything else round-trips as untyped string annotation, because *"unrecognized tags will simply
  be ignored"*.
- What the **HMMER guide documents**: the same minus `PP_cons`, which is an omission — its own
  tutorial uses it.
- What **Pfam and Sonnhammer** define: six `#=GS` features, nine `#=GR` features, the rule *"`#=GC`:
  the same features as for `#=GR` with `_cons` appended"*, and roughly 25 further `#=GF` tags.

`#=GC seq_cons` is a **Pfam artefact, not an Easel tag** — grepping Easel and HMMER for `seq_cons`
returns hits only in data files, never in `.c` or `.h`. It is a Belvu-style consensus-*symbol*
string, not residues. **VERIFIED.**

**The gap characters: the guide's prose and Easel's code disagree, and the code wins.**
**VERIFIED**, `esl_alphabet.c:273-278`:

```c
a = esl_alphabet_CreateCustom("ACDEFGHIKLMNPQRSTVWY-BJZOUX*~", 20, 29);
esl_alphabet_SetEquiv(a, '_', '-');       /* allow _ as a gap too */
esl_alphabet_SetEquiv(a, '.', '-');       /* allow . as a gap too */
esl_alphabet_SetCaseInsensitive(a);
```

So **`-`, `.` and `_` are gaps** (all index 20) and **`~` is `missing data`** at index `Kp-1`, which
is *not* a gap. `esl_abc_XIsGap` and `esl_abc_XIsMissing` are separate macros with no combined
predicate. `~` means *"this row is a fragment and does not extend here"*, written deliberately by
`tracealign.c:757-772`; `-` means *"this row spans this column and has no residue."* Collapsing the
two destroys fragment annotation that HMMER, Infernal and Dfam emit on purpose. The guide says four
gap characters on p.217 and three on p.50; Sonnhammer says two. **Model `{-, ., _}` and `{~}` as two
predicates.**

**HMMER's Stockholm output is A2M-with-dots in its case and gap semantics**, and it says so —
`tracealign.c:694-700`, verbatim:

```text
/* The reason to make a text-mode MSA rather than let Easel handle printing a digital
 * MSA is to impose HMMER's standard representation on gap characters and insertions:
 * at inserts, gaps are '.' and residues are lower-case, whereas at matches, gaps are '-'
 * and residues are upper case. */
```

It stores the same partition **twice**: once in the case, and once explicitly in `#=GC RF` (`x` at
consensus columns, `.` at inserts). **That redundancy is the lossless bridge** — see the conversion
section. Guide p.222: *"only Stockholm and SELEX formats support reference annotation."*

`esl-reformat`'s complete format set is exactly ten, and the output format is **positional, not a
flag**: `stockholm pfam a2m psiblast selex afa clustal clustallike phylip phylips`. `stockholm` and
`pfam` are the same parser and writer differing **only in line width** — 200 versus one line per
sequence (`esl_msafile_stockholm.c:338-361`). **VERIFIED.** The guide's claim at p.221 that *"Easel
currently does not read A2M format"* is **stale**: `esl_msafile_a2m_Read()` exists and is dispatched.

Produced by `hmmalign`, and by `hmmsearch`/`phmmer`/`jackhmmer`/`nhmmer` under `-A`. Consumed by
`hmmbuild`, Infernal's `cmbuild` (**Stockholm only** — it needs the consensus structure annotation),
AlphaFold 2, and MMseqs2 `convertmsa`. Pfam and Rfam distribute in it. **VERIFIED.**

### Clustal, PIR, MSF, PHYLIP and the long tail

These matter for interoperability and nothing else. The delegate's findings, compressed:

**Clustal `.aln`** — **VERIFIED, and it corrects the brief**: ClustalW **2.x writes `CLUSTAL 2.1
multiple sequence alignment`**, with no `W` and no parentheses (`AlignmentOutput.cpp:1669`). The
`CLUSTAL W (1.81)` form is 1.x, and survives mainly as hardcoded literals in third-party writers —
EMBOSS, BioPerl, Biopython. Clustal Omega writes `CLUSTAL O(1.2.4) multiple sequence alignment`,
with no space before the paren (`src/squid/clustal.c`). Readers are loose: Easel's `clustallike`
only requires the first line to contain the word `alignment`, and its own examples include MUSCLE's
and MAFFT's headers.

The conservation line is `*` identical, `:` all residues within one of nine strong Gonnet-PAM250
groups, `.` all within one of eleven weak groups, space otherwise — including **any column
containing a gap**. Both implementations carry the identical group lists, and **both disable the
`:`/`.` tests for nucleotide data**, so a DNA `.aln` conservation line holds only `*` and spaces.
Gap character is `-` only. **VERIFIED.**

**PIR/NBRF and MODELLER** — `>P1;code`, a free-text second line, sequence in blocks of ten,
terminated by `*`, gaps `-`. MODELLER's variant replaces line 2 with ten colon-separated fields;
field 1 has **five** legal values (`sequence`, `structure`, `structureX`, `structureN`,
`structureM`), a dot means "unspecified" for both residue number and chain id, and `/` is a chain
break. Rich per-sequence metadata, no per-column channel at all. **VERIFIED** from the MODELLER
manual.

**MSF/GCG** — `!!AA_MULTIPLE_ALIGNMENT 1.0`, then a `MSF: … Check: … ..` line, then `//`. **Gaps are
`.` internally and `~` terminally** — confirmed exactly as asked, from squid's `msf.c` and from
EMBOSS's own docs (*"msf format … uses '.' as the gap character inside the sequences and '~' as the
gap character at the terminal ends"*). But it is **not universal**: ClustalW writes `.` everywhere
including the flanks, and EMBOSS writes `CompCheck:` where squid writes `Check:`. A reader must
treat both as gaps unconditionally. **VERIFIED.**

**PHYLIP** — **VERIFIED** from Felsenstein's own docs: *"The name should be ten characters in
length, and either terminated by a Tab character or filled out to the full ten characters by blanks
if shorter."* So **names may contain spaces**, and splitting a strict PHYLIP line on whitespace is
wrong. Interleaved versus sequential is *"selected by the I option"* — **a runtime menu choice, not
a file property**, so the two are not distinguishable by inspection. `.` is explicitly rejected as a
gap; space is explicitly not a gap. "Relaxed PHYLIP" appears **nowhere** in Felsenstein's
documentation — RAxML caps names at 256 and PhyML at 100, both requiring whitespace separation,
which is the exact opposite of the strict rule.

**Not relevant to a protein package**, each for a stated reason: **SELEX** (Easel's own header calls
it *"largely obsolete"*; it is the one format where whitespace is a gap); **NEXUS** (a phylogenetics
container whose gap, missing and match symbols are all *declared per file* — `GAP` has no default at
all — and which Easel does not support); **MAF** (its own spec opens *"stores multiple alignments at
the DNA level between entire genomes"*; every row is `assembly.chromosome` plus offset and strand,
and a `Protein` has no genome coordinate — manufacturing one is exactly the hidden acquisition
**direct support only** forbids). **PSI-BLAST `.psi`** is marginal: it is a protein format, its case
convention is load-bearing (*"In each column, all letters must be in upper case, or all letters must
be in lower case"* — the case tells PSI-BLAST whether a column feeds the PSSM), but nothing in this
package takes a `.psi`, so owning a writer would violate direct support only.

### Comparison table

| | A3M | A2M | Aligned FASTA | Stockholm | Clustal `.aln` | PIR/MODELLER | MSF/GCG | PHYLIP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Match-column gap | `-` | `-` | `-` | `-`, also `.` `_` | `-` | `-` | `.` | `-`, also `?` |
| Insert-column pad | **absent** | `.`, optional | n/a | `.` by convention | n/a | n/a | `~` terminal | n/a |
| Missing data | — | — | — | **`~`**, distinct | — | — | `~` reused | `?` |
| Case meaningful | **yes** | **yes** | **no** | **yes**, by convention | no | no | no | no |
| Rows equal length | **no** | dotted yes, dotless no | yes | yes per block | yes per block | no | yes | yes |
| Match-state mask | implicit, in the case | implicit, in the case | **none** | **`#=GC RF`**, explicit | none | none | none | none |
| Per-file metadata | one `#` line | — | — | **`#=GF`** | — | `C;` `R;` | `MSF:` `Check:` | — |
| Per-sequence metadata | FASTA header | FASTA header | FASTA header | **`#=GS`** | — | **ten fields** | `Name:` `Weight:` | — |
| Per-column track | pseudo-records `ss_*` | — | — | **`#=GC`** | conservation `*:.` | — | — | — |
| Per-residue track | — | — | — | **`#=GR`** | ClustalW `!SS_` | — | — | — |
| Relevant here | **yes** | **yes** | **yes** | **yes** | interop only | MODELLER only | legacy only | phylogenetics only |

### The conversions, and where information is lost

**A3M to aligned FASTA is not lossless. The loss is exactly one bit per column: whether it was a
match column.**

The converter must do this, and there is no shortcut. **VERIFIED** from two implementations that
agree character for character — Easel's `esl_msafile_a2m.c:512-530` and `reformat.pl:484-524`:

1. Split every row into `M` match characters (`[A-Z]` or `-`) interleaved with `M+1` insertion runs.
2. For each junction `j`, take `nins[j] = max` over rows of that run's length.
3. The padded length is `L = M + Σ nins[j]`.
4. Rewrite each row: its insertion residues **left-justified**, then `.` padding to `nins[j]`, then
   the match character.
5. For aligned FASTA, rewrite every `.` as `-`.

After step 5 a `-` in an insert column is byte-identical to a `-` in a match column, and since
aligned FASTA readers ignore case, nothing distinguishes them. **Step 4 also invents an alignment
among the inserted residues** — A3M never claimed those runs were mutually aligned, and
left-justifying them asserts that they are.

**MEASURED** — `reformat.pl a3m fas` does not uppercase; it only maps `.` to `-`, so its output
*incidentally* still carries the case marking and reproduced the guide's published FASTA block byte
for byte. But that is a courtesy of the writer, not a property of the format: `reformat.pl:463`
uppercases the whole alignment the moment it reads anything back as `fas`.

**Aligned FASTA to A3M is lossless only relative to a match-state rule you supply**, and both
toolchains default to the same one: **`-M first`**. `reformat.pl:535` sets it automatically for a3m
and a2m output; Easel states it in five lines (`esl_msafile_a2m.c:436-441`) and documents the
fallback: *"If it does not [have a valid RF line], then as a fallback, the first sequence in the
alignment is considered to be the consensus."* Take row 1 as the query, make every column where it
gaps an insert column, lowercase those residues, and **drop the gap characters in insert columns
entirely** — which is where A3M's compactness comes from. **VERIFIED.**

**MEASURED, and this is the sharp edge:** the round trip works when, and only when, the query row
has no insertions. On an A3M whose first row carries lowercase, `-M first` **promotes the query's
own inserts to match columns**:

```text
ORIGINAL A3M (10 match states)  ->  a2m                ->  fas
>query  ACDEFqqqGHIKL               ACDEFQQQ..GHIKL        ACDEFqqq--GHIKL
>s2     ACDEFGHIKL                  ACDEF---..GHIKL        ACDEF-----GHIKL
>s3     ACDEFwwwwwGHIKL             ACDEFWWWwwGHIKL        ACDEFwwwww GHIKL
```

The match-state count went 10 to 13, and both round trips differ from the original. That is not a
bug — it is the documented default applied to input that violates the invariant. **A3M is
self-describing only under "row 1 is all match states."** That holds for anything hhblits or
`result2msa` produced; anything else needs an out-of-band mask.

**A2M and A3M convert both ways, information-losslessly but not byte-losslessly.** A2M to A3M is
`tr/.//d` (`reformat.pl:678`), exactly the guide's definition. A3M to A2M recomputes the padding and
**MEASURED** byte-identical on the guide's example. Two things do not survive a round trip that
starts at A2M, both **MEASURED**:

- **All-dot insert columns vanish.** Nothing in the A3M records that a column with no insertions in
  any row ever existed, because `nins[j]` is a max over actual insertions.
- **Intra-insert justification is lost.** `ab..`, `.ab.` and `..ab` are three A2M strings mapping to
  the same A3M string `ab`; both implementations restore them left-justified.

Neither loss is semantic — UCSC says the dots *"carry no information"*, and HMMER's guide is
explicit that *"this representation only works if all insertions relative to consensus are
considered to be unaligned characters."* Lossless in information, lossy in layout.

**Stockholm to A3M loses the entire markup channel** unless the converter is written to keep it.
AlphaFold's `convert_stockholm_to_a3m` (`parsers.py:215-271`) is the canonical real-world converter
and drops, in order: every `#=GF` line; every `#=GC` line **including `SS_cons` and `RF`**; every
`#=GR` line; every `#=GS` feature except `DE`; sequence weights; and the gathering, noise and
trusted cutoffs. It then re-derives match states **from the first row's gaps rather than from
`RF`** — so if `RF` and row 1 disagree, which they can after several jackhmmer iterations, the
conversion silently changes the match-state assignment. MMseqs2's `convertmsa` keeps only `#=GF ID`
or `#=GF AC`. **VERIFIED.**

**The lossless bridge exists and it is `#=GC RF`.** Easel's A2M reader *reconstructs* it while
un-padding (`esl_msafile_a2m.c:525` writes `.` at insert columns and `x` at consensus columns), and
`reformat.pl a3m sto` writes it from the other side as `$refline =~ s/[a-z]/-/g`. **MEASURED:**
`a3m -> sto -> a3m` came back byte-identical for the sequences — though descriptions were lost and
names longer than 32 characters were hard-truncated by `%-32.32s`.

**So A3M to Stockholm is the archival direction**, because Stockholm is a strict superset that can
hold the match-state mask explicitly plus arbitrary annotation at four scopes. Stockholm to A3M is
the lossy one.

## 2. What biotite already covers

### The version gate, which is the whole answer

**VERIFIED** by listing `src/biotite/sequence/io/` and grepping `sequence/align/alignment.py` and
`sequence/io/fasta/convert.py` at each tag:

| Capability | First biotite version | Available at the repo's 1.4.0? |
| --- | --- | --- |
| `Alignment` class, `trace_from_strings` | ≤ 1.4.0 | **yes**, but see below |
| `fasta.get_alignment` / `set_alignment` | ≤ 1.4.0 | yes — and banned by ADR-0002 |
| `Alignment.from_strings(strings, sequence_factory, gap_character)` | **1.6.0** | **no** |
| Vectorised `trace_from_strings` | **1.6.0** | **no** |
| `fasta.get_a3m_alignments` / `set_a3m_alignments` | **1.6.0** | **no** |
| `biotite.sequence.io.clustal` | **1.7.0** | **no** |

At 1.4.0, `Alignment.from_strings` does not exist, and `trace_from_strings(seq_str_list)` is a
**pure-Python double loop with the gap character hardcoded to `-`** — one Python-level iteration per
`(column, sequence)` cell, so an MSA of `n` rows and `m` columns costs `m × n` interpreted
iterations. **VERIFIED** by reading the 1.4.0 source; the cost is **INFERRED** arithmetic from it.
1.6.0 replaced it with `np.frombuffer` and a per-row mask.

The upgrade is not free. **VERIFIED** against `api.anaconda.org/dist/conda-forge/biotite/...`:
the py313 linux-64 builds of **1.6.0 and 1.7.1 both declare `numpy >=1.26,<2.4`**, where **1.4.0
declares `numpy >=1.23,<3`**. That is the same conda-forge cap
[the sibling note](biotite-coverage-of-v1.md#why-the-lock-holds-biotite-at-140) verified: newest
biotite or newest numpy, not both. That note found nothing needing either side of the trade. **This
note finds the first thing that does** — every biotite feature relevant to MSAs sits behind it.

### What is in biotite at all

**VERIFIED** by grepping the whole v1.7.1 tarball: `a2m`, `stockholm`, `hhblits` and `hmmer` occur
**zero times** in `src/`. `sequence/io/` holds `clustal, fasta, fastq, genbank, gff` and
`general.py`, whose format dispatch covers only FASTA, FASTQ and GenBank and has **no alignment
loader at all**.

So biotite reads and writes exactly three MSA formats — aligned FASTA, A3M and Clustal — and the
last two only above 1.4.0.

### `Alignment`: what it holds, and the one thing it does not

**VERIFIED**, `sequence/align/alignment.py`. It holds three fields: `sequences`, a list of **ungapped**
biotite `Sequence` objects; `trace`, an `(m, n)` integer array where `-1` is a gap; and `score`.

**It carries no names.** `Alignment` has no `name` or `label` attribute anywhere, which is why every
converter takes them separately — `clustal.set_alignment(file, alignment, seq_names, line_length)`
and `fasta.set_alignment(file, alignment, seq_names)`. Anything in this package that holds an
`Alignment` must hold the headers beside it.

Three further consequences, all **VERIFIED** from the source:

- `trace_from_strings` requires **equal-length** strings and takes **one** gap character. An A3M
  cannot be fed to it directly, and an A2M needs `.` and `-` normalised to one symbol first.
- `get_gapped_sequences()` renders every gap as `-`, and the sequences were uppercased on
  construction. **Case and the dot/dash distinction do not survive a round trip through
  `Alignment`.**
- The trace is `np.full(..., dtype=int)`, so int64 on linux-64: `m × n × 8` bytes. A ColabFold-sized
  MSA of 100,000 rows by 1,000 columns is 800 MB of trace for 100 MB of text. **INFERRED**
  arithmetic.

Set against that, one thing is genuinely well designed: **`from_strings` takes a
`sequence_factory`**, so `protein.seq.to_protein_sequence` slots in as that factory and the string
never touches biotite's converters. That is the ADR-0002-compatible door — and it is the 1.6.0
feature.

### The converters, again, exactly where MSA support would land

**VERIFIED**, and this is the finding with teeth.

```python
# biotite/sequence/io/fasta/convert.py — and, copied verbatim, clustal/convert.py
def _convert_to_protein(seq_str: str) -> ProteinSequence:
    """Replace selenocysteine with cysteine and pyrrolysine with lysine."""
    return ProteinSequence(seq_str.upper().replace("U", "C").replace("O", "K"))
```

| Reader | Route | ADR-0002 |
| --- | --- | --- |
| `fasta.get_alignment` | `Alignment.from_strings(..., partial(_convert_to_sequence, ...))` | **banned** |
| `fasta.get_a3m_alignments` | `_convert_to_sequence` for the query, `_convert_to_protein` for each target | **banned** |
| `clustal.get_alignment` | a second, independent copy of `_convert_to_sequence` | **banned** |
| `FastaFile` / `ClustalFile` | raw strings, no alphabet | **clean** |
| `application.MSAApp` | `FastaFile` plus `trace_from_strings` on the caller's own `Sequence` objects | **clean** — but see below |

Two details make this worse than it looks:

- With no `seq_type` the auto-detect tries `NucleotideSequence` **first**, the same misdetection the
  sibling note measured — `"MKTU"` comes back as DNA.
- **`clustal/convert.py` is a second copy of the same function**, not an import of it. An upstream
  fix to one would not fix the other. **VERIFIED.**

**The repo's guard does not cover any of them.** `tests/test_io_fasta.py` walks `src/protein/**/*.py`
for `_BANNED_CONVERTERS = {"get_sequence", "get_sequences", "to_sequence", "convert_letter_3to1"}`.
`get_alignment`, `get_a3m_alignments` and `set_alignment` are not in that set, so a module calling
`fasta.get_a3m_alignments` would pass `pixi run check` while silently rewriting every `U` and `O` in
every row of every MSA. **VERIFIED** by reading the test. If MSA support lands, that set grows —
independently of whether biotite is ever upgraded.

### The A3M reader, judged on its merits

If the version gate were not there, is `get_a3m_alignments` the right thing to adopt? **VERIFIED**
by reading it:

- **It returns `N-1` pairwise `Alignment` objects, not one MSA.** That is the correct information
  model, and it matches AlphaFold's and Boltz's independently. It is also `N-1` Python objects and
  `N-1` traces for one file.
- **`_is_gap` is `== ord("-")` and nothing else.** A dotted A2M handed to it is mis-parsed: `.` is
  neither lowercase nor `-`, so it is read as a residue in a match column.
- **`FastaFile.read` raises `InvalidFileError: File starts with '#' instead of '>'`** on any A3M
  carrying a leading `#` line — HH-suite's name line or ColabFold's cardinality line. `read_iter`
  does not raise; it **silently discards** everything before the first `>`. Both **VERIFIED** from
  `fasta/file.py`.
- Its own test fixture, `tests/sequence/data/1a00_A_uniref90.a3m`, is an OpenProteinSet UniRef90
  A3M: 3,391 records, a 141-residue all-uppercase query, 2,350 lines carrying lowercase, and **zero
  dots in any sequence line** — confirming that real A3M has no dots.

`ClustalFile`, by contrast, is exactly the shape `io/fasta.py` already adopts: a
`MutableMapping[str, str]` from name to gapped string with no alphabet anywhere. It **drops the
conservation line** on read (skipping any line matching `[ *:.]+`), always writes the header
`CLUSTAL W multiple sequence alignment`, requires the first line to start with `CLUSTAL` — so
MUSCLE's `MUSCLE (3.8) …` header would be rejected (**INFERRED**) — and has no `read_iter`.

### Does biotite align?

`align_multiple` exists and is a genuine progressive aligner: Feng-Doolittle distances, UPGMA guide
tree, "once a gap, always a gap". **It is not usable for a homologue MSA.** **VERIFIED** from
`multiple.pyx`: `_get_distance_matrix` runs `align_optimal` — full Needleman-Wunsch — on **every
pair**, `n(n+1)/2` alignments, and retains every resulting `Alignment` in an `n × n` object array.

biotite says so itself, in its own tutorial (`doc/tutorial/sequence/align_multiple.rst`):

> `align_multiple()` is only recommended for strongly related sequences or exotic sequence types.
> When high accuracy or computation time matters, other MSA programs deliver better results.

**Verdict: fine for a handful of sequences, wrong for anything a search returns.** `align_optimal`
is the right tool for a pairwise question and is not in scope here. `SequenceProfile.from_alignment`
and `.to_consensus` exist and are the reason to hold an `Alignment` at all.

### Coverage verdict

| Piece | What already exists | Verdict | Why |
| --- | --- | --- | --- |
| The `Alignment` type | `biotite.sequence.align.Alignment` | **Hold it** | Three fields, ungapped sequences plus a trace. It is the right thing to hold, and holding it is what the map says to do. |
| Sequence names | nothing on `Alignment` | **Build ours** | `Alignment` has no name field; both converters take `seq_names` separately. Whatever holds an `Alignment` holds the headers too. |
| String to `Alignment` | `Alignment.from_strings(..., sequence_factory)` | **Adopt — at 1.6.0** | The factory parameter is the ADR-0002 seam: pass `to_protein_sequence` and no converter runs. Absent at 1.4.0. |
| Aligned FASTA, record layer | `FastaFile.read_iter` / `write_iter` | **Adopt** | Already adopted by `io/fasta.py`. An aligned FASTA *is* a FASTA at the record layer; `fasta.read_records` reads one today, gaps and all. |
| Aligned FASTA to sequences | `fasta.get_alignment` | **Build ours** | `U` to `C`, `O` to `K`, silently; nucleotide misdetection; and `.` is not in its gap set. |
| A3M | `fasta.get_a3m_alignments` (1.6.0+) | **Build ours** | Banned converters, `-` as the only gap, and `FastaFile.read` raises on the `#` line. The *model* — `N-1` pairwise alignments — is worth copying. |
| Clustal `.aln` | `clustal.ClustalFile` (1.7.0+) | **Adopt the file layer if ever needed** | Pure `MutableMapping[str, str]`, the `FastaFile` shape. Its `get_alignment` is banned. Nothing here needs `.aln`. |
| Stockholm | **nothing** | **Build ours, or decline** | Zero occurrences in biotite. `esl-reformat` from the `hmmer` package converts it; owning a parser buys nothing unless jackhmmer lands. |
| A2M, PIR, MSF, PHYLIP, NEXUS | **nothing** | **Decline** | No consumer in this package, and each is a different gap convention to get wrong. |
| Multiple alignment itself | `align_multiple` | **Do not use** | `n(n+1)/2` full Needleman-Wunsch runs, and upstream's own tutorial says use another program. |
| Driving an aligner | `application.MSAApp`, `ClustalOmegaApp`, `MafftApp`, `Muscle5App` | **Build ours if ever needed** | It does **not** rewrite residues — the one clean converter path in biotite. But `LocalApp` imports `subprocess`, which `external.py` exists to be the only place for, and `MSAApp` renames every sequence to `str(i)`. |
| Profiles and consensus | `SequenceProfile.from_alignment`, `.to_consensus` | **Adopt** | The payoff for holding an `Alignment`, and it needs nothing new. |

## 3. Generating an MSA

### MMseqs2: no `easy-msa`, three verbs instead

**VERIFIED, three independent ways at the repo's exact pin `18-8cc5c`**: grepping every `"easy-*"`
literal out of `src/MMseqsBase.cpp` returns exactly six commands — `easy-cluster`, `easy-linclust`,
`easy-linsearch`, `easy-rbh`, `easy-search`, `easy-taxonomy`; `src/workflow/` has no `EasyMsa.cpp`;
`data/workflow/` has no `easymsa.sh`; and the user guide mentions `easy-msa` zero times. Master adds
`easy-proteomecluster` and `easy-proteomesearch`, still no `easy-msa`.

**So there is nothing to thinly wrap.** MSA generation is always the explicit pipeline:

```bash
mmseqs createdb query.fasta  queryDB
mmseqs createdb target.fasta targetDB

# -a stores the backtrace; without it result2msa realigns every hit itself
mmseqs search queryDB targetDB resultDB tmp -a

mmseqs result2msa queryDB targetDB resultDB msaDB --msa-format-mode 5

mmseqs unpackdb msaDB out_dir --unpack-name-mode 1 --unpack-suffix .a3m
```

`--msa-format-mode` takes seven values, from `Parameters.h` and the help string in
`Parameters.cpp:121`. **VERIFIED**, identical at the pin:

| Value | Constant | Writes |
| --- | --- | --- |
| 0 | `FORMAT_MSA_CA3M` | binary cA3M database, needs `convertca3m` |
| 1 | `FORMAT_MSA_CA3M_CONSENSUS` | binary cA3M with consensus |
| **2** | `FORMAT_MSA_FASTADB` | **aligned FASTA — the default** |
| 3 | `FORMAT_MSA_FASTADB_SUMMARY` | aligned FASTA plus a `#`-prefixed header summary |
| 4 | `FORMAT_MSA_STOCKHOLM_FLAT` | Stockholm flat file, `# STOCKHOLM 1.0` … `//` |
| **5** | `FORMAT_MSA_A3M` | **A3M** |
| 6 | `FORMAT_MSA_A3M_ALN_INFO` | A3M plus per-line alignment info |

Mode 3 **adds** a `#` line rather than removing one — `result2msa.cpp` writes
`"#" + summaryPrefix + "-" + queryKey + "|" + summarizedHeaders`. **VERIFIED.**

**`result2msa` output is a real alignment, not concatenated hits.** `result2msa.cpp:260` calls
`MultipleAlignment::computeMSA`, which threads every target onto the query via its pairwise backtrace
and propagates the union of query gaps. The user guide agrees, p.78: *"MMseqs2 mmseqs result2msa can
produce an MSA using a centre star alignment without insertions in the query."* **VERIFIED.**

**But it is a centre-star, query-anchored MSA.** Columns are defined by the query; two targets are
aligned to each other only through their common anchoring on it. That is category (b) below, not
category (a). And *"without insertions in the query"* is precisely the A3M invariant from Part 1 —
so `result2msa` output is well-formed A3M by construction.

Two traps, both **VERIFIED** from source:

- **Without `-a` on the search, `result2msa` silently recomputes** every pairwise alignment itself
  (`result2msa.cpp:248-256`, comment `// Recompute if not all the backtraces are present`). It
  works either way; one way pays twice.
- **`result2msa` disables filtering by default, overriding the global default.** `result2msa.cpp:20-21`
  is literally `// do not filter by default` / `par.filterMsa = 0;`. The global `filterMsa = 1`
  applies to `result2profile`. Filtering is `--filter-msa 1` plus `--max-seq-id` (0.9),
  `--qid` (0.0), `--qsc` (-20.0), `--cov` (0.0), `--diff` (1000) and `--filter-min-enable` (0).
  `filterresult` is the standalone equivalent.

**The output is an ffindex database and needs `unpackdb`.** **VERIFIED**: `--unpack-name-mode`
defaults to 1, naming each file by its accession through `<db>.lookup` and falling back to the
numeric key with a logged warning if there is no lookup file; `--unpack-suffix` defaults to `""` and
writes gzip if it ends `.gz`; the writer strips the trailing `\0`. Neighbouring verbs worth knowing
by name: `result2dnamsa`, `convertmsa` (Stockholm to MSA database), `msa2profile`, `msa2result`,
`convertca3m`, `pairaln`, and a hidden `filtera3m`.

MMseqs2's newest release is `18-8cc5c`, 2025-07-27 — the version this repo pins. Master is active
past the cutoff (HEAD 2026-08-23) with no new release. **VERIFIED.**

### ColabFold

`colabfold_search` is a pure `subprocess` driver over `mmseqs` with a resumability trick; the old
shell entry point is a stub that prints *"Do not use this script."* **VERIFIED**,
`colabfold/mmseqs/search.py`. The UniRef leg runs, in order: `createdb`, `search --num-iterations 3
-a -e 0.1 --max-seqs 10000`, `mvdb`, `lndb`, `expandaln`, `align`, `filterresult`, `result2msa
--msa-format-mode 6`; then the same shape against the environmental database; then `mergedbs` and
`unpackdb`. The complex path adds `pairaln` twice and `result2msa --msa-format-mode 5`.

Databases are `--db1 uniref30_2302_db`, `--db2` templates (empty by default; the artifact is
pdb100, with "pdb70" surviving as legacy naming), `--db3 colabfold_envdb_202108_db`, `--db4
spire_ctg10_2401_db`. **CORRECTION to the brief: `setup_databases.sh` states no per-file sizes** —
the only figure published is the README's aggregate. **VERIFIED.**

Running fully locally works and is documented, and the stated cost is the point:

> First create a directory for the databases on a disk with sufficient storage (940GB (!))

and, on memory:

> the batch searches will require a machine with about 128GB RAM or, if the databases are to be kept
> permamently in RAM, with over 1TB RAM.

ColabFold's README pins `mmseqs2=18.8cc5c` — *"Please use this version if you want to obtain the
same MSAs as the server"* — which is **this repo's exact pin**. **VERIFIED.**

**The public API server is explicitly not for bulk**, printed to stderr on every default run:

> WARNING: You are welcome to use the default MSA server, however keep in mind that it's a limited
> shared resource only capable of processing a few thousand MSAs per day. Please submit jobs only
> from a single IP address.

No numeric rate limit is published; the token-bucket numbers in `MsaServer/config.json` are a
commented-out self-hosting example, not policy. Its client-side backoff is thin — on
`requests.exceptions.Timeout` it retries immediately with no sleep. **VERIFIED.** Given this
package's **bulk, not per-ID** rule, the API server is the wrong shape regardless of etiquette.

Output is **one `.a3m` per query**, renamed from `0.a3m`, `1.a3m`, … to the job name. `pair.a3m`
exists only as an intermediate that is deleted before exit on the local path. ColabFold **v1.6.2,
2026-07-14, is post-cutoff** — it fixes `--pair-mode paired`/`unpaired` crashing on complexes and
adds `--af3-json` export. **VERIFIED.**

### The classical alternatives

**VERIFIED** against `api.anaconda.org` and the upstream repositories, 2026-09-03.

| Tool | Native output | conda | Latest | Alive? |
| --- | --- | --- | --- | --- |
| **HHblits** (`hhsuite`) | **A3M** (`-oa3m`) | bioconda 3.3.0 | 3.3.0, 2020-08-25 | barely; last commit 2025-08-12 |
| **jackhmmer** (`hmmer`) | **Stockholm** (`-A`) | bioconda 3.4 | 3.4, 2023-08-15 | yes, slowly |
| **MAFFT** | aligned FASTA; `--clustalout`, `--phylipout` | **conda-forge 7.526**, bioconda 7.525 | 7.526, 2024-04 | yes |
| **MUSCLE** | aligned FASTA **only** in v5 | bioconda 5.3 and 3.8.1551 | 5.3, 2024-11-11 | yes |
| **Clustal Omega** | `--outfmt` fa/clu/msf/phy/selex/st/vie | bioconda 1.2.4 | 1.2.4, **2016-12-20** | **dead, ~10 years** |
| **FAMSA** | aligned FASTA only | bioconda 2.4.1 | 2.5.2, 2026-03-23 | yes, active |
| **Kalign 3** | `-f fasta\|msf\|clu` | **conda-forge 3.6.0**; bioconda has `kalign3` 3.4.0 | v3.6.0, 2026-05-16 | yes, active |
| **T-Coffee** | Clustal by default; nine formats | bioconda 13.46.2.7c9e712d | 2025-10-02 | yes, low |
| **probcons** | aligned FASTA; `-clustalw` | bioconda 1.12 | 1.12, tarball dated **2007** | **dead** |

Corrections worth carrying: **Clustal Omega 1.2.4 is 2016-12-20, not 2018** — its ChangeLog calls it
*"made code gcc-6 compliant (no new command-line flags)"* over 1.2.3. **MUSCLE v5 dropped every
non-FASTA writer** (`-clwout`, `-msfout`, `-physout` are gone). There is **no bioconda `kalign`** and
no **`muscle5`** package. `conda-forge` and `bioconda` overlap on exactly MAFFT and Kalign, and
conda-forge is newer for both; everything else is bioconda-only. Every row has a linux-64 build, so
`platforms = ["linux-64"]` never constrains the choice. Post-cutoff: **Kalign 3.6.0 and conda-forge
`kalign` 3.6.0, both 2026-05-16**; **FAMSA 2.5.2, 2026-03-23**.

**The two general-purpose converters ride along with tools you might want anyway.** `esl-reformat`
ships inside the **bioconda `hmmer` package** — HMMER's own `make install` deliberately does *not*
install the Easel miniapps, but the recipe's test block runs `esl-reformat -h`. It covers stockholm,
a2m, afa, psiblast, clustal and phylip — **but not a3m**, which is why AlphaFold 2 had to write its
own `convert_stockholm_to_a3m`. The a3m-aware converter is `reformat.pl`, which ships with
`hhsuite`. **VERIFIED.**

### The two use cases are different tools

| | (a) Align proteins I already have | (b) Find homologues and align them to a query |
| --- | --- | --- |
| Input | a FASTA of `N` sequences you chose | one sequence plus a large database |
| Output | a symmetric `N`-row alignment | a query-anchored profile MSA |
| Tools | MAFFT (safe default), FAMSA (large families, best maintained), Kalign 3 (small and fast), MUSCLE 5, Clustal Omega (frozen at 2016), T-Coffee | MMseqs2, HHblits, jackhmmer; ColabFold is a packaged opinion about doing this with MMseqs2 |
| Shape here | takes a **collection** and no database — **no class owns it** | takes a `Protein` and a `Database` — **exactly `search()`'s shape** |
| This repo has | nothing | **the search half already** |

**(b) is "search then align", and this package already owns the search.** `Protein.search(db)` runs
`mmseqs easy-search`; the MSA is the same search with `-a`, followed by two more verbs on the same
binary. That is why the two halves of the question have such different costs.

### What the folding tools want

**AlphaFold 3 takes A3M strings inline in its JSON.** **VERIFIED**, `docs/input.md`: per protein
chain, `unpairedMsa` and `pairedMsa` are A3M **strings**, with `unpairedMsaPath` / `pairedMsaPath`
as file-path alternatives added in JSON version 2. Three hard requirements on the A3M: it is A3M
(lowercase insertions, `-` gaps); *"The first sequence is exactly equal to the query sequence"*; and
with lowercase removed every row is exactly the query's length. **Both fields must be set or both
unset** — you cannot provide one and leave the other null. Templates are separate, a list of mmCIF
strings with explicit index mappings, not embedded in the MSA. AF3's own pipeline is jackhmmer and
nhmmer only — **it dropped HHblits entirely**, which is why its database footprint is ~630 GB
unpacked against AlphaFold 2's 2.62 TB.

**AlphaFold 2 reads both**: `jackhmmer -A` Stockholm against UniRef90, MGnify and small BFD, and
`hhblits -oa3m` A3M against BFD plus UniRef30. `parsers.py` exposes `parse_stockholm`, `parse_a3m`
and `convert_stockholm_to_a3m`. **VERIFIED.**

**Boltz** takes `msa: <path>.a3m` in its YAML, or a two-column CSV (`sequence`, `key`) for
multi-chain, where *"Sequences with the same key are mutually aligned."* **Chai-1** takes
`.aligned.pqt`, *"similar to an a3m file, but has additional columns"*, and ships a3m conversion.
**ESMFold takes no MSA at all** — it is a single-sequence predictor. **VERIFIED.**

**So A3M is the lingua franca.** Only HMMER's branch speaks Stockholm, and the converter that ships
with HMMER cannot reach A3M.

## 4. Recommendation

Not a design — the options, and what each costs.

### What is already free

**`fasta.read_records` reads an aligned FASTA or an A3M today, byte for byte.** It is `FastaFile.read_iter`
with gzip, no alphabet anywhere. Only `read_proteins` refuses, because `-` is outside
`protein.seq.ALPHABET`. **VERIFIED** by reading `io/fasta.py`. The one caveat is that `read_iter`
silently swallows a leading `#` line rather than raising as `FastaFile.read` would.

So "read an MSA file" is already possible at the record layer. What is missing is anything that
understands the columns.

### Option A — `io/a3m.py`, record layer only

One module, mirroring `io/fasta.py`'s two-layer shape: read and write `(header, row)` pairs, plus two
pure functions between A3M and the rectangular form — `expand` (the `nins[j]` max-and-pad algorithm
from Part 1) and `compress` (lowercase where row 1 gaps, then drop insert-column gaps).

- **Costs:** one file, no dependency change, no new binary, no biotite version bump. Testable with a
  ten-line fixture and no network, which `tests/_guards.py` requires anyway.
- **Buys:** the only format AF3, Boltz, ColabFold and HHblits all speak. Aligned FASTA falls out of
  `expand`; nothing else is needed to hand an MSA to a folding tool.
- **Does not buy:** column arithmetic. Rows stay strings.

Adding a format is adding a file, which `io/__init__.py` says outright. This is the shape the repo
already has.

### Option B — hold biotite's `Alignment`

A frozen dataclass holding an `Alignment` plus the names — because `Alignment` has none — and
probably a match-column mask, because the trace does not record which columns were match states.
`Embedding` is the precedent for "a frozen dataclass is what one call returns".

- **Costs:** three fields that must stay in sync; an int64 trace of `m × n × 8` bytes; and
  **`Alignment.from_strings` is a 1.6.0 feature**, so this option carries the numpy downgrade. At
  1.4.0 it means constructing the trace by hand, which is either a Python double loop or our own
  numpy.
- **Buys:** `SequenceProfile.from_alignment`, `to_consensus`, `get_sequence_identity`,
  `remove_gaps` — real column arithmetic, for free, from a library already depended on.
- **The seam is clean when it exists:** `Alignment.from_strings(rows, to_protein_sequence)` passes
  every row through this package's one door and never touches biotite's converters.

The honest reading is that **Option B is worth taking only when something needs a profile or a
consensus**. Until then it is a class with no caller, and `search/mmseqs.py`'s precedent — a search
answers with a `DataFrame`, not with `Hit` objects — argues for strings.

### Option C — generation, and where the verb goes

`result2msa` and `unpackdb` are two more verbs on `MmseqsLikeTool`, beside `easy_search` and
`convertalis`. Both fit `run_to`'s freshness rule; both take the same ffindex paths the class
already handles; **no new binary**, because `mmseqs` is already in `REQUIRED_TOOLS`. `result2msa`
needs `search -a` rather than `easy-search`, which is the one genuinely new piece — `easy_search`
does not expose the intermediate databases.

**Direct support only decides the surface, and it decides it twice:**

- MMseqs2 takes a **sequence and a database** directly, so `Protein.msa(db)` is the exact parallel of
  `Protein.search(db)` — same argument, same tool, one more verb on `SearchMixin` or a sibling
  mixin. The module function beneath it is bulk-shaped, taking many proteins in one `result2msa`
  call, which is what `search()` already does badly and this could do well.
- Aligning a set you already have takes a **collection and no database**. No class in this package
  is a collection of proteins, so under the same rule it is a **module-level function, not a
  method** — and it needs a third binary.

### What the third binary actually costs

`REQUIRED_TOOLS == ("mmseqs", "foldseek")` is pinned by a test, and `doctor()` raises if any is
missing. **VERIFIED** from `tests/test_external.py`. Adding MAFFT or FAMSA therefore makes
`protein doctor` fail for everyone until they install a tool most of them will not use, or forces
`REQUIRED_TOOLS` to grow an optional tier it does not have. That is a real cost against the
restraint rule, paid by every user, for a use case nobody in this repo has asked for yet.

### The shape of the smallest honest answer

1. **`io/a3m.py`**, record layer plus `expand`/`compress`. Free, and it is the format everything
   downstream wants.
2. **`result2msa` and `unpackdb` on `MmseqsLikeTool`**, plus whatever exposes `search -a`. No new
   dependency.
3. **`Protein.msa(db)`** over a bulk module function, mirroring `search`.
4. **Grow `_BANNED_CONVERTERS`** to name `get_alignment`, `get_a3m_alignments` and `set_alignment`.
   That is a one-line diff and it is worth doing **whether or not any of the rest lands**, because
   the hole exists today.
5. **Decline** Stockholm, A2M, Clustal, MSF, PIR and PHYLIP until a consumer exists; decline
   `align_multiple`; decline the third binary.

Step 4 is the only one this note argues for unconditionally.

## Open items

- **Whether `Alignment` is held at all** is a question about `SequenceProfile`, not about MSAs. If
  nothing needs a profile or a consensus, rows of text are the honest representation and the numpy
  trade never comes up.
- **The biotite upgrade now has a reason.** The sibling note found nothing needing 1.6.0 or 1.7.1.
  A3M reading, `Alignment.from_strings` and Clustal I/O all sit behind it, and every build since
  1.5.0 carries conda-forge's `numpy <2.4` cap. Whether that is worth `numpy` 2.5.2 to 2.3.5 is a
  scoping question, not a technical one — and Option A needs none of it.
- **`_BANNED_CONVERTERS` has a hole today**, independent of everything else here.
- **A3M's invariant is not checked anywhere.** "Row 1 is all match states" is what makes A3M
  self-describing, and violating it silently changes the match-state count on any round trip. If
  `io/a3m.py` lands, that invariant is a test, not a comment.
