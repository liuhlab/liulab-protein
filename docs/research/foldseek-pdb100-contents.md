---
search:
  exclude: true
---

# What `foldseek databases PDB` stores, and whether a chain's coordinates come back out

Research note for [issue #4](https://github.com/liuhlab/liulab-protein/issues/4). Answers a
claim in [issue #1](https://github.com/liuhlab/liulab-protein/issues/1).

## Answer

`foldseek databases PDB` (pdb100) stores, per chain: the **observed** amino-acid sequence, the
3Di structural sequence, a text header, and **C-alpha coordinates only**. No backbone N/C/O,
no side chains, no ligands, no waters, no author residue numbering, no insertion codes, no
alternate locations, no B-factors, no occupancies.

A named chain's C-alpha trace **does** come back out, to within 0.001 A — the precision the
PDB file format itself writes — as a CA-only PDB file, via `foldseek createsubdb` then
`foldseek convert2pdb`. Every chain is present, not only the cluster representatives.

Issue #1's `pdb["1UBQ_A"]` returning a `Protein` "carrying that chain's structure file" is
**wrong as written**, on four counts, in rising order of how much they cost to fix:

1. **The key is not `1UBQ_A`.** It is `1ubq-assembly1_A`. pdb100 is built from **biological
   assembly** files, not from deposited asymmetric units, and the assembly number is part of
   the name. `1UBQ_A` matches nothing, and neither does `1ubq_A`.
2. **`pdb` is the wrong database to look in.** `1ubq-assembly1_A` is *absent* from the
   searchable `pdb100` index — ubiquitin chain A is not a 100%-identity cluster
   representative. Only `pdb100_seq` has it.
3. **What comes back is not "that chain's structure file".** It is a synthetic CA-only PDB
   renumbered from 1.
4. **Foldseek must be shelled out to twice, and it writes files.**

All-atom coordinates need a separate bulk source: RCSB's assembly mmCIF archive by rsync,
113.1 GiB, CC0 — see [Bulk coordinate sources](#bulk-coordinate-sources). The smallest
correct alternative to the `PDB` class as specified is in [Verdict](#verdict-on-issue-1).

## Method and provenance

| | |
| --- | --- |
| Binary exercised | foldseek `10.941cd33`, bioconda, the default pixi env on `GPU71FM` |
| Database inspected | the real pdb100 at `/scratch/zhoulab/hanliu/protein/db/pdb` on `GPU71FM`, downloaded under issue #6 |
| Source read | `github.com/steineggerlab/foldseek`, `master` and tag `10-941cd33`, fetched 2026-09-02 |

Everything below is marked **VERIFIED** (measured on the real database or the running binary)
or **INFERRED**. Line references to `master` are given where `master` and the `10-941cd33`
tag agree; where they differ it is said so.

**Where documentation and observation disagree, observation wins, and it did.** This note was
first drafted from the foldseek sources alone. Two claims drawn from them were wrong, both
from
[`util/update_webserver_pdb/single-script.sh`](https://github.com/steineggerlab/foldseek/blob/master/util/update_webserver_pdb/single-script.sh),
the build script the lab ships:

- The script rsyncs the **divided asymmetric-unit** mmCIF tree and names chains `1ubq_A`. The
  published database is built from **biological assemblies** and names them
  `1ubq-assembly1_A`. The checked-in script does not describe the pipeline that made the
  artefact.
- Reasoning from that script, this note said pdb100 holds asymmetric units and cannot
  represent assemblies. The truth is the exact reverse: it holds assemblies and cannot
  represent asymmetric units.

Treat that script as a sketch of the method, not a specification of the output.

## 1. The file set, and what each `.dbtype` marks

`foldseek databases PDB <name> tmp` downloads
`https://foldseek.steineggerlab.workers.dev/pdb100.tar.gz` — 2,326,827,389 bytes,
`last-modified: Fri, 19 Dec 2025` (HTTP `HEAD`, **VERIFIED**) — and unpacks it onto `<name>`
([`data/structdatabases.sh:144-152`](https://github.com/steineggerlab/foldseek/blob/10-941cd33/data/structdatabases.sh#L144-L152)).
`<name>.version` reads `250101 PDB_DATE`, so the snapshot is the PDB as of 2025-01-01 —
**VERIFIED** on the extracted database.

**On disk it is 4.3 GB, not 2.33 GB** — **VERIFIED** by `du -sh`. The advertised figure is the
compressed tarball.

There are **two parallel databases**, and this is the single most important structural fact
about pdb100. Sizes below are from `ls -l` on the real extraction — **VERIFIED**:

| | Representative tier | Full tier |
| --- | --- | --- |
| Amino acids | `pdb100` (73.5 MB) | `pdb100_seq.0` -> `pdb100`, `pdb100_seq.1` (343.0 MB) |
| 3Di | `pdb100_ss` (73.5 MB) | `pdb100_seq_ss.0` -> `pdb100_ss`, `.1` (343.0 MB) |
| C-alpha | `pdb100_ca` (459.8 MB) | `pdb100_seq_ca.0` -> `pdb100_ca`, `.1` (2,094.1 MB) |
| Headers | `pdb100_h` (30.7 MB) | `pdb100_seq_h.0` -> `pdb100_h`, `.1` (119.4 MB) |
| Entries | **324,204** | **1,562,678** |

The full tier is a **split** ffindex database: `.0` is a symlink to the representative data
file and `.1` holds everything else, with one index over both. Shared between the tiers:
`pdb100.lookup` (50.3 MB, 1,562,678 lines), `pdb100.source` (325,897 lines),
`pdb100_mapping`, `pdb100_taxonomy` (741.3 MB), and the cluster database `pdb100_clu`
(324,204 entries). `pdb100_seq.lookup`, `pdb100_seq.source`, `pdb100_seq_mapping` and
`pdb100_seq_taxonomy` are symlinks to those.

`.dbtype` values, read off the real database — **VERIFIED**:

| File | `.dbtype` int32 | Constant |
| --- | --- | --- |
| `pdb100`, `pdb100_ss`, `pdb100_seq` | `0` | `DBTYPE_AMINO_ACIDS` |
| `pdb100_h` | `12` | `DBTYPE_GENERIC_DB` |
| `pdb100_ca`, `pdb100_seq_ca` | `101` | `DBTYPE_CA_ALPHA` |
| `pdb100_clu` | `6` | `DBTYPE_CLUSTER_RES` |

Names from
[`lib/mmseqs/src/commons/Parameters.h:69-89`](https://github.com/steineggerlab/foldseek/blob/master/lib/mmseqs/src/commons/Parameters.h#L69-L89)
and
[`src/commons/LocalParameters.cpp:6`](https://github.com/steineggerlab/foldseek/blob/10-941cd33/src/commons/LocalParameters.cpp#L6)
(`const int LocalParameters::DBTYPE_CA_ALPHA = 101;`).

**The `.dbtype` does not tell you which database is the 3Di one.** `pdb100` and `pdb100_ss`
both carry `0`, and are byte-for-byte the same size because 3Di is one character per residue.
Only the `_ss` suffix distinguishes them. A `Database` class must not sniff `.dbtype` to
decide what it is holding.

There is no `_id` database, so original residue numbers are stored nowhere. (`master` added
an optional one — `idbw` at
[`structcreatedb.cpp:363-366`](https://github.com/steineggerlab/foldseek/blob/master/src/strucclustutils/structcreatedb.cpp#L363-L366)
— but the published pdb100 predates it.) There is no `_fcz` foldcomp database either.

## 2. Coordinates: C-alpha only, and how lossy

### Only C-alpha is ever written

`createdb` reads full structures with gemmi and writes one coordinate array per chain:
C-alpha.
[`structcreatedb.cpp:404-433`](https://github.com/steineggerlab/foldseek/blob/master/src/strucclustutils/structcreatedb.cpp#L404-L433)
holds the function's only coordinate writer, `cadbw`, reached by two branches that both write
C-alpha and differ only in encoding. N, C and C-beta are read only to derive the 3Di states
([`:321-325`](https://github.com/steineggerlab/foldseek/blob/master/src/strucclustutils/structcreatedb.cpp#L321-L325))
and are then discarded. If the input is itself CA-only, PULCHRA rebuilds a backbone in memory
purely to get 3Di, and that too is discarded
([`:305-320`](https://github.com/steineggerlab/foldseek/blob/master/src/strucclustutils/structcreatedb.cpp#L305-L320)).
**VERIFIED** by reading the source, corroborated by the README's memory formula, which prices
a residue at `6 bytes Ca + 1 3Di byte + 1 AA byte`
([README, Memory requirements](https://github.com/steineggerlab/foldseek#memory-requirements))
— six bytes is three int16s, not three float32s, and by the emitted PDB files, which contain
`CA` atoms and nothing else.

### The encoding

Default is `--coord-store-mode 2`, "C-alpha as difference (uint16_t)" (`foldseek createdb -h`
on 10.941cd33 — **VERIFIED**). Layout, from
[`src/commons/Coordinate16.h`](https://github.com/steineggerlab/foldseek/blob/10-941cd33/src/commons/Coordinate16.h):
per axis, one `int32` start in thousandths of an angstrom followed by `n-1` `int16` deltas.
The three axes are stored consecutively, x then y then z. Encoding truncates —
`int32_t last = (int)(data[0] * 1000)` (`:61`); decoding divides by `1000.0f` (`:34`).

Entry size is `(n-1)*3*2 + 3*4 + 1` bytes plus the writer's NUL. Measured: a 141-residue
chain occupied 854 bytes, a 76-residue chain 464 — both exact — **VERIFIED**.

**It is lossy, by 0.001 A.** Round-tripping 1UBQ chain A through `createdb` and `convert2pdb`
and diffing against `_atom_site.Cartn_*` in the deposited mmCIF gave
`max abs coord diff = 0.001000 A over 76 rows` — **VERIFIED** on GPU71FM. That is the
truncation quantum and no worse: deltas are computed between already-quantised values, so
error does not accumulate along a chain
([`Coordinate16.h:60-73`](https://github.com/steineggerlab/foldseek/blob/10-941cd33/src/commons/Coordinate16.h#L60-L73)).
The legacy PDB format writes three decimals, so this is the precision a `.pdb` file carries
anyway. The same coordinates came back from the real pdb100 — **VERIFIED**.

Two things put `float32` in a `_ca` database instead, so a reader must decide by entry length
as foldseek does, not by assuming the compressed layout:

- A consecutive delta over 32.767 A, which a chain break can produce, drops that chain to
  plain `float32`, 12 bytes per residue
  ([`structcreatedb.cpp:405-420`](https://github.com/steineggerlab/foldseek/blob/master/src/strucclustutils/structcreatedb.cpp#L405-L420));
  the reader detects it by entry length
  ([`Coordinate16.h:16-18`](https://github.com/steineggerlab/foldseek/blob/10-941cd33/src/commons/Coordinate16.h#L16-L18)).
- `foldseek compressca <DB> <caDB> --coord-store-mode 1` re-encodes to float32 — **VERIFIED**:
  entry sizes went 854 to 1693 bytes. It recovers nothing; the 0.001 A was lost at `createdb`
  time.

## 3. Getting one chain out

Yes, in two commands. `foldseek convert2pdb` is the only structure emitter:

```console
$ foldseek convert2pdb -h
usage: foldseek convert2pdb <i:Db> <o:pdbFile|pdbDir> [options]
 --pdb-output-mode INT   PDB output mode:
                         0: Single multi-model PDB file
                         1: One PDB file per chain
                         2: One PDB file per complex [0]
```

Run against a whole database it writes one file per entry, so extract first. This recipe was
run end to end against the **real pdb100** on GPU71FM — **VERIFIED**:

```sh
DB=/scratch/zhoulab/hanliu/protein/db/pdb/pdb100

# 1. name -> integer key. Note pdb100_seq, not pdb100: see the trap below.
KEY=$(awk -F'\t' '$2=="1ubq-assembly1_A"{print $1; exit}' "$DB.lookup")   # 1190456

# 2. one-entry sub-database. --subdb-mode 1 soft-links the data, so this is O(1)
#    in the size of pdb100 and copies nothing.
printf '%s\n' "$KEY" > keys.txt
foldseek createsubdb keys.txt "${DB}_seq" one --subdb-mode 1

# 3. convert2pdb needs .lookup, and needs the header DATA files, which
#    createsubdb does not link when the source header db is split (see below)
ln -sf "$DB.lookup"      one.lookup
ln -sf "${DB}_h"         one_h.0
ln -sf "${DB}_seq_h.1"   one_h.1

# 4. emit
foldseek convert2pdb one outdir --pdb-output-mode 1
# -> outdir/1ubq-assembly1_A.pdb
```

### Three traps, all hit while doing this

**Use `pdb100_seq`, not `pdb100`.** `1ubq-assembly1_A` resolves through the shared lookup to
key 1190456, and that key is **absent** from `pdb100.index` and `pdb100_ca.index`, present in
`pdb100_seq.index` and `pdb100_seq_ca.index` — **VERIFIED**. The representative tier covers
324,204 of 1,562,678 named chains, 20.7%. A lookup that succeeds on the name and then reads
the representative database fails for four chains in five, and ubiquitin is one of them. This
is exactly the confusion in
[foldseek issue #258](https://github.com/steineggerlab/foldseek/issues/258).

**`createsubdb` does not link a split header database.** It links `one_h.dbtype` and
`one_h.index` but not `one_h.0` / `one_h.1`, and `convert2pdb` then dies with
`No datafile could be found for .../one_h!` — **VERIFIED**. Link the two data files by hand,
as above.

**Do not use `--id-mode 1`.** Its help says "Select DB entries based on 1: FASTA identifiers
(.lookup)", which is exactly what is wanted, and in 10.941cd33 it is broken: foldseek passes
one parameter string to all three sub-database stages, so `--id-mode 1` leaks to the `_ss`
stage, which has no `.lookup`, and the workflow dies with `Cannot open lookup file ...` —
**VERIFIED**. `master` fixed it by resetting `par.dbIdMode = 0` for the second stage
([`createstructsubdb.cpp:25-27`](https://github.com/steineggerlab/foldseek/blob/master/src/strucclustutils/createstructsubdb.cpp#L25-L27));
the tag has one shared `CREATESTRUCTSUBDB_PAR`
([tag](https://github.com/steineggerlab/foldseek/blob/10-941cd33/src/strucclustutils/createstructsubdb.cpp#L25)).
Resolve the name to a key from `.lookup` yourself.

### What the emitted file actually is

Real output from the real pdb100 — **VERIFIED**:

```text
TITLE     1ubq-assembly1_A STRUCTURE OF UBIQUITIN REFINED AT 1.8 ANGSTROMS RESOL
TITLE    2UTION
ATOM      1  CA  MET A   1      26.266  25.413   2.842
ATOM      2  CA  GLN A   2      26.850  29.021   3.898
...
ATOM     76  CA  GLY A  76      40.373  39.813  33.944
```

78 lines: two `TITLE`, 76 `ATOM`, every one named `CA`. No `TER`, no `END`, no occupancy, no
B-factor, no element column — the writer is one `fprintf`
([`convert2pdb.cpp:211`](https://github.com/steineggerlab/foldseek/blob/10-941cd33/src/strucclustutils/convert2pdb.cpp#L211)):

```c
fprintf(threadHandle, "ATOM  %5d  CA  %s %c%4d    %8.3f%8.3f%8.3f\n",
        (int)(j + 1), aa3, chainName[0], int(j + 1), ...);
```

Read that format string for the three defects it bakes in:

- **Residues are renumbered from 1.** Both the atom serial and the residue number are the loop
  index; author numbering is not stored and cannot be recovered. Demonstrated on 6M0J chain E:
  deposited `auth_seq_id` runs 333 to 526 across 195 CA records, and foldseek emits 194 atoms
  numbered 1 to 194, having dropped one alternate location — **VERIFIED**. Gaps in the model
  close up silently, so sequence position is not model position.
- **`chainName[0]`.** The PDB chain column holds one character, so a multi-character chain
  label is truncated. **180,077 of 1,562,678 pdb100 chain labels are longer than one
  character** — **VERIFIED** — for example `6dwu-assembly44_DU`. Nearly 12% of the database
  cannot round-trip its own chain identifier through this writer.
- **`aa3` is a 26-entry table** indexed by the one-letter code
  ([`convert2pdb.cpp:127`](https://github.com/steineggerlab/foldseek/blob/10-941cd33/src/strucclustutils/convert2pdb.cpp#L127)),
  so `X` becomes `XAA` rather than the real modified-residue name.

`master` has grown `--pdb-output-format` for mmCIF output and a per-chain mmCIF split. That is
**not** in 10.941cd33 — the installed binary's help lists `--pdb-output-mode` only.

### The other verbs the ticket asked about

`foldseek -h` prints 55 lines, but foldseek inherits the MMseqs2 command set and much of it is
callable while unlisted: `convert2fasta`, `prefixid`, `lndb`, `mvdb` and `createtsv` all
answer `-h` on 10.941cd33 — **VERIFIED**. Do not conclude from `foldseek -h` that a verb is
missing.

- **`convert2fasta`** exists and is the efficient bulk reader, but only for sequence.
  `foldseek convert2fasta pdb100_seq out.fasta` writes `>1ubq-assembly1_A <TITLE>` plus the
  observed amino-acid sequence for every chain in one pass — **VERIFIED**. Pointed at
  `pdb100_seq_ss` it emits 3Di instead, after `foldseek lndb pdb100_seq_h pdb100_seq_ss_h`
  supplies the header database it complains about — **VERIFIED**. No coordinates either way.
- **`structureto3didescriptor`** runs the opposite direction: it reads PDB/mmCIF files and
  writes a four-column TSV of name, amino-acid sequence and 3Di — **VERIFIED**. It consumes
  structures and cannot produce one.
- **`createsubdb`** — covered above.

## 4. How chains are keyed

### The key is `1ubq-assembly1_A`

`pdb100.lookup` is `key<TAB>name<TAB>file-number`. Verbatim first and last lines from the real
file — **VERIFIED**:

```text
0	200l-assembly1_A	0
1	101m-assembly1_A	1
2	201l-assembly1_A	2
3	201l-assembly2_B	3
...
1562676	8zz0-assembly1_F	325896
1562677	8zz0-assembly1_G	325896
```

The name has three parts and every one of them will trip a naive parser:

**Entry id, lowercase.** 224,856 distinct ids — **VERIFIED**. `1UBQ` never appears.

**Assembly number, and it is part of the identity.** `201l` appears as both
`201l-assembly1_A` and `201l-assembly2_B`; the maximum seen is `6dwu-assembly44_DU` —
**VERIFIED**. `pdb100.source` names 325,897 assembly files against 224,856 entries, so an
entry averages 1.45 assemblies. "The chain A of entry 1abc" is not a unique address in pdb100
unless the assembly is named too.

**Chain label, with an optional `-<N>` symmetry-copy suffix.** 641,030 of 1,562,678 names —
41% — end in `-<N>` — **VERIFIED**. `2ggg-assembly1_C`, `_C-2`, `_C-3` and `_C-4` are four
distinct keys pointing at the same source file. **Splitting on the last `-` therefore gives
the wrong chain.** Split on the last `_`, then strip a trailing `-<digits>` if the remainder
is to be read as a chain.

The suffix is not foldseek's invention. RCSB's assembly mmCIF for 2GGG carries
`label_asym_id` values `A A-2 A-3 A-4 C C-2 C-3 C-4` with `auth_asym_id` set to `?`
throughout — **VERIFIED** by downloading and parsing that file. Foldseek copies the label it
finds. So the pdb100 chain label is the assembly file's label: the author chain id for the
first copy, and `<chain>-<N>` for the Nth symmetry copy.

That the base label follows the author chain id, not `label_asym_id`, is confirmed by 6M0J,
which is `6m0j-assembly1_A` and `6m0j-assembly1_E` — **VERIFIED**; `E` is 6M0J's author chain
for the RBD, whose `label_asym_id` is different. 4HHB is `4hhb-assembly1_A` through `_D`.

### "pdb100" means 100% identity, not 100k entries

The lab clusters at 100% sequence identity with 95% coverage and keeps the representatives
([`single-script.sh:51-57`](https://github.com/steineggerlab/foldseek/blob/master/util/update_webserver_pdb/single-script.sh#L51-L57)):

```sh
foldseek cluster pdb_seq pdb_clu tmp -c 0.95 --min-seq-id 1.0 --cov-mode 0
foldseek createsubdb pdb_clu pdb_seq pdb --subdb-mode 1
```

So the "100" is redundancy removal at identity 1.0. The counts bear it out: 1,562,678 chains
cluster to 324,204 representatives — **VERIFIED**.

### Which number answers "how many entries"

Three different numbers, and they answer three different questions — all **VERIFIED**:

| Number | What it counts | Where |
| --- | --- | --- |
| 224,856 | PDB entries | distinct ids in `pdb100.lookup` |
| 325,897 | biological assemblies | lines in `pdb100.source` |
| **1,562,678** | **chains — every key you can name** | lines in `pdb100.lookup`, `pdb100_seq.index` |
| **324,204** | **chains you can search** | lines in `pdb100.index`, `pdb100_ca.index`, `pdb100_clu.index` |

A `Database.status()` should report the last two as a pair. Reporting either alone is
misleading: 1,562,678 overstates what `foldseek search` will match against, and 324,204
understates what can be retrieved.

## Verdict on issue #1

> `PDB` understands PDB ids, chains and the asymmetric unit, so `pdb["1UBQ_A"]` returns a
> `Protein` carrying that chain's structure file.

Unbuildable as written, and the phrase to strike hardest is "the asymmetric unit" — pdb100 is
the one thing it is not. pdb100 holds biological assemblies. The asymmetric unit is what is
**absent**, and no amount of parsing recovers it.

The smallest correct alternative, in rising order of cost:

1. **`pdb["1ubq-assembly1_A"]` returns a `Protein` with sequence and metadata, and no
   structure.** Read `pdb100_seq` for the sequence and `pdb100_seq_h` / `pdb100_mapping` for
   the title and taxid. The ffindex format is a three-column TSV index over a NUL-delimited
   data file, so a read-only Python reader needs no subprocess; `convert2fasta` is the bulk
   alternative. Nothing about this object is a lie.
2. **Give the class a real key type, not a string convention.** A `(entry, assembly, chain,
   copy)` tuple that renders to the foldseek name and parses back. Anything that accepts
   `"1UBQ_A"` has to guess an assembly, and a guess of `assembly1` is wrong whenever an entry
   has several. Prefer failing loudly with the candidate names — a `pdb.chains("1ubq")` that
   lists what exists is more useful than an accessor that silently picks one.
3. **Add `.ca_structure()`, returning a path to a CA-only PDB.** The recipe in section 3. Name
   it so no caller reads it as the deposited structure, and document the renumbering. It is
   enough for `foldseek_search`, TM-align, and everything foldseek itself does.
4. **All-atom `.structure` is backed by the RCSB assembly mirror, not by pdb100.** A separate
   `Database`. See below.

Two more things worth writing into the class:

- pdb100's sequence is the **observed** sequence, not SEQRES. A residue with no modelled
  coordinates is simply absent from the string, so pdb100 sequence indices do not correspond
  to UniProt indices.
- The searchable tier holds 20.7% of the named chains. `search` results and `__getitem__`
  therefore draw on different databases, and the class should be explicit about which.

## Bulk coordinate sources

Priced against live servers on 2026-09-02. Only one family carries all-atom coordinates for an
arbitrary PDB chain.

### Recommendation: rsync the RCSB assembly mmCIF archive

```sh
rsync -rlpt -v -z --delete --port=33444 \
  rsync.rcsb.org::ftp_data/assemblies/mmCIF/divided/ /path/to/assemblies
```

| | |
| --- | --- |
| Size | **113.1 GiB**, 368,372 `-assemblyN.cif.gz` files |
| Layout | `divided/<middle two chars>/<lowercase id>-assembly<N>.cif.gz` |
| Contents | all chains, all atoms, HETATM, ligands, B-factors, symmetry copies expanded |
| Licence | **CC0 1.0**, per [rcsb.org/pages/policies](https://www.rcsb.org/pages/policies) |

Size and file count measured directly by a recursive `rsync --list-only` and summed —
**VERIFIED**. `1ubq-assembly1.cif.gz` and `2ggg-assembly{1,2}.cif.gz` were confirmed present
at those paths — **VERIFIED**.

**This is the archive that keys 1:1 with pdb100.** Every line of `pdb100.source` is one of
these filenames, so `1ubq-assembly1_A` maps to
`divided/ub/1ubq-assembly1.cif.gz` plus label `A` with no translation and no network call.
That correspondence is why this beats the deposited-structure archive for this project, even
though it is larger.

### The alternative: the deposited asymmetric-unit archive

```sh
rsync -rlpt -v -z --delete --port=33444 \
  rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ /path/to/mmCIF
```

Verbatim from [wwpdb.org/ftp/pdb-ftp-sites](https://www.wwpdb.org/ftp/pdb-ftp-sites) —
**VERIFIED**. **85.1 GiB**, 259,016 `.cif.gz` files, layout `mmCIF/ub/1ubq.cif.gz`, weekly
Wednesday refresh from 00:00 UTC, CC0. `mmCIF/ub/1ubq.cif.gz` is byte-identical to
`https://files.rcsb.org/download/1UBQ.cif.gz` — **VERIFIED** — so the mirror is an exact
substitute for the per-ID REST calls this project rules out, not an approximation.

Take this one if `Protein` should mean the deposited chain, keyed by entry plus
`auth_asym_id`. It does **not** key compatibly with pdb100: mapping an assembly chain label
back to an author chain needs the assembly file anyway, so a project that wants both ends up
mirroring both, at 198 GiB.

Two traps on either archive, both **VERIFIED** independently here:

- **Use port 33444.** The rsync default 873 times out at RCSB —
  `connect timeout: 132.249.213.18`.
- **Budget the measured size, not the published one.** RCSB's [File Download
  Services](https://www.rcsb.org/docs/programmatic-access/file-download-services) page still
  links [`rsyncPDB.sh`](https://files.wwpdb.org/pub/pdb/software/rsyncPDB.sh), which claims
  the mmCIF coordinates are "Aproximately 24 GB". That script is stamped 2014 and the archive
  has grown 3.5x since.

Mirrors on the standard port 873, if RCSB is slow from the cluster:
`rsync.ebi.ac.uk::pub/databases/pdb/data/...` (PDBe) and `ftp.pdbj.org::ftp_data/...` (PDBj).
FTP itself has been deprecated across the archive since 2024-11-01.

Worth taking alongside either: `derived_data/pdb_seqres.txt.gz`, 66.8 MB — the SEQRES sequence
of every chain, which is what pdb100's observed sequences should be checked against.

### Foldcomp: no PDB set exists, and the format cannot hold one

Three independent disqualifiers, all **VERIFIED**:

1. Every published set is predicted structures — `afdb_uniprot_v4` (1.10 TB),
   `highquality_clust30` (120.4 GB), `afdb_swissprot_v4` (3.05 GB), `h_sapiens` (234 MB),
   `e_coli` (21.8 MB). No experimental-structure set exists.
2. It could not be built either. The [README](https://github.com/steineggerlab/foldcomp):
   "Foldcomp currently only supports compression of single chain PDB files". The
   [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10085514) (Kim, Mirdita, Steinegger,
   *Bioinformatics* 39(4) btad153, 2023) adds "without missing residues". Experimental entries
   are routinely multi-chain and almost always have disordered gaps.
3. The paper is explicit that it discards metadata: "Foldcomp is not meant to replace the
   PDB/mmCIF format, since these contain valuable meta-information that is discarded by
   Foldcomp."

Worth recording because it is a common misconception: foldcomp is **not** backbone-only. It
stores side chains at 13 bytes per residue, with reconstruction error 0.08 A backbone and
0.14 A all-atom. The accuracy is fine; the data does not exist in this format and the format
cannot represent it. Licence MIT.

### No foldseek database keeps more than C-alpha

`--coord-store-mode` offers `1: C-alpha as float`, `2: C-alpha as difference (uint16_t)` and,
in `compressca`, `3: Plain text list of floats`. **Every mode is C-alpha** — so not even a
local build can produce an all-atom foldseek database. The `--write-foldcomp` expert flag
exists ([`LocalParameters.cpp:21`](https://github.com/steineggerlab/foldseek/blob/master/src/commons/LocalParameters.cpp#L21),
default `0` at
[`:392`](https://github.com/steineggerlab/foldseek/blob/master/src/commons/LocalParameters.cpp#L392))
and writes a side `_fcz` database, but it inherits every foldcomp limit above, and no
published database uses it — the extracted pdb100 has no `_fcz` member. The AlphaFold foldseek
databases are the same shape: `afdb50.tar.gz` 122.6 GB, `afdb_swissprot.tar.gz` 1.56 GB,
`cath50.tar.gz` 1.01 GB, all C-alpha. **VERIFIED**. Foldseek is GPL-3.0.

### BinaryCIF has no bulk distribution

**VERIFIED negative.** `data/structures/divided/` on the rsync server lists `XML-extatom,
XML-noatom, XML, mmCIF, nmr_chemical_shifts, nmr_data, nmr_restraints, nmr_restraints_v2, pdb,
structure_factors` — no `bcif`. RCSB serves BinaryCIF only per entry from
`https://models.rcsb.org/<id>.bcif.gz`, the per-ID pattern this project rules out. For the
record `1ubq.bcif.gz` is 21,254 bytes against `1ubq.cif.gz` at 26,915, so a bulk bcif archive
would be near 67 GiB (**INFERRED** by scaling) — but none is published.

### RCSB sequence clusters are not a coordinate source

`https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{30,...,100}.txt`, about
22.5 MB each, are entity-id lists with no coordinates — **VERIFIED**. They are a redundancy
filter to apply over a real source, and the natural way to reproduce something pdb100-like
against a local mirror.

### Currency

pdb100's version file says `250101 PDB_DATE` and its files are dated 2025-07-25, so it is
rebuilt occasionally, not weekly. The RCSB archives refresh every Wednesday. A weekly mirror
will hold entries pdb100 has never seen, and `search` will silently not find them.

## Open items

- The `.source` file numbers in `pdb100.lookup` point into `pdb100.source`, giving a direct
  chain-to-assembly-file map. Nothing here needed it, but a `PDB` class joining pdb100 to a
  local assembly mirror should use it rather than re-deriving the filename from the key.
- Whether to read ffindex directly in Python or shell out per lookup. The format is trivial
  and these databases are immutable, so a read-only reader carries no consistency risk. A
  design question, not a research one.
