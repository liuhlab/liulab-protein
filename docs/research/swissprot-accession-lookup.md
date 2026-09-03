---
search:
  exclude: true
---

# Pulling one Swiss-Prot entry by accession from a local MMseqs2 database

Research note for [#3](https://github.com/liuhlab/liulab-protein/issues/3). Written
2026-09-02.

## Answer

**Yes.** `swissprot["P12345"]` is a local, offline lookup, and MMseqs2 ships a verb built
for exactly it:

```sh
mmseqs view swissprot --id-list P12345 --id-mode 1                      # sequence
mmseqs view swissprot --id-list P12345 --id-mode 1 --idx-entry-type 2   # ...fails, see below
```

The header — entry name, description, organism, gene — is in the parallel `_h` database and
comes back too, either by resolving the numeric key first or by `createsubdb` +
`convert2fasta`. No network call, no index build, no `mmseqs createindex`.

Two things found along the way are load-bearing and were **not** anticipated by
[#1](https://github.com/liuhlab/liulab-protein/issues/1):

1. `mmseqs databases` builds the database with `createdb --gpu 1`, which stores sequences
   **numerically encoded, and lossily**: `U`, `O` and `X` all become `X`; `B` becomes `D`;
   `Z` becomes `E`; `J` becomes `L`. A `Protein` built from this database can never carry
   selenocysteine or pyrrolysine. This bears directly on `seq.py`'s `AMBIGUOUS` set.
2. `0.094 GB` is the size of the compressed source FASTA, not of the database on disk.

## How this was checked

| Source | What |
| --- | --- |
| Wiki | <https://github.com/soedinglab/MMseqs2/wiki>, commit `beea65f`, cloned from `MMseqs2.wiki.git` |
| Source | <https://github.com/soedinglab/MMseqs2>, tag `18-8cc5c` (commit `8cc5ce3`), the tag matching the installed binary |
| Binary | `mmseqs` 18.8cc5c, the conda dependency in this repo's default pixi environment, on `GPU71FM` |

Everything under **VERIFIED** was run against that binary on `GPU71FM` on a four-entry
synthetic FASTA with real UniProt-style headers (`>sp|P12345|AATM_RABIT ... OS=... OX=...
GN=GOT2 PE=1 SV=2`). **No database was downloaded.** Probe scripts and outputs are at
`/scratch/zhoulab/hanliu/protein/.research-swissprot-probe` on `GPU71FM`.

Anything marked **INFERRED** is arithmetic or reading, not a measurement, and is flagged
where it appears.

## 1. What `mmseqs databases UniProtKB/Swiss-Prot` writes

The recipe downloads `uniprot_sprot.fasta.gz` from ExPASy and hands it to `createdb`
([`databases.sh:118-125`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/data/workflow/databases.sh#L118-L125)),
then — because Swiss-Prot is declared `hasTaxonomy = true`
([`Databases.cpp:48-53`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/workflow/Databases.cpp#L48-L53)) —
builds the taxonomy sidecars and finally moves the release note into `<name>.version`
([`databases.sh:483-499`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/data/workflow/databases.sh#L483-L499)).

VERIFIED (file set from `createdb --gpu 1`, byte sizes from the four-entry probe):

| File | Holds |
| --- | --- |
| `<name>` | the sequence data: entries concatenated, addressed only through `.index` |
| `<name>.index` | `key \t offset \t length`, one line per entry, tab separated |
| `<name>.dbtype` | 4 bytes, little-endian. `00 00 08 00` here: amino acid (`0`) with the GPU-extended bit (`8 << 16`) |
| `<name>.lookup` | `key \t accession \t fileNumber`, one line per entry |
| `<name>.source` | `fileNumber \t filename`, e.g. `0\tmini.fasta` — names the input files |
| `<name>_h` | the FASTA headers, `>` stripped, each terminated `\n\0` |
| `<name>_h.index` | the same three columns, **same key space** as `<name>.index` |
| `<name>_h.dbtype` | `0c 00 00 00` = 12 = `DBTYPE_GENERIC_DB` |

INFERRED, from the shell script only (not run, because that means downloading):
`mmseqs databases` additionally writes `<name>.version`, `<name>_mapping` (accession → NCBI
taxon id, parsed out of the `OX=` field of each header) and `<name>_taxonomy`
([`databases.sh:483-499`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/data/workflow/databases.sh#L483-L499),
suffix table at
[`DBReader.cpp:1179-1183`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/commons/DBReader.cpp#L1179-L1183)).

The wiki documents the same layout at
[Sequence database format](https://github.com/soedinglab/MMseqs2/wiki#sequence-database-format)
and
[MMseqs2 database format](https://github.com/soedinglab/MMseqs2/wiki#mmseqs2-database-format).

### The offset arithmetic

`.index` length **includes the trailing newline and null byte**; the payload is
`length - 2` bytes at `offset`. The wiki states this
([Sequence database format](https://github.com/soedinglab/MMseqs2/wiki#sequence-database-format):
"The real sequence length is two characters shorter (`$3 - 2`)"), and the writer confirms it
([`createdb.cpp:279-285`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/createdb.cpp#L279-L285)).

VERIFIED on the probe. Header database, plain `dd`, no MMseqs2 involved:

```text
$ awk -F'\t' '$2=="P12345"{print $1}' gpudb.lookup        # -> 3
$ awk -F'\t' '$1==3{print $2, $3}' gpudb_h.index          # -> 290 115
$ dd if=gpudb_h bs=1 skip=290 count=113
sp|P12345|AATM_RABIT Aspartate aminotransferase, mitochondrial OS=Oryctolagus cuniculus OX=9986 GN=GOT2 PE=1 SV=2
```

The same three-line recipe against the **sequence** database returns bytes, not letters. See
section 7.

## 2. The keys are opaque numeric ids; `.lookup` is the map

VERIFIED. `createdb` assigns each entry a numeric id and **shuffles** the database, so the
key has no relation to input order and none at all to the accession. The four-entry probe,
same input in both cases:

```text
plaindb.lookup            gpudb.lookup
0  P12345  0              0  Q6GZX4  0
1  Q6GZX4  0              1  P0A031  0
2  P0A031  0              2  Q197F8  0
3  Q197F8  0              3  P12345  0
```

Column 2 **is** the UniProt accession. `createdb` fills it with
`Util::parseFastaHeader`
([`createdb.cpp:258-259`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/createdb.cpp#L258-L259)),
which recognises the `sp|` prefix and returns the text between the two pipes
([`Util.cpp:117-190`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/commons/Util.cpp#L117-L190), `sp|` at [line 136](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/commons/Util.cpp#L136)) —
so `>sp|P12345|AATM_RABIT ...` yields exactly `P12345`, **not** `AATM_RABIT` and not the
whole `sp|P12345|AATM_RABIT`. `tr|` (TrEMBL) is in the same table; Swiss-Prot is all `sp|`.
The wiki names the same file
([Create a seqTaxDB by manual annotation](https://github.com/soedinglab/MMseqs2/wiki#create-a-seqtaxdb-by-manual-annotation-of-a-sequence-database):
"a tab-separated `sequenceDB.lookup` file that contains numeric-db-id, Accession … and
File") and lists the recognised header prefixes under
[Identifier parsing](https://github.com/soedinglab/MMseqs2/wiki#identifier-parsing).

So "P12345" → key is one hop through `.lookup`. MMseqs2 does that hop for you behind
`--id-mode 1`; the package can equally do it in Python.

VERIFIED: `<name>_h` has **no `.lookup` of its own**. `--id-mode 1` against the header
database fails with `Cannot open lookup file gpudb_h.lookup!`
([`DBReader.cpp:138-144`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/commons/DBReader.cpp#L138-L144)).
The header database is addressed by numeric key only.

## 3. The verb is `mmseqs view`

`getdbkeys` **does not exist** — VERIFIED both ways: no such symbol in the source, and
`mmseqs getdbkeys --help` fails on the binary. The candidates that do exist:

| Verb | Fits? |
| --- | --- |
| **`view`** | **yes** — prints named entries to stdout, accepts accessions |
| `createsubdb` | yes, but writes a database, not text; use when you want a real sub-database |
| `convert2fasta` | whole-database export only; no id selection |
| `prefixid` | prefixes every entry with its key; whole-database, not a selector |
| direct `.index` read | works for headers; needs a decode step for sequences (section 7) |

`view` is registered as "Print DB entries given in `--id-list` to stdout"
([`MMseqsBase.cpp:846-852`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/MMseqsBase.cpp#L846-L852))
and takes `--id-list`, `--id-mode` and `--idx-entry-type`
([`Parameters.cpp:1214-1218`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/commons/Parameters.cpp#L1214-L1218)).
`--id-mode 1` resolves each name through the reverse lookup
([`view.cpp:25-42`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/view.cpp#L25-L42)).

### The exact command lines

VERIFIED, all four:

```sh
# sequence, by accession — one entry
mmseqs view swissprot --id-list P12345 --id-mode 1

# sequences, by accession — many at once, one call
mmseqs view swissprot --id-list P12345,Q6GZX4 --id-mode 1

# header, by NUMERIC KEY (--id-mode 1 does not work with --idx-entry-type 2)
mmseqs view swissprot --id-list 3 --idx-entry-type 2

# a real one-entry sub-database, by accession
mmseqs createsubdb <(printf 'P12345\n') swissprot P12345db --id-mode 1
mmseqs convert2fasta P12345db P12345.fasta
```

A missing accession is a **warning on stderr and a skipped entry**, not a non-zero exit:
`Could not find NOSUCH1 in lookup`
([`view.cpp:34-38`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/view.cpp#L34-L38)).
VERIFIED. A `SwissProt.__getitem__` that must raise `KeyError` cannot rely on the exit
code — it has to notice the empty output.

The `--idx-entry-type 2` restriction is a real limitation, VERIFIED and explained by the
source: `view` calls `getLookupIdByAccession` on whichever reader `--idx-entry-type`
selected, and with `2` that is the header reader, which has no `.lookup`
([`view.cpp:8-30`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/view.cpp#L8-L30)).
Resolve the accession from `.lookup` first, then ask for the header by key.

`createsubdb --id-mode 1` uses the same mechanism
([`createsubdb.cpp:26-56`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/createsubdb.cpp#L26-L56))
and is documented on the wiki under
[Manipulating databases](https://github.com/soedinglab/MMseqs2/wiki#manipulating-databases).

## 4. The full UniProt header comes back

VERIFIED. `createsubdb` symlinks the ancillary files of the parent — including `_h`, `_h.index`
and `.lookup` — into the new database
([`createsubdb.cpp:91`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/createsubdb.cpp#L91),
`DBFiles::SEQUENCE_ANCILLARY` at
[`DBReader.h:44-48`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/commons/DBReader.h#L44-L48)).
`convert2fasta` then walks the sub-database's one key and pulls that key's header out of the
symlinked full header database
([`convert2fasta.cpp:41-55`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/convert2fasta.cpp#L41-L55)).
Actual output of the two-line recipe above:

```text
>sp|P12345|AATM_RABIT Aspartate aminotransferase, mitochondrial OS=Oryctolagus cuniculus OX=9986 GN=GOT2 PE=1 SV=2
MALLHSGRVLPGIAAAFHPGLAAAASARASSWWTHVEMGPPDPILGVTEAFKRDTNSKKMNLGVGAYRDDNGKPYVLPSVRKAEAQIAAKNLDKEYLPIGGLAEFCKASAELALGENSEV
```

Every field `SwissProt` wants is there: accession `P12345`, entry name `AATM_RABIT`,
description, `OS=` organism, `OX=` taxon, `GN=` gene, `PE=`, `SV=`. It is the source FASTA
header verbatim, so parsing it is UniProt FASTA parsing and nothing MMseqs2-specific
([`createdb.cpp:248-256`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/createdb.cpp#L248-L256)
copies header bytes through unchanged; the probe round-trip confirms it).

## 5. Cost

Retrieval loads **the whole `.index` and, in `--id-mode 1`, the whole `.lookup` into RAM**.
Both `view` and `createsubdb` open the reader with `USE_INDEX|USE_DATA` plus
`USE_LOOKUP_REV`
([`view.cpp:26-29`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/view.cpp#L26-L29),
[`createsubdb.cpp:26-31`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/createsubdb.cpp#L26-L31)).
`DBReader::open` then allocates one `LookupEntry` per line and **sorts them by accession**
([`DBReader.cpp:138-156`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/commons/DBReader.cpp#L138-L156)),
so the accession hop is a binary search over a sorted array
([`DBReader.cpp:702-712`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/commons/DBReader.cpp#L702-L712)).
The data file is memory-mapped, so it costs address space, not resident memory.

Sizing, for Swiss-Prot **N = 575,748** entries (VERIFIED today from
`https://rest.uniprot.org/uniprotkb/search?query=reviewed:true&size=0`, header
`x-total-results`; release 2026_03 of 02-Sep-2026):

| Structure | Per entry | For N | Basis |
| --- | --- | --- | --- |
| `Index[]` | 24 B | **~13.8 MB** | `sizeof` measured with `g++ -O2` on `GPU71FM` |
| `LookupEntry[]` | 48 B | **~27.6 MB** | same; accessions are 6 or 10 chars so `std::string` stays inside its small-string buffer and adds no heap |
| Total resident | | **~41 MB** | INFERRED (sum of the two) |

INFERRED, on-disk: `.index` is roughly 22 bytes per line (~13 MB) and `.lookup` roughly 16
(~9 MB), from the observed column widths scaled to N. Not measured.

Wall time: `mmseqs view` on the four-entry probe was 7 ms. At Swiss-Prot scale the cost is
dominated by reading and sorting ~9 MB of `.lookup`, which is sub-second but is paid **on
every process launch**. INFERRED — not measured at scale.

That per-call constant is the argument for the package holding its own index rather than
shelling out per accession.

### About "0.094 GB"

VERIFIED: `Content-Length` on
`https://ftp.expasy.org/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz`
is **93,801,562 bytes = 0.0938 GB**. So #1's figure is the compressed download, not the
database. INFERRED: the built database is several times that — the residues alone are
uncompressed in `<name>`, and `<name>_h` holds every header. Budget the disk accordingly.

## 6. Bulk export — what a self-built index would use

`convert2fasta` is the whole-database export and is the only route back to FASTA
([wiki](https://github.com/soedinglab/MMseqs2/wiki#sequence-database-format): "Sequence
database can be converted back to FASTA only with `convert2fasta`"):

```sh
mmseqs convert2fasta swissprot swissprot.fasta          # >header + sequence, every entry
mmseqs convert2fasta swissprot swissprot_h.fasta --use-header-file
```

VERIFIED on the probe: the first writes standard FASTA with the complete original headers,
in database (shuffled) order.

But the package does not need it. `.lookup`, `.index` and `_h.index` are **already** the
bulk index, they are already local, and they are plain tab-separated text:

* `accession → key` is `.lookup` columns 2 and 1.
* `key → (offset, length)` is `.index`, for the sequence, and `_h.index`, for the header.
* Read `length - 2` bytes at `offset`.

For headers that is the whole story and it is pure Python — VERIFIED above with `dd`. For
sequences, see the next section.

Building an `accession → (seq_offset, seq_len, hdr_offset, hdr_len)` map at registration
time is a single pass over two text files of a few million lines, and it makes
`swissprot["P12345"]` two `seek`+`read` calls with no subprocess. That fits the project's
preference for bulk local data exactly, and `tests/_guards.py` has nothing to block.

## 7. Two findings that contradict #1

### The sequences are numerically encoded, and the encoding is lossy

`mmseqs databases` calls `createdb ... --gpu 1` **unconditionally** for every FASTA-sourced
database, Swiss-Prot included
([`databases.sh:305-320`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/data/workflow/databases.sh#L305-L320)).
VERIFIED: `mmseqs databases --help` exposes only `--tsv`, `--force-reuse`,
`--remove-tmp-files`, `--compressed`, `--threads` and `-v` — there is **no way to turn it
off**.

That sets `SEQUENCE_SPLIT_MODE_GPU`
([`createdb.cpp:349-351`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/createdb.cpp#L349-L351), which also turns low-complexity masking on),
which stores each residue as its substitution-matrix index, pads each entry to a 4-byte
boundary with byte `20`, and stamps the GPU bit into `.dbtype`
([`createdb.cpp:231-243`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/createdb.cpp#L231-L243),
[`createdb.cpp:782-784`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/util/createdb.cpp#L782-L784)).
`DBReader` decodes on the way out through `getUnpadded`, against a **21-letter** table —
`ACDEFGHIKLMNPQRSTVWYX`, with `code - 32` for soft-masked residues
([`DBReader.cpp:536-559`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/src/commons/DBReader.cpp#L536-L559)).
Twenty-one letters cannot represent twenty-six.

VERIFIED round trip, same FASTA into both database types:

```text
input                : MAUOBZJXWYACDEFGHIKLMNPQRSTVW
createdb --gpu 1     : MAXXDELXWYACDEFGHIKLMNPQRSTVW     <- what `mmseqs databases` gives you
createdb (plain)     : MAUOBZJXWYACDEFGHIKLMNPQRSTVW
```

`U` → `X`, `O` → `X`, `B` → `D`, `Z` → `E`, `J` → `L`. Raw bytes at that offset are
`0a 00 34 34 02 03 09 34 …` — no ASCII anywhere.

Consequences for this package:

* A `Protein` built from a `mmseqs databases`-built Swiss-Prot database **can never carry**
  `U`, `O`, `B`, `Z` or `J`. #1 says `seq.py` accepts those six "because UniProt and the ESM
  tokenizers accept all six" — true of UniProt, but not of this database. `SwissProt`
  returns sequences drawn from a 21-letter alphabet. Selenoproteins come back with `X`.
* A direct byte-offset read of `<name>` yields **bytes 0-20, not letters**, and soft-masked
  residues arrive with `+32`. Any Python fast path must apply the same decode table. Headers
  need no decoding.
* The lossless route is to run `createdb` yourself, without `--gpu`, on
  `uniprot_sprot.fasta.gz` — which the package would have to download itself, since
  `databases.sh` deletes the FASTA after `createdb`
  ([`databases.sh:311-313`](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/data/workflow/databases.sh#L311-L313)).
  This is worth an ADR: `download()` delegating to `mmseqs databases` is the simple choice
  and it silently changes the data.

Checked and **not** a problem: `easy-search` runs against a GPU-mode database on CPU with no
GPU flag and returns correct hits. VERIFIED on the probe. The search lane is unaffected.

### The rest of #1's database section holds up

VERIFIED: the immutability claim is right, and the wiki says so in the same words —
"The data file of the databases cannot be altered easily since any change would break the
offset in the `.index` file"
([Manipulating databases](https://github.com/soedinglab/MMseqs2/wiki#manipulating-databases)).
The file set #1 lists (`<name>`, `.index`, `.dbtype`, `.lookup`, `<name>_h`) is right as far
as it goes; it misses `.source`, `.version`, `_mapping` and `_taxonomy`.

VERIFIED: `swissprot["P12345"]` really is behaviour a declaration-table row cannot carry, so
the case for `SwissProt` being a subclass survives — the mechanism is just not the one #1
left unspecified.

## Recommendation

Build the accession index once, at registration, from `.lookup` + `.index` + `_h.index`;
serve `swissprot["P12345"]` with two `seek`+`read` calls and the 21-letter decode table.
Keep `mmseqs view --id-mode 1` as the reference implementation to test against — it is the
same answer, one subprocess and ~41 MB slower.

Then decide, in an ADR, whether `download()` may keep using `mmseqs databases` given that it
quietly rewrites five residue codes.
