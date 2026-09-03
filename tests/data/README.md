# Test fixtures

Small, subsampled **real** files — never a large one, and never a made-up one where a real one
would do. Nothing in the suite reaches the network (`tests/_guards.py`), so anything a test
needs to read lives here.

Every fixture added here gets a row in a table on this page, in the same commit as the file:
what it is, the URL or command its bytes came from, and every way it departs from the source.
A fixture whose provenance is not written down is one nobody can check later, and being
checkable is the whole reason to prefer a real file to an invented one.

## FASTA

Fetched on GPU71FM, 2026-09-03, from UniProt's REST API, which serves one Swiss-Prot entry
per accession as FASTA:

```bash
curl -sS https://rest.uniprot.org/uniprotkb/<accession>.fasta
```

| File | What it is | Departs from the source |
| --- | --- | --- |
| `uniprot_p01308.fasta` | `P01308`, human insulin, 110 aa — one record | nothing |
| `uniprot_three.fasta` | `P01308`, `P69905`, `P07203` — three records | the three responses concatenated, in that order |

`uniprot_three.fasta` is three records so that a reader can be tested on more than one, and
`P07203` — glutathione peroxidase 1 — is in it because it carries a `U`. It is one of the 285
Swiss-Prot entries that make ADR-0002 a decision rather than a preference: biotite's
converters would write `C` there, and this package writes `X`.

Neither file is subsampled. A Swiss-Prot record is a few hundred residues, so the whole entry
is smaller than an excerpt plus the note explaining what was cut.

## Search hits

Run on GPU71FM, 2026-09-03, against the real databases under
`/scratch/zhoulab/hanliu/protein/db/`. Both runs asked for the columns the package asks for,
so each file is what `protein.search` reads back:

```bash
mmseqs easy-search q.fasta db/swissprot/swissprot hits.tsv tmpdir \
  -s 1.0 --threads 4 --max-seqs 20 \
  --format-output query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits

foldseek easy-search 1ubq.cif db/pdb/pdb100 fshits.tsv fstmp \
  -s 1.0 --threads 4 --max-seqs 20 \
  --format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,alntmscore,lddt
```

| File | What it is | Departs from the source |
| --- | --- | --- |
| `mmseqs_hits_p01308.tsv` | 20 Swiss-Prot hits for `P01308`, human insulin | nothing |
| `foldseek_hits_1ubq.tsv` | 20 pdb100 hits for `1UBQ`, ubiquitin | nothing |

`--max-seqs 20` is why each file holds 20 rows. The query for the first was the insulin
sequence in `uniprot_p01308.fasta`; the query for the second was `1UBQ.cif`, downloaded from
`https://files.rcsb.org/download/1UBQ.cif`. Neither query file is kept here — a hit table is
what the code parses, and the structure is 103 kB.

The two files are also the evidence for two claims the code makes. MMseqs2 reports `pident`
as a percentage, so an identical hit reads `100.000`; Foldseek reports `fident` as a
fraction, so its best hit reads `0.973`. And Foldseek's last two columns are `alntmscore`
and `lddt`, with no `q3di` anywhere: that column does not exist in Foldseek 10-941cd33, and
asking for it fails the whole search.

## Structures

Downloaded on GPU71FM, 2026-09-03, from the RCSB file server, which serves one deposited
entry per id:

```bash
curl -sS -O https://files.rcsb.org/download/<id>.cif      # or .pdb
```

All three are stored gzipped, which is also how RCSB's bulk rsync tree serves them, so the
reader's gzip branch is exercised by the files a real mirror would leave. A test writes a
decompressed copy into its own directory where a plain file is wanted.

| File | What it is | Departs from the source |
| --- | --- | --- |
| `1ubq.cif.gz` | `1UBQ`, ubiquitin — one protein chain `A`, 76 residues, plus 58 waters | gzipped |
| `1bna.cif.gz` | `1BNA`, the Dickerson dodecamer — two DNA chains `A` and `B`, plus waters | gzipped |
| `1l2y_2models.pdb.gz` | `1L2Y`, Trp-cage — one 20-residue chain, NMR | gzipped; models 3-38 and the `MASTER` record dropped, `END` re-added |

The three entries are each in it for a reason:

| Entry | Why |
| --- | --- |
| `1UBQ` | chain A is `P0CG48` in SIFTS and `P62988` in this very file, which is the join the package gets right |
| `1UBQ` | its waters carry chain label `A` too, so a sequence that did not filter would read 58 residues too long |
| `1BNA` | a chain that is not a protein, so `.kind` has something to answer `nucleic` for and `.uniprot` something to answer `()` for |
| `1BNA` | two chains, so `structure["A"]` has a wrong answer available to it |
| `1L2Y` | the PDB format, and more than one model — an X-ray entry cannot test `.models` |

The `1L2Y` cut keeps the publisher's own bytes for every line it keeps:

```bash
awk 'BEGIN{m=0} /^MODEL /{m++} {if (m<=2) print} /^ENDMDL/{if(m>=2){print "END"; exit}}' \
  1l2y.pdb > 1l2y_2models.pdb
```

The download was 959,202 bytes, `md5:c19b6b883f76be35a1e8ef4765245197`, 11,842 lines; the cut
is 789 lines and holds models 1 and 2 whole. Uncut it is 138 kB gzipped, which is more than a
fixture should weigh to prove that a stack has a depth.

## SIFTS

Fetched on GPU71FM, 2026-09-03, from the EBI's flat-file tree, which publishes one current
release and overwrites it weekly in place:

```bash
curl -sS -O https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz
```

The download was 6,211,584 bytes, `md5:f92297379aa659822b69abe3db5c1984`, and its own first
line says `# 2026/08/30 - 13:24 | PDB: 35.26 | UniProt: 2026.03`. That release is gone from
the server as soon as the next one lands, so the digest records what was cut rather than
pinning what can be fetched again.

| File | What it is | Departs from the source |
| --- | --- | --- |
| `sifts_pdb_chain_uniprot_slice.tsv` | 91 of 1,033,045 rows, for ten entries | gunzipped; only the rows for those ten entries kept |

Both header lines are kept verbatim, and so are the line endings: **the release line ends
`LF` and every line after it ends `CRLF`**, which the reader has to strip and a
re-normalised fixture would stop testing.

The ten entries are each in it for a reason, and together they cover every shape the
reader and the two verbs have to answer for:

| Entry | Why |
| --- | --- |
| `101m` | one row, one chain, one accession — the ordinary case |
| `102l` | two segments for one `(pdb, chain, accession)` triple |
| `10ad` | `res_end - res_beg != sp_end - sp_beg`, so no offset is definable |
| `10eg` | chain labels `A` and `a` in one entry, so case is part of the name |
| `10lk` | multi-character chain labels (`Q1`, `S3`), which 17% of rows carry |
| `11sy` | four chains of one entry mapped to `P0CG48` |
| `1cmx` | a second entry for `P0CG48`, so the reverse direction spans entries |
| `1ubq` | chain A is `P0CG48` here and `P62988` in the mmCIF — the round trip |
| `8uqe` | chain B carries four accessions, the most any chain does |
| `9on4` | has a chain **labelled `NA`**, between chains `MA` and `OA` |
