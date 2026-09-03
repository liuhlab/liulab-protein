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
