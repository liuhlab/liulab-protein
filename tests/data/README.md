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
