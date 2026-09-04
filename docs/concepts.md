# The three things you work with

`Protein`, `Structure` and `Chain`. A protein is a sequence. A structure is a set of
coordinates. A chain is one polymer inside a structure.

A protein turns up in many structures, and a structure holds many proteins. This page says
how you make each one and how you get from one to another.

## A protein

A `Protein` is one UniProt accession's sequence. It carries no coordinates.

Build one from residues and an accession:

```python
from protein import Protein

p = Protein("MKTAYIAKQRQISFVKSHFSRQ", id="P12345")
p.length  # 22
```

Or read one out of a local database:

```python
from protein.db import SwissProt

p = SwissProt()["P0CG48"]
```

That reads the files on disk and touches no network.

`p.sequence` is a biotite sequence, not a string. So `p.sequence == "MKT"` is `False`, and
`str(p.sequence)` is how you get the letters. A slice such as `p[10:20]` is a plain string.

## A structure

A `Structure` is one set of coordinates. Name it by its PDB id:

```python
from protein import Structure

s = Structure("1UBQ")
```

Nothing is read until you ask a question. The first question that needs the file takes it
from the local cache, or downloads it once from RCSB. On a machine with no network you get
`CoordinatesNotDownloadedError`, and the message names the command that fills the cache.

You can also open a file you already have:

```python
s = Structure.from_file("1ubq.cif.gz")
```

`s.chain_ids` lists the labels in the file. `s.atoms` is every atom of the first model:
protein, nucleic acid, ligand and water alike. `s.models` holds all the models of an NMR
entry.

## A chain

Reach a chain through its structure, by label:

```python
chain = Structure("1UBQ")["A"]
chain.id  # '1UBQ_A'
```

Case is part of a label, so `"a"` and `"A"` are two different chains. Ask for a label that
is not there and you get a `KeyError` listing the ones that are. `s.chains` gives you all of
them.

## Moving between them

Three calls do most of the work.

| From | To | Call |
| --- | --- | --- |
| a structure | one of its chains | `structure["A"]` |
| a chain | the accessions it belongs to | `chain.uniprot` |
| a protein | the entries it appears in | `protein.structures` |

`chain.uniprot` is a tuple of accessions.

`protein.structures` is a pandas `DataFrame`, one row per mapped segment. The columns are
`pdb`, `chain`, `accession`, then the residue range the segment covers in the chain
(`res_beg`, `res_end`) and in the UniProt sequence (`sp_beg`, `sp_end`). Take both `pdb` and
`chain` from a row: an entry id alone does not say which chain to open.

Here is the walk from an accession to a structure and back:

```python
from protein import Protein, Structure

p = Protein("MQIFVKTLTG", id="P0CG48")
row = p.structures.iloc[0]
chain = Structure(row["pdb"])[row["chain"]]
chain.uniprot  # ('P0CG48',)
```

An empty frame means no structure has been solved for that accession. A `Protein` you built
without an `id` raises `ValueError` instead, because there is nothing to look up.

## What a chain gives you that a protein does not

Coordinates. `chain.atoms` holds them, and `len(chain)` counts atoms.

A chain is also not always a protein. Ask `chain.kind` before you ask for a sequence:

- `"protein"` — `chain.sequence` is an amino-acid sequence.
- `"nucleic"` — the chain is DNA or RNA, and `chain.sequence` is a nucleotide sequence.
- `"other"` — ligand or water only. `chain.sequence` raises `ValueError`, and `chain.atoms`
  is all there is.

The sequence is what was solved for. A residue with no coordinates is not in it, so a
disordered loop is missing rather than filled in.

`chain.uniprot` can hand back none, one or several accessions:

- `()` for a nucleic acid chain, a ligand chain, or an entry the map does not cover. That is
  a real answer and not a missing one.
- One accession in the ordinary case.
- Several for a chain built from more than one protein. You get all of them.

## Where the accession answer comes from

`chain.uniprot` answers from SIFTS, the EBI's map from PDB chains to UniProt accessions. The
map is kept apart from the coordinate files and worked out again against current UniProt.

The structure file carries a cross-reference of its own, and the two can disagree. For
`1UBQ` chain A the file says `P62988`. This package answers `P0CG48`.

Prepare the map before you ask for an accession:

```bash
protein sifts prepare
```

Until you do, `chain.uniprot` and `protein.structures` raise `SiftsNotDownloadedError`.
[Set up your data](data.md) covers the command.

One case skips the map. A structure you folded yourself carries the accessions you handed
the model, and its chains answer with those.

## Databases

A database is a directory on disk that MMseqs2 or Foldseek searches. You register it under a
name, and the name is what you pass:

```python
p.search("swissprot")
p.msa("uniref30")
Structure("1UBQ").search("pdb")
```

There are two ways to register one. Use `adopt` when the files are already on disk, and
`download` when they are not:

```bash
protein db adopt swissprot /path/to/swissprot
protein db download pdb
```

[Set up your data](data.md) walks through both.

## Next

- [Search a database](guides/search.md) — hit tables, and what the columns mean.
- [Work with structures](guides/structures.md) — chains, coordinates and 3D views.
