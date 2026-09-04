---
search:
  exclude: true
---

# 5. A predicted structure's accession is provenance, not a join

`Structure` takes an optional per-chain accession map. `Chain.uniprot` answers from it where
one is present, and asks SIFTS only where it is absent.

**SIFTS is still the only join.** The two facts look alike and are not the same. A
depositor's `_struct_ref_seq` is a claim about somebody else's entry, frozen at deposition and
re-curated since — `1UBQ` chain A is `P62988` in the file and `P0CG48` in SIFTS. An accession
on a prediction is the input the file was written from. Reading an accession back out of a
written prediction is the option that would erase that difference, so nothing in this package
reads `_struct_ref`, in any code path.

**Why carry it at all.** A structure folded from a known accession answered `()`, and so does
a deposited entry SIFTS maps nothing to. One spelling for two different facts, and a caller
sorting a batch of predictions cannot tell them apart.

**The map is read per structure, not per chain.** A structure that carries one answers every
chain from it, `()` included, and never reaches SIFTS. A folded complex holds DNA chains with
no accession, and falling through for those would ask SIFTS about an id that is no PDB entry
— which raises where nobody has prepared the map.

**It does not survive the file, and nothing is built to make it.** No sidecar file, no custom
mmCIF category. `Structure.from_file` still defaults its id to the file stem and carries no
map, so a reopened prediction answers `()` again. That is a limit rather than an oversight,
and a test pins it: the fixture it reopens holds `P62988` in its own cross-reference, so any
other answer would mean the file had been read.

What it costs: `Chain.uniprot` now has two sources and its return value does not say which
answered. `Structure.accessions` is readable, so the question can still be asked; putting the
answer in the tuple would push the distinction onto every caller that does not care.
