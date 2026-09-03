---
search:
  exclude: true
---

# 3. `download()` delegates, and a folded alphabet is labelled rather than hidden

`Database.download()` runs `mmseqs databases` / `foldseek databases` and writes a record.
It does not fetch the source FASTA and build the database itself.

Measured on GPU71FM (#3, #18): the MMseqs2 recipe hardcodes `createdb --gpu 1`, ungated, and
that encodes residues against a 21-letter table. `U` and `O` become `X`, `B` becomes `D`,
`Z` becomes `E`, `J` becomes `L`. Plain `createdb` is lossless. So Swiss-Prot obtained the
way #1 specifies can never return selenocysteine.

The fold is accepted as a scoping decision. Those residues are out of scope here. The
standard twenty are untouched, so 575,303 of Swiss-Prot's 575,503 entries are byte-perfect,
and `protein.seq` already coerces `U`, `O` and `J` to `X` on the way in (ADR-0001). The real
stake is `B` and `Z` in 200 entries. Against that: building our own downloader means owning
a URL, an unpack, a checksum and a resume for every name the registry ever carries.

It buys no speed, and an argument for it framed as performance is wrong. GPU search does
need a GPU database, but `mmseqs makepaddedseqdb` derives one from a lossless database in a
single command, verified byte-identical. Simplicity is the whole justification.

**Labelled, not hidden.** `Database.is_gpu_encoded` reads the four `.dbtype` bytes —
`00 00 08 00` GPU, `00 00 00 00` plain — and `status()` and `protein db status` carry the
consequence in the caller's terms. It is named for the mechanism: four bytes prove how a
database was encoded and nothing about the FASTA behind it, so `is_lossless` would be a lie
for a database built cleanly from damaged input. `foldseek databases PDB` answers `False`,
so this is a Swiss-Prot problem rather than a `Database` problem.

**No warning at retrieval.** Nothing distinguishes a folded `D` from a real one, so a warning
would fire identically on the 575,303 good entries; `filterwarnings = ["error"]` would then
force a targeted ignore, disabling it the moment it was written.

**The reversal path**, which is what makes this a deferral rather than a dead end:

```sh
mmseqs createdb uniprot_sprot.fasta swissprot-faithful   # lossless
mmseqs makepaddedseqdb swissprot-faithful swissprot-gpu  # derived, not re-downloaded
```

`adopt()` takes either kind and records which it found.

This sits beside ADR-0002 rather than against it. We never lie at our own boundary; we may
adopt a third-party artifact that already has, provided we label it.
