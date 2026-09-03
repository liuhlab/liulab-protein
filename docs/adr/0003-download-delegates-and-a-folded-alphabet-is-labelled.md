---
search:
  exclude: true
---

# 3. `download()` delegates, and a folded alphabet is labelled rather than hidden

`Database.download()` runs `mmseqs databases` / `foldseek databases` and writes a record.
It does not fetch the source FASTA and build the database itself.

The MMseqs2 recipe hardcodes `createdb --gpu 1`, ungated, and that encodes residues against a
21-letter table: `U` and `O` become `X`, `B` becomes `D`, `Z` becomes `E`, `J` becomes `L`.
Plain `createdb` is lossless. So a Swiss-Prot obtained this way can never return
selenocysteine.

The fold is accepted as a scoping decision. `U`, `O` and `J` are out of scope here and
`protein.seq` coerces them to `X` on the way in anyway (ADR-0001), and the standard twenty are
untouched. The real stake is `B` and `Z`. Against that: building our own downloader means
owning a URL, an unpack, a checksum and a resume for every name the registry ever carries.

It buys no speed, and an argument for it framed as performance is wrong. GPU search does need
a GPU database, but `mmseqs makepaddedseqdb` derives one from a lossless database in a single
command. Simplicity is the whole justification.

**Labelled, not hidden.** `Database.is_gpu_encoded` reads the `.dbtype` header, and `status()`
and `protein db status` carry the consequence in the caller's terms. It is named for the
mechanism: those bytes prove how a database was encoded and nothing about the FASTA behind
it, so `is_lossless` would be a lie for a database built cleanly from damaged input.
`foldseek databases PDB` answers `False`, so this is a Swiss-Prot problem rather than a
`Database` problem.

**No warning at retrieval.** Nothing distinguishes a folded `D` from a real one, so a warning
would fire on every faithful entry as well; `filterwarnings = ["error"]` would then force a
targeted ignore, disabling it the moment it was written.

**The reversal path**, which is what makes this a deferral rather than a dead end:

```sh
mmseqs createdb uniprot_sprot.fasta swissprot-faithful   # lossless
mmseqs makepaddedseqdb swissprot-faithful swissprot-gpu  # derived, not re-downloaded
```

`adopt()` takes either kind and records which it found.

This sits beside ADR-0002 rather than against it. We never lie at our own boundary; we may
adopt a third-party artifact that already has, provided we label it.
