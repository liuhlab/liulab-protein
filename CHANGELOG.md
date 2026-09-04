# Changelog

Every change worth knowing about, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version numbers are CalVer tags
of the form `vYYYY.M.PATCH`, and the tag is where the version comes from — nothing here
sets one.

## [Unreleased]

### Added

- **`protein.xref`, which says what gene a UniProt accession names.** Swiss-Prot fills in a
  taxon id for every entry it hands back, and nothing read it. `gene_stems_for` takes a list
  of accessions and one taxon id, and answers with the gene ids `liulab-genome` maps them to.
  `species_for` turns a taxon id into a species name, or `None` when no set covers it, so a
  caller can check first rather than catch an error.

  **Three species are covered — human, mouse and worm — and Swiss-Prot holds every species.**
  Most entries therefore reach no set at all, and that is kept apart from a real miss. A taxon
  with no set raises `TaxonNotCoveredError`, because the question cannot be asked of it. An
  accession that was asked and matched nothing rides back in the answer's `unresolved` list,
  so a list of accessions never comes back shorter without saying so.

  **The module owns nothing.** `liulab-genome` fetches the data, stores it and records which
  release answered, so there is no new set to prepare, no cache and no new command. One
  direction lives here. Going from a gene to its proteins starts from a species rather than
  from a protein, so it stays a plain `genome.xref` call.

- **A nucleic alphabet, so DNA and RNA get in.** `protein.seq.to_nucleotide_sequence` takes
  the four bases, the eleven IUPAC ambiguity codes and `U`, in either case, and hands back
  biotite's `NucleotideSequence`. `U` becomes `T` and says so; anything else raises the same
  error a bad protein sequence does. There is no `DNA` class and no `RNA` class.

  `Chain.sequence` now answers for a nucleic chain instead of refusing, and refuses only a
  chain that is neither — a ligand or the solvent. `Chain.kind` is unchanged and still says
  which type will come back. The guard that kept DNA away from ESM-C moved to `ESMC.embed`.

- **A structure can say what it was produced from.** `Structure` takes an optional
  `accessions` map, one entry per chain, and `Chain.uniprot` answers from it rather than
  asking SIFTS. A structure folded from a known accession used to answer `()`, which is what
  a deposited entry SIFTS maps nothing to also answers.

  **SIFTS is still the only join.** An accession here is an input the file was written from,
  never a cross-reference read back out of a file — nothing reads `_struct_ref`. Provenance
  does not survive being written, so a prediction reopened from disk answers `()` again.
