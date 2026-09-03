# Context

## Glossary

Entries are alphabetical. **Data dir**, **Completion marker**, **Prepared set** and
**Freshness** are `liulab-genome`'s words, defined in its `CONTEXT-MAP.md` and used here
unchanged — this package imports `genome.store` rather than restating them.

### 3Di

Foldseek's structural alphabet: twenty letters, one per residue, naming the geometry of that
residue's nearest tertiary contact. It turns a fold into a string, which is how Foldseek
searches structures with MMseqs2's machinery. A **Database** searched with Foldseek stores
one.

This package never computes a 3Di string and never reads one.

### Chain

One polymer inside a **Structure**, addressed by its author chain id — `structure["A"]`. It
holds `.atoms` filtered from the parent's array, `.kind` (protein, nucleic or other),
`.sequence` read from residue names, and `.uniprot`, the accessions **SIFTS** gives it.

A chain is not a **Protein**. It carries coordinates, it may be nucleic acid or ligand with
no accession at all, and it may carry several. Foldseek takes coordinates, so `search()`
lives here and on `Structure`, never on `Protein`.

### Database

A large, immutable set of local ffindex files that an **External tool** searches:
`SequenceDatabase` with MMseqs2, `StructureDatabase` with Foldseek. **A directory plus a
completion record is the registration; a name addresses a directory; nothing is persisted
centrally.**

Immutable is literal: the index holds byte offsets into the data file, so a change makes a new
database rather than an edit, and no mutating verb is exposed. Two ways in — `adopt` a set
already on disk, or `download` one through the tool's own downloader. **SIFTS** is not a
Database; it is a prepared set.

### Embedding

**A class, not just a word.** A frozen dataclass holding one sequence's `(L, d_model)`
float32 CPU array with BOS and EOS stripped, plus the three facts that identify it: the
source it came from, the checkpoint slug, and the layer normalised to a non-negative index.
`np.asarray(e)` is the array; `.mean()` is the per-sequence vector.

It is what `ESMC.embed` returns, and the only thing in the embedding lane a caller keeps.

### External tool

A native binary this package drives rather than reimplements: `mmseqs` and `foldseek`.
`protein.external` is the one module that touches `subprocess`, and it is where a tool's
location, version, freshness rule and install instructions live.

Foldseek vendors MMseqs2, so the two share a command grammar and a database format. That
shared half is `MmseqsLikeTool`; what genuinely differs — Foldseek's `fident` against
MMseqs2's `pident`, and Foldseek's extra columns — is named on the subclasses.

### Protein

**Identified by a UniProt accession.** It holds a biotite `ProteinSequence` and its metadata,
and it carries no coordinates, so it has no `foldseek_search()` — *a method exists on a class
where the tool takes that thing directly, never where the class would first have to acquire
something else.*

It has no `embed()` either, and that is a second rule: **resident state gets an object; a
subprocess does not.** ESM-C holds its weights across calls, so they became the `ESMC` object
you construct and keep; mmseqs holds nothing between calls, so `search()` stays a method. The
asymmetry between `p.search(db)` and `ESMC().embed(p)` is this rule, not an oversight.

### SIFTS

The EBI's re-curated map between PDB chains and UniProt accessions, and the only join
between this package's two namespaces. **An accession addresses a sequence; a PDB id
addresses a structure.** Neither owns the other, and the map is many-to-many both ways: a
chain may carry several accessions, and an accession many entries.

It is segment-level, not chain-level. A row is an entry, an author chain, an accession and
two residue ranges, so one chain-and-accession pair may carry several rows. The two ranges
are not always the same length, so both come back verbatim and no offset is computed.

A structure file carries its own cross-reference and it disagrees, because the file holds the
depositor's reference frozen at deposition: `1UBQ` chain A is `P62988` in the mmCIF's
`_struct_ref_seq` and `P0CG48` in SIFTS. The join therefore reads SIFTS alone, so that
`Protein.structures` and `Chain.uniprot` round-trip.

A prepared set under the **Data dir**, not a **Database**. See `protein.sifts`.

### Structure

One PDB entry's asymmetric unit, **addressed by a PDB id**. It holds the path to a
coordinate file, parsed into biotite's `AtomArray` on first use and cached, and the
**Chain**s in it.

A peer of **Protein**, never a part of one. The two are many-to-many in both directions and
**SIFTS** is the only join; a structure also holds nucleic acids and ligands that have no
protein at all. The asymmetric unit is forced, because SIFTS keys on its author chains.
Coordinates come from the local cache, filled from RCSB on a miss — never from `pdb100`,
which is C-alpha only and renumbers residues.
