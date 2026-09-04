# Context

## Glossary

Entries are alphabetical. **Data dir**, **Completion marker**, **Prepared set**,
**Freshness**, **Xref set**, **Gene id stem** and **Namespace** are `liulab-genome`'s words,
defined in its `CONTEXT-MAP.md` and used here unchanged — this package imports
`genome.store` and `genome.xref` rather than restating them.

### 3Di

Foldseek's structural alphabet: twenty letters, one per residue, naming the geometry of that
residue's nearest tertiary contact. It turns a fold into a string, which is how Foldseek
searches structures with MMseqs2's machinery. A **Database** searched with Foldseek stores
one.

This package never computes a 3Di string and never reads one.

### Chain

One polymer inside a **Structure**, addressed by its author chain id — `structure["A"]`. It
holds `.atoms` filtered from the parent's array, `.kind` (protein, nucleic or other),
`.sequence` read from residue names, and `.uniprot`, the accessions **SIFTS** gives it — or
the ones its structure was produced from, where it carries them.

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

### FoldingRequest

**A class, not just a word.** What one structure prediction takes in: one entry per chain,
each naming its kind — protein, DNA or RNA — its sequence, the accession it came from, and,
for a protein chain, an **MSA**. Built at the call site and dropped after the fold, which is
what separates it from a **Structure**: lifetime, not content.

It carries **no output path**. Where the answer is written is not an input, so one request
folds to two destinations.

**Every check the model does not make lives here**, because past its own sibling-row check
the model degrades in silence: an alignment whose query row is not the chain's sequence, or
whose length disagrees, is refused rather than cut or gap-filled. A nucleic chain refuses an
alignment outright — the model accepts one there and drops it. A protein chain given none
gets the depth-1 alignment on its own sequence.

Chain labels are derived from position, since nobody named these chains. See `protein.fold`.

### MSA

A multiple sequence alignment, held as `(header, row)` pairs of plain text with row 0 the
query. **A3M is the only format read or written.**

**Case is the match state.** In A3M an uppercase residue or a `-` occupies a column, and a
lowercase residue is an insertion that occupies none. An alignment that has been uppercased
has lost the one thing separating A3M from aligned FASTA, which is why an `MSA` holds strings
and never biotite's `Alignment`.

**Query-anchored**, and checked at construction: row 0 carries no lowercase, and every row
shares a match-state count. At construction rather than in the reader, so a parsed alignment
and a generated one meet the same rule. The check is shape and never residues — a row
spelling `U` is well-formed A3M, whatever this package's own alphabet says.

A header is carried byte-for-byte because a `key=<taxon>` field in it is what pairs the chains
of a complex downstream; a row without one folds as if related to nothing. A leading `#` line
is carried for the same reason: it can encode a complex's chain layout, and biotite's FASTA
reader drops it.

See `protein.msa` and `protein.io.a3m`.

### Prediction

A **Structure** a model produced rather than a crystallographer, written into a directory the
caller names. **Never under the Data dir**, which holds reference and input data and no
user's outputs, so the directory is a required argument that defaults nowhere.

**Its name is a fact about the molecule**: user-given, else the one accession its
**FoldingRequest** names, else a short hash of the sequences. The checkpoint and every
sampler setting stay out — neither says what was folded.

**A name already held is weighed against the residues on disk.** The same sequence is a cache
hit and the GPU is never started; a different one raises, because a mutant can carry a
reference accession and would otherwise overwrite the reference or silently reuse it.
`overwrite=` is how a caller says they meant it. The residues, not a provenance record, are
what make that survive the process. The accepted edge: settings are not in the path, so a
re-fold with a different seed hits the cache.

See `protein.fold.predictions`.

### Protein

**Identified by a UniProt accession.** It holds a biotite `ProteinSequence` and its metadata,
and it carries no coordinates, so it has no `foldseek_search()` — *a method exists on a class
where the tool takes that thing directly, never where the class would first have to acquire
something else.*

It has no `embed()` either, and that is a second rule: **resident state gets an object; a
subprocess does not.** ESM-C holds its weights across calls, so they became the `ESMC` object
you construct and keep; mmseqs holds nothing between calls, so `search()` stays a method. The
asymmetry between `p.search(db)` and `ESMC().embed(p)` is this rule, not an oversight.

### SAE activation

**A class, not just a word.** What one sparse-autoencoder call over an **Embedding** gives
back: two `(L, k)` arrays — which codebook features fired at each residue, and how hard —
plus one reconstruction loss per residue.

A peer of **Embedding**, never one of them. It is sparse over a codebook far wider than
`d_model`, so that width would be a lie here, and `.mean()` would divide a feature's presence
by protein length. `.max()` is the per-sequence vector and there is no `.mean()` at all. See
ADR-0007.

The pair is lossless: top-k fills exactly `k` slots per row whether or not all `k` are
non-zero. `.dense()` materialises the whole codebook on request and nothing holds it.
Indices are stored in the narrowest unsigned type the codebook needs, values in float16.

It carries which SAE, which parent, which layer, which codebook and whether normalisation
happened, so two activation sets are comparable or provably not.

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

One set of coordinates and the **Chain**s in it, named by an id the constructor does not
police. A PDB entry is the ordinary case: give the entry id and the file comes from the local
cache, filled from RCSB on a miss — never from `pdb100`, which is C-alpha only and renumbers
residues. `from_file` takes any coordinate file and names it after the file, so a prediction
is a `Structure` like any other. The path is held and parsed into biotite's `AtomArray` on
first use.

A deposited entry holds its **asymmetric unit**, forced rather than chosen: SIFTS keys on AU
author chains, and many entries have several assemblies.

A peer of **Protein**, never a part of one. The two are many-to-many both ways and **SIFTS**
is the only join; a structure also holds nucleic acids and ligands with no protein at all.

A structure may carry the accessions it was **produced from**, one per chain, and
`Chain.uniprot` answers from those rather than asking SIFTS. That is provenance, not a join —
an input the file was written from, never a cross-reference read back out of it.

### Xref

The map between a UniProt accession and the **Gene id stem**s naming the same gene: a join
reaching out of this package rather than between its two namespaces, and one it does not
own. `liulab-genome` holds the sets, their releases and both directions; `protein.xref`
adds the one step genome cannot take from a **Protein**, since a set is built for a species
and a Swiss-Prot header carries a taxon id.

Only the accession direction lives here. The other starts from a species and a stem, takes
no **Protein**, and is `genome.xref` written plainly.

The accession is the key and `GN=` is not. A symbol goes through another verb against a
source that is not the identifier default, and matches previous and alias spellings, so it
carries ambiguity the accession does not.

Two misses stay apart. A taxon no set covers cannot be asked at all and raises; an accession
that was asked and named nothing comes back in the answer's `unresolved`. Swiss-Prot is the
whole of UniProt and few taxa are covered, so the first is the ordinary outcome.

Not a **Prepared set** and not a **Database**: it owns no bytes. See `protein.xref`.
