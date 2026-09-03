# Context

## Glossary

### SIFTS

The EBI's re-curated map between PDB chains and UniProt accessions, and the only join
between this package's two namespaces. **An accession addresses a sequence; a PDB id
addresses a structure.** Neither owns the other, and the map is many-to-many both ways:
1.00% of chains carry more than one accession, and one accession reaches a median of 2
entries and a maximum of 3,668.

It is segment-level, not chain-level. A row is an entry, an author chain, an accession and
two residue ranges, so one chain-and-accession pair may carry several rows — up to 33. Both
ranges come back verbatim: 2.2% of segments have unequal range lengths, so no offset is
computed and no caller is handed one.

A structure file carries its own cross-reference and it disagrees. `1UBQ` chain A is
`P62988` in the mmCIF's `_struct_ref_seq` and `P0CG48` in SIFTS, because the file holds the
depositor's reference frozen at deposition and SIFTS holds PDBe's re-curated one. The join
therefore reads SIFTS alone, so that `Protein.structures` and `Chain.uniprot` round-trip.

Stored as a prepared set at `<LIULAB_DATA>/protein/sifts/pdb_chain_uniprot.tsv.gz`. See
`protein.sifts`.
