# How it fits together

This page is the shape of the package, so the guides make sense. Read it once, start to
finish.

## Three peers, not a hierarchy

A `Protein` is one UniProt accession's sequence. A `Structure` is one PDB entry. A `Chain`
is one polymer inside a structure.

The three are peers. A protein turns up in many structures. A structure holds many
proteins. Neither one owns the other, and there is no tree to walk down.

A chain is not a protein wearing a different hat. It carries coordinates, and a protein
carries none. It may be DNA or RNA. It may be a ligand with no accession at all. It may
also carry several accessions at once. Ask `chain.kind` before you ask for a sequence, and
it will tell you which of those you are holding.

## SIFTS is the only join

A structure file makes its own claim about where a chain came from. The claim sits in
`_struct_ref_seq`, and **this package never reads it**. Not in any code path.

The join comes from SIFTS instead. SIFTS is the EBI's map between PDB chains and UniProt
accessions. It is kept apart from the coordinate files and worked out again against current
UniProt.

The two disagree, and here is the case to remember. For `1UBQ` chain A the file says
`P62988`. SIFTS says `P0CG48`. This package answers with SIFTS.

The reason is the depositor's record. It was frozen on the day of deposition, and it
describes somebody else's entry as that entry stood back then. UniProt has moved since.

The map runs many-to-many both ways. One chain may carry several accessions, and one
accession may show up in many entries.

One exception is worth naming. A predicted structure can carry the accession it was folded
from. That is provenance: the input the file was written from, not a cross-reference read
back out of it. ADR-0005 has the reasoning.

## Methods and objects you keep

Two rules decide whether you call a method or build something first.

The first rule: a method goes on the class the tool takes directly. Foldseek takes
coordinates. So the search by shape lives on `Structure` and on `Chain`, and never on
`Protein`. A protein would have to go and find coordinates first, and this package hides no
step like that. See [Search a database](guides/search.md).

The second rule: anything holding a lot of state in memory is a class you build and keep.
`ESMC()` and `ESMFold2()` hold model weights, so you make one and use it many times. mmseqs
holds nothing between calls, so `search()` stays a method.

That is why `p.search(db)` and `ESMC().embed(p)` do not look alike. It is the rule at work,
not an oversight.

## What a database is here

A database is a directory of files that an outside tool searches. mmseqs searches a
sequence database. Foldseek searches a structure database.

A directory on disk with a completion record beside it **is** the registration. There is no
central list and nothing is written down elsewhere. The directory name is the name you
type.

There are two ways to get one. `adopt` writes a record beside files that are already there,
which is the usual case on a cluster. `download` hands the job to the tool's own downloader
and then records what it left behind.

These databases are immutable, and that word is literal. The index holds byte offsets into
the data file. Change a byte and every offset after it is wrong. So a change makes a new
database rather than an edit, and no verb here will modify one.
[Set up your data](data.md) walks through both routes.

## Prepared sets are the other big files

Two large sets do not come from a search tool at all. The SIFTS map comes from the EBI. The
SAE feature descriptions come from the ESM project.

Nothing searches either one, so neither is a database. They are prepared sets. You prepare
one once, and then it is read many times. Each has a fixed home and a status you can ask
for, and [Set up your data](data.md) covers both.

## Where the types come from

This package holds biotite's types and never subclasses them. A `Protein` holds a biotite
`ProteinSequence`. A `Structure` parses into a biotite `AtomArray`.

It calls biotite's file and array layer, and it stays away from the convenience converters
next door. The reason is worth stating plainly. `fasta.get_sequence` and
`structure.to_sequence` rewrite `U` to `C` and `O` to `K` without a word. Selenocysteine
comes back as cysteine, and nothing tells you. A silent rewrite of a residue is a wrong
answer nobody sees.

This package folds letters as well. `U`, `O` and `J` all become `X`, which means unknown.
The difference is that it warns you each time and names what it changed. ADR-0002 has the
full argument.
