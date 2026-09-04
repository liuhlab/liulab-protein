# API reference

Built from the docstrings in `src/protein/`, so this page and the code cannot drift apart.
Write the docstring; this page follows.

## The examples are tests

`pixi run check` runs the `Examples` blocks in `src/protein/`, so an example that no longer
matches its code fails the tests. Write one where it makes the object easier to use, and
leave it out where it would not. An example nobody keeps up to date is worse than none.

Keep an example cheap, offline and deterministic: it has to give the same answer on any
machine, with no network. A line that cannot do that needs `# doctest: +SKIP` at the end of
that line, and the marker covers only its own line.

## The protein

A `Protein` is one UniProt accession's sequence. `protein.seq` is the alphabet behind it,
and the error you catch when a sequence carries something that is not a residue.

::: protein.core.Protein

::: protein.seq

## Alignments

An `MSA` is one alignment, anchored on the query in its first row and held as text.

::: protein.msa

## Structures

A `Structure` is one PDB entry, and a `Chain` is one polymer in it. They are peers of
`Protein`, not parts of one.

::: protein.structure
    options:
      members: false

::: protein.structure.structure

::: protein.structure.chain

## Embedding

`ESMC` holds the weights, and `Embedding` is what one call over them gives back.

::: protein.embed
    options:
      members: false

::: protein.embed.esm

::: protein.embed.embedding

## Search

One lane, two queries: a sequence goes to MMseqs2, coordinates go to Foldseek. Both answer
with a `pandas.DataFrame` read by the same parser.

::: protein.search
    options:
      members: false

::: protein.search.mixin

::: protein.search.mmseqs

::: protein.search.foldseek

## Databases

Each block below lists only what its own module writes, so nothing here is documented twice.

::: protein.db
    options:
      members:
        - DECLARED
        - KINDS
        - Declaration
        - database_class
        - open_database

::: protein.db.base

::: protein.db.swissprot

## The PDB to UniProt map

::: protein.sifts

## The UniProt to gene map

::: protein.xref

## Files

::: protein.io.a3m

::: protein.io.fasta

::: protein.io.structure

## Where the big files live

::: protein.store

## Driving mmseqs and foldseek

::: protein.external

## The command line

The whole module, because typer builds `protein.cli:app` out of plain functions and their
docstrings. The root app holds two commands; each lane ships the rest beside the code it
calls.

::: protein.cli

::: protein.db.cli

::: protein.embed.cli

::: protein.search.cli

::: protein.structure.cli
