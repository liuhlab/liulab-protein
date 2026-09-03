# API reference

Built from the docstrings in `src/protein/`, so this page and the code cannot drift apart.
Write the docstring; this page follows.

## The examples are tests

`pixi run check` runs the `Examples` blocks in `src/protein/`, so an example that no longer
matches its code fails the tests.

Write one where it makes the object easier to use, and leave it out where it would not.
An example nobody keeps up to date is worse than none.

Keep an example cheap, offline and deterministic. It has to give the same answer on any
machine, with no network. A line that cannot do that needs `# doctest: +SKIP` at the end of
that line.

Two things about that marker are easy to get backwards:

- It covers only the line it sits on. It does not carry to the line below. In a block that
  mixes lines that run with lines that cannot, each line that cannot run needs its own.
- A trailing comment written as plain prose looks just like a marker and is not one. Only
  the `# doctest:` form is read as one.

## The protein

A `Protein` is one UniProt accession's sequence. `protein.seq` is the alphabet behind it,
and the error you catch when a sequence carries something that is not a residue.

::: protein.core.Protein

::: protein.seq

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

A package that names things it does not define. Each block below lists only what its own
module writes, so nothing on this page is documented twice.

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

## Files

::: protein.io.fasta

::: protein.io.structure

## Where the big files live

::: protein.store

## Driving mmseqs and foldseek

::: protein.external

## The command line

The whole module, because typer makes every verb a plain function with a docstring, and
`protein.cli:app` — the object `[project.scripts]` registers — is built from them. The root
app holds two commands; each lane ships the rest beside the code they call.

::: protein.cli

::: protein.db.cli

::: protein.embed.cli

::: protein.search.cli

::: protein.structure.cli
