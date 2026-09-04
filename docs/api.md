# API reference

Every class, function and argument in the package, built from the code itself.

Looking for how to do something rather than what a call takes? Start with
[Getting started](start.md) or one of the guides.

## The protein

A `Protein` is one UniProt accession's sequence. `protein.seq` is the alphabet behind it,
and the error you catch when a sequence carries something that is not a residue.

::: protein.core.Protein

::: protein.seq

## Alignments

An `MSA` is one alignment, anchored on the query in its first row and held as text. One
module per job: the class, the search that fills one, and the MUSCLE run that lines one up.
The commands over them are with every other lane's, below.

::: protein.msa
    options:
      members: false

::: protein.msa.msa

::: protein.msa.mmseqs

::: protein.msa.muscle

## Structures

A `Structure` is one PDB entry, and a `Chain` is one polymer in it. They are peers of
`Protein`, not parts of one.

::: protein.structure
    options:
      members: false

::: protein.structure.structure

::: protein.structure.chain

::: protein.structure.view

## Embedding

`ESMC` holds the weights, and `Embedding` is what one call over them gives back. Running a
sparse autoencoder over an embedding gives back a `SaeActivation`, which holds which features
fired and how strongly.

::: protein.embed
    options:
      members: false

::: protein.embed.esm
    options:
      members: false

::: protein.embed.esm.esmc

::: protein.embed.embedding

::: protein.embed.esm.sae

::: protein.embed.esm.features

## Folding

`ESMFold2` holds the weights, what goes in is a chain or a list of them — as plain
dictionaries, if that is what you have — and a `Structure` written into a directory you
named is what comes back.

::: protein.fold
    options:
      members: false

::: protein.fold.request

::: protein.fold.esmfold

::: protein.fold.predictions

## Search

One lane, two queries: a sequence goes to MMseqs2, coordinates go to Foldseek. Both answer
with a `pandas.DataFrame` read by the same parser.

::: protein.search
    options:
      members: false

::: protein.search.mixin

::: protein.search.target

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

## Prepared sets

A `PreparedSet` is the half of a prepared set a caller meets after it is built: its status,
its cached read, and the two commands that prepare it and say what is here.

::: protein.prepared

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

::: protein.fold.cli

::: protein.msa.cli

::: protein.search.cli

::: protein.structure.cli
