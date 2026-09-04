# Build an alignment

There are two ways to build an alignment here. This page covers both, and what the file
they hand back means.

## Two ways in

`Protein.msa(db)` searches a database and gives back the alignment in memory. It works the
way `Protein.search(db)` does, which gives back a table of hits.

```python
from protein import Protein

p = Protein("MKTAYIAKQRQISFVKSHFSRQ", id="P12345")
msa = p.msa("uniref30")
msa.depth  # how many rows the search found
```

You name the database on every call. Nothing ships with this package, and nothing is picked
for you. A shallow set standing in for a deep one is a wrong answer that looks right.

`align` takes a set you already hold. The sequences may have come from a paper, from a
colleague, or from an earlier search. MUSCLE lines them up, and the result is anchored on
the one you name:

```python
from protein.msa import align

msa = align({"P01308": "MKTAYIAK", "Q6YK33": "MKTAWIAK"}, query="P01308")
msa.depth  # 2
msa.to_a3m()  # the alignment as text
```

The query goes in row 0. The columns where it has a gap become lowercase insertions. That
is what the folding tools expect.

## Nothing is written unless you say where

Neither call takes an output path. The search runs in scratch space and cleans up after
itself. What you get back is a value in memory.

`write` is how it lands on disk:

```python
msa.write("p12345.a3m")  # keep it, at a path you choose
```

`align` is the same. Nothing durable is written until you call `write` and name the file.

## Case is the whole point of an A3M

A3M is FASTA with one rule on top. **Case is the match state.** An uppercase residue holds
a column, and so does a `-`. A lowercase residue is an insertion, and it holds no column.

That rule is the whole difference. An alignment that has been put into upper case has lost
the one thing separating A3M from aligned FASTA. So nothing here changes the case of
anything.

It is also why an `MSA` holds plain text. biotite's `Alignment` uppercases on construction,
so this package never holds one.

## How deep does it need to be?

Deep enough, not as deep as you can get. The folding tools that read one of these throw
away everything past about 16,000 rows. Past 1,024 rows they take a sample of what is left,
rather than the top of the list. So a few thousand rows is a floor to clear, and going far
past it buys nothing.

## Headers matter when you fold a complex

If you are folding two chains together, the header of each row matters as much as the
count. This package copies the organism the database named into a `key=` field. That is how
the folding tools tell that a row from one chain and a row from another came from the same
species.

Rows with no key still fold. But each chain then folds as though nothing related it to the
others, and nothing warns you.

Headers are carried byte for byte for that reason. A `key=` or `OX=` field reaches the
alignment, instead of being cut off with the description. A leading `#` line survives too.
That is where the chain layout of a complex is written.

## What gets refused, and when

An `MSA` is checked when it is built, not two steps later. Row 0 is the query, so it
carries no lowercase. Every row shares one match-state count. A ragged alignment is refused
at construction, and the error names the row to blame.

The check is about shape and never about residues. A row spelling `U` is well-formed A3M,
whatever this package's own alphabet says. An alignment is a file's content, and this class
holds it and hands it back.

## From the command line

```bash
protein msa search MKTAYIAKQRQISFVKSHFSRQ uniref30 p12345.a3m
protein msa align homologues.fasta insulin.a3m --query P01308
```

The output path is an argument, not an option. It is required for both, and for the same
reason there is no output path in Python. Nothing durable lands anywhere you did not name.

`search` passes the mmseqs knobs through: `--sensitivity`, `--evalue`, `--max-seqs` and
`--threads`. `--id` names the query, and that name becomes the header of row 0. `align`
needs `--query` to say which record to anchor on. The identifier the header opens with is
enough there, so you do not have to type a whole UniProt header.

Both print the query, the depth, the match-state count and the path written. Both take
`--json`.

## Read next

[Predict a structure](folding.md) is what reads one of these.
[Commands](../reference/commands.md) has every option.
