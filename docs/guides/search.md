# Search a database

Both kinds of search answer with the same table. A sequence goes to MMseqs2. A shape goes to
Foldseek. Each hands back a `pandas.DataFrame`, and one parser reads both.

## The two queries

A `Protein` searches with its residues, and that runs MMseqs2:

```python
from protein import Protein

p = Protein("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", id="P12345")
hits = p.search("swissprot")
```

A `Structure` or a `Chain` searches with its coordinates. Both of those run Foldseek:

```python
from protein import Structure

s = Structure("1UBQ")
s.search("pdb")  # every chain, one run
s["A"].search("pdb")  # one chain
```

A whole structure is one Foldseek run, not a loop over chains. Foldseek splits the chains
itself. The `query` column then names each one as `<entry>_<chain>`, such as `1UBQ_A`.

## Why a protein cannot search by shape

There is no shape search on `Protein`. Foldseek reads coordinates, and a protein holds none.
A method goes on the class the tool takes directly. It never goes on a class that would first
have to fetch something else. [How it fits together](../concepts.md) has the rest of the rule.

## You name the database every time

Nothing ships with this package, and nothing is picked for you. A shallow set standing in for
a deep one is a wrong answer that looks right.

So you pass a name. The name is a directory on disk with a completion record beside it, and
[Set up your data](../data.md) shows how to register one. A name nothing is registered under
raises `LookupError`, and the message lists the names there are.

## What comes back

One row per hit, in the tool's own column order. A search that found nothing gives you the
same columns and no rows. You never have to test for empty before you read a column.

Both tools report these:

| Column | What it holds |
| --- | --- |
| `query` | what the query was called |
| `target` | the database entry that was hit |
| `alnlen` | how long the alignment is |
| `mismatch` | how many positions differ |
| `gapopen` | how many gaps were opened |
| `qstart`, `qend` | where the alignment sits in the query |
| `tstart`, `tend` | where it sits in the target |
| `evalue` | how many hits this good you would expect by chance |
| `bits` | the bit score |

For a sequence search, `query` is the protein's `id`. A protein with no id is called `query`.
For a structure search it is the chain.

The two tools differ in one place, and it is identity. MMseqs2 reports `pident`, a
percentage. Foldseek reports `fident`, the same quantity as a fraction. Neither is renamed
into the other, so a table tells you which number it carries by the column name it has. Read
that name before you filter on it.

Foldseek also adds two columns a sequence search has no answer for. They are `alntmscore` and
`lddt`, and both score how well the two shapes agree.

## Tuning a search

Four knobs, spelled the same way for both tools. They are keyword arguments on `search`, and
flags on the command line.

| Keyword | Flag | What it does |
| --- | --- | --- |
| `sensitivity` | `-s` | Lower is faster and finds less. |
| `evalue` | `-e` | Hits above it are not reported. |
| `max_seqs` | `--max-seqs` | Caps the hits per query. |
| `threads` | `--threads` | How many cores to use. |

Leave one out and the tool's own default stands. This package does not restate defaults.

Name `threads` on a shared machine. Both tools grab every core they can see, and on a login
node that is rude.

```python
hits = p.search("swissprot", sensitivity=7.5, evalue=1e-3, max_seqs=1000, threads=8)
```

Anything else you need goes in `extra`, which is passed through unread.

## From the command line

`protein search seq` takes residues. `protein search struct` takes a coordinate file, or a
PDB id it will fetch for you.

```bash
protein search seq MKTAYIAKQRQISFVKSHFSRQ swissprot --id P12345 --threads 8
protein search struct 1UBQ pdb --chain A --threads 8
protein search seq MKTAYIAKQRQISFVKSHFSRQ swissprot | cut -f2
```

Rows go to stdout, tab-separated. The column names and the hit count go to stderr. That is
why the pipe above gets you the targets and no header. `--json` carries the same answer,
keyed by column name. [Commands](../reference/commands.md) lists every option.
