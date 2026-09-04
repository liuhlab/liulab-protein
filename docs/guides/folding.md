# Predict a structure

ESMFold2 turns sequences into coordinates, on this machine. There is no server and no token.
The weights are read from the Hugging Face cache.

[Getting started](../start.md) folds one complex end to end. Read that for the worked
example. This page is the reference for everything it did not cover.

## What you need first

Folding needs the `esm` environment, which is the one with torch in it. It also wants a GPU:

```bash
pixi install -e esm
```

## Build the model once

`ESMFold2()` loads a checkpoint when you build it. So build it once and keep it, and fold as
many things as you like against it:

```python
from protein.fold import ESMFold2

model = ESMFold2()  # loads once
model.device  # where the weights actually went
```

You name a checkpoint by its slug:

| Slug | What it is |
| --- | --- |
| `ESMFold2-Fast` | the default. It needs no alignment, and it does not refuse one |
| `ESMFold2` | the other one, loaded the same way |

A slug the package does not know fails by name, before any download starts. An alignment
reaches a Fast fold too, so give one when you have one.

`device=` says where to run. Leave it off and you get the GPU when torch can see one, and
the CPU when it cannot. The answer sits on `model.device`, so a run that fell back to the
CPU tells you so.

Extra keywords are passed on. Whatever else you give `ESMFold2()` goes to the loader, and
whatever else you give `fold` goes to the model's own `fold`. `model.model` is the loaded
model itself, for the rest.

## What goes in

One entry per chain. The plain way to write one is a dictionary naming its `kind`,
its `sequence`, and optionally an `accession` and an `alignment`:

```python
chains = [
    {"kind": "protein", "sequence": "MKTAY", "accession": "P12345"},
    {"kind": "dna", "sequence": "ACGT"},
]
```

Hand that straight to `fold()`. You do not have to build anything first:

```python
prediction = model.fold(chains, "folds")
prediction.chain_ids  # ('A', 'B')
```

Chain labels come from position, so read them off rather than guessing. `A` is the first
chain you passed and `B` is the second. Past `Z` they carry on as `AA`, `AB`.

A single chain needs no list around it, and a `Protein` can stand in for a whole entry,
carrying its accession with it:

```python
model.fold(Protein("MKTAY", id="P12345"), "folds")
```

Bare residues work too, and mean protein:

```python
model.fold("MKTAY", "folds")
```

## What gets refused

A dictionary must name its `kind`. `ACGT` is a valid protein sequence as well as a valid
strand of DNA, and nothing here guesses which you meant. Bare residues are the one exception,
and they mean protein.

Misspell a field and you get an error naming it. Nothing is silently dropped, so a mistyped
`accession` cannot quietly cost you the prediction's provenance.

Three more things raise when the request is built, before any folding starts:

- An alignment whose query row is not the chain's sequence. The message says where the two
  first differ.
- An alignment that covers a different number of positions than the chain has residues.
- Any alignment at all on a DNA or RNA chain.

A protein chain given no alignment gets one built for it: the depth-1 alignment on its own
sequence. You need do nothing about that.

To fold against a real alignment, pass it as a field. It can be an `MSA` you built, or the
path to an A3M file, which is what lets a whole request come out of a JSON document. See
[Build an alignment](alignments.md):

```python
{"kind": "protein", "sequence": "MKTAY", "alignment": "p12345.a3m"}
```

`ChainRequest` and `FoldingRequest` are still there, and a mixture of spellings is fine in
one call. Reach for `ChainRequest` when you want the object itself:

```python
from protein.fold import ChainRequest

chain = ChainRequest.of(Protein("MKTAY", id="P12345"))
chain.accession  # 'P12345'
```

## Where the answer goes

`fold` takes the output directory as an argument. You pass one every time. There is no
default:

```python
structure = model.fold(request, "folds")
```

The directory is created if it is not there.

What comes back is a `Structure`, the same class a deposited entry gives you. Read its
chains, read a sequence, or [search with it](search.md).

## Naming, repeats and overwriting

The file lands at `<directory>/<name>.cif`, and the name is:

1. what you passed as `name=`, if you passed one
2. the one accession the request names, if it names exactly one
3. a short hash of the chains' kinds and sequences

Fold the same thing twice and the second call gives you back the first answer. Nothing is
recomputed, and the card is never started.

Fold a different sequence under a name already taken and it raises `FileExistsError`. Pass
`overwrite=True` to say you meant it, or pick another name. The check reads the sequences
back off the residues on disk, so it still holds a day later and in another process. That is
what stops a mutant carrying a reference accession from landing on the reference's file.

Settings are not in the name. So re-folding with a new seed hits the answer already there,
and you need `overwrite=True` or a distinct name to get a fresh one.

## What the model says about its own answer

Per-residue confidence rides in the B-factor column of the file. That is where every viewer
looks for it, so a prediction colours by confidence with nothing more to do.

**Watch the scale.** The B-factor column runs 0 to 100. The scalars below are the same
measures as a fraction, 0 to 1. A colour range or a cutoff written for one against the other
treats every residue alike.

The scalars are on `Structure.confidence`:

```python
structure.confidence.plddt  # mean per-residue confidence, 0 to 1
structure.confidence.ptm  # the predicted TM-score, or None
structure.confidence.iptm  # the interface score, which means something for a complex
```

Ask for several diffusion samples with `num_diffusion_samples=`. The one with the highest
mean confidence is what gets written.

The pairwise matrix says whether two chains sit correctly against each other, which
per-residue confidence cannot. It goes in a sibling file beside the coordinates, named
`<name>.pairwise.npy`, and it is read only when you ask:

```python
matrix = structure.confidence.pairwise()  # one entry per pair of residues
```

Low entries mean the model is sure of where those two residues sit relative to each other.
Where the model reported no matrix, `pairwise()` raises `FileNotFoundError`.

A prediction that was already on disk comes back with `confidence` set to `None`. The
scalars do not survive the file. That covers the repeat fold above, and anything you open
with `Structure.from_file`. So keep the scalars when you first get them, or re-fold with
`overwrite=True` to get them again.

## Two warnings

`load_esmc=False` is not a memory option. Without the language model, the model still writes
a file of the right length. The structure in it is wrong. There is no error and no mark in
the file, so it warns instead.

Nothing here manages the card. There is no length cap and no memory arithmetic. A fold that
does not fit raises whatever the GPU raises.

## From the command line

`protein fold structure` folds one protein and prints what it wrote:

```bash
pixi run -e esm protein fold structure query.fasta folds
```

The query is a FASTA file holding exactly one record, or the residues themselves. A file
that exists wins, so a sequence is never read as a path somebody typo'd. The output
directory is an argument, not an option.

| Option | What it does |
| --- | --- |
| `--checkpoint` | which checkpoint: `ESMFold2-Fast` or `ESMFold2` |
| `--device` | where to run. Omitted, cuda when it is visible, else cpu |
| `--name` | what to call it. Omitted, the accession, else a hash |
| `--overwrite` | fold over a name already holding another sequence |
| `--json` | print JSON instead of plain text |

It prints where the coordinates went, whose they are, and what the model said about them. It
exits 1 if the slug is unknown, if the query is neither one FASTA record nor a protein
sequence, or if the directory holds that name over a different sequence.

The full list is on [Commands](../reference/commands.md), and the
[Python API](../api.md) has the rest.
