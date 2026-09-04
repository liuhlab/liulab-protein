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

A `FoldingRequest` holds one `ChainRequest` per chain. Each chain names its kind: `protein`,
`dna` or `rna`.

```python
from protein.fold import ChainRequest, FoldingRequest

request = FoldingRequest(
    [
        ChainRequest("protein", "MKTAY", accession="P12345"),
        ChainRequest("dna", "ACGT"),
    ]
)
request.chain_ids  # ('A', 'B')
```

Chain labels come from position in the request, so read them off `request.chain_ids` rather
than guessing. `A` is the first chain you passed and `B` is the second. Past `Z` they carry
on as `AA`, `AB`.

`ChainRequest.of` is the shortcut when you already hold a `Protein`. It carries the
accession across:

```python
from protein import Protein
from protein.fold import ChainRequest

chain = ChainRequest.of(Protein("MKTAY", id="P12345"))
chain.accession  # 'P12345'
```

That accession rides onto the prediction, and `Chain.uniprot` answers with it.

## What gets refused

Three things raise when you build the request, before any folding starts:

- An alignment whose query row is not the chain's sequence. The message says where the two
  first differ.
- An alignment that covers a different number of positions than the chain has residues.
- Any alignment at all on a DNA or RNA chain.

A protein chain given no alignment gets one built for it: the depth-1 alignment on its own
sequence. You need do nothing about that.

To fold against a real alignment, build it first and pass it in. See
[Build an alignment](alignments.md):

```python
ChainRequest.of(protein, alignment=msa)
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
