# Embed a sequence

Turn a sequence into one vector per residue with ESM-C, and read those vectors as sparse
features.

## The model is an object you keep

`ESMC()` loads the weights and holds on to them. So you build it once and reuse it:

```python
from protein import ESMC, Protein

model = ESMC()
p = Protein("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", id="P12345")
e = model.embed(p)
```

There is no `p.embed()`, and that is on purpose. Anything that stays loaded between calls
gets an object of its own. Anything that holds nothing between calls stays a method, which
is why `p.search(db)` is one: mmseqs is a program that runs and then exits. ESM-C keeps its
weights, so it is a class you build and keep.

You name a checkpoint by slug: `300m`, `600m` or `6b`. An unknown slug fails right there, by
name, before anything is downloaded. Omit `device=` and the weights go to the GPU when torch
can see one, and to the CPU when it cannot. Either way `model.device` says where they went,
so a run that quietly fell back to the CPU tells you.

ESM-C needs torch, which lives in its own environment:

```bash
pixi install -e esm
pixi run -e esm protein esm embed query.fasta
```

Run every command on this page that way. See [Install](../install.md).

## What one call gives back

An `Embedding`. It holds a `(L, d_model)` float32 array on the CPU. The model adds a start
and an end token, and both are cut off before you see the array. So `L` is the residue
count, and `len(e)` is the length of the sequence you handed in.

It is frozen, so nothing can edit a measurement after the fact.

It also carries the three facts that say what it is:

- `source` — the accession or chain key it came from.
- `checkpoint` — the slug, such as `300m`.
- `layer` — which hidden state, as a non-negative index.

A bare array carries none of that. Found again an hour later, it cannot say which model or
which layer made it, and two arrays from different ones are not comparable.

```python
import numpy as np

np.asarray(e)  # the array itself
e.shape  # (L, d_model)
e.mean()  # the per-sequence vector, (d_model,)
e.as_json()  # source, checkpoint, layer and shape
```

Every numpy function takes an `Embedding` straight, so you rarely reach for `e.array`.

## Choosing a layer

`embed()` takes `layer=`, and it indexes the hidden states the way the model does. Layer 0
is what the embedding layer put out. The last layer is the last hidden state. Negative
numbers count back from the end, and the default is `-1`.

`model.n_layers` is how many transformer layers the checkpoint has, and there is one more
hidden state than that. A layer outside the range raises, and the message spells the range
out.

What comes back is stored normalised. Ask for `layer=-1` and the `Embedding` records the
non-negative index instead. That is what lets you compare two of them later, or check one
against something that expects a named layer.

There is one thing to decide. A sparse autoencoder covers particular layers, and its slug
names them, so pass the layer yours covers if you plan to encode with one. Otherwise leave
`layer` alone and take the last hidden state.

## Sparse features

A sparse autoencoder splits an embedding row into named features. `SAE(slug)` loads one and
`encode()` runs it over an `Embedding`:

```python
from protein.embed import SAE

sae = SAE("300m-layer23-k64-cb16384")
activation = sae.encode(model.embed(p, layer=23))
```

`SAE` has no default slug, and the missing default is the point. `ESMC()` defaults to its
smallest checkpoint, so an `SAE()` default would pair the two wrong on the first try. Name
both halves or neither.

What comes back is a `SaeActivation`. It is a peer of `Embedding` and not one of them. It
holds which features fired at each residue and how hard, not a dense row per residue.
`.max()` is the per-sequence vector, and there is no `.mean()` at all: an average over
length would report the same feature as weaker in a long protein than in a short one. The
codebook is also far wider than `d_model`, so that width would be a lie here. ADR-0007 has
the whole trade-off.

Only the slots that fired are stored. `activation.dense()` builds the full matrix when you
ask for it, and nothing keeps a copy.

## The check that stops a plausible wrong answer

Hand an autoencoder the wrong model's embedding, or the right model's wrong layer, and the
shapes can still agree. The multiply then works. Numbers come out, and they look like
features. The only thing that says otherwise is how badly the decoder rebuilds the row, and
nothing looks at that for you.

So `encode` checks three things first: which backbone the embedding came from, which layer
it is, and how wide the row is. If one of them disagrees it raises, and the message names
that one. Each check reads a fact the two objects already recorded, never a tensor, so all
three run with no GPU.

`normalize=True` scales the magnitudes by the checkpoint's own per-feature numbers. Those
are buffers that default to ones. So asking for them where a checkpoint ships none raises,
rather than scaling every feature by one and calling that normalised. Whichever way it went,
`activation.normalized` says so.

Every activation also carries one reconstruction loss per residue: how badly the decoder
rebuilt that residue. Those numbers are the measured backstop on whether the pairing made
sense.

## Feature descriptions

An activation names a feature by number. The publisher writes a label and a description for
each number, and one SAE checkpoint has them: `6b-layer60-k64-cb16384`. Ask for any other
slug and `descriptions` raises rather than answering nothing, because a description written
against one codebook says nothing about another's features.

Fetch the whole set once:

```bash
protein esm features prepare
protein esm features status
```

That is the one step in this lane that needs the network, so run it on a login node. See
[Set up your data](../data.md).

Then join by index, against an activation from that same SAE:

```python
from protein.embed.esm.features import descriptions

frame = descriptions("6b-layer60-k64-cb16384")
frame.loc[activation.indices[0]]  # what fired at the first residue
```

The frame is read once and then shared, so do not change it in place.

## From the command line

`protein esm embed` takes a FASTA file holding exactly one record:

```bash
protein esm embed query.fasta --checkpoint 300m --layer -1 --out p12345.npy
```

| Option | What it does |
| --- | --- |
| `--checkpoint` | Which slug: `300m`, `600m` or `6b`. |
| `--layer` | Which hidden state; 0 is the embedding layer, -1 the last. |
| `--device` | Where to run. Omitted, cuda when visible, else cpu. |
| `--out` | Write the array here as a `.npy` file. |
| `--json` | Print JSON instead of plain text. |

It prints the provenance and never the numbers, so `--out` is how you keep them. It exits 1
when the slug is unknown, the layer is out of range, or the file is not exactly one protein.

The [Commands](../reference/commands.md) page has the rest.
