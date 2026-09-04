# Embed a sequence

Turn a sequence into one vector per residue with ESM-C, and read those vectors as sparse
features.

## What you need first

ESM-C needs torch, which lives in the `esm` environment. It also wants a GPU:

```bash
pixi install -e esm
pixi run -e esm protein esm embed query.fasta
```

Run every command on this page that way. See [Install](../install.md).

## Embed one sequence

Build `ESMC()` once, keep it, and call `.embed()` as often as you like:

```python
from protein import ESMC, Protein

model = ESMC()
p = Protein("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", id="P12345")
e = model.embed(p)
```

Name a checkpoint by slug: `300m`, `600m` or `6b`. An unknown slug fails right there, by
name, before anything is downloaded. Omit `device=` and the weights go to the GPU when torch
can see one, and to the CPU when it cannot. Either way `model.device` says where they went,
so a run that quietly fell back to the CPU tells you.

`embed()` takes a `Protein` or a `Chain`. Hand it a plain string and it raises, because a
string cannot say afterwards what was embedded.

## What one call gives back

An `Embedding`. It holds a `(L, d_model)` float32 array on the CPU. The model adds a start
and an end token, and both are cut off before you see the array. So `L` is the residue
count, and `len(e)` is the length of the sequence you handed in.

It also carries the three facts that say what it is:

- `source` — the accession or chain key it came from.
- `checkpoint` — the slug, such as `300m`.
- `layer` — which hidden state, as a non-negative index.

That is what lets you tell two embeddings apart an hour later. Arrays from different models
or different layers are not comparable.

```python
import numpy as np

np.asarray(e)  # the array itself
e.shape  # (L, d_model)
e.mean()  # the per-sequence vector, (d_model,)
e.as_json()  # source, checkpoint, layer and shape
```

Every numpy function takes an `Embedding` straight, so you rarely reach for `e.array`. You
cannot edit one in place.

## Choosing a layer

`embed()` takes `layer=`, and it indexes the hidden states the way the model does. Layer 0
is what the embedding layer put out. The last layer is the last hidden state. Negative
numbers count back from the end, and the default is `-1`.

`model.n_layers` is how many transformer layers the checkpoint has, and there is one more
hidden state than that. A layer outside the range raises, and the message spells the range
out.

What comes back is stored normalised. Ask for `layer=-1` and the `Embedding` records the
matching non-negative index, so you can compare two of them later.

Pick the layer with your next step in mind. A sparse autoencoder covers particular layers,
and its slug names them, so pass the layer yours covers if you plan to encode with one.
Otherwise leave `layer` alone and take the last hidden state.

## Sparse features

A sparse autoencoder splits an embedding row into named features. `SAE(slug)` loads one and
`encode()` runs it over an `Embedding`:

```python
from protein.embed import SAE

sae = SAE("300m-layer23-k64-cb16384")
activation = sae.encode(model.embed(p, layer=23))
```

`SAE` has no default slug. Name the backbone and the autoencoder, or neither.

What comes back is a `SaeActivation`. It holds which features fired at each residue, and how
strongly. `.max()` is the per-sequence vector: each feature's strongest hit anywhere in the
protein. There is no `.mean()`, because an average over length would report the same feature
as weaker in a long protein than in a short one.

Only the strongest features per residue are stored. `activation.dense()` builds the full
matrix when you ask for it, and nothing keeps a copy.

Every activation also carries one reconstruction loss per residue: how badly the decoder
rebuilt that residue. Read those when you want to know whether the pairing worked.

## What encode refuses

Pair the wrong backbone with an autoencoder, or the right backbone's wrong layer, and
`encode` raises. The message names the one that disagrees.

You want that error. The shapes can agree anyway, so the multiply works, numbers come out,
and they look like features. Nothing in the numbers themselves would tell you.

`normalize=True` scales the magnitudes by the checkpoint's own per-feature numbers. Ask for
it where a checkpoint ships none and it raises, rather than scaling every feature by one and
calling that normalised. Whichever way it went, `activation.normalized` says so.

## Feature descriptions

An activation names a feature by number. Labels and descriptions are published for one
autoencoder: `6b-layer60-k64-cb16384`. Ask for any other slug and `descriptions` raises.

Fetch the whole set once:

```bash
protein esm features prepare
protein esm features status
```

That is the one step in this lane that needs the network, so run it on a login node. See
[Set up your data](../data.md).

Then join by index, against an activation from that same autoencoder:

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

It prints where the embedding came from and never the numbers themselves, so `--out` is how
you keep them. It exits 1 when the slug is unknown, the layer is out of range, or the file
does not hold exactly one protein.

The [Commands](../reference/commands.md) page has the rest.
