---
search:
  exclude: true
---

# 7. An SAE activation is a peer of `Embedding`, not one

A sparse autoencoder takes an `Embedding` row for row and gives back, per residue, which
codebook features fired and how hard. `SaeActivation` is a frozen value object of its own,
and `Embedding` is untouched.

Making it an `Embedding` was the cheaper move: one type, one set of tests, and every caller
that already takes an embedding takes this too. It fails on the three facts that type
promises. `d_model` would name the codebook, which is wider than the model by more than an
order of magnitude and is not a model dimension at all. `.mean()` would divide a feature's
presence by protein length, so the same domain fired in a long protein reads weaker than in
a short one — the reduction that is right for a dense embedding is wrong here, and it would
be the one already on the class. And `__array__` would have to pick one of three arrays.

Widening `Embedding` with a fourth fact fails for the same reason from the other side: it
would make every embedding carry a codebook it does not have.

**What the peer costs.** Two types where a reader expected one, and a caller writing
generic code over "whatever a model returned" has to branch. Nothing is shared between them
— not the array plumbing, not the provenance fields, not `repr` — so a change to how
provenance is spelled is now two edits. That duplication is accepted: what looks shared is
two different contracts that happen to rhyme.

**What the peer buys.** Every name on it is true. `.max()` is the per-sequence vector and
there is no `.mean()` at all, so the misleading reduction is not one keystroke away.
`.dense()` materialises the codebook only when asked, and the pair it materialises from is
lossless, because top-k fills exactly `k` slots per row whether or not all `k` are non-zero.
The seven facts it carries — source, parent, layer, SAE, codebook size, `k` and whether
normalisation happened — are what make two activation sets comparable or provably not.

It is numpy and nothing else, so all of that is checked in the gate with no weights.
