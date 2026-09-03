---
search:
  exclude: true
---

# 1. A protein validates its sequence at construction

`Protein.__init__` routes its input through `protein.seq.to_protein_sequence`, which raises
`InvalidResidueError` on anything outside the alphabet and folds `U`, `O` and `J` to `X` with
a warning. An in-memory `Protein` therefore cannot hold a stop symbol, a gap, a digit or a
space, and it carries a biotite `ProteinSequence` from its first line.

This departs from `liulab-genome`'s ADR-0005, which declines the same check on two grounds.
Neither carries here. Scanning every character of a chromosome costs too much — but biotite
has to walk a protein string to build a `ProteinSequence` anyway, so the check rides along on
a pass the constructor was already making. And a reference full of `N` runs would fail a
strict constructor — but the code a protein database writes for the same thing, `X`, is in
this alphabet. The uncertainty this package meets has an accepted spelling; the one genome
meets does not.

What it costs: a caller holding text of unknown quality can no longer build a `Protein` and
inspect it, and must repair the string first. That is why `InvalidResidueError` carries
`.offenders` as data rather than only a message, and why the fold warns instead of raising.

What it buys: every consumer downstream — an ESM-C tokenizer, an mmseqs query FASTA, a
`Chain` sequence compared against one — can skip the check, and a stray `*` fails beside the
file it came from rather than deep inside a subprocess.

The guarantee is about construction, and only that. Assigning `p.sequence` afterwards reaches
around the check, and nothing tries to stop it — a guard at the door is worth having without
being a cage.
