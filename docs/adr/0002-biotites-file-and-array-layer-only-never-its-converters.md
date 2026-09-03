---
search:
  exclude: true
---

# 2. Call biotite's file and array layer only, never its converters

This package takes biotite for parsing and for the values parsing produces — `FastaFile` with
`read_iter` and `write_iter`, `pdbx.CIFFile`, `get_structure`, `filter_amino_acids`,
`ProteinSequence` and `AtomArray` — and never for its convenience converters.
`fasta.get_sequence`, `fasta.get_sequences`, `structure.to_sequence` and
`ProteinSequence.convert_letter_3to1` are banned by name.

The reason is measured, not stylistic. biotite's `ProteinSequence` alphabet holds the twenty
residues plus `B`, `Z`, `X` and `*`, with no `U`, `O` or `J`. Its converters do not fail on
the three it lacks: they rewrite `U` to `C` and `O` to `K`, silently. Those are different
residues — selenocysteine reported as cysteine — in 285 Swiss-Prot entries. A package that
hands one back has lied at its own boundary, which is the one thing this package does not do.
The fourth has the defect from the other side: `convert_letter_3to1("SEC")` answers `C` where
`structure.info.one_letter_code` answers `U`.

So the string-to-sequence step is ours and there is exactly one of it,
`protein.seq.to_protein_sequence`. It folds those three codes to `X`, which means unknown and
is true, warns with the accession named, and refuses `*` although biotite accepts it. `X` is
also what `mmseqs databases` writes for `U` and `O`, so a downloaded database and a freshly
parsed FASTA agree about them.

What it costs: a little more code than the one-line converter, and a rule that has to be
known before it is obeyed. So it is not left to review — a test walks `src/protein/**/*.py`
and fails on any reference to those names.

What it buys: every sequence in this package entered through one door, and adopting a file
that folded its own alphabet before we met it stays a labelled choice rather than an
accident.
