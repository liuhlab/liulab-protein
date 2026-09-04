---
search:
  exclude: true
---

# 2. Call biotite's file and array layer only, never its converters

This package takes biotite for parsing and for the values parsing produces — `FastaFile` with
`read_iter` and `write_iter`, `pdbx.CIFFile`, `get_structure`, `filter_amino_acids`,
`ProteinSequence` and `AtomArray` — and never for its convenience converters.
`fasta.get_sequence`, `fasta.get_sequences`, `structure.to_sequence` and
`ProteinSequence.convert_letter_3to1` are banned by name, and so are the two below.

biotite's `ProteinSequence` alphabet holds the twenty residues plus `B`, `Z`, `X` and `*`,
with no `U`, `O` or `J`. Its converters do not fail on the three it lacks: they rewrite `U` to
`C` and `O` to `K`, silently — selenocysteine reported as cysteine. A package that hands one
back has lied at its own boundary, which is the one thing this package does not do. The
fourth has the defect from the other side: `convert_letter_3to1("SEC")` answers `C` where
`structure.info.one_letter_code` answers `U`.

So the string-to-sequence step is ours and there is exactly one of it,
`protein.seq.to_protein_sequence`. It folds those three codes to `X`, which means unknown and
is true, warns with the accession named, and refuses `*` although biotite accepts it. `X` is
also what `mmseqs databases` writes for `U` and `O`, so a downloaded database and a freshly
parsed FASTA agree about them.

What it costs: a little more code than the one-line converter, and a rule that has to be
known before it is obeyed. So it is not left to review — a test walks `src/protein/**/*.py`
and fails on any reference to those names.

The guard names two alignment converters too. `fasta.get_a3m_alignments` reads an A3M through
the same rewrite; `fasta.set_alignment` is the write half, and an `Alignment` uppercases its
sequences and renders every gap as `-`, so what leaves through one is no longer the A3M that
came in — in A3M, case is the match state.

It does **not** name `get_alignment`. The walk matches bare names, and
`Muscle5App.get_alignment()` is a safe accessor over sequences this package built and handed
in, so nothing is re-parsed. Banning the name would fire on correct work, and a rule shown to
do that is evidence against the rule.

What it buys: every sequence in this package entered through one door, and adopting a file
that folded its own alphabet before we met it stays a labelled choice rather than an
accident.
