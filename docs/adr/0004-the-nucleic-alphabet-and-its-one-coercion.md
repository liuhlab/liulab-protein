---
search:
  exclude: true
---

# 4. The nucleic alphabet, and its one coercion

Nucleic acids enter through `protein.seq.to_nucleotide_sequence` as
`biotite.sequence.NucleotideSequence`. There is no `DNA` class and no `RNA` class.

Neither would have anything to be identified by. A `Protein` is a UniProt accession and a
`Structure` is a file; a strand handed in for folding is a string of bases and nothing else. A
class over it would hold one field and add a name, and this package holds biotite's types
rather than wrapping them (ADR-0002). `NucleotideSequence` also already carries the
distinction such a split would encode badly: its `ambiguous` flag chooses between `ACGT` and
`ACGT` plus the eleven IUPAC codes, and it chooses from the symbols it is given, so no caller
does.

**Accepted:** the four bases, the eleven ambiguity codes, and `U`, case-insensitively.
**Coerced:** `U` becomes `T`, loudly. **Refused:** everything else, as `InvalidResidueError`
carrying its offenders as data — one error shape for both alphabets.

The coercion is the nucleic twin of folding `U`, `O` and `J` to `X` (ADR-0001). biotite's
nucleotide alphabet holds no uracil, so RNA spelled the way everyone spells it would otherwise
raise `AlphabetError` deep inside biotite. Thymine sits where uracil does, so the bases
survive and only the spelling is lost — which is why the fold warns rather than saying
nothing. It reuses `ResidueCoercionWarning`, so `pyproject.toml` needs no new
`filterwarnings` entry.

`Chain.sequence` therefore answers with either of biotite's two types, and refuses only a
chain of neither kind. `kind` is what says which will come back, and it is unchanged.

What it costs. Reading an RNA chain warns once, and a caller who wanted to know whether they
were handed DNA or RNA cannot ask the sequence afterwards — for a chain the answer is `kind`,
and for a folding request it is the input. The other cost is that `ESMC.embed` had been relying
on `Chain.sequence` refusing a DNA chain; `A`, `C`, `G`, `T` and `N` are protein letters too,
so the refusal moved to the tokenizer's own door, where it is about the sequence type rather
than about `kind`.

Widening the alphabet is a later decision with a later reason. Modified bases and a gap column
both have real uses; neither has one here yet.
