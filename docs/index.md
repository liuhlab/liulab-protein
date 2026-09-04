# liulab-protein

Handling protein sequence related tasks.

A `Protein` is one UniProt accession's sequence. A `Structure` is one PDB entry, and a
`Chain` is one polymer inside it. The three are peers rather than parents and children: a
protein turns up in many structures, and a structure holds many proteins. The SIFTS map is
what joins them.

On top of those sit the jobs. Search sequences with MMseqs2, search shapes with Foldseek,
line up homologues with MUSCLE, embed a sequence with ESM-C, and predict a structure with
ESMFold2.

```python
from protein import Protein, Structure

p = Protein("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", id="P12345")
p.length  # 33
p.search("swissprot")  # a DataFrame of MMseqs2 hits

s = Structure("1UBQ")  # coordinates arrive on first use
s.chain_ids  # ('A',)
s["A"].uniprot  # ('P0CG48',) — from SIFTS, not from the file
s["A"].search("pdb")  # a DataFrame of Foldseek hits
```

Every one of those has a command as well. `protein --help` lists them.

## Where to go next

| If you want to | Read |
| --- | --- |
| watch it do one real job, start to finish | [Getting started](start.md) |
| know why the pieces are shaped this way | [How it fits together](concepts.md) |
| get it running | [Install](install.md), then [Set up your data](data.md) |
| find similar sequences or similar folds | [Search a database](guides/search.md) |
| build an alignment | [Build an alignment](guides/alignments.md) |
| turn a sequence into vectors | [Embed a sequence](guides/embedding.md) |
| predict a structure | [Predict a structure](guides/folding.md) |
| look up a command or a class | [Commands](reference/commands.md), [Python API](api.md) |
| send a change | [Contributing](contributing.md) |

Some notes here are written for coding agents rather than for people. Conventions live under
`docs/agents/`, decision records under `docs/adr/`, and research notes under
`docs/research/`. None of those show up in the menu or the search box, and each page is
still reachable by its own URL.
