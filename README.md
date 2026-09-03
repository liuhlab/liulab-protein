# liulab-protein

Handling protein sequence related tasks.

A `Protein` is one UniProt accession's sequence. A `Structure` is one PDB entry, and a
`Chain` is one polymer inside it. The three are peers, joined by the SIFTS map rather than
owned by each other. On top of them: sequence search with MMseqs2, shape search with
Foldseek, and embeddings from ESM-C.

## Set it up

This repo uses [pixi](https://pixi.sh) and nothing else — no pip, no conda, no uv.
Clone it, then:

```bash
pixi install
```

`mmseqs` and `foldseek` come with it. Ask the package whether it can see them:

```bash
pixi run protein doctor
```

Embedding needs torch, which is heavy enough to live in its own environment:

```bash
pixi install -e esm
```

## Use it

```python
from protein import Protein, Structure

p = Protein("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", id="P12345")
p.length  # 33
p.search("swissprot")  # a DataFrame of MMseqs2 hits

s = Structure("1UBQ")  # coordinates arrive on first use
s["A"].uniprot  # ('P0CG48',) — from SIFTS, not from the file
s["A"].search("pdb")  # a DataFrame of Foldseek hits
```

There is a command for each of those:

```text
protein version | doctor
protein db        list | adopt | download | status
protein esm       embed
protein search    seq | struct
protein sifts     prepare | status
protein structure fetch | show
```

## Point it at the data

Databases are gigabytes and shared, so they live under the lab data directory rather than
in the repo. Name it once:

```bash
export LIULAB_DATA=/scratch/zhoulab/hanliu
```

Everything lands under `$LIULAB_DATA/protein/`: databases in `db/<name>/`, the SIFTS map in
`sifts/`, cached coordinates in `structures/`. Register a database that is already on disk
with `protein db adopt`, or fetch one with `protein db download`. The docs explain both.

## Check your work

```bash
pixi run check
```

That runs the linters, the type checker and the tests. It reports every failure at once, so read to
the bottom before you fix anything.

## Read the docs

The site is at <https://liuhlab.github.io/liulab-protein/>.
Build it yourself with `pixi run docs-build`.

## Set up your agent

Skills for coding agents live in `skills/`. Link them into each agent's own folder:

```bash
python skills/install.py --target all
```

If you work on the lab's clusters, add the shared plugin once per machine:

```text
/plugin marketplace add liuhlab/liulab-compute-skills
/plugin install lab-compute@liulab
```
