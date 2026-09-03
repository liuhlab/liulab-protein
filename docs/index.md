# liulab-protein

Handling protein sequence related tasks.

Three classes carry the work. A `Protein` is one UniProt accession's sequence. A `Structure`
is one PDB entry, and a `Chain` is one polymer inside it. They are peers, not parents and
children: a protein can appear in many structures, and a structure can hold many proteins.
The SIFTS map is what joins them.

On top of that sit three things you can do. Search sequences with MMseqs2, search shapes
with Foldseek, and embed a sequence with ESM-C.

## Install it

The repo uses [pixi](https://pixi.sh) and nothing else. No pip, no conda, no uv. Clone the
repo, then:

```bash
pixi install
```

That reads `pyproject.toml` and builds the environment from the lock file, so you get the
same versions the tests ran on. It also installs `mmseqs` and `foldseek`, so there is
nothing to go and fetch by hand. Ask the package whether it can see them:

```bash
pixi run protein doctor
```

Embedding is the one exception. ESM-C needs torch, which is heavy enough that it lives in
its own environment:

```bash
pixi install -e esm
pixi run -e esm protein esm embed query.fasta
```

## Use it

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

Embedding is a class you build and keep, because the weights are large and somebody has to
own them:

```python
from protein import ESMC

model = ESMC()  # loads once
model.embed(p).shape  # (33, 960)
```

Everything above has a command as well:

```text
protein version | doctor
protein db        list | adopt | download | status
protein esm       embed
protein search    seq | struct
protein sifts     prepare | status
protein structure fetch | show
```

Every command takes `--json`. The [API reference](api.md) has the rest, built from the
docstrings in `src/`.

## Where the big files live

Databases and maps are gigabytes, they are shared, and they are the same files
`liulab-genome` reads. So there is one data directory for the lab and this package takes a
subdirectory of it, rather than inventing a second root.

Name it with `LIULAB_DATA`:

```bash
export LIULAB_DATA=/scratch/zhoulab/hanliu
```

If you do not, a couple of well-known lab paths are tried, and then `~/liulab_data`.
Everything this package writes lands under `$LIULAB_DATA/protein/`:

| Path | What it holds |
| --- | --- |
| `db/<name>/` | one registered database |
| `sifts/` | the PDB to UniProt map |
| `structures/` | coordinate files, cached as you ask for them |
| `.work/` | scratch space a search makes and then removes |

A database is registered by being there. There is no central list: a directory with a
completion record in it is a registered database, and its directory name is the name you
type. That means two ways in.

Use `adopt` when the files are already on disk, which is the usual case on a cluster. It
writes a record beside them and copies nothing:

```bash
protein db adopt swissprot /path/to/swissprot
```

Use `download` when they are not. It hands the job to `mmseqs databases` or `foldseek
databases`, then registers what they left behind:

```bash
protein db download swissprot
protein db list
```

The SIFTS map is smaller and comes from the EBI rather than from either tool:

```bash
protein sifts prepare
```

Both of those need the network, so run them on a login node. The lab's compute nodes have
none, and a job that dies for a file you could have fetched in a second is a wasted job.

## Check your work

One command runs the linters, the type checker and the tests:

```bash
pixi run check
```

It runs every step, then prints all the failures at once. Read to the bottom before you
fix anything.

Some notes here are written for coding agents, not for people. Conventions go under
`docs/agents/`, decision records under `docs/adr/`, and research notes under
`docs/research/`. Nothing in those three directories shows up in the menu or the search
box, and a page written there is still reachable by its own URL.
