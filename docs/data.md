# Set up your data

Point the package at the lab's data directory, then bring in the databases you search.

## One data directory for the lab

Databases and maps are gigabytes, and the lab shares one copy of them. Everything this
package reads and writes lives under one directory that you name.

Name it with `LIULAB_DATA`:

```bash
export LIULAB_DATA=/path/to/lab-data
```

Leave it unset and a couple of well-known lab locations are tried in turn. The first one
that is there wins. If neither is, you get `~/liulab_data`.

Everything this package writes lands under `$LIULAB_DATA/protein/`:

| Path | What it holds |
| --- | --- |
| `db/<name>/` | one registered database |
| `esm/sae-features/` | what each of an SAE's features means |
| `sifts/` | the PDB to UniProt map |
| `structures/` | coordinate files, cached as you ask for them |
| `.work/` | scratch space a search makes and then removes |

None of those directories is made until something writes to it.

## Register a database

A database is registered by being there. There is no central list. A directory with a
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
```

Then ask what you have. `list` shows every name registered here, plus every name this
package knows how to fetch. `status` reports on one:

```bash
protein db list
protein db status swissprot
```

## Two sets from their own publishers

Two smaller sets come from neither search tool. The PDB to UniProt map comes from the EBI,
and the SAE feature descriptions from the ESM project:

```bash
protein sifts prepare
protein esm features prepare
```

## Fetch on a login node

`download`, `sifts prepare` and `esm features prepare` all need the network. So run them on
a login node. The lab's compute nodes have none, and a job that dies for a file you could
have fetched in a second is a wasted job.

## Next

[Search a database](guides/search.md) puts one to work. [Commands](reference/commands.md)
lists the rest.
