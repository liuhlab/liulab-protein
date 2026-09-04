# Commands

`protein --help` lists every command, and each command has a `--help` of its own. Every one
of them takes `--json`. Each lane's commands live beside the code they call.

## Every command

| Command | What it does |
| --- | --- |
| `protein version` | Print the installed package version. |
| `protein doctor` | Report which native tools are here, and what version each is. |
| `protein db list` | List every database registered here, and every one this package can fetch. |
| `protein db adopt` | Register a database that is already on disk, without copying it. |
| `protein db download` | Fetch a database with the tool's own downloader, then register it. |
| `protein db status` | Say what is on disk for one database. |
| `protein esm embed` | Embed the one record in a FASTA file, and report what came back. |
| `protein esm features prepare` | Download the SAE feature descriptions and store them. |
| `protein esm features status` | Say whether the feature descriptions are here. |
| `protein fold structure` | Fold one sequence with ESMFold2 and report what was written. |
| `protein msa search` | Search a database with one sequence, and write the alignment it found. |
| `protein msa align` | Line up a FASTA of sequences with MUSCLE, anchored on the one you name. |
| `protein search seq` | Search one sequence against a database with MMseqs2. |
| `protein search struct` | Search a structure, or one chain of it, against a database with Foldseek. |
| `protein sifts prepare` | Download the SIFTS map and store it. |
| `protein sifts status` | Say which SIFTS release is here. |
| `protein structure fetch` | Put one PDB entry's coordinates in the cache. |
| `protein structure show` | List a structure's chains, and what is known about each. |

## version and doctor

Neither belongs to a lane. `doctor` exits 1 when a tool is missing, and the message names
the command that installs it. Run it once after you install.

## db

The four database commands. See [Set up your data](../data.md) for where the files go.

- `adopt` copies nothing. It points a name in the data dir at the path you give it, and
  writes a record beside the real files. Use it on a cluster, where the files are usually
  there already.
- `download` does not do the downloading. `mmseqs databases` and `foldseek databases` fetch,
  and this registers what they leave. It needs a network, so run it on a login node.
- `--kind sequence` or `--kind structure` is only for a name this package does not know.
  A declared name, and any name you adopted once, already says which it is.
- `--force` writes a fresh record over one that is there.
- None of the four changes a database. These files are read-only by design, so there is no
  `remove`, no `rebuild` and no `index`.

## esm

Embedding, and the feature descriptions that name what an SAE feature means. See
[Embed a sequence](../guides/embedding.md).

- The FASTA must hold exactly one record. A file with more is refused.
- The numbers are never printed. `--out` writes the array as a `.npy` file, and that is how
  you keep them.
- `--layer` picks the hidden state. 0 is the embedding layer and -1 is the last.
- `--device` is left out by default. It picks cuda when torch can see a GPU, and cpu when it
  cannot. Name one to override that.
- `features prepare` needs the network, so run it on a login node.

## fold

One sequence in, one predicted structure on disk. See
[Predict a structure](../guides/folding.md).

- The output directory is an argument, not an option. The lab data dir holds input and
  reference data, never your outputs, so you have to say where the prediction goes.
- The query is a FASTA file with one record, or the residues themselves. A file that exists
  wins, so a mistyped path is not folded as a sequence.
- The command folds from the sequence alone. There is no flag for an alignment. To hand one
  in, use the [Python API](../api.md).
- `--name` says what to call it. Left out, it uses the accession, and a hash when there is
  none.
- Folding over a name that already holds a different sequence fails. `--overwrite` allows it.
- `--device` works the way it does for `esm embed`.
- Per-residue confidence is the coordinate file's own B-factor column.

## msa

Two ways to build an alignment. See [Build an alignment](../guides/alignments.md).

- Both take the output path as an argument. Nothing durable lands anywhere you did not name.
- `search` takes the residues themselves, then the name of a registered sequence database.
  The alphabet is checked before mmseqs starts.
- `align` needs `--query`, which says what goes in row 0. Give the header, or the identifier
  the header opens with.
- The depth is whatever the search found, and no floor is enforced. A search that matched
  nothing writes the query alone.

## search

One query against one database. See [Search a database](../guides/search.md).

- Rows go to stdout, tab-separated. The column names and the count go to stderr, so
  `protein search seq ... | cut -f2` is the list of targets and nothing else.
- Identity is `pident` for `seq` and `fident` for `struct`. The first is a percentage and
  the second is a fraction. Neither is renamed into the other.
- Without `--chain`, `struct` runs Foldseek once over every chain at once. The `query`
  column then says which chain each hit belongs to.
- `--chain` wants the label exactly as the file spells it. Case counts, and not every label
  is one character.
- `--threads` is worth naming on a shared machine. The default is every core.

## sifts

The PDB to UniProt map, which is the only join between the two namespaces. See
[Set up your data](../data.md).

- `prepare` needs the network, so run it on a login node. Already prepared, it fetches
  nothing and tells you what is there.
- `status` reads the record on disk and nothing else. It says which release you have, never
  whether a newer one exists.
- `protein structure show` fails until the map is prepared, and its message names the
  command that prepares it.

## structure

Fill the coordinate cache, and look inside an entry.

- `fetch` is the one step here that needs a network. Run it before a job starts, because
  the compute nodes have none.
- `show` takes a coordinate file, or a PDB entry id. An id that is not cached is fetched.
- An empty `uniprot` cell is a real answer. It means a nucleic-acid chain, a ligand chain,
  or an entry SIFTS never curated.
- Searching a structure is not here. It is `protein search struct`, beside the sequence
  search.

## The --json flag

`--json` prints one JSON object instead of the plain lines. The answer is the same, keyed
by name, so nothing has to read a printed table. Use it whenever something other than a
person reads the output.

Every command takes it. Without it, `protein search` and `protein structure show` print a
tab-separated table, and the rest print one `key: value` per line. On a `prepare` command,
`--json` also turns the progress bar off.
