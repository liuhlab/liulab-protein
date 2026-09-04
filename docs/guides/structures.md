# Work with structures

A `Structure` is one PDB entry, and a `Chain` is one polymer inside it. This page covers
getting one, reading what is in it, and looking at it in 3D.

## Get a structure

Name a PDB entry:

```python
from protein import Structure

s = Structure("1UBQ")
```

Nothing is read yet. The coordinates arrive the first time you ask for atoms, from a local
cache or from RCSB when the cache does not have them. [Set up your data](../data.md) says
where that cache lives.

A machine with no network raises `CoordinatesNotDownloadedError`. Fill the cache from a login
node first:

```bash
protein structure fetch 1UBQ
```

For a file you already hold, use `from_file`:

```python
s = Structure.from_file("model.cif")
```

It takes mmCIF or PDB, gzipped or not. The structure is named after the file. Pass `id=` to
call it something else.

## What is inside

`chain_ids` gives you the labels, in file order:

```python
s.chain_ids  # ('A',)
```

Index by label for one chain. `chains` hands you all of them:

```python
first = s["A"]
for chain in s.chains:
    print(chain.id, chain.kind)
```

Case is part of a chain label, so `a` and `A` are two chains. A label that is not there
raises `KeyError`, and the message lists the labels that are.

`atoms` is every atom of the first model, as a biotite `AtomArray`:

```python
s.atoms.array_length()
```

Ligands, water and nucleic acids are all in there. Nothing is filtered out. An NMR entry has
several models, and `models` holds them all.

## Read one chain

Three things a chain answers:

```python
chain = Structure("1UBQ")["A"]
chain.kind  # 'protein'
chain.sequence  # the residues that were solved
chain.uniprot  # ('P0CG48',)
```

Check `kind` first. It is `"protein"`, `"nucleic"` or `"other"`. A chain may hold DNA or RNA,
and then `sequence` is a nucleotide sequence rather than a protein one. A chain of only
ligand or water is `"other"`, and asking it for a sequence raises `ValueError`.

`sequence` is what was solved for. A residue with no coordinates is missing from it, rather
than filled in from what the entry says it should be.

`uniprot` is a tuple, and it may be empty. `()` is a real answer: a nucleic chain, a ligand
chain, or an entry SIFTS never curated all give you that. A chain can also belong to more
than one accession, which is why the answer is never a bare string.

The accessions come from SIFTS, so that map has to be prepared here. Without it you get
`SiftsNotDownloadedError`, and [Set up your data](../data.md) has the command that fixes it.
The coordinate file carries a cross-reference of its own, and the two can differ. `1UBQ`
chain A is `P62988` in the file and `P0CG48` here.

## See it in 3D

`view()` builds a viewer from the coordinates a structure holds and hands back
[py3Dmol](https://pypi.org/project/py3Dmol/)'s own object. The viewer below was made by
running the call under it while this site was built:

```python exec="true" html="true" source="above"
from protein import Structure

ubiquitin = Structure.from_file("docs/fixtures/1ubq.cif", id="1UBQ")
print(ubiquitin.view(width="100%", height=420).write_html())
```

Those coordinates are [`1ubq.cif`](../fixtures/1ubq.cif), the file RCSB serves for that entry.
It is kept in the repository so that building this site needs no network. Your own code would
say `Structure("1UBQ")`.

Drag to turn it, scroll to zoom. The colour runs from blue at the start of the chain to red
at the end. The ribbon comes from the coordinates alone: the file records no helix or sheet,
so the viewer works them out.

## In a notebook

Call `show()`:

```python
from protein import Structure

Structure("1UBQ").view().show()
```

Displaying a structure shows no viewer. Call `view()` when you want one.

## As a page of its own

`write_html()` gives you the HTML. Hand it an open file and it writes a whole page you can
open in a browser or send to somebody:

```python
with open("1ubq.html", "w") as page:
    Structure("1UBQ").view().write_html(page)
```

Open the file yourself. py3Dmol will take a path instead, but it opens that file and never
closes it.

The page carries the coordinates inside it and loads 3Dmol.js from a CDN, so it needs a
network to open but no files beside it.

## Show one chain

A chain has the same method, and shows that chain and nothing else:

```python
s = Structure("1BNA")
s["A"].view()  # one strand of the Dickerson dodecamer, not both
```

## Draw something else

Because what comes back is py3Dmol's viewer, every 3Dmol.js call is there. Pass `style={}` to
take the drawing over yourself:

```python
viewer = Structure("1UBQ").view(style={})
viewer.setStyle({"stick": {"colorscheme": "greenCarbon"}})
viewer.addSurface("VDW", {"opacity": 0.6})
viewer.show()
```

A predicted structure carries its per-residue confidence in the B-factor column, so colouring
by it is one call. That column runs from 0 to 100:

```python
confidence = {"prop": "b", "gradient": "roygb", "min": 50, "max": 100}
viewer = prediction.view(style={"cartoon": {"colorscheme": confidence}})
```

Watch the scale. The B-factor column runs 0 to 100, while `prediction.confidence.plddt` is
the same measure as a fraction between 0 and 1.

## Search by shape

A structure and a chain both search by shape, and both run Foldseek:

```python
s.search("pdb")  # every chain, one run
s["A"].search("pdb")  # one chain
```

Each gives you back a table of hits. [Search a database](search.md) has the columns and the
knobs.
