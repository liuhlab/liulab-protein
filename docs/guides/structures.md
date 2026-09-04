# Work with structures

A `Structure` is one PDB entry, and a `Chain` is one polymer inside it. This page is about
looking at one.

## See it in 3D

`view()` builds a viewer from the coordinates a structure holds and hands back
[py3Dmol](https://pypi.org/project/py3Dmol/)'s own object. The viewer below was made by
running the call under it while this site was built:

```python exec="true" html="true" source="above"
from protein import Structure

ubiquitin = Structure.from_file("docs/fixtures/1ubq.cif", id="1UBQ")
print(ubiquitin.view(width="100%", height=420).write_html())
```

Those coordinates are [`1ubq.cif`](../fixtures/1ubq.cif), the file RCSB serves for that
entry, kept in the repository so that building this site needs no network. Your own code
would say `Structure("1UBQ")` and let the coordinates arrive on first use.

Drag to turn it, scroll to zoom. The colour runs from blue at the start of the chain to red
at the end, and the ribbon is drawn from the coordinates alone: the file carries no record of
where its helix and sheets are, so the viewer works them out.

## In a notebook

Call `show()`:

```python
from protein import Structure

Structure("1UBQ").view().show()
```

Printing a structure draws nothing, and that is on purpose. A structure holds a path and
reads the file only when something asks for atoms, so a viewer that appeared every time you
displayed one would parse the file — and download it, on a machine that did not have it yet.
Ask for the viewer, and you choose when that happens.

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

## One chain

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

A predicted structure carries its per-residue confidence in the B-factor column, which is
where every viewer looks for it, so colouring by it is one call:

```python
viewer = prediction.view(
    style={"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb", "min": 0, "max": 1}}}
)
```
