---
search:
  exclude: true
---

# 8. A view is asked for, and py3Dmol is held rather than wrapped

`Structure.view()` and `Chain.view()` build a viewer from the atoms the object holds and
return py3Dmol's own `view`. Neither class gains `_repr_html_`.

**py3Dmol is held, the way biotite's types are (ADR-0002).** What it does that is worth
taking is small and annoying to redo: the promise that loads 3Dmol.js once per page around
the AMD globals a notebook leaves lying about, and a `__getattr__` that forwards every
3Dmol.js call. Returning its object hands over that whole library. A wrapper would have to
grow a method per style and would still be the smaller thing. It is one pure-Python module of
seven kilobytes, and it is already in `pixi.lock` — the `esm` package depends on it.

**It is not mirrored into `[tool.pixi.dependencies]`.** Every other runtime dependency is,
so that pixi takes it from conda-forge. That recipe makes IPython a hard dependency where
the wheel makes it an extra, and nothing here imports IPython, so mirroring would put IPython
and its dozen dependencies in the environment that runs the gate.

**A view is asked for, never produced by displaying a structure.** A notebook calls
`_repr_html_` on every display. `Structure` holds a path and parses on first use, and
`Structure.__repr__` is written not to parse, because printing one in a debugger would
otherwise reach RCSB. An HTML repr reinstates exactly that reach, on every cell that ends in
a structure. So displaying one prints `Structure('1UBQ')`, and `view()` is the call that says
you meant it.

**The atoms are serialised, not the file.** A chain has no file of its own, and one path for
both is what makes a chain's view hold that chain rather than the entry around it.

What it costs. The page carries the coordinates inline and loads 3Dmol.js from a CDN, so a
saved page opens only where there is a network, and a large entry makes a large page. Past
`view` itself nothing is typed: a misspelled 3Dmol.js call is forwarded happily and fails in
the browser rather than at the call.
