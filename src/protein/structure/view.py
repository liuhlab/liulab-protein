"""Coordinates in a 3D viewer: py3Dmol built, held, and handed back.

**py3Dmol is the viewer and its own object is what a caller gets.** It writes the ``<div>``
and the ``<script>`` that load 3Dmol.js from a CDN, and its ``__getattr__`` forwards every
3Dmol.js call, so returning it hands over the whole API rather than the slice this module
thought of. A wrapper would grow a method per style and end up the smaller thing. That is the
same rule biotite's types are held under (ADR-0002), applied to a second library.

**A view is asked for, never produced by displaying a structure.** Neither
:class:`~protein.structure.structure.Structure` nor :class:`~protein.structure.chain.Chain`
gains ``_repr_html_``: a notebook calls that on every display, which would parse the file and
could fetch it from RCSB — the reach :meth:`~protein.structure.structure.Structure.__repr__`
is written to avoid. ADR-0008.

**The atoms are serialised, never the file.** A chain has no file of its own, so one path for
both is what makes a chain's view hold that chain. What is written carries ``atom_site`` and
no secondary structure; 3Dmol.js computes its own, so a cartoon still draws.

Examples
--------
>>> from protein import Structure
>>> view(Structure("1UBQ").atoms, name="1UBQ")             # doctest: +SKIP
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    import py3Dmol
    from biotite.structure import AtomArray

__all__ = ["DEFAULT_HEIGHT", "DEFAULT_STYLE", "DEFAULT_WIDTH", "VIEW_FORMAT", "view"]

#: What the coordinates are handed to 3Dmol.js as. mmCIF, which is the one format this
#: package writes, and the one that survives a chain label of more than one character.
VIEW_FORMAT = "cif"

#: How big the viewer is when nobody says. py3Dmol's own defaults, in pixels; a string is
#: taken as CSS, so ``width="100%"`` fills the column it sits in.
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

#: What is drawn when no style is named: one ribbon per chain, coloured N to C. Water and
#: ligands carry no cartoon, so an entry that is only those draws nothing until a caller
#: says otherwise.
DEFAULT_STYLE: dict[str, Any] = {"cartoon": {"color": "spectrum"}}


def view(
    atoms: AtomArray,
    *,
    name: str,
    width: int | str = DEFAULT_WIDTH,
    height: int | str = DEFAULT_HEIGHT,
    style: Mapping[str, Any] | None = None,
) -> py3Dmol.view:
    """Return a py3Dmol viewer holding ``atoms``.

    Parameters
    ----------
    atoms : biotite.structure.AtomArray
        What to show. Every atom given is written, so filtering is the caller's.
    name : str
        What to call the data block — the structure's or the chain's own id.
    width, height : int or str, optional
        The viewer's size. An integer is pixels; a string is CSS, so ``"100%"`` fills the
        column the viewer sits in.
    style : mapping, optional
        A 3Dmol.js style, e.g. ``{"stick": {}}``. Defaults to :data:`DEFAULT_STYLE`. An
        empty mapping draws nothing, which is how a caller takes the styling over.

    Returns
    -------
    py3Dmol.view
        The viewer. ``.show()`` draws it in a notebook, ``.write_html()`` returns the HTML a
        page embeds, and ``.write_html(handle)`` writes a whole page into an **open file** —
        hand it a path instead and py3Dmol opens that file and never closes it. Every
        3Dmol.js call — ``setStyle``, ``addSurface``, ``zoomTo`` — is forwarded by it.

    Examples
    --------
    >>> from protein import Structure
    >>> viewer = view(Structure("1UBQ").atoms, name="1UBQ")     # doctest: +SKIP
    >>> with open("1ubq.html", "w") as page:                    # doctest: +SKIP
    ...     viewer.write_html(page)
    """
    # Deferred like every other optional-at-import cost here: nothing that merely imports
    # `protein.structure` should pay for the viewer.
    import py3Dmol

    from protein.io import structure as _io

    return py3Dmol.view(
        data=_io.to_text(atoms, name=name),
        format=VIEW_FORMAT,
        # py3Dmol takes a CSS string as readily as a number — turning an int into one is the
        # first thing it does — but it annotates neither parameter, so pyright reads `int`
        # off the default and refuses the string half of what the library accepts.
        width=width,  # pyright: ignore[reportArgumentType]
        height=height,  # pyright: ignore[reportArgumentType]
        style=dict(DEFAULT_STYLE if style is None else style),
    )
