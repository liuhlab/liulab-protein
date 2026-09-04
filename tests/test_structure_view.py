"""Tests for `Structure.view()` and `Chain.view()` — what the viewer is handed.

The claim worth holding is the one a reader cannot check by eye: the coordinates embedded in
the page parse back to exactly the atoms the object holds, so a chain's view shows that chain
and a structure's shows the whole entry. Everything runs over the committed entries in
`tests/data`, whose provenance is in `tests/data/README.md`.

Nothing here renders anything. py3Dmol writes a `<div>` and a `<script>` and needs no browser
and no IPython to do it; `show()` is py3Dmol's own and needs a notebook, so it is not tested.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from biotite.structure import AtomArray

from protein import Chain, Structure
from protein.io import structure as io
from protein.structure import view as view_module

_DATA = Path(__file__).resolve().parent / "data"
_UBQ = _DATA / "1ubq.cif.gz"
_BNA = _DATA / "1bna.cif.gz"

#: How py3Dmol hands coordinates to 3Dmol.js: `addModel(<json string>, <json format>, ...)`.
#: Matching it is deliberate — this is the one place a page's content can be read back, and a
#: py3Dmol that stopped spelling it this way should fail here rather than ship a blank box.
_MODEL = re.compile(r'addModel\(("(?:[^"\\]|\\.)*"),("[a-z]+"|undefined)')


@pytest.fixture
def ubq() -> Structure:
    """`1UBQ`, named as its entry so a chain key reads `1UBQ_A`."""
    return Structure.from_file(_UBQ, id="1UBQ")


@pytest.fixture
def bna() -> Structure:
    """`1BNA`, the one committed entry with two chains."""
    return Structure.from_file(_BNA, id="1BNA")


def _page(viewer: Any) -> str:
    """Return the HTML one viewer writes.

    py3Dmol's `write_html` answers `None` when it is handed a path, so its type is
    `str | None` and every caller here would otherwise carry the same two lines.
    """
    html = viewer.write_html()
    assert isinstance(html, str)
    return html


def _model(html: str) -> tuple[str, str]:
    """Return the coordinates and the format name one page hands the viewer."""
    match = _MODEL.search(html)
    assert match is not None, "no addModel call in the page"
    return json.loads(match[1]), json.loads(match[2])


def _reparsed(html: str, tmp_path: Path) -> AtomArray:
    """Parse a page's own coordinates back into an atom array."""
    text, _ = _model(html)
    path = tmp_path / "embedded.cif"
    path.write_text(text, encoding="utf-8")
    return io.read_atoms(path)


# --- what reaches the viewer ---------------------------------------------------


def test_a_structures_page_carries_every_atom_it_holds(ubq: Structure, tmp_path: Path) -> None:
    embedded = _reparsed(_page(ubq.view()), tmp_path)
    assert embedded.array_length() == ubq.atoms.array_length()


def test_a_chains_page_carries_that_chain_and_no_other(bna: Structure, tmp_path: Path) -> None:
    # The whole reason the chain is serialised rather than filtered in the viewer: `1BNA` has
    # two chains, and a page for one of them holds one of them.
    chain: Chain = bna["A"]
    embedded = _reparsed(_page(chain.view()), tmp_path)
    assert embedded.array_length() == chain.atoms.array_length()
    assert embedded.array_length() < bna.atoms.array_length()
    assert io.chain_ids(embedded) == ("A",)


def test_the_coordinates_are_handed_over_as_mmcif(ubq: Structure) -> None:
    text, spelling = _model(_page(ubq.view()))
    assert spelling == view_module.VIEW_FORMAT == "cif"
    assert text.startswith("data_1UBQ")


def test_a_chains_page_is_named_after_the_chain(ubq: Structure) -> None:
    # So a saved page says which chain it holds, the way Foldseek's query column does.
    text, _ = _model(_page(ubq["A"].view()))
    assert text.startswith("data_1UBQ_A")


# --- what the page is ----------------------------------------------------------


def test_the_page_holds_a_viewer_and_the_library_that_draws_it(ubq: Structure) -> None:
    html = _page(ubq.view())
    assert "3dmolviewer_" in html
    assert "3Dmol-min.js" in html
    assert "createViewer" in html


def test_written_into_an_open_file_it_is_a_page_a_browser_opens(
    ubq: Structure, tmp_path: Path
) -> None:
    # An open file rather than a path: handed a path, py3Dmol opens it and never closes it,
    # and `filterwarnings = ["error"]` turns that unclosed handle into a failed test.
    target = tmp_path / "1ubq.html"
    with target.open("w", encoding="utf-8") as handle:
        ubq.view().write_html(handle)
    page = target.read_text(encoding="utf-8")
    assert page.startswith("<html>")
    assert "3Dmol-min.js" in page


# --- the knobs -----------------------------------------------------------------


def test_a_size_is_pixels_when_it_is_a_number_and_css_when_it_is_a_string(
    ubq: Structure,
) -> None:
    assert "width: 640px" in _page(ubq.view())
    assert "width: 100%" in _page(ubq.view(width="100%"))


def test_the_default_style_is_one_ribbon_coloured_along_the_chain(ubq: Structure) -> None:
    assert '"cartoon": {"color": "spectrum"}' in _page(ubq.view())


def test_an_empty_style_hands_the_drawing_to_the_caller(ubq: Structure) -> None:
    # `{}` is a style and not a missing argument, so it must not fall back to the default.
    # py3Dmol writes no `setStyle` call for one, which leaves 3Dmol.js drawing nothing until
    # the caller says what to draw — the point of asking for it.
    html = _page(ubq.view(style={}))
    assert "setStyle" not in html
    assert "spectrum" not in html
    assert "addModel" in html


# --- what deliberately is not here ---------------------------------------------


@pytest.mark.parametrize("held", [Structure, Chain])
def test_neither_class_renders_itself_when_it_is_displayed(held: type) -> None:
    # ADR-0008. A notebook calls `_repr_html_` on every display, which would parse the file
    # and could fetch it from RCSB — the reach `Structure.__repr__` is written to avoid.
    assert not hasattr(held, "_repr_html_")
