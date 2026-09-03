"""A package, so the root ``conftest.py`` can load ``tests._guards`` as a plugin."""

from __future__ import annotations

import re

#: The SGR escape sequences rich writes.
_SGR = re.compile(r"\x1b\[[0-9;]*m")


def plain_text(rendered: str) -> str:
    """Return ``rendered`` as the characters a reader sees, with the styling removed.

    Typer prints ``--help`` through rich, and **a substring of that output is not a
    substring of the help**: with colour on, rich's option highlighter styles the first dash
    separately from the rest, so ``"--json" in result.output`` is ``False``. Colour is off
    under a bare ``ssh`` and on in CI. Assert against this, never against ``result.output``.
    """
    return _SGR.sub("", rendered)
