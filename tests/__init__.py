"""A package, so the root ``conftest.py`` can load ``tests._guards`` as a plugin."""

from __future__ import annotations

import re

#: Every SGR escape sequence rich writes — `\x1b[` then the parameters then `m`.
_SGR = re.compile(r"\x1b\[[0-9;]*m")


def plain_text(rendered: str) -> str:
    """Return ``rendered`` as the characters a reader sees, with the styling removed.

    Typer prints ``--help`` through rich, and **a substring of that output is not a
    substring of the help**. With colour on, rich's option highlighter styles the first
    dash separately from the rest, so ``--json`` is written as
    ``\\x1b[1;36m-\\x1b[0m\\x1b[1;36m-json\\x1b[0m`` and ``"--json" in result.output`` is
    ``False``. Colour is off under a bare ``ssh`` and on under GitHub Actions, which is why
    #17 found this as a red CI job beside a green gate. Assert against this, never against
    ``result.output`` directly.
    """
    return _SGR.sub("", rendered)
