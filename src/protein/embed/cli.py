"""The ``protein esm`` sub-app — one FASTA record in, one embedding out.

A thin Typer wrapper over :class:`protein.embed.ESMC`: it reads the record, constructs the
model, makes one call and renders. It ships from this package so that what the command
prints and what :class:`~protein.embed.Embedding` holds change in one place.

The input is a FASTA file holding **exactly one** record, because
:meth:`protein.core.Protein.from_fasta` refuses a file holding more — which is also how the
one-at-a-time rule reaches the command line without a flag saying so.

Nothing here imports torch either: :mod:`protein.embed.esm` keeps that inside method bodies,
so mounting this sub-app costs ``protein --help`` nothing.

Examples
--------
>>> from protein.embed.cli import app
>>> [command.name for command in app.registered_commands]
['embed']
"""

from __future__ import annotations

import json as _json
from pathlib import Path as _Path
from typing import Annotated as _Annotated

import numpy as _np
import typer

from protein.core import Protein as _Protein
from protein.embed.esm import CHECKPOINTS as _CHECKPOINTS
from protein.embed.esm import ESMC as _ESMC

#: What this sub-app catches. A checkpoint slug that is not in the table, a ``layer``
#: outside the model's range, a FASTA holding no record or more than one, and a residue
#: outside the alphabet are all ``ValueError``s, each already carrying its next action. A
#: FASTA that is not there and an ``--out`` that cannot be written are ``OSError``s. What
#: the hub raises for a checkpoint it cannot serve is neither shape reliably, so nothing
#: here translates it: it reaches the caller as itself.
_EMBED_ERRORS = (ValueError, OSError)

app = typer.Typer(help="Embed a protein sequence with ESM-C.", no_args_is_help=True)


# `Annotated` rather than the call-in-the-default spelling the root app uses: ruff's B008
# passes a `str`- or `bool`-annotated default and fails a `Path`-annotated one, and a command
# taking two paths written in two styles reads worse than one written in the newer style
# throughout.
@app.command("embed")
def embed(
    fasta: _Annotated[_Path, typer.Argument(help="A FASTA file holding exactly one record.")],
    checkpoint: _Annotated[
        str,
        typer.Option("--checkpoint", help=f"Which ESM-C checkpoint: {', '.join(_CHECKPOINTS)}."),
    ] = "300m",
    layer: _Annotated[
        int,
        typer.Option("--layer", help="Which hidden state; 0 is the embedding layer, -1 the last."),
    ] = -1,
    device: _Annotated[
        str | None,
        typer.Option("--device", help="Where to run. Omitted, cuda when visible, else cpu."),
    ] = None,
    out: _Annotated[
        _Path | None,
        typer.Option("--out", help="Write the (L, d_model) float32 array here as a .npy file."),
    ] = None,
    json: _Annotated[bool, typer.Option("--json", help="Emit JSON instead of plain text.")] = False,
) -> None:
    """Embed the one record in FASTA and report what came back.

    Prints the provenance and never the numbers: an embedding is megabytes of float32 and a
    terminal is not where those go. ``--out`` is how you keep them.

    Exits with code 1 if the checkpoint slug is unknown, the layer is out of range, or the
    FASTA cannot be read as exactly one protein.
    """
    try:
        protein = _Protein.from_fasta(fasta)
        embedding = _ESMC(checkpoint, device=device).embed(protein, layer=layer)
        if out is not None:
            _np.save(out, embedding.array)
    except _EMBED_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err

    written = embedding.as_json()
    written["out"] = str(out) if out is not None else None
    if json:
        typer.echo(_json.dumps(written))
        return
    for key, value in written.items():
        typer.echo(f"{key}: {value}")
