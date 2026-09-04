"""The ``protein fold`` sub-app — one sequence in, one predicted structure on disk.

``structure`` is a thin Typer wrapper over :class:`protein.fold.ESMFold2`: it reads the
query, constructs the model, folds once and renders what was written. **The output directory
is an argument and not an option**, because it is not optional — the lab **Data dir** holds
reference and input data and never a user's outputs.

The query is a FASTA file holding exactly one record, or the residues themselves. A file that
exists wins, so a sequence is never read as a path somebody typo'd.

Nothing on this path imports torch until a fold actually runs: the module that holds the
weights keeps that import inside a method body.

Examples
--------
>>> from protein.fold.cli import app
>>> [command.name for command in app.registered_commands]
['structure']
"""

from __future__ import annotations

import json as _json
from pathlib import Path as _Path
from typing import Annotated as _Annotated
from typing import Any as _Any

import typer

from protein.core import Protein as _Protein
from protein.fold.esmfold import CHECKPOINTS as _CHECKPOINTS
from protein.fold.esmfold import DEFAULT_CHECKPOINT as _DEFAULT_CHECKPOINT
from protein.fold.esmfold import ESMFold2 as _ESMFold2
from protein.fold.request import FoldingRequest as _FoldingRequest

#: What this command catches. Each already carries its next action, so the command prints the
#: message and exits 1. ``FileExistsError`` — a held name over a different sequence — is an
#: :class:`OSError`, and it names ``--overwrite`` itself.
_FOLD_ERRORS = (ValueError, OSError)

app = typer.Typer(help="Predict a structure with ESMFold2.", no_args_is_help=True)


# `Annotated` rather than the call-in-the-default spelling the root app uses: ruff's B008
# fails a `Path`-annotated default, and one style throughout reads better than two.
@app.command("structure")
def fold_structure(
    query: _Annotated[
        str,
        typer.Argument(help="A FASTA file holding exactly one record, or the residues."),
    ],
    out: _Annotated[
        _Path,
        typer.Argument(help="The directory to write the prediction into. Created if missing."),
    ],
    checkpoint: _Annotated[
        str,
        typer.Option("--checkpoint", help=f"Which checkpoint: {', '.join(_CHECKPOINTS)}."),
    ] = _DEFAULT_CHECKPOINT,
    device: _Annotated[
        str | None,
        typer.Option("--device", help="Where to run. Omitted, cuda when visible, else cpu."),
    ] = None,
    name: _Annotated[
        str | None,
        typer.Option("--name", help="What to call it. Omitted, the accession, else a hash."),
    ] = None,
    overwrite: _Annotated[
        bool,
        typer.Option("--overwrite", help="Fold over a name already holding another sequence."),
    ] = False,
    json: _Annotated[bool, typer.Option("--json", help="Emit JSON instead of plain text.")] = False,
) -> None:
    """Fold QUERY into OUT and report what was written.

    Prints where the coordinates went and what the model said about them; per-residue
    confidence is the file's own B-factor column.

    Exits with code 1 if the checkpoint slug is unknown, the query is neither one FASTA
    record nor a protein sequence, or OUT already holds this name over a different sequence.
    """
    try:
        request = _FoldingRequest(_read(query))
        model = _ESMFold2(checkpoint, device=device)
        structure = model.fold(request, out, name=name, overwrite=overwrite)
    except _FOLD_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    _render(_reported(structure), json=json)


def _read(query: str) -> _Protein:
    """Return the protein ``query`` names: an existing FASTA file, else the residues.

    The path is tried first because a file that exists is not a sequence somebody typed.
    """
    path = _Path(query)
    if path.exists():
        return _Protein.from_fasta(path)
    return _Protein(query)


def _reported(structure: _Any) -> dict[str, _Any]:
    """Return what the command prints: where it went, whose it is, and how good it is."""
    fields: dict[str, _Any] = {
        "id": structure.id,
        "path": str(structure.path),
        "chains": list(structure.accessions or {}),
    }
    if structure.confidence is not None:
        fields.update(structure.confidence.as_json())
    return fields


def _render(fields: dict[str, _Any], *, json: bool) -> None:
    """Print one result, as JSON or as one ``key: value`` line each."""
    if json:
        typer.echo(_json.dumps(fields))
        return
    for key, value in fields.items():
        typer.echo(f"{key}: {','.join(value) if isinstance(value, list) else value}")
