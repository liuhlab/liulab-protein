# liulab-protein

Protein sequences, the structures they appear in, and the large local databases both are
searched against. Distribution name **`liulab-protein`**, import name **`protein`**.

The thing easiest to get wrong is that this package has **two namespaces, two jobs**. A
`Protein` is identified by a UniProt accession; a PDB id addresses a `Structure`. Neither
owns the other, they are many-to-many in both directions, and **SIFTS is the only join** —
never an mmCIF's own `_struct_ref_seq`, which disagrees with it.

## Restraint

Generalizable, lightweight, uncustomized — in that order, and ahead of thorough.

Every gate, lint rule, cap and check is paid by everyone who works here afterwards. They
arrive one at a time, each reasonable alone, and nothing measures the sum. So before adding
one:

- **Does it generalize?** A rule shaped by one situation belongs where that situation is.
  Evidence that it helps there is not evidence about here.
- **What is the total?** Count what someone must already satisfy before writing any code.
- **Would a narrower rule do?** Prefer narrowing to forbidding a legitimate practice.

**A measurement outranks a hypothesis.** When a rule is shown to fire on correct work, that
is evidence; a defect it might also catch is not. Declining a rule, or removing one that
misfires, is as much a contribution as adding one.

## Toolchain

- **pixi** only. Never bare pip, uv, or conda. `pyproject.toml` is the single source of
  truth for dependencies, environments, and tasks.
- **Python 3.13**, declared by `requires-python` and the pixi pin. Write it anywhere else —
  a ruff target, a pyright version — and `conformance` holds that copy to the floor.
- **hatchling + hatch-vcs**. The version comes from the newest git tag. Never hand-edit one.
- **`platforms = ["linux-64"]`**, so nothing resolves on macOS. The gate runs in CI on
  `ubuntu-latest`, or on GPU71FM, which is the only host this project uses.
- Environments: `default` (the gate), `test`, `docs`, and `esm` — the only one with torch.

## Architecture

```text
src/protein/
├── core.py      Protein: a biotite ProteinSequence plus metadata, validated at construction
├── seq.py       the alphabet; folds U/O/J to X with a warning, rejects the rest
├── external.py  THE subprocess boundary — nothing else imports subprocess
├── store.py     <LIULAB_DATA>/protein/, reusing liulab-genome's data root
├── sifts.py     the PDB-to-UniProt map, a prepared set, and the one join
├── io/          fasta.py, structure.py — one module per format
├── db/          Database, SequenceDatabase, StructureDatabase, SwissProt
├── embed/       ESMC holds the weights; Embedding is what one call returns
├── search/      mmseqs.py owns the hit table; foldseek.py reuses its parser
└── structure/   Structure and Chain
```

Four rules shape all of it:

- **Direct support only.** A method exists on a class where the tool takes that thing
  directly, never where the class would first have to acquire something else. Foldseek takes
  coordinates, so structural `search()` is on `Structure` and `Chain`, not on `Protein`.
- **Resident state gets an object; a subprocess does not.** `ESMC()` holds 1.33 GB of
  weights, so it is a class you construct and keep. mmseqs holds nothing between calls, so
  `search()` stays a method.
- **Hold biotite's types, never subclass them**, and call its file and array layer only.
  `fasta.get_sequence` and `structure.to_sequence` rewrite `U` to `C` and `O` to `K` in
  silence. That is ADR-0002.
- **Bulk, not per-ID.** `tests/_guards.py` blocks the network inside tests, so a per-ID
  remote call is untestable by construction.

`liulab-genome` is the sibling this package mirrors and also depends on: `genome.store`
supplies the data root, the completion records and the prepared-set pipeline. Read it
(`~/pkg/liulab-genome`) before inventing anything.

## Commands

| To | Run |
| --- | --- |
| run the gate | `pixi run check` — `check-static` plus `test` |
| run one step | `pixi run lint` / `fmt-check` / `typecheck` / `vale` / `markdownlint` / `conformance` / `test` |
| run one test | `pixi run test tests/test_core.py::test_the_sequence_is_a_biotite_protein_sequence` |
| fix formatting | `pixi run fmt` |
| build the site | `pixi install -e docs && pixi run docs-build` — its own CI job, not part of `check` |
| run the ESM lane | `pixi run -e esm pytest -m model` — by hand, needs the weights |

`check.sh` runs every step and reports **all** failures, not just the first, so read to the
bottom before fixing anything.

Five traps in the gate:

- **`--doctest-modules` runs every `Examples` block in `src/` as a test.** Anything that
  loads a model, touches a database or shells out needs `# doctest: +SKIP`, and the marker
  covers only its own line. `ELLIPSIS` is off, so a multi-line message needs a real test.
- **`filterwarnings = ["error"]`.** Each tolerated warning gets a targeted entry in
  `pyproject.toml` with a comment saying why. Never a blanket ignore.
- **`src/` is collected**, so every module is imported at test time. Keep heavy imports in
  method bodies: `import protein` must never import torch.
- **The markers are a boolean partition.** `-m 'not model'` is the gate; `-m model` is the
  ESM lane and runs nowhere else.
- **A substring of `--help` output is not a substring of the help.** Rich styles the first
  dash of an option on its own, so `--json` is two spans and colour is off under `ssh` and
  on in CI. Assert through `tests.plain_text`; reproduce CI with `FORCE_COLOR=1`.

## Writing rules

Three rules, all enforced by `vale`: be concise; agent-facing documents have word caps;
human-facing prose avoids jargon and stays readable. Read `docs/agents/writing.md` before
writing either kind — the caps are lower than you expect, and this file is subject to one.

## Read next

| When | Read |
| --- | --- |
| Before changing code | `CONTEXT.md`, then any ADR covering the area |
| Recording vocabulary or a decision | `docs/agents/domain.md` |
| Filing or working an issue | `docs/agents/issue-tracker.md` |
| Labelling someone else's issue | `docs/agents/triage-labels.md` |
| Writing anything | `docs/agents/writing.md` |
