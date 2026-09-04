# Contributing

This page is how to send a change. For the rules that shape the code itself, read [How it
fits together](concepts.md).

## One command runs everything

```bash
pixi run check
```

That is the static checks plus the tests. It runs every step, then prints all the failures
at once. Read to the bottom before you fix anything. The first red block is often not the
one that matters.

## Run one step

Each step is a task of its own, so you can run the one you broke.

| To | Run |
| --- | --- |
| lint the code | `pixi run lint` |
| check the formatting | `pixi run fmt-check` |
| check the types | `pixi run typecheck` |
| check the prose | `pixi run vale` |
| check the Markdown | `pixi run markdownlint` |
| check the repo rules | `pixi run conformance` |
| run the tests | `pixi run test` |

Name a test and you get only that one:

```bash
pixi run test tests/test_core.py::test_the_sequence_is_a_biotite_protein_sequence
```

Formatting is the one failure you never fix by hand:

```bash
pixi run fmt
```

## Four traps in the gate

### Docstring examples run as tests

pytest collects `src/` as well as `tests/`. Every `Examples` block it finds there is run.
So an example that loads a model, opens a database or calls an outside tool needs a
`# doctest: +SKIP` marker. The marker covers its own line and no other line.

### A warning is a failure

Warnings are turned into errors. Say you have one you mean to keep. Give it its own entry
in `pyproject.toml`, with a comment saying why you kept it. Never a blanket ignore.

### Every module is imported

Collecting `src/` imports each module in it. A heavy import at the top of a file is then
paid by the whole test run. Keep those imports inside the method that needs them.
`import protein` must never pull in torch, and a test holds it to that.

### The tests split in two

The markers cut the tests into two sets with nothing in common. The gate runs everything
that is not marked `model`. The other lane runs by hand and needs the ESM-C weights:

```bash
pixi run -e esm pytest -m model
```

## The checks read your prose as well

Every page a person browses is checked twice. Once against a list of lab jargon. Once for
reading grade, which has to come out at 11 or below.

The grade is a ratio over sentence structure. Shorter words move it hardly at all. Shorter
sentences move it a lot. So when a sentence runs to three clauses, cut it into two
sentences instead. `docs/agents/writing.md` has the rest, and the rule files sit in
`styles/Lab/`.

## Sending a change

`main` is protected and takes no direct push. Work goes on a branch and lands through a
pull request, with `check`, `test` and `docs` green. Nobody is exempt.

Issues live on the
[GitHub repository](https://github.com/liuhlab/liulab-protein/issues).

## Building the site

```bash
pixi install -e docs
pixi run docs-build
```

The site is its own CI job, not part of `check`. It needs an environment nothing else
installs, which is why it stands apart.

The build has no network. A page cannot go and fetch a thing while it is built. So any
file a page needs is committed under `docs/fixtures/`.
