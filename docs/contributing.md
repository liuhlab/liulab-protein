# Contributing

## Run the checks

```bash
pixi run check
```

That is the static checks plus the tests. It runs every step, then prints all the
failures at once. Read to the bottom before you fix anything. The first red block is
often not the one that matters.

## Fix the formatting

Formatting is the one failure you never fix by hand:

```bash
pixi run fmt
```

It rewrites the Python blocks inside the Markdown pages too, so a file under `docs/` can
change when you run it.

## Run one step, or one test

Each step is a task of its own, so you can run the one you broke.

| To | Run |
| --- | --- |
| lint the code | `pixi run lint` |
| check the formatting | `pixi run fmt-check` |
| check the types | `pixi run typecheck` |
| check the Markdown | `pixi run markdownlint` |
| run the tests | `pixi run test` |

Name a test and you get only that one:

```bash
pixi run test tests/test_core.py::test_the_sequence_is_a_biotite_protein_sequence
```

## Build the site

```bash
pixi install -e docs
pixi run docs-build
```

To read your pages as you write them, `pixi run docs` serves the site and reloads on
each save.

## Send it

`main` is protected and takes no direct push. Work on a branch, then open a pull request
with the checks green.

Issues live on the
[GitHub repository](https://github.com/liuhlab/liulab-protein/issues).
