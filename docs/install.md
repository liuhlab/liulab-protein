# Install

One command gets you the package and the three tools it drives.

## pixi and nothing else

The repo uses [pixi](https://pixi.sh). No pip, no conda, no uv. Clone the repo, then:

```bash
pixi install
```

That builds the environment from the lock file, so you get the same versions the tests ran
on. It also installs `mmseqs`, `foldseek` and `muscle`. There is nothing to go and fetch by
hand.

Ask the package whether it can see all three:

```bash
pixi run protein doctor
```

It names any tool that is missing, and exits with an error.

## The esm environment

Embedding is the one exception. ESM-C needs torch, which is heavy. So it sits in its own
environment:

```bash
pixi install -e esm
```

Commands there take the `-e esm` flag:

```bash
pixi run -e esm protein esm embed query.fasta
```

That environment also brings in the CUDA header files. The fast GPU code is built the first
time you run it, and the build reads those headers. Without them it fails, the real error is
hidden, and you are left on a path many times slower. The environment points the compiler at
its own copy of the headers, so there is nothing to set by hand.

## Linux only

The repo resolves for `linux-64` and for nothing else. It will not install on macOS. Use a
Linux machine or the cluster.

## Next

[Set up your data](data.md) is the next step. [Contributing](contributing.md) covers running
the checks.
