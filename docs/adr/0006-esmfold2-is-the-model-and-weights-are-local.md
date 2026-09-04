---
search:
  exclude: true
---

# 6. ESMFold2 is the model, and weights are local

`ESMFold2` loads a checkpoint from the Hugging Face cache and folds on this machine. There is
no Forge client, no token and no per-call remote request.

**Local, because the alternative is a second architecture.** A tokened per-molecule HTTP call
contradicts *bulk, not per-ID* and is untestable behind `tests/_guards.py`, which blocks the
network in every test. Nothing about the design would be shared with it.

**The cache is adopted as it is.** No mirror under the **Data dir**, no **Completion marker**,
no second copy to keep fresh, and the package never sets `HF_HOME`. Whether the lab points one
at shared storage is ops, and it changes no line here. `doctor()` and `store.py` are untouched.

**A slug, not a repository.** `CHECKPOINTS` maps `ESMFold2-Fast` and `ESMFold2` to their
repositories, so an unknown name fails by name before a download starts — the same reason
`ESMC` takes one. Both are reachable; the fast one is the default, and it neither needs an
alignment nor refuses one. `msa_encoder.enabled: false` means its pair-conditioning module is
never built, but the profile and deletion features are computed a level above it, so an
alignment still reaches a Fast fold. The hosted SDK's warning that Fast "will ignore any MSA
provided" describes the server and not this code, so nothing here forks on the checkpoint.

**The card is not managed.** No length ceiling, no VRAM arithmetic, no OOM handling: a fold
that does not fit raises whatever CUDA raises. A cap measured on one card is a limit invented
for every other. The one guard is `load_esmc=False`, which is not a memory option — it returns
an mmCIF of the right length holding a wrong structure and emits nothing. A loud crash needs
no guard; a silent wrong answer does.

**The fused kernels are selected by default.** They fall back to the reference path themselves
where they cannot be compiled, and the `esm` environment exists to let them compile, so
choosing them costs a caller nothing and not choosing them costs everyone an order of
magnitude.

What it costs. The lane runs nowhere without a GPU and a filled cache, so CI never folds
anything: the `model` marker is the whole of its coverage. And the parameters this class
names are ours to keep current — `fold` forwards the rest, and `model` is the loaded object
for anything neither reaches.
