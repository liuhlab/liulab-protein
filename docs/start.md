# Getting started: fold the AP-1 complex

This page folds one real complex end to end, then checks the answer against a crystal
structure.

AP-1 is a transcription factor made of two proteins, FOS and JUN. Neither one folds alone.
Together they clamp a short run of DNA and switch a gene on. You start from two gene names
and finish with coordinates.

You need a GPU, the ESM environment (`pixi install -e esm`), and the `swissprot` database
registered.

## From a gene name to an accession

You know these genes as `FOS` and `JUN`. This package works on UniProt accessions. The hop
from one to the other lives in the sibling package `liulab-genome`, not here.

It takes two calls, not one. HGNC is the publisher that carries human gene symbols. The
Alliance is the one that carries human identifiers. They are two different files, so you ask
each for the half it holds.

```python
from genome.xref import XrefSet

matched = XrefSet.for_symbols("Homo sapiens").match_symbols(["FOS", "JUN"])
stems = matched.gene_id_stems

XrefSet("Homo sapiens").from_stems(stems, "uniprot").resolved
```

`for_symbols` picks HGNC for you. The second call maps each gene back to the accessions it
names. The two this page needs are `P01100` for FOS and `P05412` for JUN.

The reverse hop, accession to gene, is `protein.xref` and it lives here.

## The sequences

```python
from protein.db import SwissProt

swissprot = SwissProt()
fos = swissprot["P01100"]  # Protein('P01100', 380 aa)
jun = swissprot["P05412"]  # Protein('P05412', 331 aa)
```

That reads the local Swiss-Prot database and touches no network. Registering it is one
command, on [Set up your data](data.md).

## Trim to the bZIP domain

Fold either protein whole and you spend the run on disorder. Outside the bZIP domain, both
FOS and JUN have no fixed shape. What you asked about is how the two grip DNA. Folding all
the rest buries it. The bZIP is also the piece that was crystallised.

FOS has its bZIP domain at residues 137 to 200. JUN has one at 252 to 315. The crystal
begins two residues later in each. Take the crystal's range, and the check at the end of
this page compares like with like.

**Watch the numbering.** UniProt counts residues from 1 and includes both ends. Python
slicing does neither. It counts from 0 and stops one short. So FOS 139 to 200 is
`fos[138:200]`, and JUN 254 to 315 is `jun[253:315]`.

```python
fos_bzip = fos[138:200]  # FOS 139-200
jun_bzip = jun[253:315]  # JUN 254-315
len(fos_bzip), len(jun_bzip)  # (62, 62)
```

Both come out at 62 residues. Check that before you fold. An off-by-one here is quiet. You
still get a structure, and it is about the wrong residues.

A slice of a `Protein` is a plain string. It is not a `Protein`. A subsequence has no
accession, and nothing here will invent one.

## The DNA, as a duplex

AP-1 binds double-stranded DNA. The site is seven bases, `TGACTCA`. Fold those seven on
their own and you get something that looks fine and means nothing. Seven bases are not a
duplex. There is nothing there for two proteins to clamp.

Give the model both strands. These are the two the crystal used:

```python
top = "AATGGATGAGTCATAGGAGA"
bottom = "TTCTCCTATGACTCATCCAT"
```

The two pair along their whole length, bar one base at each 5' end. `TGACTCA` sits in the
middle of the second strand.

## Fold it

One `ChainRequest` per chain, in the order you want them:

```python
from protein import ChainRequest, ESMFold2, FoldingRequest

request = FoldingRequest(
    [
        ChainRequest("protein", fos_bzip, accession="P01100"),
        ChainRequest("protein", jun_bzip, accession="P05412"),
        ChainRequest("dna", top),
        ChainRequest("dna", bottom),
    ]
)
request  # FoldingRequest(4 chains, 164 residues)
request.chain_ids  # ('A', 'B', 'C', 'D')
```

The two accessions are provenance. They ride along to the answer, so a folded chain can say
where it came from.

```python
model = ESMFold2()  # the weights load once
prediction = model.fold(request, "folds")
prediction.chain_ids
```

A chain label comes from its place in the request. Nobody named these chains, so read the
labels off `prediction.chain_ids` rather than assuming them.

The output directory is required and defaults nowhere. What comes back is a `Structure`, the
same class a deposited entry gets. It also carries `prediction.confidence`, which is what
the model said about its own answer.

The default checkpoint is `ESMFold2-Fast`. It needs no alignment, which is why this page
never builds one. For a deeper run that reads one, see
[Build an alignment](guides/alignments.md).

## Check it against the crystal

Glover and Harrison solved this complex in 1995, in Nature 373:257-261. The entry is `1FOS`,
"Two human c-Fos:c-Jun:DNA complexes", at 3.05 angstroms. It is the thing you just
predicted, measured.

```python
from protein import Structure

crystal = Structure("1FOS")
crystal.chain_ids
```

Chains E and G are FOS 139 to 200. F and H are JUN 254 to 315. A and C are one DNA strand,
B and D the other. Two copies of the complex sit in the asymmetric unit, which is why every
chain has a twin.

```python
crystal["E"].uniprot  # ('P01100',)
```

That accession comes from SIFTS, not from the file. Every mmCIF carries a cross-reference of
its own, and the two disagree. SIFTS is the only join this package reads.

Now the line that matters:

```python
str(crystal["E"].sequence) == fos_bzip  # True
```

The slice you cut out of Swiss-Prot, by UniProt numbering, is the same 62 residues a
crystallographer put in a tube in 1995. Nothing asserted that. The trim checked itself.

## Where next

- [Predict a structure](guides/folding.md) — naming a prediction, repeated chains, bigger
  complexes.
- [How it fits together](concepts.md) — why `Protein` and `Structure` are peers.
