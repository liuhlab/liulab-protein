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

One entry per chain, in the order you want them. Each names what it is, what it reads, and
the accession it came from:

```python
from protein import ESMFold2

model = ESMFold2()  # the weights load once
prediction = model.fold(
    [
        {"kind": "protein", "sequence": fos_bzip, "accession": "P01100"},
        {"kind": "protein", "sequence": jun_bzip, "accession": "P05412"},
        {"kind": "dna", "sequence": top},
        {"kind": "dna", "sequence": bottom},
    ],
    "folds",
)
prediction.chain_ids  # ('A', 'B', 'C', 'D')
```

**Always name the `kind`.** `ACGT` is a valid protein sequence as well as a valid strand of
DNA, so nothing here guesses which you meant. Misspell any of the four field names and you
get an error rather than a silently dropped field.

The two accessions ride along to the answer, so a folded chain can say which entry its
sequence came from.

A chain label comes from its place in the list. Nobody named these chains, so read the
labels off `prediction.chain_ids` rather than assuming them.

You have to pass an output directory. There is no default. What comes back is a `Structure`,
the same thing you get for a deposited entry, and it carries `prediction.confidence` with
what the model said about its own answer.

The default checkpoint is `ESMFold2-Fast`, which needs no alignment. To fold against a
deeper one, see [Build an alignment](guides/alignments.md).

## Look at it

`view()` gives you a viewer you can turn and zoom. Colour it by the B-factor column and you
are colouring by the model's own confidence, residue by residue:

```python exec="true" html="true" source="above"
from protein import Structure

prediction = Structure.from_file("docs/fixtures/ap1.cif", id="ap1")
confidence = {"prop": "b", "gradient": "roygb", "min": 50, "max": 100}
print(
    prediction.view(
        width="100%", height=460, style={"cartoon": {"colorscheme": confidence}}
    ).write_html()
)
```

Drag to turn it, scroll to zoom. Blue is confident and red is not.

That is the real answer to the fold above, kept in the repository as
[`ap1.cif`](fixtures/ap1.cif) so this page needs no GPU to draw it. Two long helices wind
around each other and lie down in the groove of the duplex. The ends of the helices are red,
which is what you should expect: they are the loose ends of a piece cut out of a longer
protein.

The numbers agree with the picture:

```python
prediction.confidence.plddt  # 0.87
prediction.confidence.iptm  # 0.74
```

`plddt` is the mean per-residue confidence and `iptm` scores how well the chains sit against
each other, so `iptm` is the one to read for a complex. Both run from 0 to 1. The B-factor
column the viewer colours by holds the same per-residue measure scaled to 0 to 100.

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

That accession comes from SIFTS, not from the file. The file carries a cross-reference of its
own, and the two can differ. This package always answers with SIFTS.

Now the line that matters:

```python
str(crystal["E"].sequence) == fos_bzip  # True
```

The slice you cut out of Swiss-Prot, by UniProt numbering, is the same 62 residues a
crystallographer put in a tube in 1995. Nothing asserted that. The trim checked itself.

## Where next

- [Work with structures](guides/structures.md) — open the prediction and the crystal, and
  look at them in 3D.
- [Predict a structure](guides/folding.md) — naming a prediction, repeated chains, bigger
  complexes.
- [The three things you work with](concepts.md) — moving between proteins, structures and
  chains.
