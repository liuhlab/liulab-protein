"""The UniProt-to-gene hop: which gene an accession names, asked of `liulab-genome`.

A **Protein** is addressed by a UniProt accession and a gene by a **Gene id stem**, and
:mod:`genome.xref` already holds the map: ``uniprot`` is a first-class **Namespace** there,
published by the identifier default of every species it covers. What this module adds is the
one step genome cannot take from a **Protein** — an ``XrefSet`` is built for a species, and
what a Swiss-Prot header carries is ``metadata["taxon_id"]``.

**It wraps a package, not a publisher**, which is the whole difference from
:mod:`protein.sifts`. Sifts declares a URL and a reader, so it owns a **Prepared set** and
earns :func:`~protein.sifts.prepare`, :func:`~protein.sifts.status` and a CLI. ``XrefSet``
owns its fetch, its slice, its **Completion marker** and its **Data dir** root already, so
nothing here owns any of them and nothing caches it. ``XrefSetNotDownloadedError`` travels
out untouched; it already names the call to make on a login node.

**One direction, because only one of them starts from a Protein.** Stem to accession is
asked of a species and a stem, which is
:meth:`~genome.xref.xref.XrefSet.from_stems` with nothing added.

**The accession is the key and ``GN=`` is not.** A symbol is answered by a different verb
against a source that is not the identifier default, matching previous and alias spellings.

**Two misses, kept apart.** A taxon no set covers cannot be asked at all and raises; an
accession that was asked and named nothing rides back in
:attr:`~genome.xref.xref.ResolvedStems.unresolved`.

Examples
--------
>>> from protein import xref
>>> xref.species_for(9606)
'Homo sapiens'
>>> xref.species_for(7227) is None
True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# The module, never the function: a name bound here is a second reference no
# `monkeypatch.setattr` on it would reach. `ResolvedStems` is the exemption, bound so a
# caller can annotate what this module hands back.
from genome import xref as genome_xref
from genome.xref import ResolvedStems

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "ResolvedStems",
    "TaxonNotCoveredError",
    "gene_stems_for",
    "species_for",
]


class TaxonNotCoveredError(LookupError):
    """No **Xref set** covers this taxon, so the question cannot be asked of it at all.

    Distinct from an accession that was asked and named nothing, which comes back in
    ``unresolved``. Swiss-Prot is the whole of UniProt and the curated table covers a few
    species, so this is the ordinary outcome rather than an edge case — :func:`species_for`
    answers the same question without raising.

    Examples
    --------
    >>> issubclass(TaxonNotCoveredError, LookupError)
    True
    """


def species_for(taxon_id: int) -> str | None:
    """Return the species whose **Xref set** covers ``taxon_id``, or ``None`` if none does.

    Reads genome's curated table and nothing else, so it downloads nothing and answers for
    a taxon that has no set.

    Parameters
    ----------
    taxon_id : int
        An NCBI taxonomy id, as ``OX=`` gives it and ``metadata["taxon_id"]`` holds it.

    Returns
    -------
    str or None
        The species as the curated table spells it, which is what ``XrefSet`` is built
        from. ``None`` when no row carries this taxon.

    Examples
    --------
    >>> species_for(9606)
    'Homo sapiens'
    >>> species_for(7227) is None
    True
    """
    for row in genome_xref.xref_table():
        if row.ncbi_taxid == taxon_id:
            return row.species
    return None


def gene_stems_for(accessions: Iterable[str], taxon_id: int) -> ResolvedStems:
    """Return the **Gene id stem**s the covering release says each accession names.

    Parameters
    ----------
    accessions : iterable of str
        UniProt accessions, in the order they should come back. They are asked on the
        caller's own spelling, so the answer zips against the caller's rows.
    taxon_id : int
        The taxon all of them belong to. One call reads one set, because an answer names
        one species, one source and one release — a batch spanning taxa is not expressible
        and is asked a taxon at a time.

    Returns
    -------
    genome.xref.ResolvedStems
        The accessions that named stems, mapped to every stem each names, and the ones that
        named none — with the species, source and release that answered.

    Raises
    ------
    TaxonNotCoveredError
        If no **Xref set** covers ``taxon_id``.
    genome.xref.XrefSetNotDownloadedError
        If the covering set is not prepared here and this machine cannot fetch it.

    Examples
    --------
    >>> gene_stems_for(["P04637"], 9606).resolved          # doctest: +SKIP
    {'P04637': ('ENSG00000141510',)}
    """
    species = species_for(taxon_id)
    if species is None:
        raise TaxonNotCoveredError(
            f"no xref set covers taxon {taxon_id}, so nothing here names a gene for its "
            f"accessions. Covered: {_covered()}."
        )
    return genome_xref.XrefSet(species).to_stems(accessions, genome_xref.UNIPROT)


def _covered() -> str:
    """Spell the taxa the curated table carries, for a refusal to name."""
    species = {row.ncbi_taxid: row.species for row in genome_xref.xref_table()}
    return ", ".join(f"{taxon} ({species[taxon]})" for taxon in sorted(species))
