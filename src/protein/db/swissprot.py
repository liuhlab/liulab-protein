"""Swiss-Prot: the one **Database** that reads its own headers.

``swissprot["P12345"]`` gives back a :class:`~protein.core.Protein` whose accession, entry
name, description, organism, taxon id and gene are all filled.

**The header split is in two layers.** :func:`protein.io.fasta.split_header` does the plain
FASTA one, which leaves a UniProt header's first token whole; resolving *that* into a
database prefix, an accession and an entry name is a UniProtKB convention rather than a FASTA
one, so it is here. :func:`parse_uniprot_header` is public because a caller meets the same
grammar in any UniProt FASTA.

**Retrieval is offline and it is not bulk** — see
:meth:`~protein.db.base.SequenceDatabase.entry`. A job wanting many entries should search or
export rather than loop. Nothing warns that the residues may already have been folded,
because nothing distinguishes a folded ``D`` from a real one;
:attr:`~protein.db.base.Database.is_gpu_encoded` carries the label instead, and ADR-0003 the
argument.

Examples
--------
>>> from protein.db.swissprot import SwissProt, parse_uniprot_header
>>> SwissProt().name, SwissProt().source
('swissprot', 'UniProtKB/Swiss-Prot')
>>> parse_uniprot_header("sp|P01308|INS_HUMAN Insulin OS=Homo sapiens OX=9606").accession
'P01308'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from protein.db.base import SequenceDatabase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from protein.core import Protein

__all__ = [
    "INTEGER_FIELDS",
    "UNIPROT_FIELDS",
    "SwissProt",
    "UniProtHeader",
    "parse_uniprot_header",
]

#: UniProt's two-letter header fields, mapped to what this package calls them in
#: :attr:`protein.core.Protein.metadata`. A field with no name here is kept under its own two
#: letters rather than dropped.
UNIPROT_FIELDS: Mapping[str, str] = MappingProxyType(
    {
        "OS": "organism",
        "OX": "taxon_id",
        "GN": "gene",
        "PE": "protein_existence",
        "SV": "sequence_version",
    }
)

#: The fields whose values are numbers, read as :class:`int` when they parse as one, so
#: ``metadata["taxon_id"]`` can be joined on rather than being text that looks like a number.
INTEGER_FIELDS: frozenset[str] = frozenset({"OX", "PE", "SV"})

#: The whitespace before a ``KEY=`` field, which is where the description ends. A lookahead,
#: so the split keeps the key: a value may hold spaces and only the keys are anchored.
_FIELD_BOUNDARY = re.compile(r"\s+(?=[A-Z]{2}=)")

#: The UniProtKB identifier grammar: ``<db>|<accession>|<entry name>``, where ``db`` is
#: ``sp`` for a reviewed entry and ``tr`` for an unreviewed one. Anything else is left whole
#: rather than sliced into the wrong fields.
_IDENTIFIER = re.compile(r"^(?P<entry_type>sp|tr)\|(?P<accession>[^|]+)\|(?P<entry_name>[^|]+)$")


@dataclass(frozen=True)
class UniProtHeader:
    """One UniProt FASTA header, resolved into the things it actually names.

    Attributes
    ----------
    accession : str or None
        The stable identifier, e.g. ``"P01308"``. ``None`` when the header does not carry
        the ``db|accession|name`` grammar.
    entry_name : str or None
        The mnemonic, e.g. ``"INS_HUMAN"``. **Not stable** — that is why it is not the
        accession.
    description : str or None
        The protein name: what follows the identifier, up to the first ``KEY=`` field.
    entry_type : str or None
        ``"sp"`` for a reviewed Swiss-Prot entry, ``"tr"`` for TrEMBL.
    identifier : str
        The first whitespace-delimited token, verbatim, whatever grammar it turned out to be.
    fields : dict
        The ``KEY=value`` fields, under :data:`UNIPROT_FIELDS`' names where there is one and
        under the raw two letters where there is not.

    Examples
    --------
    >>> header = parse_uniprot_header("sp|P01308|INS_HUMAN Insulin OS=Homo sapiens OX=9606")
    >>> header.entry_name, header.description
    ('INS_HUMAN', 'Insulin')
    >>> header.fields
    {'organism': 'Homo sapiens', 'taxon_id': 9606}
    """

    identifier: str
    accession: str | None = None
    entry_name: str | None = None
    description: str | None = None
    entry_type: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)


def parse_uniprot_header(header: str) -> UniProtHeader:
    """Resolve one UniProt FASTA header into an accession, an entry name and its fields.

    A header that does not carry the ``db|accession|name`` grammar keeps its first token in
    :attr:`~UniProtHeader.identifier` and leaves the three resolved names ``None``. Nothing
    is guessed: a UniRef or a locally built FASTA names things differently, and slicing one
    on pipes would put the wrong text in the accession.

    Parameters
    ----------
    header : str
        The header with its ``>`` already stripped.

    Returns
    -------
    UniProtHeader
        What the header names.

    Examples
    --------
    >>> full = parse_uniprot_header(
    ...     "sp|P12345|AATM_RABIT Aspartate aminotransferase, mitochondrial "
    ...     "OS=Oryctolagus cuniculus OX=9986 GN=GOT2 PE=1 SV=2"
    ... )
    >>> full.accession, full.entry_type
    ('P12345', 'sp')
    >>> full.description
    'Aspartate aminotransferase, mitochondrial'
    >>> full.fields["organism"], full.fields["taxon_id"], full.fields["gene"]
    ('Oryctolagus cuniculus', 9986, 'GOT2')
    >>> parse_uniprot_header("MyProtein some text").accession is None
    True
    """
    from protein.io import fasta

    identifier, remainder = fasta.split_header(header)
    identifier = identifier or ""
    match = _IDENTIFIER.match(identifier)

    description: str | None = None
    fields: dict[str, Any] = {}
    if remainder:
        head, *rest = _FIELD_BOUNDARY.split(remainder)
        description = head or None
        for chunk in rest:
            key, _, value = chunk.partition("=")
            fields[UNIPROT_FIELDS.get(key, key)] = _typed(key, value)

    return UniProtHeader(
        identifier=identifier,
        accession=match["accession"] if match else None,
        entry_name=match["entry_name"] if match else None,
        description=description,
        entry_type=match["entry_type"] if match else None,
        fields=fields,
    )


def _typed(key: str, value: str) -> Any:
    """Return ``value`` as an :class:`int` for the numeric fields, else as it was written."""
    if key in INTEGER_FIELDS:
        try:
            return int(value)
        except ValueError:
            return value
    return value


class SwissProt(SequenceDatabase):
    """UniProtKB/Swiss-Prot as a local MMseqs2 database, addressed by accession.

    Adds two things to :class:`~protein.db.base.SequenceDatabase`: an accession is folded to
    upper case on the way in, because ``.lookup`` is byte-matched; and the header on the way
    out is resolved by :func:`parse_uniprot_header` into ``id``, ``name``, ``description``
    and :attr:`~protein.core.Protein.metadata`.

    ``metadata["taxon_id"]`` is what :func:`protein.xref.gene_stems_for` joins on. The link
    itself lives there rather than here, because it is asked of a list and not of one entry.

    Examples
    --------
    >>> SwissProt()
    SwissProt('swissprot')
    >>> SwissProt().name, SwissProt().TOOL_NAME
    ('swissprot', 'mmseqs')
    """

    NAME: ClassVar[str | None] = "swissprot"
    SOURCE: ClassVar[str | None] = "UniProtKB/Swiss-Prot"

    def key_for(self, name: str) -> str:
        """Return the numeric key ``name`` is stored under, folding the accession first.

        Parameters
        ----------
        name : str
            A UniProt accession, in either case and with surrounding space allowed.

        Returns
        -------
        str
            The numeric key.

        Raises
        ------
        KeyError
            If Swiss-Prot carries no such accession — which includes every TrEMBL one.

        Examples
        --------
        >>> SwissProt().key_for("p12345")                        # doctest: +SKIP
        '415743'
        """
        return super().key_for(name.strip().upper())

    def _to_protein(self, name: str, header: str, sequence: str) -> Protein:
        """Build the **Protein**, with the UniProt header resolved into its parts."""
        from protein.core import Protein

        parsed = parse_uniprot_header(header)
        return Protein(
            sequence,
            id=parsed.accession or name.strip().upper(),
            name=parsed.entry_name,
            description=parsed.description,
            metadata={
                "header": header,
                "database": self.name,
                "entry_type": parsed.entry_type,
                **parsed.fields,
            },
        )
