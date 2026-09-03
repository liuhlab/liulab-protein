"""Swiss-Prot: the one **Database** that reads its own headers.

``swissprot["P12345"]`` gives back a :class:`~protein.core.Protein` whose accession, entry
name, description, organism, taxon id and gene are all filled — and that is what earns this
class a subclass where ``pdb`` is a row in a declaration table. A name and a URL are a row;
knowing what ``sp|P12345|AATM_RABIT Aspartate aminotransferase, mitochondrial OS=Oryctolagus
cuniculus OX=9986 GN=GOT2 PE=1 SV=2`` means is behaviour.

**The header split is in two layers, and both are needed.**
:func:`protein.io.fasta.split_header` does the plain FASTA one — first token, then free text
— so a UniProt header lands whole as ``sp|P12345|AATM_RABIT``. Resolving *that* into a
database prefix, an accession and an entry name is a UniProtKB convention rather than a FASTA
one, so it is here. :func:`parse_uniprot_header` is public because it is the same grammar a
caller meets in any UniProt FASTA, not only in this database.

**Retrieval is offline and it is not bulk.** Two ``mmseqs view`` calls per entry, roughly a
quarter of a second each — see :meth:`~protein.db.base.SequenceDatabase.entry`. A job that
wants thousands of entries should search or export, not loop.

**The residues may already have been folded before we saw them.** ``mmseqs databases`` builds
with ``createdb --gpu 1``, which encodes ``B`` to ``D``, ``Z`` to ``E`` and ``U``/``O`` to
``X``. Nothing is warned about at retrieval, because nothing distinguishes a folded ``D``
from a real one; :attr:`~protein.db.base.Database.is_gpu_encoded` and
:meth:`~protein.db.base.Database.status` carry the label instead. ADR-0003 is the whole
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
#: :attr:`protein.core.Protein.metadata`. A field UniProt adds later is kept under its own
#: two letters rather than dropped — a header this package cannot name is still a header it
#: must not lose.
UNIPROT_FIELDS: Mapping[str, str] = MappingProxyType(
    {
        "OS": "organism",
        "OX": "taxon_id",
        "GN": "gene",
        "PE": "protein_existence",
        "SV": "sequence_version",
    }
)

#: The three fields whose values are numbers. Read as :class:`int` when they parse as one and
#: left as text when they do not, so ``metadata["taxon_id"]`` is something `liulab-genome`'s
#: cross-reference tables can be joined on rather than a string that looks like a number.
INTEGER_FIELDS: frozenset[str] = frozenset({"OX", "PE", "SV"})

#: The whitespace before a ``KEY=`` field, which is where the description ends and the
#: annotation begins. A lookahead, so the split keeps the key: values hold spaces
#: (``OS=Oryctolagus cuniculus``) and only the two-letter keys are anchored.
_FIELD_BOUNDARY = re.compile(r"\s+(?=[A-Z]{2}=)")

#: The UniProtKB identifier grammar: ``<db>|<accession>|<entry name>``, where ``db`` is
#: ``sp`` for a reviewed entry and ``tr`` for an unreviewed one. Anything else is not this
#: grammar and is left whole rather than sliced into the wrong fields.
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

    The header this reads is the source FASTA's own, byte-for-byte: ``createdb`` copies the
    bytes through unchanged and ``mmseqs view`` hands them back, so this is UniProt FASTA
    parsing rather than anything MMseqs2-specific.

    A header that does not carry the ``db|accession|name`` grammar keeps its first token in
    :attr:`~UniProtHeader.identifier` and leaves the three resolved names ``None``. Nothing
    is guessed: a UniRef or a locally built FASTA is a different naming scheme, and slicing
    it on pipes would put the wrong text in the accession.

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

    Adds two things to :class:`~protein.db.base.SequenceDatabase`, and they are what earn it
    a class of its own: an accession is folded to upper case on the way in, because that is
    the one spelling UniProt uses and ``.lookup`` is byte-matched; and the header on the way
    out is resolved by :func:`parse_uniprot_header` into ``id``, ``name``, ``description``
    and :attr:`~protein.core.Protein.metadata`.

    This is also where the `liulab-genome` link will land: ``metadata["taxon_id"]`` and
    ``metadata["gene"]`` are what ``genome.xref`` joins on.

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
