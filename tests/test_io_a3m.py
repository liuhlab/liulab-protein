"""A3M in and out: the comment line, the case, and the round trip that has to be exact."""

from __future__ import annotations

from pathlib import Path

from protein.io import a3m

_DATA = Path(__file__).resolve().parent / "data"

_PAIRED = _DATA / "colabfold_pair.a3m"
_HHBLITS = _DATA / "hhblits_5ahw_slice.a3m"


# --- reading -----------------------------------------------------------------


def test_a_leading_comment_line_comes_back_with_its_hash_and_no_newline() -> None:
    comment, _ = a3m.read_records(_PAIRED)
    assert comment == "#12,10\t1,1"


def test_a_file_that_opens_with_a_record_has_no_comment_and_loses_no_record() -> None:
    comment, records = a3m.read_records(_HHBLITS)
    assert comment is None
    assert len(records) == 5


def test_the_first_record_is_not_swallowed_along_with_the_comment() -> None:
    _, records = a3m.read_records(_PAIRED)
    assert records[0] == ("101", "MKTAYIAKQRQISHFSRQLEER")


def test_lowercase_insert_columns_are_read_as_they_were_written() -> None:
    _, records = a3m.read_records(_HHBLITS)
    assert "dvinttielgvtpsvrqeqefaveikerr" in records[1][1]


def test_a_header_is_read_byte_for_byte_so_the_key_field_survives() -> None:
    _, records = a3m.read_records(_PAIRED)
    assert [header for header, _ in records[1:]] == [
        "UniRef100_A0A0A0MRZ7 key=9606",
        "UniRef100_Q8N6T3 key=10090",
    ]


# --- writing -----------------------------------------------------------------


def test_a_row_is_written_on_one_line_however_long_it_is() -> None:
    row = "M" * 500
    assert a3m.format_records([("a", row)]) == f">a\n{row}\n"


def test_the_comment_is_written_first_and_verbatim() -> None:
    text = a3m.format_records([("a", "MKT")], comment="#3\t1")
    assert text.splitlines()[0] == "#3\t1"


def test_no_comment_means_the_text_opens_with_the_first_header() -> None:
    assert a3m.format_records([("a", "MKT")]).startswith(">a\n")


def test_write_records_writes_what_format_records_returns(tmp_path: Path) -> None:
    records = [("a", "MKT"), ("b", "MKkT")]
    written = tmp_path / "out.a3m"
    a3m.write_records(written, records, comment="#3\t1")
    assert written.read_text(encoding="utf-8") == a3m.format_records(records, comment="#3\t1")


def test_write_records_consumes_a_generator(tmp_path: Path) -> None:
    written = tmp_path / "out.a3m"
    a3m.write_records(written, ((header, "MKT") for header in ("a", "b")))
    assert written.read_text(encoding="utf-8") == ">a\nMKT\n>b\nMKT\n"


# --- the round trip ----------------------------------------------------------


def test_a_colabfold_shaped_file_round_trips_byte_for_byte(tmp_path: Path) -> None:
    comment, records = a3m.read_records(_PAIRED)
    again = tmp_path / "again.a3m"
    a3m.write_records(again, records, comment=comment)
    assert again.read_bytes() == _PAIRED.read_bytes()


def test_a_real_hhblits_file_round_trips_byte_for_byte(tmp_path: Path) -> None:
    comment, records = a3m.read_records(_HHBLITS)
    again = tmp_path / "again.a3m"
    a3m.write_records(again, records, comment=comment)
    assert again.read_bytes() == _HHBLITS.read_bytes()
