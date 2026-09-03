"""Tests for protein.store — one data root, borrowed rather than declared a second time.

Three claims: this package files under liulab-genome's `LIULAB_DATA`, the
module-not-the-function import discipline is real enough to monkeypatch through, and
registration is genome's `completion` module rather than a second copy of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from genome.store import completion, data_dir

from protein import store
from protein.store import (
    PROTEIN_SUBDIR,
    RegistrationError,
    UnfinishedRegistrationError,
    protein_data_dir,
    work_dir,
)


def test_the_package_files_under_the_lab_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LIULAB_DATA", str(tmp_path))
    assert protein_data_dir() == tmp_path / "protein"


def test_the_subdirectory_name_is_spelled_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LIULAB_DATA", str(tmp_path))
    assert protein_data_dir().name == PROTEIN_SUBDIR


def test_asking_where_something_goes_creates_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LIULAB_DATA", str(tmp_path))
    assert not protein_data_dir().exists()
    assert not work_dir().exists()


def test_the_root_comes_from_genomes_module_so_a_patch_on_it_is_seen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The module is imported, never the function, so this patch reaches the call.
    monkeypatch.setattr(
        data_dir, "prepared_data_dir", lambda subdir: tmp_path / "elsewhere" / subdir
    )
    assert protein_data_dir() == tmp_path / "elsewhere" / "protein"


def test_the_work_area_is_the_hidden_directory_under_the_package_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LIULAB_DATA", str(tmp_path))
    assert work_dir() == tmp_path / "protein" / completion.WORK_DIR_NAME


def test_a_registration_is_a_directory_plus_a_completion_record(tmp_path: Path) -> None:
    directory = tmp_path / "protein" / "db" / "swissprot"
    directory.mkdir(parents=True)
    built = directory / "swissprot"
    built.write_text("ffindex", encoding="utf-8")

    # No `tools=`: asking one for its version is a subprocess, and the record is what is
    # under test.
    record = completion.build_record(directory, kind="database", name="swissprot", files=[built])
    completion.write_record(directory, record)

    assert completion.read_record(directory) == record
    assert completion.check_registration(directory, repair="protein db adopt swissprot") == record


def test_a_directory_holding_files_but_no_record_is_an_unfinished_registration(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "protein" / "db" / "swissprot"
    directory.mkdir(parents=True)
    (directory / "swissprot").write_text("half a download", encoding="utf-8")
    with pytest.raises(UnfinishedRegistrationError):
        completion.check_registration(directory, repair="protein db adopt swissprot")


def test_the_unfinished_state_is_catchable_as_the_error_this_package_re_exports(
    tmp_path: Path,
) -> None:
    # The one exemption to importing modules rather than names: a caller has to be able to
    # name what it catches.
    assert issubclass(UnfinishedRegistrationError, RegistrationError)
    assert RegistrationError is completion.RegistrationError


def test_nothing_callable_is_re_exported_from_this_module() -> None:
    # A callable borrowed from genome and bound here would be the second reference the
    # module-import rule exists to prevent.
    borrowed = {
        name
        for name in store.__all__
        if callable(getattr(store, name))
        and not isinstance(getattr(store, name), type)
        and getattr(store, name).__module__ != store.__name__
    }
    assert borrowed == set()
