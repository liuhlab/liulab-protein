"""Tests for protein.external — the one module that shells out to a native binary.

Nothing here needs a tool installed: a stub on `PATH` is a real binary as far as this module
is concerned, and the command grammar needs no binary at all, because one
`monkeypatch.setattr(ExternalTool, "run", ...)` catches every invocation.
"""

from __future__ import annotations

import ast
import os
import stat
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest

import protein.external as external
from protein.external import (
    DEFAULT_VERSION_ARGS,
    NO_VERSION_REPORTED,
    REQUIRED_TOOLS,
    ExternalTool,
    Foldseek,
    InstalledTool,
    Mmseqs,
    MmseqsLikeTool,
    Muscle,
    RecordingTool,
    ToolNotFoundError,
    clear_version_cache,
    doctor,
    is_fresh,
)

#: The package's source tree, found from this file so the test moves with the repo.
_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "protein"


@pytest.fixture(autouse=True)
def _forget_versions() -> Iterator[None]:
    """Clear the path-keyed version cache around every test.

    A stub written at one tmp_path must never be answered from what another test learned.
    """
    clear_version_cache()
    yield
    clear_version_cache()


@pytest.fixture
def on_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[str, str], Path]:
    """Return a helper that puts an executable stub on `PATH` under a chosen tool name.

    The interpreter is pointed somewhere empty too, or the resolver's second lookup would
    find the tools the pixi environment really has.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(external.sys, "executable", str(tmp_path / "nowhere" / "python"))

    def install(name: str, body: str) -> Path:
        script = bin_dir / name
        script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return script

    return install


@pytest.fixture
def run_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace `ExternalTool.run` on the base class and collect every call's arguments.

    One patch, both adapters, both tools — the property `run_to` is written for.
    """
    calls: list[list[str]] = []

    def record(
        self: ExternalTool, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True
    ) -> str:
        calls.append(list(args))
        return ""

    monkeypatch.setattr(ExternalTool, "run", record)
    return calls


@pytest.fixture
def data_root(liulab_data: Path) -> Path:
    """Return this package's root under the per-test data dir the suite's guard points at."""
    return liulab_data / "protein"


# --- the freshness rule ------------------------------------------------------


def test_an_output_that_does_not_exist_is_never_fresh(tmp_path: Path) -> None:
    assert not is_fresh(tmp_path / "absent", [])


def test_an_empty_output_is_never_fresh(tmp_path: Path) -> None:
    output = tmp_path / "empty"
    output.touch()
    assert not is_fresh(output, [])


def test_an_output_newer_than_every_input_is_fresh(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("in", encoding="utf-8")
    output = tmp_path / "output"
    output.write_text("out", encoding="utf-8")
    assert is_fresh(output, [source])


def test_an_output_older_than_one_input_is_stale(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.write_text("out", encoding="utf-8")
    source = tmp_path / "source"
    source.write_text("in", encoding="utf-8")
    # mtimes can tie at filesystem resolution, so the newer input is made explicit.
    output_time = output.stat().st_mtime
    os.utime(source, (output_time + 10, output_time + 10))
    assert not is_fresh(output, [source])


def test_an_input_that_does_not_exist_is_ignored_rather_than_treated_as_new(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.write_text("out", encoding="utf-8")
    assert is_fresh(output, [tmp_path / "never-written"])


# --- the recording adapter ---------------------------------------------------


def test_the_recording_tool_keeps_every_call_in_order() -> None:
    tool = RecordingTool("mmseqs")
    tool.run(["createdb", "sp.fasta", "sp"])
    tool.run(["createindex", "sp", "tmp"], cwd=Path("/data"), capture=False)
    assert [call.args[0] for call in tool.calls] == ["createdb", "createindex"]
    assert tool.calls[1].cwd == Path("/data")
    assert tool.calls[1].capture is False


def test_a_captured_run_returns_what_the_tool_wrote() -> None:
    assert RecordingTool("mmseqs", stdout="hit\n").run(["easy-search"]) == "hit\n"


def test_a_non_zero_exit_names_the_tool_the_code_and_the_arguments() -> None:
    tool = RecordingTool("foldseek")
    tool.exit_code = 2
    with pytest.raises(RuntimeError) as failure:
        tool.run(["easy-search", "q.pdb"])
    message = str(failure.value)
    assert "foldseek failed (exit 2)" in message
    assert "'easy-search'" in message


def test_a_captured_failure_carries_the_tools_own_output() -> None:
    tool = RecordingTool("mmseqs", stdout="Invalid Command")
    tool.exit_code = 1
    with pytest.raises(RuntimeError, match="Invalid Command"):
        tool.run(["nonsense"])


def test_the_on_run_hook_sees_each_call_as_it_is_made(tmp_path: Path) -> None:
    def leave_the_output_behind(call: external.ToolCall) -> None:
        (tmp_path / call.args[-1]).write_text("built", encoding="utf-8")

    tool = RecordingTool("mmseqs", on_run=leave_the_output_behind)
    tool.run(["createdb", "sp.fasta", "sp"])
    assert (tmp_path / "sp").read_text(encoding="utf-8") == "built"


# --- run_to and its one load-bearing line ------------------------------------


def test_run_to_skips_the_call_entirely_when_the_output_is_fresh(tmp_path: Path) -> None:
    output = tmp_path / "sp"
    output.write_text("built", encoding="utf-8")
    tool = RecordingTool("mmseqs")
    assert tool.run_to(["createdb"], output=output, inputs=[]) == output
    assert tool.calls == []


def test_run_to_runs_when_the_output_is_missing(tmp_path: Path) -> None:
    tool = RecordingTool("mmseqs")
    tool.run_to(["createdb"], output=tmp_path / "sp", inputs=[])
    assert len(tool.calls) == 1


def test_run_to_runs_again_when_told_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "sp"
    output.write_text("built", encoding="utf-8")
    tool = RecordingTool("mmseqs")
    tool.run_to(["createdb"], output=output, inputs=[], overwrite=True)
    assert len(tool.calls) == 1


def test_run_to_goes_out_through_run_so_one_patch_catches_every_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Patching `run` on the ABC has to stop a `run_to` made by either adapter, or the
    # suite has a hole a real binary falls through.
    caught: list[list[str]] = []

    def record(
        self: ExternalTool, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True
    ) -> str:
        caught.append(list(args))
        return ""

    monkeypatch.setattr(ExternalTool, "run", record)
    RecordingTool("mmseqs").run_to(["createdb"], output=tmp_path / "sp", inputs=[])
    InstalledTool("mmseqs").run_to(["createdb"], output=tmp_path / "sp", inputs=[])
    assert caught == [["createdb"], ["createdb"]]


# --- install instructions ----------------------------------------------------


def test_the_install_message_names_the_conda_package_not_the_binary() -> None:
    assert "pixi add mmseqs2" in RecordingTool("mmseqs").install_instructions()


def test_a_tool_the_table_does_not_know_installs_under_its_own_lowercased_name() -> None:
    assert "pixi add someothertool" in RecordingTool("SomeOtherTool").install_instructions()


def test_the_install_message_quotes_the_tools_homepage_when_there_is_one() -> None:
    assert "steineggerlab/foldseek" in RecordingTool("foldseek").install_instructions()


# --- locating a real binary --------------------------------------------------


def test_a_binary_on_path_is_located_there(on_path: Callable[[str, str], Path]) -> None:
    stub = on_path("mmseqs", "exit 0")
    assert InstalledTool("mmseqs").path == str(stub)


def test_a_binary_beside_the_running_interpreter_is_found_when_path_lacks_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_bin = tmp_path / "envs" / "default" / "bin"
    env_bin.mkdir(parents=True)
    sibling = env_bin / "mmseqs"
    sibling.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sibling.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(external.sys, "executable", str(env_bin / "python"))
    assert InstalledTool("mmseqs").path == str(sibling)


def test_a_missing_binary_raises_with_its_install_instructions_as_the_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(external.sys, "executable", str(tmp_path / "nowhere" / "python"))
    with pytest.raises(ToolNotFoundError, match="pixi add foldseek"):
        _ = InstalledTool("foldseek").path


# --- versions ----------------------------------------------------------------


def test_the_two_tools_are_asked_version_because_they_reject_the_double_dash_flag() -> None:
    assert InstalledTool("mmseqs").version_args == ("version",)
    assert InstalledTool("foldseek").version_args == ("version",)
    assert InstalledTool("someothertool").version_args == DEFAULT_VERSION_ARGS


def test_a_tool_is_asked_with_the_arguments_its_table_entry_names(
    on_path: Callable[[str, str], Path],
) -> None:
    on_path(
        "mmseqs",
        'if [ "$1" = "version" ]; then echo 18.8cc5c; exit 0; fi\n'
        'echo "Invalid Command: $1" >&2\nexit 1',
    )
    assert InstalledTool("mmseqs").version == "18.8cc5c"


def test_a_tool_that_runs_but_declines_to_identify_itself_answers_empty(
    on_path: Callable[[str, str], Path],
) -> None:
    on_path("mmseqs", "exit 1")
    assert InstalledTool("mmseqs").version == ""


def test_only_the_first_line_of_a_chatty_version_is_reported(
    on_path: Callable[[str, str], Path],
) -> None:
    on_path("mmseqs", "echo 18.8cc5c\necho and some banner")
    assert InstalledTool("mmseqs").version == "18.8cc5c"


def test_a_version_written_only_to_stderr_is_still_read(
    on_path: Callable[[str, str], Path],
) -> None:
    on_path("mmseqs", "echo 18.8cc5c >&2")
    assert InstalledTool("mmseqs").version == "18.8cc5c"


def test_the_version_is_asked_once_per_binary_and_remembered_across_objects(
    on_path: Callable[[str, str], Path], tmp_path: Path
) -> None:
    counter = tmp_path / "asks"
    on_path("mmseqs", f'echo x >> "{counter}"\necho 18.8cc5c')
    assert InstalledTool("mmseqs").version == "18.8cc5c"
    assert InstalledTool("mmseqs").version == "18.8cc5c"
    assert counter.read_text(encoding="utf-8").count("x") == 1


def test_clearing_the_cache_sends_the_next_ask_back_to_the_binary(
    on_path: Callable[[str, str], Path], tmp_path: Path
) -> None:
    counter = tmp_path / "asks"
    on_path("mmseqs", f'echo x >> "{counter}"\necho 18.8cc5c')
    _ = InstalledTool("mmseqs").version
    clear_version_cache()
    _ = InstalledTool("mmseqs").version
    assert counter.read_text(encoding="utf-8").count("x") == 2


# --- doctor ------------------------------------------------------------------


def test_doctor_reports_a_version_for_each_required_tool(
    on_path: Callable[[str, str], Path],
) -> None:
    on_path("mmseqs", "echo 18.8cc5c")
    on_path("foldseek", "echo 10.941cd33")
    on_path("muscle", "echo muscle 5.3.linux64")
    assert doctor() == {
        "mmseqs": "18.8cc5c",
        "foldseek": "10.941cd33",
        "muscle": "muscle 5.3.linux64",
    }


def test_doctor_lists_a_tool_that_is_present_but_will_not_say_what_it_is(
    on_path: Callable[[str, str], Path],
) -> None:
    on_path("mmseqs", "echo 18.8cc5c")
    on_path("foldseek", "exit 1")
    on_path("muscle", "echo muscle 5.3.linux64")
    assert doctor()["foldseek"] == NO_VERSION_REPORTED


def test_doctor_raises_when_a_required_tool_is_missing(
    on_path: Callable[[str, str], Path],
) -> None:
    on_path("mmseqs", "echo 18.8cc5c")
    with pytest.raises(ToolNotFoundError, match="pixi add foldseek"):
        doctor()


def test_the_required_tools_are_the_three_this_package_drives() -> None:
    assert REQUIRED_TOOLS == ("mmseqs", "foldseek", "muscle")


def test_a_missing_muscle_fails_doctor_like_any_other_required_tool(
    on_path: Callable[[str, str], Path],
) -> None:
    # There is no optional tier: the aligner is installed and checked like the searchers.
    on_path("mmseqs", "echo 18.8cc5c")
    on_path("foldseek", "echo 10.941cd33")
    with pytest.raises(ToolNotFoundError, match="pixi add muscle"):
        doctor()


def test_muscle_is_asked_for_its_version_the_way_muscle_spells_it() -> None:
    assert Muscle().version_args == ("-version",)
    assert Mmseqs().version_args == ("version",)


def test_muscle_reports_the_version_line_its_binary_prints(
    on_path: Callable[[str, str], Path],
) -> None:
    on_path("muscle", "echo muscle 5.3.linux64")
    assert Muscle().version == "muscle 5.3.linux64"


# --- the shared command grammar ----------------------------------------------


def test_the_two_tools_report_the_identity_column_each_one_spells() -> None:
    assert Mmseqs().format_columns[2] == "pident"
    assert Foldseek().format_columns[2] == "fident"


def test_the_shared_columns_are_the_same_and_in_the_same_order() -> None:
    shared = tuple(column for column in Mmseqs().format_columns if column != "pident")
    also = tuple(
        column
        for column in Foldseek().format_columns
        if column not in {"fident", *Foldseek().EXTRA_COLUMNS}
    )
    assert shared == also


def test_only_foldseek_adds_the_structural_columns() -> None:
    assert Foldseek().EXTRA_COLUMNS == ("alntmscore", "lddt")
    assert Mmseqs().EXTRA_COLUMNS == ()


def test_no_format_column_is_q3di_because_foldseek_rejects_that_code() -> None:
    # Asking for it fails the whole search, so neither tool may name it.
    assert "q3di" not in Foldseek().format_output
    assert "q3di" not in Mmseqs().format_output


def test_the_format_output_argument_is_the_columns_joined_by_commas() -> None:
    assert Mmseqs().format_output == ",".join(Mmseqs().format_columns)


def test_each_tool_knows_its_own_binary_and_conda_package() -> None:
    assert (Mmseqs().name, Mmseqs().package) == ("mmseqs", "mmseqs2")
    assert (Foldseek().name, Foldseek().package) == ("foldseek", "foldseek")


def test_createdb_names_every_input_before_the_database(run_calls: list[list[str]]) -> None:
    Mmseqs().createdb([Path("a.fasta"), Path("b.fasta")], Path("sp"))
    assert run_calls == [["createdb", "a.fasta", "b.fasta", "sp"]]


def test_createindex_takes_a_scratch_directory_and_answers_the_idx_path(
    data_root: Path, run_calls: list[list[str]]
) -> None:
    assert Mmseqs().createindex(Path("sp")) == Path("sp.idx")
    verb, database, work = run_calls[0]
    assert (verb, database) == ("createindex", "sp")
    assert Path(work).parent == data_root / ".work"


def test_easy_search_asks_for_this_tools_columns(
    data_root: Path, run_calls: list[list[str]]
) -> None:
    Foldseek().easy_search(Path("q.cif"), Path("pdb100"), Path("hits.tsv"))
    call = run_calls[0]
    assert call[:4] == ["easy-search", "q.cif", "pdb100", "hits.tsv"]
    assert call[5:7] == ["--format-output", Foldseek().format_output]


def test_extra_arguments_are_passed_through_after_the_format(
    data_root: Path, run_calls: list[list[str]]
) -> None:
    Mmseqs().easy_search(Path("q.fa"), Path("sp"), Path("hits.tsv"), extra=["-s", "7.5"])
    assert run_calls[0][-2:] == ["-s", "7.5"]


def test_convertalis_renders_an_alignment_database_as_the_same_columns(
    run_calls: list[list[str]],
) -> None:
    Mmseqs().convertalis(Path("q"), Path("sp"), Path("aln"), Path("hits.tsv"))
    assert run_calls[0][:5] == ["convertalis", "q", "sp", "aln", "hits.tsv"]
    assert run_calls[0][5:7] == ["--format-output", Mmseqs().format_output]


def test_cluster_takes_a_scratch_directory(data_root: Path, run_calls: list[list[str]]) -> None:
    assert Mmseqs().cluster(Path("sp"), Path("sp_clu")) == Path("sp_clu")
    verb, database, clusters, work = run_calls[0]
    assert (verb, database, clusters) == ("cluster", "sp", "sp_clu")
    assert Path(work).parent == data_root / ".work"


# --- the scratch directory neither tool cleans -------------------------------


def test_the_scratch_directory_exists_while_the_command_runs(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []

    def record(
        self: ExternalTool, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True
    ) -> str:
        seen.append(Path(args[4]).is_dir())
        return ""

    monkeypatch.setattr(ExternalTool, "run", record)
    Mmseqs().easy_search(Path("q.fa"), Path("sp"), Path("hits.tsv"))
    assert seen == [True]


def test_the_scratch_directory_is_removed_when_the_command_finishes(
    data_root: Path, run_calls: list[list[str]]
) -> None:
    Mmseqs().easy_search(Path("q.fa"), Path("sp"), Path("hits.tsv"))
    assert not Path(run_calls[0][4]).exists()


def test_the_scratch_directory_is_removed_when_the_command_fails(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[Path] = []

    def fail(
        self: ExternalTool, args: Sequence[str], *, cwd: Path | None = None, capture: bool = True
    ) -> str:
        made.append(Path(args[4]))
        raise RuntimeError("mmseqs failed")

    monkeypatch.setattr(ExternalTool, "run", fail)
    with pytest.raises(RuntimeError):
        Mmseqs().easy_search(Path("q.fa"), Path("sp"), Path("hits.tsv"))
    assert not made[0].exists()


def test_the_scratch_directory_says_which_command_left_it(
    data_root: Path, run_calls: list[list[str]]
) -> None:
    Foldseek().easy_search(Path("q.cif"), Path("pdb100"), Path("hits.tsv"))
    assert Path(run_calls[0][4]).name.startswith("foldseek-easy-search-")


def test_the_scratch_root_is_the_one_the_store_names(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The module is imported rather than the function, so this patch is seen.
    import protein.store

    elsewhere = data_root.parent / "somewhere-else"
    monkeypatch.setattr(protein.store, "work_dir", lambda: elsewhere)
    with Mmseqs().scratch_dir() as scratch:
        assert scratch.parent == elsewhere


# --- the boundary itself -----------------------------------------------------


def test_external_is_the_only_module_in_this_package_that_imports_subprocess() -> None:
    boundary = _SOURCE_ROOT / "external.py"
    offenders = [
        module.relative_to(_SOURCE_ROOT).as_posix()
        for module in sorted(_SOURCE_ROOT.rglob("*.py"))
        if module != boundary and _imports_subprocess(module)
    ]
    assert offenders == [], (
        f"{offenders} import subprocess. protein.external is the package's one process "
        f"boundary; drive the tool through an ExternalTool instead."
    )


def _imports_subprocess(module: Path) -> bool:
    """Whether ``module`` names :mod:`subprocess` in an import statement."""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "subprocess" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "subprocess":
            return True
    return False


def test_every_adapter_and_every_tool_is_one_external_tool() -> None:
    # The stand-in being a full ExternalTool is what makes the freshness rule and the
    # failure message the same code in a test as in a run.
    assert issubclass(RecordingTool, ExternalTool)
    assert issubclass(InstalledTool, ExternalTool)
    assert issubclass(MmseqsLikeTool, ExternalTool)
    assert issubclass(Mmseqs, MmseqsLikeTool)
    assert issubclass(Foldseek, MmseqsLikeTool)
