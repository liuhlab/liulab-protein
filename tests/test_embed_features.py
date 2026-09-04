"""Tests for the SAE feature descriptions — the prepared set, the reader, and the join.

Everything runs over `tests/data/esm_sae_features_slice.json`, real records whose provenance
is in `tests/data/README.md`. `genome.store.fetch.fetch_url` is monkeypatched through its
module, so the pipeline runs whole against the slice and the network is never reached.

The claims worth holding are that the set lands under the ESM provider directory, that the
index reads back as the type an activation stores, that a description survives the round trip
through a tab-separated file, and that a slug with no published descriptions raises.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from genome.store import completion, fetch
from typer.testing import CliRunner

from protein.cli import app as root_app
from protein.embed import SaeActivation
from protein.embed.esm import features
from protein.embed.esm.features import (
    SaeFeaturesFormatError,
    SaeFeaturesNotDownloadedError,
    app,
)

from . import plain_text

_SLICE = Path(__file__).parent / "data" / "esm_sae_features_slice.json"

#: The feature indices the slice keeps, in the publisher's own order.
_KEPT = (0, 19, 10425, 16383)


def _records() -> dict[int, dict[str, Any]]:
    """The slice's records, read straight from the JSON, as the publisher wrote them."""
    document = json.loads(_SLICE.read_text(encoding="utf-8"))
    return {record["feature_index"]: record for record in document["data"]}


def _document(*records: dict[str, Any]) -> str:
    """The publisher's envelope around ``records``."""
    return json.dumps({"data": list(records)})


@pytest.fixture
def publisher_file(tmp_path: Path) -> Path:
    """The committed slice, served the way the publisher serves it."""
    served = tmp_path / "publisher" / "features.json"
    served.parent.mkdir(parents=True)
    shutil.copyfile(_SLICE, served)
    return served


@pytest.fixture
def prepared_features(monkeypatch: pytest.MonkeyPatch, publisher_file: Path) -> Path:
    """Run the whole prepared-set pipeline against the slice, and return the stored file."""

    def fake_fetch(url: str, dest_dir: Path, **kwargs: Any) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / (kwargs.get("fname") or "features.json")
        shutil.copyfile(publisher_file, target)
        return target

    monkeypatch.setattr(fetch, "fetch_url", fake_fetch)
    return features.prepare(progressbar=False).path


# --- the source ---------------------------------------------------------------


def test_the_set_lands_under_the_provider_that_produced_the_bytes(tmp_path: Path) -> None:
    # The Data dir groups by what produced the bytes, which is why this is not a sibling of
    # `sifts/` but a tenant of `esm/`.
    assert features.features_data_dir() == tmp_path / "protein" / "esm" / "sae-features"
    assert features.source().path.name == features.STORED_NAME


def test_the_whole_codebook_is_one_request_and_needs_no_token() -> None:
    # Bulk, not per-ID: the publisher also serves one record per feature, and enumerating
    # the set that way would be one request per index.
    assert features.source().url == features.FEATURES_URL
    assert features.FEATURES_URL.endswith("/features")


def test_the_source_pins_nothing_because_the_publisher_keeps_no_archive() -> None:
    assert features.source().checksum is None


def test_the_repair_is_delete_and_rebuild_and_names_the_prepare_command() -> None:
    assert features.source().repair.endswith(features.PREPARE_COMMAND)


# --- the reader ---------------------------------------------------------------


def test_the_reader_records_which_sae_the_descriptions_belong_to(tmp_path: Path) -> None:
    measured = features.read_features(
        iter([_document(_records()[0])]), tmp_path / "out.tsv.gz", origin="slice"
    )
    assert measured["sae"] == features.DESCRIBED_SAE
    assert measured["codebook_size"] == features.CODEBOOK_SIZE
    assert measured["rows"] == 1


def test_the_reader_keeps_the_three_fields_the_publisher_serves_in_bulk(
    prepared_features: Path,
) -> None:
    frame = features.descriptions(features.DESCRIBED_SAE)
    assert frame.index.name == "feature_index"
    assert list(frame.columns) == ["label", "description"]


def test_the_reader_sorts_by_feature_index_whatever_order_the_publisher_served(
    tmp_path: Path,
) -> None:
    records = _records()
    staged = tmp_path / "out.tsv.gz"
    features.read_features(iter([_document(records[16383], records[0])]), staged, origin="slice")
    stored = pd.read_csv(staged, sep="\t")
    assert list(stored["feature_index"]) == [0, 16383]


def test_the_reader_refuses_a_document_that_is_not_json(tmp_path: Path) -> None:
    with pytest.raises(SaeFeaturesFormatError, match="not JSON"):
        features.read_features(iter(["<html>"]), tmp_path / "out.tsv.gz", origin="slice")


def test_the_reader_refuses_an_envelope_the_publisher_re_shaped(tmp_path: Path) -> None:
    with pytest.raises(SaeFeaturesFormatError, match="re-shaped"):
        features.read_features(
            iter([json.dumps([_records()[0]])]), tmp_path / "out.tsv.gz", origin="slice"
        )


def test_the_reader_refuses_a_record_missing_one_of_the_three_fields(tmp_path: Path) -> None:
    with pytest.raises(SaeFeaturesFormatError, match="does not carry"):
        features.read_features(
            iter([_document({"feature_index": 1, "label": "x"})]),
            tmp_path / "out.tsv.gz",
            origin="slice",
        )


@pytest.mark.parametrize("index", [-1, 16384, "7", 1.0])
def test_the_reader_refuses_an_index_the_stored_type_could_not_hold(
    tmp_path: Path, index: object
) -> None:
    # An index outside the codebook does not fit the type the activations store, so it would
    # be read back as a different feature.
    with pytest.raises(SaeFeaturesFormatError, match="codebook runs"):
        features.read_features(
            iter([_document({"feature_index": index, "label": "x", "description": "y"})]),
            tmp_path / "out.tsv.gz",
            origin="slice",
        )


def test_the_reader_refuses_a_feature_named_twice(tmp_path: Path) -> None:
    record = _records()[0]
    with pytest.raises(SaeFeaturesFormatError, match="more than once"):
        features.read_features(
            iter([_document(record, record)]), tmp_path / "out.tsv.gz", origin="slice"
        )


def test_the_reader_refuses_a_response_that_carries_no_descriptions(tmp_path: Path) -> None:
    with pytest.raises(SaeFeaturesFormatError, match="no descriptions"):
        features.read_features(iter([_document()]), tmp_path / "out.tsv.gz", origin="slice")


def test_nothing_is_placed_when_the_reader_refuses(tmp_path: Path) -> None:
    staged = tmp_path / "out.tsv.gz"
    with pytest.raises(SaeFeaturesFormatError):
        features.read_features(iter(["not json"]), staged, origin="slice")
    assert not staged.exists()


# --- the pipeline -------------------------------------------------------------


def test_the_marker_records_which_sae_and_how_many_descriptions(
    prepared_features: Path,
) -> None:
    record = completion.read_record(features.features_data_dir())
    assert record is not None
    assert record.kind == "esm-sae-features"
    assert record.source_url == features.FEATURES_URL
    assert record.details["sae"] == features.DESCRIBED_SAE
    assert record.details["rows"] == len(_KEPT)


def test_preparing_a_set_that_is_already_here_fetches_nothing(
    monkeypatch: pytest.MonkeyPatch, prepared_features: Path
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("a prepared set must not be fetched again")

    monkeypatch.setattr(fetch, "fetch_url", refuse)
    assert features.prepare(progressbar=False).path == prepared_features


def test_a_fetch_this_machine_cannot_make_names_the_login_node_and_the_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreachable(*args: Any, **kwargs: Any) -> Path:
        raise OSError("no route to host")

    monkeypatch.setattr(fetch, "fetch_url", unreachable)
    with pytest.raises(SaeFeaturesNotDownloadedError) as raised:
        features.prepare(progressbar=False)
    assert features.PREPARE_COMMAND in str(raised.value)
    assert "login node" in str(raised.value)


# --- descriptions ---------------------------------------------------------------


def test_the_feature_index_reads_back_as_the_type_an_activation_stores(
    prepared_features: Path,
) -> None:
    # The whole point of spelling the dtype once: a feature index out of this frame and a
    # feature index out of `SaeActivation.indices` are the same type.
    frame = features.descriptions(features.DESCRIBED_SAE)
    assert frame.index.dtype == SaeActivation.index_dtype(features.CODEBOOK_SIZE)
    assert frame.index.dtype == np.dtype(np.uint16)


def test_every_kept_feature_comes_back_under_its_own_index(prepared_features: Path) -> None:
    frame = features.descriptions(features.DESCRIBED_SAE)
    assert list(frame.index) == sorted(_KEPT)
    assert frame.loc[0, "label"] == _records()[0]["label"]


def test_a_description_survives_the_round_trip_through_a_tab_separated_file(
    prepared_features: Path,
) -> None:
    # The last feature's text carries double quotes, an em dash and non-ASCII letters, all
    # of which a naive writer or a mis-declared encoding would damage.
    frame = features.descriptions(features.DESCRIBED_SAE)
    for index in _KEPT:
        assert frame.loc[index, "label"] == _records()[index]["label"]
        assert frame.loc[index, "description"] == _records()[index]["description"]
    assert '"' in str(frame.loc[16383, "description"])


def test_the_caller_joins_the_frame_against_an_activations_indices(
    prepared_features: Path,
) -> None:
    # A frozen numpy value object never touches the filesystem, so this is the join and it
    # is the caller's to make.
    activation = SaeActivation(
        np.array([[0, 19], [10425, 16383]], dtype=np.uint16),
        np.ones((2, 2), dtype=np.float16),
        np.zeros(2, dtype=np.float32),
        "P12345",
        "6b",
        60,
        features.DESCRIBED_SAE,
        features.CODEBOOK_SIZE,
        2,
    )
    named = features.descriptions(activation.sae).loc[activation.indices[0]]
    assert list(named["label"]) == [_records()[0]["label"], _records()[19]["label"]]


@pytest.mark.parametrize("sae", ["300m-layer23-k64-cb16384", "600m-layer27-k64-cb16384"])
def test_a_checkpoint_with_no_published_descriptions_raises_by_name(
    prepared_features: Path, sae: str
) -> None:
    with pytest.raises(ValueError, match="no feature descriptions are published"):
        features.descriptions(sae)


def test_the_refusal_names_the_one_checkpoint_that_has_them(prepared_features: Path) -> None:
    with pytest.raises(ValueError, match="no feature descriptions") as raised:
        features.descriptions("300m-layer23-k64-cb16384")
    assert features.DESCRIBED_SAE in str(raised.value)


def test_an_unprepared_set_raises_rather_than_answering_nothing() -> None:
    with pytest.raises(SaeFeaturesNotDownloadedError) as raised:
        features.descriptions(features.DESCRIBED_SAE)
    assert features.PREPARE_COMMAND in str(raised.value)
    assert "login node" in str(raised.value)


def test_the_table_is_read_once_and_then_held(prepared_features: Path) -> None:
    assert features.descriptions(features.DESCRIBED_SAE) is features.descriptions(
        features.DESCRIBED_SAE
    )


def test_clearing_the_cache_re_reads_from_disk(prepared_features: Path) -> None:
    first = features.descriptions(features.DESCRIBED_SAE)
    features.clear_cache()
    assert features.descriptions(features.DESCRIBED_SAE) is not first


# --- status ---------------------------------------------------------------------


def test_status_says_nothing_is_prepared_when_nothing_is(tmp_path: Path) -> None:
    found = features.status()
    assert found.prepared is False
    assert found.rows is None
    assert found.path == tmp_path / "protein" / "esm" / "sae-features" / features.STORED_NAME


def test_status_reads_the_marker_and_names_the_checkpoint(prepared_features: Path) -> None:
    found = features.status()
    assert found.prepared is True
    assert found.sae == features.DESCRIBED_SAE
    assert found.codebook_size == features.CODEBOOK_SIZE
    assert found.rows == len(_KEPT)
    assert found.completed_at is not None


def test_status_reads_the_marker_and_not_the_table(
    monkeypatch: pytest.MonkeyPatch, prepared_features: Path
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("status must not read the table")

    monkeypatch.setattr("protein.prepared._read_table", refuse)
    assert features.status().rows == len(_KEPT)


# --- the CLI ---------------------------------------------------------------------


def test_every_command_is_registered_under_a_name_it_was_given() -> None:
    assert [command.name for command in app.registered_commands] == ["prepare", "status"]


def test_every_command_takes_json() -> None:
    # `plain_text`, not `result.output`: rich styles the first dash of `--json` separately.
    for command in app.registered_commands:
        assert command.name is not None
        result = CliRunner().invoke(app, [command.name, "--help"])
        assert "--json" in plain_text(result.output), command.name


def test_a_bare_invocation_prints_help_instead_of_nothing() -> None:
    result = CliRunner().invoke(app, [])
    assert "prepare" in plain_text(result.output)
    assert "status" in plain_text(result.output)


def test_the_sub_app_is_mounted_under_the_lane_that_owns_it() -> None:
    result = CliRunner().invoke(root_app, ["esm", "features", "--help"])
    assert result.exit_code == 0
    assert "prepare" in plain_text(result.output)


def test_prepare_stores_the_set_and_then_prints_what_is_here(
    monkeypatch: pytest.MonkeyPatch, publisher_file: Path
) -> None:
    def fake_fetch(url: str, dest_dir: Path, **kwargs: Any) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / "features.json"
        shutil.copyfile(publisher_file, target)
        return target

    monkeypatch.setattr(fetch, "fetch_url", fake_fetch)
    result = CliRunner().invoke(app, ["prepare", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["sae"] == features.DESCRIBED_SAE


def test_prepare_exits_one_and_says_where_to_run_it_when_the_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreachable(*args: Any, **kwargs: Any) -> Path:
        raise OSError("no route to host")

    monkeypatch.setattr(fetch, "fetch_url", unreachable)
    result = CliRunner().invoke(app, ["prepare"])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert features.PREPARE_COMMAND in result.output


def test_status_answers_json_when_asked(prepared_features: Path) -> None:
    result = CliRunner().invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == features.status().as_json()


def test_status_prints_one_line_per_field_when_not_asked_for_json(
    prepared_features: Path,
) -> None:
    result = CliRunner().invoke(app, ["status"])
    assert result.exit_code == 0
    assert f"sae: {features.DESCRIBED_SAE}" in result.output
    assert f"rows: {len(_KEPT)}" in result.output
