"""The harness's own two promises, tested rather than assumed."""

import os
import socket
from pathlib import Path

import pytest
import requests

from ._guards import LIULAB_DATA_ENV, NetworkAccessError


def test_a_requests_call_is_blocked_and_names_the_url_it_wanted() -> None:
    with pytest.raises(NetworkAccessError, match=r"https://rest\.uniprot\.org/"):
        requests.get("https://rest.uniprot.org/", timeout=1)


def test_a_raw_internet_socket_is_blocked() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkAccessError, match=r"192\.0\.2\.1:80"),
    ):
        sock.connect(("192.0.2.1", 80))


def test_a_unix_socket_still_works_because_pytest_needs_one(tmp_path: Path) -> None:
    # The guard is families, not sockets: AF_UNIX is local IPC, and blocking it breaks the
    # runner itself. Connecting to a path that does not exist must fail as the OS would.
    with (
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock,
        pytest.raises(FileNotFoundError),
    ):
        sock.connect(str(tmp_path / "absent.sock"))


def test_the_data_root_points_at_this_test_and_not_the_lab(
    tmp_path: Path, liulab_data: Path
) -> None:
    assert liulab_data == tmp_path
    assert os.environ[LIULAB_DATA_ENV] == str(tmp_path)
