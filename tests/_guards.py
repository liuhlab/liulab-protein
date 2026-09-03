"""The two guards every test runs behind: no network, and a data root of its own.

Both are autouse and neither can be opted out of, which is what makes "no test reaches the
network" and "no test writes into the lab's real data" guarantees rather than habits. A module
rather than a conftest so the root ``conftest.py`` can load them as a plugin.

The network guard is also what makes this package's bulk-not-per-ID rule enforceable: a code
path that queries a remote API one accession at a time is untestable by construction here.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, NoReturn

import pytest
import requests

#: The environment variable naming the lab's data root, spelled out rather than imported so
#: that a guard cannot fail to load.
LIULAB_DATA_ENV = "LIULAB_DATA"

#: Address families that leave this machine. ``AF_UNIX`` is local IPC, never the network, and
#: pytest itself uses it.
_NETWORK_FAMILIES = frozenset({socket.AF_INET, socket.AF_INET6})

#: What to do instead, carried by every blocked call.
_OFFLINE_HELP = (
    "No test may reach the network. Put the bytes in tests/data, with their provenance in "
    "tests/data/README.md, and monkeypatch the one call that would have fetched them."
)


class NetworkAccessError(RuntimeError):
    """Raised when a test attempts to reach the network."""


def _blocked(call: str, target: str) -> NetworkAccessError:
    """Return the error a blocked call raises: what was attempted, where, and the fix."""
    return NetworkAccessError(f"blocked network call: {call} {target}\n\n{_OFFLINE_HELP}")


def _address_text(address: Any) -> str:
    """Render a socket address as ``host:port`` when it has that shape."""
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return str(address)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make reaching the network a failure, in every test, without being asked for.

    Two cuts, because one is not enough: ``requests.Session.request``, which every
    ``requests`` call funnels through — biotite's ``rcsb.fetch`` and ``uniprot.fetch``
    included — and ``socket.socket.connect``/``connect_ex`` as the backstop under everything
    that is not ``requests``. Nothing opts out; a test that needs bytes stands them in from
    ``tests/data``.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def blocked_request(
        self: requests.Session, method: Any, url: Any, *args: Any, **kwargs: Any
    ) -> NoReturn:
        raise _blocked(f"requests {str(method).upper()}", str(url))

    def blocked_connect(sock: socket.socket, address: Any) -> None:
        if sock.family in _NETWORK_FAMILIES:
            raise _blocked("socket connect", _address_text(address))
        real_connect(sock, address)

    def blocked_connect_ex(sock: socket.socket, address: Any) -> int:
        if sock.family in _NETWORK_FAMILIES:
            raise _blocked("socket connect_ex", _address_text(address))
        return real_connect_ex(sock, address)

    monkeypatch.setattr(requests.sessions.Session, "request", blocked_request)
    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked_connect_ex)


@pytest.fixture(autouse=True)
def liulab_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the data root at this test's own directory, in every test, unasked.

    The one place the root is set, because "no test writes into the lab's real reference
    data" is a guarantee only if nothing can slip past. Request it to name the root:
    ``liulab_data / "protein"`` is where this package's files land.
    """
    monkeypatch.setenv(LIULAB_DATA_ENV, str(tmp_path))
    return tmp_path
