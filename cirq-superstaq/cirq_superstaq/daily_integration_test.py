"""Integration checks that run daily (via Github action) between client and prod server."""

import os

import cirq
import numpy as np
import pytest

import cirq_superstaq


@pytest.fixture
def service() -> cirq_superstaq.Service:
    token = os.getenv("TEST_USER_TOKEN")
    service = cirq_superstaq.Service(api_key=token)
    return service


def test_aqt_compile(service: cirq_superstaq.Service) -> None:
    qubits = cirq.LineQubit.range(8)
    circuit = cirq.Circuit(cirq.H(qubits[4]))
    expected = cirq.Circuit(
        cirq.rz(np.pi / 2)(qubits[4]),
        cirq.rx(np.pi / 2)(qubits[4]),
        cirq.rz(np.pi / 2)(qubits[4]),
    )
    assert service.aqt_compile(circuit).circuit == expected
    assert service.aqt_compile([circuit]).circuits == [expected]
    assert service.aqt_compile([circuit, circuit]).circuits == [expected, expected]


def test_get_balance(service: cirq_superstaq.Service) -> None:
    assert isinstance(service.get_balance(pretty_output=False), float)


def test_tsp(service: cirq_superstaq.Service) -> None:
    out = service.tsp(["Chicago", "San Francisco", "New York City", "New Orleans"])
    assert out.map_link == [
        "https://www.google.com/maps/dir/?api=1&origin=Chicago&destination=Chicago"
        "&waypoints=San+Francisco%7CNew+Orleans%7CNew+York+City%7C"
    ]
