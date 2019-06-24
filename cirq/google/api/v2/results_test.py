import numpy as np
import pytest

import cirq
from cirq.google.api import v2


@pytest.mark.parametrize('reps', range(1, 100, 7))
def test_pack_bits(reps):
    data = np.random.randint(2, size=reps, dtype=bool)
    packed = v2.pack_bits(data)
    assert isinstance(packed, bytes)
    assert len(packed) == (reps + 7) // 8
    unpacked = v2.unpack_bits(packed, reps)
    np.testing.assert_array_equal(unpacked, data)


def q(row: int, col: int) -> cirq.GridQubit:
    return cirq.GridQubit(row, col)


def _check_measurement(m, key, qubits, slot):
    assert m.key == key
    assert m.qubits == qubits
    assert m.slot == slot


def test_find_measurements_simple_schedule():
    circuit = cirq.Circuit()
    circuit.append(cirq.measure(q(0, 0), q(0, 1), q(0, 2), key='k'))
    # schedule = ct.circuit_to_schedule(device, circuit)
    measurements = v2.find_measurements(circuit)

    assert len(measurements) == 1
    m = measurements[0]
    _check_measurement(m, 'k', [q(0, 0), q(0, 1), q(0, 2)], 0)


def test_multiple_measurements_different_slots():
    circuit = cirq.Circuit()
    circuit.append(cirq.measure(q(0, 0), q(0, 1), key='k0'))
    circuit.append(cirq.measure(q(0, 2), q(0, 0), key='k1'))
    # schedule = ct.circuit_to_schedule(device, circuit)
    measurements = v2.find_measurements(circuit)

    assert len(measurements) == 2
    m0, m1 = measurements
    _check_measurement(m0, 'k0', [q(0, 0), q(0, 1)], 0)
    _check_measurement(m1, 'k1', [q(0, 2), q(0, 0)], 1)


def test_multiple_measurements_shared_slots():
    circuit = cirq.Circuit()
    circuit.append([
        cirq.measure(q(0, 0), q(0, 1), key='k0'),
        cirq.measure(q(0, 4), q(0, 3), key='k1')
    ])
    circuit.append([
        cirq.measure(q(0, 2), q(0, 0), q(0, 1), key='k2'),
        cirq.measure(q(0, 3), q(0, 4), key='k3')
    ])
    # schedule = ct.circuit_to_schedule(device, circuit)
    measurements = v2.find_measurements(circuit)

    assert len(measurements) == 4
    m0, m1, m2, m3 = measurements
    _check_measurement(m0, 'k0', [q(0, 0), q(0, 1)], 0)
    _check_measurement(m1, 'k1', [q(0, 4), q(0, 3)], 0)
    _check_measurement(m2, 'k2', [q(0, 2), q(0, 0), q(0, 1)], 1)
    _check_measurement(m3, 'k3', [q(0, 3), q(0, 4)], 1)
