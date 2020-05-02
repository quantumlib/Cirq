from typing import List, Sequence, Tuple

import itertools
import numpy as np
import pytest
import sympy

import cirq


def test_parameterized_gates():
    t = sympy.Symbol('t')
    with pytest.raises(ValueError):
        cphase_gate = cirq.CZPowGate(exponent=t)
        fsim_gate = cirq.google.SYC
        cirq.decompose_cphase_into_two_fsim(cphase_gate, fsim_gate=fsim_gate)

    with pytest.raises(ValueError):
        cphase_gate = cirq.CZ
        fsim_gate = cirq.FSimGate(theta=t, phi=np.pi / 2)
        cirq.decompose_cphase_into_two_fsim(cphase_gate, fsim_gate=fsim_gate)

    with pytest.raises(ValueError):
        cphase_gate = cirq.CZ
        fsim_gate = cirq.FSimGate(theta=np.pi / 2, phi=t)
        cirq.decompose_cphase_into_two_fsim(cphase_gate, fsim_gate=fsim_gate)


def test_invalid_qubits():
    with pytest.raises(ValueError):
        cirq.decompose_cphase_into_two_fsim(cphase_gate=cirq.CZ,
                                            fsim_gate=cirq.google.SYC,
                                            qubits=cirq.LineQubit.range(3))


def test_circuit_structure():
    circuit = cirq.decompose_cphase_into_two_fsim(cirq.CZ,
                                                  fsim_gate=cirq.google.SYC)
    assert len(circuit) == 5

    # First moment consists of 0- and 1-qubit operations.
    assert set(len(op.qubits) for op in circuit[0].operations) == {0, 1}

    # Second moment consists of a single Sycamore gate.
    assert len(circuit[1]) == 1
    assert isinstance(circuit[1].operations[0].gate, cirq.google.SycamoreGate)

    # Third moment consists of 1-qubit operations only.
    assert set(len(op.qubits) for op in circuit[2].operations) == {1}

    # Fourth moment consists of a single Sycamore gate.
    assert len(circuit[3]) == 1
    assert isinstance(circuit[3].operations[0].gate, cirq.google.SycamoreGate)

    # Fifth moment consists of 1-qubit operations only.
    assert set(len(op.qubits) for op in circuit[4].operations) == {1}


def assert_decomposition_valid(cphase_gate, fsim_gate):
    u_expected = cirq.unitary(cphase_gate)
    circuit = cirq.decompose_cphase_into_two_fsim(cphase_gate,
                                                  fsim_gate=fsim_gate)
    u_actual = cirq.unitary(circuit)
    assert np.allclose(u_actual, u_expected)


@pytest.mark.parametrize(
    'exponent', (-5.5, -3, -1.5, -1, -0.65, -0.2, 0, 0.1, 0.75, 1, 1.5, 2, 5.5))
def test_decomposition_to_sycamore_gate(exponent):
    cphase_gate = cirq.CZPowGate(exponent=exponent)
    assert_decomposition_valid(cphase_gate, cirq.google.SYC)


@pytest.mark.parametrize(
    'theta, phi',
    itertools.product(
        (-2.4 * np.pi, -np.pi / 11, np.pi / 9, np.pi / 2, 1.4 * np.pi),
        (-1.4 * np.pi, -np.pi / 9, np.pi / 11, np.pi / 2, 2.4 * np.pi)))
def test_valid_cphase_exponents(theta, phi):
    fsim_gate = cirq.FSimGate(theta=theta, phi=phi)
    valid_exponent_intervals = (
        cirq.compute_cphase_exponents_for_fsim_decomposition(fsim_gate))
    assert valid_exponent_intervals

    for min_exponent, max_exponent in valid_exponent_intervals:
        margin = 1e-8
        min_exponent += margin
        max_exponent -= margin
        assert min_exponent < max_exponent
        for exponent in np.linspace(min_exponent, max_exponent, 3):
            for d in (-2, 0, 4):
                cphase_gate = cirq.CZPowGate(exponent=exponent + d)
                assert_decomposition_valid(cphase_gate, fsim_gate=fsim_gate)
                cphase_gate = cirq.CZPowGate(exponent=-exponent + d)
                assert_decomposition_valid(cphase_gate, fsim_gate=fsim_gate)


def complement_intervals(intervals: Sequence[Tuple[float, float]]
                        ) -> Sequence[Tuple[float, float]]:
    """Computes complement of union of intervals in [0, 2]."""
    complements: List[Tuple[float, float]] = []
    a = 0.0
    for b, c in intervals:
        complements.append((a, b))
        a = c
    complements.append((a, 2.0))
    return tuple((a, b) for a, b in complements if a < b)


@pytest.mark.parametrize('theta, phi',
                         itertools.product(
                             (-2.3 * np.pi, -np.pi / 7, np.pi / 5, 1.8 * np.pi),
                             (-1.7 * np.pi, -np.pi / 5, np.pi / 7, 2.5 * np.pi))
                        )
def test_invalid_cphase_exponents(theta, phi):
    fsim_gate = cirq.FSimGate(theta=theta, phi=phi)
    valid_exponent_intervals = (
        cirq.compute_cphase_exponents_for_fsim_decomposition(fsim_gate))
    invalid_exponent_intervals = complement_intervals(valid_exponent_intervals)
    assert invalid_exponent_intervals

    for min_exponent, max_exponent in invalid_exponent_intervals:
        margin = 1e-8
        min_exponent += margin
        max_exponent -= margin
        assert min_exponent < max_exponent
        for exponent in np.linspace(min_exponent, max_exponent, 3):
            with pytest.raises(ValueError):
                cphase_gate = cirq.CZPowGate(exponent=exponent)
                assert_decomposition_valid(cphase_gate, fsim_gate=fsim_gate)


@pytest.mark.parametrize('bad_fsim_gate', (cirq.FSimGate(
    theta=0, phi=0), cirq.FSimGate(theta=np.pi / 4, phi=np.pi / 2)))
def test_invalid_fsim_gate(bad_fsim_gate):
    with pytest.raises(ValueError):
        cirq.decompose_cphase_into_two_fsim(cirq.CZ, fsim_gate=bad_fsim_gate)
