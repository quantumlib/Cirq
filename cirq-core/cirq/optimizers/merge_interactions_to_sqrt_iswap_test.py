from typing import Callable, List

import pytest

import cirq


def assert_optimizes(before: cirq.Circuit, expected: cirq.Circuit):
    actual = cirq.Circuit(before)
    opt = cirq.MergeInteractionsToSqrtIswap()
    opt.optimize_circuit(actual)

    # Ignore differences that would be caught by follow-up optimizations.
    followup_optimizations: List[Callable[[cirq.Circuit], None]] = [
        cirq.merge_single_qubit_gates_into_phased_x_z,
        cirq.EjectPhasedPaulis().optimize_circuit,
        cirq.EjectZ().optimize_circuit,
        cirq.DropNegligible().optimize_circuit,
        cirq.DropEmptyMoments().optimize_circuit,
    ]
    for post in followup_optimizations:
        post(actual)
        post(expected)

    assert actual == expected, f'ACTUAL {actual} : EXPECTED {expected}'


def assert_optimization_not_broken(circuit):
    """Check that the unitary matrix for the input circuit is the same (up to
    global phase and rounding error) as the unitary matrix of the optimized
    circuit."""
    u_before = circuit.unitary()
    cirq.MergeInteractions().optimize_circuit(circuit)
    u_after = circuit.unitary()

    cirq.testing.assert_allclose_up_to_global_phase(u_before, u_after, atol=1e-8)


def test_clears_paired_cnot():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.CNOT(a, b)]),
                cirq.Moment([cirq.CNOT(a, b)]),
            ]
        ),
        expected=cirq.Circuit(),
    )


def test_cnots_separated_by_single_gates_correct():
    a, b = cirq.LineQubit.range(2)
    assert_optimization_not_broken(
        cirq.Circuit(
            cirq.CNOT(a, b),
            cirq.H(b),
            cirq.CNOT(a, b),
        )
    )


def test_czs_separated_by_single_gates_correct():
    a, b = cirq.LineQubit.range(2)
    assert_optimization_not_broken(
        cirq.Circuit(
            cirq.CZ(a, b),
            cirq.X(b),
            cirq.X(b),
            cirq.X(b),
            cirq.CZ(a, b),
        )
    )


def test_inefficient_circuit_correct():
    t = 0.1
    v = 0.11
    a, b = cirq.LineQubit.range(2)
    assert_optimization_not_broken(
        cirq.Circuit(
            cirq.H(b),
            cirq.CNOT(a, b),
            cirq.H(b),
            cirq.CNOT(a, b),
            cirq.CNOT(b, a),
            cirq.H(a),
            cirq.CNOT(a, b),
            cirq.Z(a) ** t,
            cirq.Z(b) ** -t,
            cirq.CNOT(a, b),
            cirq.H(a),
            cirq.Z(b) ** v,
            cirq.CNOT(a, b),
            cirq.Z(a) ** -v,
            cirq.Z(b) ** -v,
        )
    )


def test_optimizes_single_iswap():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.ISWAP(a, b))
    assert_optimization_not_broken(c)
    cirq.MergeInteractionsToSqrtIswap().optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 2


def test_optimizes_single_inv_sqrt_iswap():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SQRT_ISWAP(a, b)**-1)
    assert_optimization_not_broken(c)
    cirq.MergeInteractionsToSqrtIswap().optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 1


def test_optimizes_single_iswap_require3():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.ISWAP(a, b))
    assert_optimization_not_broken(c)
    cirq.MergeInteractionsToSqrtIswap(require_three_sqrt_iswap=True).optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 3


def test_optimizes_single_inv_sqrt_iswap_require3():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SQRT_ISWAP(a, b)**-1)
    assert_optimization_not_broken(c)
    cirq.MergeInteractionsToSqrtIswap(require_three_sqrt_iswap=True).optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 3
