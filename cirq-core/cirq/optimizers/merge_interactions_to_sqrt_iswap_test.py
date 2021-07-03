from typing import Callable, List

import cirq


def assert_optimizes(before: cirq.Circuit, expected: cirq.Circuit, **kwargs):
    actual = cirq.Circuit(before)
    opt = cirq.MergeInteractionsToSqrtIswap(**kwargs)
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


def assert_optimization_not_broken(circuit: cirq.Circuit):
    """Check that the unitary matrix for the input circuit is the same (up to
    global phase and rounding error) as the unitary matrix of the optimized
    circuit."""
    u_before = circuit.unitary()
    c_sqrt_iswap = circuit.copy()
    cirq.MergeInteractionsToSqrtIswap().optimize_circuit(c_sqrt_iswap)
    u_after = c_sqrt_iswap.unitary()

    cirq.testing.assert_allclose_up_to_global_phase(u_before, u_after, atol=2e-8)

    # Also test optimization with SQRT_ISWAP_INV
    c_sqrt_iswap_inv = circuit.copy()
    cirq.MergeInteractionsToSqrtIswap(use_sqrt_iswap_inv=True).optimize_circuit(c_sqrt_iswap_inv)
    u_after2 = c_sqrt_iswap_inv.unitary()

    cirq.testing.assert_allclose_up_to_global_phase(u_before, u_after2, atol=2e-8)


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


def test_simplifies_sqrt_iswap():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(
        before=cirq.Circuit(
            [
                # SQRT_ISWAP**8 == Identity
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
            ]
        ),
    )


def test_simplifies_sqrt_iswap_inv():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(
        use_sqrt_iswap_inv=True,
        before=cirq.Circuit(
            [
                # SQRT_ISWAP**8 == Identity
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP_INV(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b)]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment([cirq.SQRT_ISWAP_INV(a, b)]),
            ]
        ),
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
    c = cirq.Circuit(cirq.SQRT_ISWAP(a, b) ** -1)
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
    c = cirq.Circuit(cirq.SQRT_ISWAP(a, b) ** -1)
    assert_optimization_not_broken(c)
    cirq.MergeInteractionsToSqrtIswap(require_three_sqrt_iswap=True).optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 3
