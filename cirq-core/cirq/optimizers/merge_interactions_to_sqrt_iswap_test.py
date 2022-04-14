# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import pytest

import cirq


def assert_optimizes(before: cirq.Circuit, expected: cirq.Circuit, **kwargs):
    """Check that optimizing the circuit ``before`` produces the circuit ``expected``.

    The optimized circuit is cleaned up with follow up optimizations to make the
    comparison more robust to extra moments or extra gates nearly equal to
    identity that don't matter.

    Args:
        before: The input circuit to optimize.
        expected: The expected result of optimization to compare against.
        **kwargs: Any extra arguments to pass to the
            ``MergeInteractionsToSqrtIswap`` constructor.
    """
    actual = before.copy()
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
    ):
        opt = cirq.MergeInteractionsToSqrtIswap(**kwargs)
        opt.optimize_circuit(actual)

    # Ignore differences that would be caught by follow-up optimizations.
    followup_transformers: List[cirq.TRANSFORMER] = [
        cirq.merge_single_qubit_gates_to_phased_x_and_z,
        cirq.eject_phased_paulis,
        cirq.eject_z,
        cirq.drop_negligible_operations,
        cirq.drop_empty_moments,
    ]
    for transform in followup_transformers:
        actual = transform(actual).unfreeze(copy=False)
        expected = transform(expected).unfreeze(copy=False)

    assert actual == expected, f'ACTUAL {actual} : EXPECTED {expected}'


def assert_optimization_not_broken(circuit: cirq.Circuit, **kwargs):
    """Check that the unitary matrix for the input circuit is the same (up to
    global phase and rounding error) as the unitary matrix of the optimized
    circuit."""
    u_before = circuit.unitary(sorted(circuit.all_qubits()))
    c_sqrt_iswap = circuit.copy()
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
    ):
        cirq.MergeInteractionsToSqrtIswap(**kwargs).optimize_circuit(c_sqrt_iswap)
    u_after = c_sqrt_iswap.unitary(sorted(circuit.all_qubits()))

    # Not 1e-8 because of some unaccounted accumulated error in some of Cirq's linalg functions
    cirq.testing.assert_allclose_up_to_global_phase(u_before, u_after, atol=1e-6)

    # Also test optimization with SQRT_ISWAP_INV
    c_sqrt_iswap_inv = circuit.copy()
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
    ):
        cirq.MergeInteractionsToSqrtIswap(use_sqrt_iswap_inv=True).optimize_circuit(
            c_sqrt_iswap_inv
        )
    u_after2 = c_sqrt_iswap_inv.unitary(sorted(circuit.all_qubits()))

    cirq.testing.assert_allclose_up_to_global_phase(u_before, u_after2, atol=1e-6)


def test_clears_paired_cnot():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(
        before=cirq.Circuit([cirq.Moment([cirq.CNOT(a, b)]), cirq.Moment([cirq.CNOT(a, b)])]),
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
        expected=cirq.Circuit([cirq.Moment([cirq.SQRT_ISWAP(a, b)])]),
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
        expected=cirq.Circuit([cirq.Moment([cirq.SQRT_ISWAP_INV(a, b)])]),
    )


def test_works_with_tags():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.SQRT_ISWAP(a, b).with_tags('mytag1')]),
                cirq.Moment([cirq.SQRT_ISWAP(a, b).with_tags('mytag2')]),
                cirq.Moment([cirq.SQRT_ISWAP_INV(a, b).with_tags('mytag3')]),
            ]
        ),
        expected=cirq.Circuit([cirq.Moment([cirq.SQRT_ISWAP(a, b)])]),
    )


def test_no_touch_single_sqrt_iswap():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        [
            cirq.Moment(
                [cirq.ISwapPowGate(exponent=0.5, global_shift=-0.5).on(a, b).with_tags('mytag')]
            )
        ]
    )
    assert_optimizes(before=circuit, expected=circuit)


def test_no_touch_single_sqrt_iswap_inv():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        [
            cirq.Moment(
                [cirq.ISwapPowGate(exponent=-0.5, global_shift=-0.5).on(a, b).with_tags('mytag')]
            )
        ]
    )
    assert_optimizes(before=circuit, expected=circuit, use_sqrt_iswap_inv=True)


def test_cnots_separated_by_single_gates_correct():
    a, b = cirq.LineQubit.range(2)
    assert_optimization_not_broken(cirq.Circuit(cirq.CNOT(a, b), cirq.H(b), cirq.CNOT(a, b)))


def test_czs_separated_by_single_gates_correct():
    a, b = cirq.LineQubit.range(2)
    assert_optimization_not_broken(
        cirq.Circuit(cirq.CZ(a, b), cirq.X(b), cirq.X(b), cirq.X(b), cirq.CZ(a, b))
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
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
    ):
        cirq.MergeInteractionsToSqrtIswap().optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 2


def test_optimizes_single_inv_sqrt_iswap():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SQRT_ISWAP_INV(a, b))
    assert_optimization_not_broken(c)
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
    ):
        cirq.MergeInteractionsToSqrtIswap().optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 1


def test_init_raises():
    with pytest.raises(ValueError, match='must be 0, 1, 2, or 3'):
        with cirq.testing.assert_deprecated(
            "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
        ):
            cirq.MergeInteractionsToSqrtIswap(required_sqrt_iswap_count=4)


def test_optimizes_single_iswap_require0():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(a, b), cirq.CNOT(a, b))  # Minimum 0 sqrt-iSWAP
    assert_optimization_not_broken(c, required_sqrt_iswap_count=0)
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
    ):
        cirq.MergeInteractionsToSqrtIswap(required_sqrt_iswap_count=0).optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 0


def test_optimizes_single_iswap_require0_raises():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(a, b))  # Minimum 2 sqrt-iSWAP
    with pytest.raises(ValueError, match='cannot be decomposed into exactly 0 sqrt-iSWAP gates'):
        with cirq.testing.assert_deprecated(
            "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
        ):
            cirq.MergeInteractionsToSqrtIswap(required_sqrt_iswap_count=0).optimize_circuit(c)


def test_optimizes_single_iswap_require1():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SQRT_ISWAP_INV(a, b))  # Minimum 1 sqrt-iSWAP
    assert_optimization_not_broken(c, required_sqrt_iswap_count=1)
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
    ):
        cirq.MergeInteractionsToSqrtIswap(required_sqrt_iswap_count=1).optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 1


def test_optimizes_single_iswap_require1_raises():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(a, b))  # Minimum 2 sqrt-iSWAP
    with pytest.raises(ValueError, match='cannot be decomposed into exactly 1 sqrt-iSWAP gates'):
        with cirq.testing.assert_deprecated(
            "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
        ):
            cirq.MergeInteractionsToSqrtIswap(required_sqrt_iswap_count=1).optimize_circuit(c)


def test_optimizes_single_iswap_require2():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SQRT_ISWAP_INV(a, b))  # Minimum 1 sqrt-iSWAP but 2 possible
    assert_optimization_not_broken(c, required_sqrt_iswap_count=2)
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
    ):
        cirq.MergeInteractionsToSqrtIswap(required_sqrt_iswap_count=2).optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 2


def test_optimizes_single_iswap_require2_raises():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SWAP(a, b))  # Minimum 3 sqrt-iSWAP
    with pytest.raises(ValueError, match='cannot be decomposed into exactly 2 sqrt-iSWAP gates'):
        with cirq.testing.assert_deprecated(
            "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
        ):
            cirq.MergeInteractionsToSqrtIswap(required_sqrt_iswap_count=2).optimize_circuit(c)


def test_optimizes_single_iswap_require3():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.ISWAP(a, b))  # Minimum 2 sqrt-iSWAP but 3 possible
    assert_optimization_not_broken(c, required_sqrt_iswap_count=3)
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
    ):
        cirq.MergeInteractionsToSqrtIswap(required_sqrt_iswap_count=3).optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 3


def test_optimizes_single_inv_sqrt_iswap_require3():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SQRT_ISWAP_INV(a, b))
    assert_optimization_not_broken(c, required_sqrt_iswap_count=3)
    with cirq.testing.assert_deprecated(
        "Use cirq.optimize_for_target_gateset", deadline='v1.0', count=2
    ):
        cirq.MergeInteractionsToSqrtIswap(required_sqrt_iswap_count=3).optimize_circuit(c)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 3
