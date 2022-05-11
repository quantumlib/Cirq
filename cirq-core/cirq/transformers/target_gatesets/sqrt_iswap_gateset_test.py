# Copyright 2022 The Cirq Developers
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

from typing import Optional

import cirq
import pytest
import sympy
import numpy as np


def all_gates_of_type(m: cirq.Moment, g: cirq.Gateset):
    for op in m:
        if op not in g:
            return False
    return True


def assert_optimizes(before: cirq.Circuit, expected: cirq.Circuit, **kwargs):
    cirq.testing.assert_same_circuits(
        cirq.optimize_for_target_gateset(before, gateset=cirq.SqrtIswapTargetGateset(**kwargs)),
        expected,
    )


def assert_optimization_not_broken(
    circuit: cirq.Circuit, required_sqrt_iswap_count: Optional[int] = None
):
    c_new = cirq.optimize_for_target_gateset(
        circuit,
        gateset=cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=required_sqrt_iswap_count),
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, c_new, atol=1e-6
    )
    c_new = cirq.optimize_for_target_gateset(
        circuit,
        gateset=cirq.SqrtIswapTargetGateset(
            use_sqrt_iswap_inv=True, required_sqrt_iswap_count=required_sqrt_iswap_count
        ),
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, c_new, atol=1e-6
    )


def test_convert_to_sqrt_iswap_preserving_moment_structure():
    q = cirq.LineQubit.range(5)
    op = lambda q0, q1: cirq.H(q1).controlled_by(q0)
    c_orig = cirq.Circuit(
        cirq.Moment(cirq.X(q[2])),
        cirq.Moment(op(q[0], q[1]), op(q[2], q[3])),
        cirq.Moment(op(q[2], q[1]), op(q[4], q[3])),
        cirq.Moment(op(q[1], q[2]), op(q[3], q[4])),
        cirq.Moment(op(q[3], q[2]), op(q[1], q[0])),
        cirq.measure(*q[:2], key="m"),
        cirq.X(q[2]).with_classical_controls("m"),
        cirq.CZ(*q[3:]).with_classical_controls("m"),
    )

    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=cirq.SqrtIswapTargetGateset())

    assert c_orig[-2:] == c_new[-2:]
    c_orig, c_new = c_orig[:-2], c_new[:-2]

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_orig, c_new, atol=1e-6)
    assert all(
        (
            all_gates_of_type(m, cirq.Gateset(cirq.AnyUnitaryGateFamily(1)))
            or all_gates_of_type(m, cirq.Gateset(cirq.SQRT_ISWAP))
        )
        for m in c_new
    )

    c_new = cirq.optimize_for_target_gateset(
        c_orig, gateset=cirq.SqrtIswapTargetGateset(use_sqrt_iswap_inv=True), ignore_failures=False
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_orig, c_new, atol=1e-6)
    assert all(
        (
            all_gates_of_type(m, cirq.Gateset(cirq.AnyUnitaryGateFamily(1)))
            or all_gates_of_type(m, cirq.Gateset(cirq.SQRT_ISWAP_INV))
        )
        for m in c_new
    )


@pytest.mark.parametrize(
    'gate',
    [
        cirq.CNotPowGate(exponent=sympy.Symbol('t')),
        cirq.PhasedFSimGate(theta=sympy.Symbol('t'), chi=sympy.Symbol('t'), phi=sympy.Symbol('t')),
    ],
)
@pytest.mark.parametrize('use_sqrt_iswap_inv', [True, False])
def test_two_qubit_gates_with_symbols(gate: cirq.Gate, use_sqrt_iswap_inv: bool):
    # Note that even though these gates are not natively supported by
    # `cirq.parameterized_2q_op_to_sqrt_iswap_operations`, the transformation succeeds because
    # `cirq.optimize_for_target_gateset` also relies on `cirq.decompose` as a fallback.

    c_orig = cirq.Circuit(gate(*cirq.LineQubit.range(2)))
    c_new = cirq.optimize_for_target_gateset(
        c_orig, gateset=cirq.SqrtIswapTargetGateset(use_sqrt_iswap_inv=use_sqrt_iswap_inv)
    )

    # Check that `c_new` only contains sqrt iswap as the 2q entangling gate.
    sqrt_iswap_gate = cirq.SQRT_ISWAP_INV if use_sqrt_iswap_inv else cirq.SQRT_ISWAP
    for op in c_new.all_operations():
        if cirq.num_qubits(op) == 2:
            assert op.gate == sqrt_iswap_gate

    # Check if unitaries are the same
    for val in np.linspace(0, 2 * np.pi, 10):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.resolve_parameters(c_orig, {'t': val}),
            cirq.resolve_parameters(c_new, {'t': val}),
            atol=1e-6,
        )


def test_sqrt_iswap_gateset_raises():
    with pytest.raises(ValueError, match="`required_sqrt_iswap_count` must be 0, 1, 2, or 3"):
        _ = cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=4)


def test_sqrt_iswap_gateset_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.SqrtIswapTargetGateset(), cirq.SqrtIswapTargetGateset(use_sqrt_iswap_inv=False)
    )
    eq.add_equality_group(
        cirq.SqrtIswapTargetGateset(atol=1e-6, required_sqrt_iswap_count=0, use_sqrt_iswap_inv=True)
    )
    eq.add_equality_group(
        cirq.SqrtIswapTargetGateset(atol=1e-6, required_sqrt_iswap_count=3, use_sqrt_iswap_inv=True)
    )


@pytest.mark.parametrize(
    'gateset',
    [
        cirq.SqrtIswapTargetGateset(),
        cirq.SqrtIswapTargetGateset(
            atol=1e-6, required_sqrt_iswap_count=2, use_sqrt_iswap_inv=True
        ),
    ],
)
def test_sqrt_iswap_gateset_repr(gateset):
    cirq.testing.assert_equivalent_repr(gateset)


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
    c = cirq.optimize_for_target_gateset(c, gateset=cirq.SqrtIswapTargetGateset())
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 2


def test_optimizes_single_inv_sqrt_iswap():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SQRT_ISWAP_INV(a, b))
    assert_optimization_not_broken(c)
    c = cirq.optimize_for_target_gateset(c, gateset=cirq.SqrtIswapTargetGateset())
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 1


def test_optimizes_single_iswap_require0():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(a, b), cirq.CNOT(a, b))  # Minimum 0 sqrt-iSWAP
    assert_optimization_not_broken(c, required_sqrt_iswap_count=0)
    c = cirq.optimize_for_target_gateset(
        c, gateset=cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=0)
    )
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 0


def test_optimizes_single_iswap_require0_raises():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(a, b))  # Minimum 2 sqrt-iSWAP
    with pytest.raises(ValueError, match='cannot be decomposed into exactly 0 sqrt-iSWAP gates'):
        _ = cirq.optimize_for_target_gateset(
            c, gateset=cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=0)
        )


def test_optimizes_single_iswap_require1():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SQRT_ISWAP_INV(a, b))  # Minimum 1 sqrt-iSWAP
    assert_optimization_not_broken(c, required_sqrt_iswap_count=1)
    c = cirq.optimize_for_target_gateset(
        c, gateset=cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=1)
    )
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 1


def test_optimizes_single_iswap_require1_raises():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(a, b))  # Minimum 2 sqrt-iSWAP
    with pytest.raises(ValueError, match='cannot be decomposed into exactly 1 sqrt-iSWAP gates'):
        c = cirq.optimize_for_target_gateset(
            c, gateset=cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=1)
        )


def test_optimizes_single_iswap_require2():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SQRT_ISWAP_INV(a, b))  # Minimum 1 sqrt-iSWAP but 2 possible
    assert_optimization_not_broken(c, required_sqrt_iswap_count=2)
    c = cirq.optimize_for_target_gateset(
        c, gateset=cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=2)
    )
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 2


def test_optimizes_single_iswap_require2_raises():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SWAP(a, b))  # Minimum 3 sqrt-iSWAP
    with pytest.raises(ValueError, match='cannot be decomposed into exactly 2 sqrt-iSWAP gates'):
        c = cirq.optimize_for_target_gateset(
            c, gateset=cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=2)
        )


def test_optimizes_single_iswap_require3():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.ISWAP(a, b))  # Minimum 2 sqrt-iSWAP but 3 possible
    assert_optimization_not_broken(c, required_sqrt_iswap_count=3)
    c = cirq.optimize_for_target_gateset(
        c, gateset=cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=3)
    )
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 3


def test_optimizes_single_inv_sqrt_iswap_require3():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SQRT_ISWAP_INV(a, b))
    assert_optimization_not_broken(c, required_sqrt_iswap_count=3)
    c = cirq.optimize_for_target_gateset(
        c, gateset=cirq.SqrtIswapTargetGateset(required_sqrt_iswap_count=3)
    )
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 3
