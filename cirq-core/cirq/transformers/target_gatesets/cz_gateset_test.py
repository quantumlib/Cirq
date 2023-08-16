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

from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np


def all_gates_of_type(m: cirq.Moment, g: cirq.Gateset):
    for op in m:
        if op not in g:
            return False
    return True


def assert_optimizes(
    before: cirq.Circuit,
    expected: cirq.Circuit,
    additional_gates: Optional[Sequence[Type[cirq.Gate]]] = None,
):
    if additional_gates is None:
        gateset = cirq.CZTargetGateset()
    else:
        gateset = cirq.CZTargetGateset(additional_gates=additional_gates)

    cirq.testing.assert_same_circuits(
        cirq.optimize_for_target_gateset(before, gateset=gateset, ignore_failures=False), expected
    )


def assert_optimization_not_broken(circuit: cirq.Circuit):
    c_new = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, c_new, atol=1e-6
    )
    c_new = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq.CZTargetGateset(allow_partial_czs=True), ignore_failures=False
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, c_new, atol=1e-6
    )


def test_convert_to_cz_preserving_moment_structure():
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
    # Classically controlled operations are not part of the gateset, so failures should be ignored
    # during compilation.
    c_new = cirq.optimize_for_target_gateset(
        c_orig, gateset=cirq.CZTargetGateset(), ignore_failures=True
    )

    assert c_orig[-2:] == c_new[-2:]
    c_orig, c_new = c_orig[:-2], c_new[:-2]

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_orig, c_new, atol=1e-6)
    assert all(
        (
            all_gates_of_type(m, cirq.Gateset(cirq.PhasedXZGate))
            or all_gates_of_type(m, cirq.Gateset(cirq.CZ))
        )
        for m in c_new
    )

    c_new = cirq.optimize_for_target_gateset(
        c_orig, gateset=cirq.CZTargetGateset(allow_partial_czs=True), ignore_failures=False
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_orig, c_new, atol=1e-6)
    assert all(
        (
            all_gates_of_type(m, cirq.Gateset(cirq.PhasedXZGate))
            or all_gates_of_type(m, cirq.Gateset(cirq.CZPowGate))
        )
        for m in c_new
    )


def test_clears_paired_cnot():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(
        before=cirq.Circuit(cirq.Moment(cirq.CNOT(a, b)), cirq.Moment(cirq.CNOT(a, b))),
        expected=cirq.Circuit(),
    )


def test_ignores_czs_separated_by_parameterized():
    a, b = cirq.LineQubit.range(2)
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment(cirq.CZ(a, b)),
                cirq.Moment(cirq.Z(a) ** sympy.Symbol('boo')),
                cirq.Moment(cirq.CZ(a, b)),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment(cirq.CZ(a, b)),
                cirq.Moment(cirq.Z(a) ** sympy.Symbol('boo')),
                cirq.Moment(cirq.CZ(a, b)),
            ]
        ),
        additional_gates=[cirq.ZPowGate],
    )


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
    c = cirq.optimize_for_target_gateset(c, gateset=cirq.CZTargetGateset(), ignore_failures=False)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 2


def test_optimizes_tagged_partial_cz():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit((cirq.CZ**0.5)(a, b).with_tags('mytag'))
    assert_optimization_not_broken(c)
    c = cirq.optimize_for_target_gateset(c, gateset=cirq.CZTargetGateset(), ignore_failures=False)
    assert (
        len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 2
    ), 'It should take 2 CZ gates to decompose a CZ**0.5 gate'


def test_not_decompose_czs():
    circuit = cirq.Circuit(
        cirq.CZPowGate(exponent=1, global_shift=-0.5).on(*cirq.LineQubit.range(2))
    )
    assert_optimizes(before=circuit, expected=circuit)


@pytest.mark.parametrize(
    'circuit',
    (
        cirq.Circuit(cirq.CZPowGate(exponent=0.1)(*cirq.LineQubit.range(2))),
        cirq.Circuit(
            cirq.CZPowGate(exponent=0.2)(*cirq.LineQubit.range(2)),
            cirq.CZPowGate(exponent=0.3, global_shift=-0.5)(*cirq.LineQubit.range(2)),
        ),
    ),
)
def test_decompose_partial_czs(circuit):
    circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq.CZTargetGateset(), ignore_failures=False
    )
    cz_gates = [
        op.gate
        for op in circuit.all_operations()
        if isinstance(op, cirq.GateOperation) and isinstance(op.gate, cirq.CZPowGate)
    ]
    num_full_cz = sum(1 for cz in cz_gates if cz.exponent % 2 == 1)
    num_part_cz = sum(1 for cz in cz_gates if cz.exponent % 2 != 1)
    assert num_full_cz == 2
    assert num_part_cz == 0


def test_not_decompose_partial_czs():
    circuit = cirq.Circuit(
        cirq.CZPowGate(exponent=0.1, global_shift=-0.5)(*cirq.LineQubit.range(2))
    )
    cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset(), ignore_failures=False)
    cz_gates = [
        op.gate
        for op in circuit.all_operations()
        if isinstance(op, cirq.GateOperation) and isinstance(op.gate, cirq.CZPowGate)
    ]
    num_full_cz = sum(1 for cz in cz_gates if cz.exponent % 2 == 1)
    num_part_cz = sum(1 for cz in cz_gates if cz.exponent % 2 != 1)
    assert num_full_cz == 0
    assert num_part_cz == 1


def test_avoids_decompose_when_matrix_available():
    class OtherXX(cirq.testing.TwoQubitGate):  # pragma: no cover
        def _has_unitary_(self) -> bool:
            return True

        def _unitary_(self) -> np.ndarray:
            m = np.array([[0, 1], [1, 0]])
            return np.kron(m, m)

        def _decompose_(self, qubits):
            assert False

    class OtherOtherXX(cirq.testing.TwoQubitGate):  # pragma: no cover
        def _has_unitary_(self) -> bool:
            return True

        def _unitary_(self) -> np.ndarray:
            m = np.array([[0, 1], [1, 0]])
            return np.kron(m, m)

        def _decompose_(self, qubits):
            assert False

    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(OtherXX()(a, b), OtherOtherXX()(a, b))
    c = cirq.optimize_for_target_gateset(c, gateset=cirq.CZTargetGateset(), ignore_failures=False)
    assert len(c) == 0


def test_composite_gates_without_matrix():
    class CompositeDummy(cirq.testing.SingleQubitGate):
        def _decompose_(self, qubits):
            yield cirq.X(qubits[0])
            yield cirq.Y(qubits[0]) ** 0.5

    class CompositeDummy2(cirq.testing.TwoQubitGate):
        def _decompose_(self, qubits):
            yield cirq.CZ(qubits[0], qubits[1])
            yield CompositeDummy()(qubits[1])

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(CompositeDummy()(q0), CompositeDummy2()(q0, q1))
    expected = cirq.Circuit(
        cirq.X(q0), cirq.Y(q0) ** 0.5, cirq.CZ(q0, q1), cirq.X(q1), cirq.Y(q1) ** 0.5
    )
    c_new = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq.CZTargetGateset(), ignore_failures=False
    )

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        c_new, expected, atol=1e-6
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        c_new, circuit, atol=1e-6
    )


def test_unsupported_gate():
    class UnsupportedDummy(cirq.testing.TwoQubitGate):
        pass

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(UnsupportedDummy()(q0, q1))
    assert circuit == cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
    with pytest.raises(ValueError, match='Unable to convert'):
        _ = cirq.optimize_for_target_gateset(
            circuit, gateset=cirq.CZTargetGateset(), ignore_failures=False
        )


@pytest.mark.parametrize(
    'gateset',
    [
        cirq.CZTargetGateset(),
        cirq.CZTargetGateset(
            atol=1e-6,
            allow_partial_czs=True,
            additional_gates=[
                cirq.SQRT_ISWAP,
                cirq.XPowGate,
                cirq.YPowGate,
                cirq.GateFamily(cirq.ZPowGate, tags_to_accept=['test_tag']),
            ],
        ),
        cirq.CZTargetGateset(additional_gates=()),
    ],
)
def test_repr(gateset):
    cirq.testing.assert_equivalent_repr(gateset)
