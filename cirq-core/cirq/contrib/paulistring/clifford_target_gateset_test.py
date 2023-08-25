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

import pytest
import sympy

import cirq
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset


@pytest.mark.parametrize(
    'op,expected_ops',
    (
        lambda q0, q1: (
            (cirq.X(q0), cirq.SingleQubitCliffordGate.X(q0)),
            (cirq.Y(q0), cirq.SingleQubitCliffordGate.Y(q0)),
            (cirq.Z(q0), cirq.SingleQubitCliffordGate.Z(q0)),
            (cirq.X(q0) ** 0.5, cirq.SingleQubitCliffordGate.X_sqrt(q0)),
            (cirq.Y(q0) ** 0.5, cirq.SingleQubitCliffordGate.Y_sqrt(q0)),
            (cirq.Z(q0) ** 0.5, cirq.SingleQubitCliffordGate.Z_sqrt(q0)),
            (cirq.X(q0) ** -0.5, cirq.SingleQubitCliffordGate.X_nsqrt(q0)),
            (cirq.Y(q0) ** -0.5, cirq.SingleQubitCliffordGate.Y_nsqrt(q0)),
            (cirq.Z(q0) ** -0.5, cirq.SingleQubitCliffordGate.Z_nsqrt(q0)),
            (cirq.X(q0) ** 0.25, cirq.PauliStringPhasor(cirq.PauliString([cirq.X.on(q0)])) ** 0.25),
            (cirq.Y(q0) ** 0.25, cirq.PauliStringPhasor(cirq.PauliString([cirq.Y.on(q0)])) ** 0.25),
            (cirq.Z(q0) ** 0.25, cirq.PauliStringPhasor(cirq.PauliString([cirq.Z.on(q0)])) ** 0.25),
            (cirq.X(q0) ** 0, ()),
            (cirq.CZ(q0, q1), cirq.CZ(q0, q1)),
            (cirq.measure(q0, q1, key='key'), cirq.measure(q0, q1, key='key')),
        )
    )(cirq.LineQubit(0), cirq.LineQubit(1)),
)
def test_converts_various_ops(op, expected_ops):
    before = cirq.Circuit(op)
    expected = cirq.Circuit(expected_ops, strategy=cirq.InsertStrategy.EARLIEST)
    after = cirq.optimize_for_target_gateset(
        before, gateset=CliffordTargetGateset(), ignore_failures=False
    )
    assert after == expected
    cirq.testing.assert_allclose_up_to_global_phase(
        before.unitary(), after.unitary(qubits_that_should_be_present=op.qubits), atol=1e-7
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        after.unitary(qubits_that_should_be_present=op.qubits),
        expected.unitary(qubits_that_should_be_present=op.qubits),
        atol=1e-7,
    )


def test_degenerate_single_qubit_decompose():
    q0 = cirq.LineQubit(0)

    before = cirq.Circuit(cirq.Z(q0) ** 0.1, cirq.X(q0) ** 1.0000000001, cirq.Z(q0) ** 0.1)
    expected = cirq.Circuit(cirq.SingleQubitCliffordGate.X(q0))

    after = cirq.optimize_for_target_gateset(
        before, gateset=CliffordTargetGateset(), ignore_failures=False
    )
    assert after == expected
    cirq.testing.assert_allclose_up_to_global_phase(before.unitary(), after.unitary(), atol=1e-7)
    cirq.testing.assert_allclose_up_to_global_phase(after.unitary(), expected.unitary(), atol=1e-7)


def test_converts_single_qubit_series():
    q0 = cirq.LineQubit(0)

    before = cirq.Circuit(
        cirq.X(q0),
        cirq.Y(q0),
        cirq.Z(q0),
        cirq.X(q0) ** 0.5,
        cirq.Y(q0) ** 0.5,
        cirq.Z(q0) ** 0.5,
        cirq.X(q0) ** -0.5,
        cirq.Y(q0) ** -0.5,
        cirq.Z(q0) ** -0.5,
        cirq.X(q0) ** 0.25,
        cirq.Y(q0) ** 0.25,
        cirq.Z(q0) ** 0.25,
    )

    after = cirq.optimize_for_target_gateset(
        before, gateset=CliffordTargetGateset(), ignore_failures=False
    )
    cirq.testing.assert_allclose_up_to_global_phase(before.unitary(), after.unitary(), atol=1e-7)


def test_converts_single_qubit_then_two():
    q0, q1 = cirq.LineQubit.range(2)

    before = cirq.Circuit(cirq.X(q0), cirq.Y(q0), cirq.CZ(q0, q1))

    after = cirq.optimize_for_target_gateset(
        before, gateset=CliffordTargetGateset(), ignore_failures=False
    )
    cirq.testing.assert_allclose_up_to_global_phase(before.unitary(), after.unitary(), atol=1e-7)


def test_converts_large_circuit():
    q0, q1, q2 = cirq.LineQubit.range(3)

    before = cirq.Circuit(
        cirq.X(q0),
        cirq.Y(q0),
        cirq.Z(q0),
        cirq.X(q0) ** 0.5,
        cirq.Y(q0) ** 0.5,
        cirq.Z(q0) ** 0.5,
        cirq.X(q0) ** -0.5,
        cirq.Y(q0) ** -0.5,
        cirq.Z(q0) ** -0.5,
        cirq.H(q0),
        cirq.CZ(q0, q1),
        cirq.CZ(q1, q2),
        cirq.X(q0) ** 0.25,
        cirq.Y(q0) ** 0.25,
        cirq.Z(q0) ** 0.25,
        cirq.CZ(q0, q1),
    )

    after = cirq.optimize_for_target_gateset(
        before, gateset=CliffordTargetGateset(), ignore_failures=False
    )

    cirq.testing.assert_allclose_up_to_global_phase(before.unitary(), after.unitary(), atol=1e-7)

    cirq.testing.assert_has_diagram(
        after,
        '''
0: ───Y^0.5───@───[Z]^-0.304───[X]^(1/3)───[Z]^0.446───────@───
              │                                            │
1: ───────────@────────────────────────────────────────@───@───
                                                       │
2: ────────────────────────────────────────────────────@───────
''',
    )


def test_convert_to_pauli_string_phasors():
    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(cirq.X(q0), cirq.Y(q1) ** 0.25, cirq.Z(q0) ** 0.125, cirq.H(q1))
    c_new = cirq.optimize_for_target_gateset(
        c_orig,
        gateset=CliffordTargetGateset(
            single_qubit_target=CliffordTargetGateset.SingleQubitTarget.PAULI_STRING_PHASORS
        ),
    )

    cirq.testing.assert_allclose_up_to_global_phase(c_new.unitary(), c_orig.unitary(), atol=1e-7)
    cirq.testing.assert_has_diagram(
        c_new,
        """
0: ───[X]─────────[Z]^(1/8)───

1: ───[Y]^-0.25───[Z]─────────
""",
    )


def test_already_converted():
    q0 = cirq.LineQubit(0)
    c_orig = cirq.Circuit(cirq.PauliStringPhasor(cirq.X.on(q0)))
    c_new = cirq.optimize_for_target_gateset(
        c_orig,
        gateset=CliffordTargetGateset(
            single_qubit_target=CliffordTargetGateset.SingleQubitTarget.PAULI_STRING_PHASORS
        ),
        ignore_failures=False,
    )
    assert c_new == c_orig


def test_ignore_unsupported_gate():
    class UnsupportedDummy(cirq.testing.TwoQubitGate):
        pass

    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(UnsupportedDummy()(q0, q1), cirq.X(q0) ** sympy.Symbol("theta"))
    c_new = cirq.optimize_for_target_gateset(
        c_orig, gateset=CliffordTargetGateset(), ignore_failures=True
    )
    assert c_new == c_orig


def test_fail_unsupported_gate():
    class UnsupportedDummy(cirq.testing.TwoQubitGate):
        pass

    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(UnsupportedDummy()(q0, q1))
    with pytest.raises(ValueError):
        _ = cirq.optimize_for_target_gateset(
            c_orig, gateset=CliffordTargetGateset(), ignore_failures=False
        )


def test_convert_to_single_qubit_cliffords():
    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(
        cirq.X(q0), cirq.Y(q1) ** 0.5, cirq.Z(q0) ** -0.5, cirq.Z(q1) ** 0, cirq.H(q0)
    )
    c_new = cirq.optimize_for_target_gateset(
        c_orig,
        gateset=CliffordTargetGateset(
            single_qubit_target=CliffordTargetGateset.SingleQubitTarget.SINGLE_QUBIT_CLIFFORDS
        ),
        ignore_failures=True,
    )

    assert all(isinstance(op.gate, cirq.SingleQubitCliffordGate) for op in c_new.all_operations())

    cirq.testing.assert_allclose_up_to_global_phase(c_new.unitary(), c_orig.unitary(), atol=1e-7)

    cirq.testing.assert_has_diagram(
        c_new,
        """
0: ───(X^-0.5-Z^0.5)───

1: ───Y^0.5────────────
""",
    )


def test_convert_to_single_qubit_cliffords_ignores_non_clifford():
    q0 = cirq.LineQubit(0)
    c_orig = cirq.Circuit(cirq.Z(q0) ** 0.25)
    c_new = cirq.optimize_for_target_gateset(
        c_orig,
        gateset=CliffordTargetGateset(
            single_qubit_target=CliffordTargetGateset.SingleQubitTarget.SINGLE_QUBIT_CLIFFORDS
        ),
        ignore_failures=True,
    )
    assert c_orig == c_new
