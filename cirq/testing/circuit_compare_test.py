# Copyright 2018 The Cirq Developers
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

import numpy as np

import cirq


def test_sensitive_to_phase():
    q = cirq.NamedQubit('q')

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([])
        ]),
        cirq.Circuit(),
        atol=0)

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit([
                cirq.Moment([cirq.Z(q)**0.0001])
            ]),
            cirq.Circuit(),
            atol=0)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.0001])
        ]),
        cirq.Circuit(),
        atol=0.01)


def test_sensitive_to_measurement_but_not_measured_phase():
    q = cirq.NamedQubit('q')

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit([
                cirq.Moment([cirq.measure(q)])
            ]),
            cirq.Circuit(),
            atol=1e-8)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.measure(q)])
        ]),
        cirq.Circuit([
            cirq.Moment([cirq.Z(q)]),
            cirq.Moment([cirq.measure(q)]),
        ]),
        atol=1e-8)

    a, b = cirq.LineQubit.range(2)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.measure(a, b)])
        ]),
        cirq.Circuit([
            cirq.Moment([cirq.Z(a)]),
            cirq.Moment([cirq.measure(a, b)]),
        ]),
        atol=1e-8)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.measure(a)])
        ]),
        cirq.Circuit([
            cirq.Moment([cirq.Z(a)]),
            cirq.Moment([cirq.measure(a)]),
        ]),
        atol=1e-8)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.measure(a, b)])
        ]),
        cirq.Circuit([
            cirq.Moment([cirq.T(a), cirq.S(b)]),
            cirq.Moment([cirq.measure(a, b)]),
        ]),
        atol=1e-8)

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit([
                cirq.Moment([cirq.measure(a)])
            ]),
            cirq.Circuit([
                cirq.Moment([cirq.T(a), cirq.S(b)]),
                cirq.Moment([cirq.measure(a)]),
            ]),
            atol=1e-8)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.measure(a, b)])
        ]),
        cirq.Circuit([
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.measure(a, b)]),
        ]),
        atol=1e-8)


def test_sensitive_to_measurement_toggle():
    q = cirq.NamedQubit('q')

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit([
                cirq.Moment([cirq.measure(q)])
            ]),
            cirq.Circuit([
                cirq.Moment([cirq.X(q)]),
                cirq.Moment([cirq.measure(q)]),
            ]),
            atol=1e-8)

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit([
                cirq.Moment([cirq.measure(q)])
            ]),
            cirq.Circuit([
                cirq.Moment([cirq.measure(q, invert_mask=(True,))]),
            ]),
            atol=1e-8)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.measure(q)])
        ]),
        cirq.Circuit([
            cirq.Moment([cirq.X(q)]),
            cirq.Moment([cirq.measure(q, invert_mask=(True,))]),
        ]),
        atol=1e-8)


def test_measuring_qubits():
    a, b = cirq.LineQubit.range(2)

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit([
                cirq.Moment([cirq.measure(a)])
            ]),
            cirq.Circuit([
                cirq.Moment([cirq.measure(b)])
            ]),
            atol=1e-8)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.measure(a, b, invert_mask=(True,))])
        ]),
        cirq.Circuit([
            cirq.Moment([cirq.measure(b, a, invert_mask=(False, True))])
        ]),
        atol=1e-8)

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit([
            cirq.Moment([cirq.measure(a)]),
            cirq.Moment([cirq.measure(b)]),
        ]),
        cirq.Circuit([
            cirq.Moment([cirq.measure(a, b)])
        ]),
        atol=1e-8)


@pytest.mark.parametrize(
    'circuit',
    [
        cirq.testing.random_circuit(cirq.LineQubit.range(2), 4, 0.5)
        for _ in range(5)
    ]
)
def test_random_same_matrix(circuit):
    a, b = cirq.LineQubit.range(2)
    same = cirq.Circuit.from_ops(
        cirq.TwoQubitMatrixGate(circuit.to_unitary_matrix(
            qubits_that_should_be_present=[a, b])).on(a, b))

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, same, atol=1e-8)

    circuit.append(cirq.measure(a))
    same.append(cirq.measure(a))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, same, atol=1e-8)


def test_correct_qubit_ordering():
    a, b = cirq.LineQubit.range(2)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit.from_ops(cirq.Z(a),
                              cirq.Z(b),
                              cirq.measure(b)),
        cirq.Circuit.from_ops(cirq.Z(a),
                              cirq.measure(b)),
        atol=1e-8)

    with pytest.raises(AssertionError):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            cirq.Circuit.from_ops(cirq.Z(a),
                                  cirq.Z(b),
                                  cirq.measure(b)),
            cirq.Circuit.from_ops(cirq.Z(b),
                                  cirq.measure(b)),
            atol=1e-8)


def test_known_old_failure():
    a, b = cirq.LineQubit.range(2)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        actual=cirq.Circuit.from_ops(
            cirq.PhasedXPowGate(exponent=0.61351656,
                                phase_exponent=0.8034575038876517).on(b),
            cirq.measure(a, b)),
        reference=cirq.Circuit.from_ops(
            cirq.PhasedXPowGate(exponent=0.61351656,
                                phase_exponent=0.8034575038876517).on(b),
            cirq.Z(a)**0.5,
            cirq.Z(b)**0.1,
            cirq.measure(a, b)),
        atol=1e-8)


def test_assert_same_circuits():
    a, b = cirq.LineQubit.range(2)

    cirq.testing.assert_same_circuits(
        cirq.Circuit.from_ops(cirq.H(a)),
        cirq.Circuit.from_ops(cirq.H(a)),
    )

    with pytest.raises(AssertionError) as exc_info:
        cirq.testing.assert_same_circuits(
            cirq.Circuit.from_ops(cirq.H(a)),
            cirq.Circuit(),
        )
    assert 'differing moment:\n0\n' in exc_info.value.args[0]

    with pytest.raises(AssertionError) as exc_info:
        cirq.testing.assert_same_circuits(
            cirq.Circuit.from_ops(cirq.H(a), cirq.H(a)),
            cirq.Circuit.from_ops(cirq.H(a), cirq.CZ(a, b)),
        )
    assert 'differing moment:\n1\n' in exc_info.value.args[0]

    with pytest.raises(AssertionError):
        cirq.testing.assert_same_circuits(
            cirq.Circuit.from_ops(cirq.CNOT(a, b)),
            cirq.Circuit.from_ops(cirq.ControlledGate(cirq.X).on(a, b)),
        )


def test_assert_has_diagram():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.CNOT(a, b))
    cirq.testing.assert_has_diagram(circuit, """
0: ───@───
      │
1: ───X───
""")

    expected_error = """Circuit's text diagram differs from the desired diagram.

Diagram of actual circuit:
0: ───@───
      │
1: ───X───

Desired text diagram:
0: ───@───
      │
1: ───Z───

Highlighted differences:
0: ───@───
      │
1: ───█───

"""

    # Work around an issue when this test is run in python2, where using
    # match=expected_error causes an UnicodeEncodeError.
    with pytest.raises(AssertionError) as ex_info:
        cirq.testing.assert_has_diagram(circuit, u"""
0: ───@───
      │
1: ───Z───
""")
    assert expected_error in ex_info.value.args[0]


def test_assert_has_consistent_apply_unitary():
    class IdentityReturningUnalteredWorkspace:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            return args.available_buffer

        def _unitary_(self):
            return np.eye(2)

    with pytest.raises(AssertionError):
        cirq.testing.assert_has_consistent_apply_unitary(
            IdentityReturningUnalteredWorkspace())

    class DifferentEffect:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            args.available_buffer[0] = args.target_tensor[1]
            args.available_buffer[1] = args.target_tensor[0]
            return args.available_buffer

        def _unitary_(self):
            return np.eye(2, dtype=np.complex128)

    with pytest.raises(AssertionError):
        cirq.testing.assert_has_consistent_apply_unitary(
            DifferentEffect())

    class IgnoreAxisEffect:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            args.available_buffer[0] = args.target_tensor[1]
            args.available_buffer[1] = args.target_tensor[0]
            return args.available_buffer

        def _unitary_(self):
            return np.array([[0, 1], [1, 0]])

    with pytest.raises(AssertionError, match='Not equal'):
        cirq.testing.assert_has_consistent_apply_unitary(
            IgnoreAxisEffect())

    class SameEffect:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            o = args.subspace_index(0)
            i = args.subspace_index(1)
            args.available_buffer[o] = args.target_tensor[i]
            args.available_buffer[i] = args.target_tensor[o]
            return args.available_buffer

        def _unitary_(self):
            return np.array([[0, 1], [1, 0]])

    cirq.testing.assert_has_consistent_apply_unitary(
        SameEffect())

    class BadExponent:
        def __init__(self, power):
            self.power = power

        def __pow__(self, power):
            return BadExponent(self.power * power)

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            i = args.subspace_index(1)
            args.target_tensor[i] *= self.power * 2
            return args.target_tensor

        def _unitary_(self):
            return np.array([[1, 0], [0, 2]])

    cirq.testing.assert_has_consistent_apply_unitary(
        BadExponent(1))

    with pytest.raises(AssertionError):
        cirq.testing.assert_has_consistent_apply_unitary_for_various_exponents(
            BadExponent(1),
            exponents=[1, 2],
            qubit_count=1)

    class EffectWithoutUnitary:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            return args.target_tensor

    with pytest.raises(AssertionError):
        cirq.testing.assert_has_consistent_apply_unitary(
            EffectWithoutUnitary(),
            qubit_count=1)

    class NoEffect:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            return NotImplemented

    cirq.testing.assert_has_consistent_apply_unitary(
        NoEffect(),
        qubit_count=1)

    class UnknownCountEffect:
        pass

    cirq.testing.assert_has_consistent_apply_unitary(
        UnknownCountEffect(),
        qubit_count=1)

    with pytest.raises(NotImplementedError):
        cirq.testing.assert_has_consistent_apply_unitary(
            UnknownCountEffect())

    cirq.testing.assert_has_consistent_apply_unitary(
        cirq.X)

    cirq.testing.assert_has_consistent_apply_unitary(
        cirq.X.on(cirq.NamedQubit('q')))
