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


def shift_matrix(width: int, shift: int) -> np.ndarray:
    result = np.zeros((width, width))
    for i in range(width):
        result[(i + shift) % width, i] = 1
    return result


def adder_matrix(target_width: int, source_width: int) -> np.ndarray:
    t, s = target_width, source_width
    result = np.zeros((t, s, t, s))
    for k in range(s):
        result[:, k, :, k] = shift_matrix(t, k)
    result.shape = (t * s, t * s)
    return result


def test_the_tests():
    np.testing.assert_allclose(shift_matrix(4, 1),
                               np.array([
                                   [0, 0, 0, 1],
                                   [1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                               ]),
                               atol=1e-8)
    np.testing.assert_allclose(shift_matrix(8, -1),
                               np.array([
                                   [0, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 0, 0, 0, 0, 0, 0],
                               ]),
                               atol=1e-8)
    np.testing.assert_allclose(adder_matrix(4, 2),
                               np.array([
                                   [1, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0],
                               ]),
                               atol=1e-8)


def test_arithmetic_operation_apply_unitary():

    class Add(cirq.ArithmeticOperation):

        def __init__(self, target_register, input_register):
            self.target_register = target_register
            self.input_register = input_register

        def registers(self):
            return self.target_register, self.input_register

        def with_registers(self, *new_registers):
            raise NotImplementedError()

        def apply(self, target_value, input_value):
            return target_value + input_value

    inc2 = Add(cirq.LineQubit.range(2), 1)
    np.testing.assert_allclose(cirq.unitary(inc2),
                               shift_matrix(4, 1),
                               atol=1e-8)

    dec3 = Add(cirq.LineQubit.range(3), -1)
    np.testing.assert_allclose(cirq.unitary(dec3),
                               shift_matrix(8, -1),
                               atol=1e-8)

    add3from2 = Add(cirq.LineQubit.range(3), cirq.LineQubit.range(2))
    np.testing.assert_allclose(cirq.unitary(add3from2),
                               adder_matrix(8, 4),
                               atol=1e-8)

    add2from3 = Add(cirq.LineQubit.range(2), cirq.LineQubit.range(3))
    np.testing.assert_allclose(cirq.unitary(add2from3),
                               adder_matrix(4, 8),
                               atol=1e-8)

    with pytest.raises(ValueError, match='affected by the operation'):
        _ = cirq.unitary(Add(1, cirq.LineQubit.range(2)))

    with pytest.raises(ValueError, match='affected by the operation'):
        _ = cirq.unitary(Add(1, 1))

    np.testing.assert_allclose(cirq.unitary(Add(1, 0)), np.eye(1))

    cirq.testing.assert_has_consistent_apply_unitary(
        Add(cirq.LineQubit.range(2), cirq.LineQubit.range(2)))


def test_arithmetic_operation_qubits():

    class Three(cirq.ArithmeticOperation):

        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

        def registers(self):
            return self.a, self.b, self.c

        def with_registers(self, *new_registers):
            return Three(*new_registers)

        def apply(self, target_value, input_value):
            raise NotImplementedError()

    q0, q1, q2, q3, q4, q5 = cirq.LineQubit.range(6)
    op = Three([q0], [], [q4, q5])
    assert op.qubits == (q0, q4, q5)
    assert op.registers() == ([q0], [], [q4, q5])

    op2 = op.with_qubits(q2, q4, q1)
    assert op2.qubits == (q2, q4, q1)
    assert op2.registers() == ([q2], [], [q4, q1])

    op3 = op.with_registers([q0, q1, q3], [q5], 1)
    assert op3.qubits == (q0, q1, q3, q5)
    assert op3.registers() == ([q0, q1, q3], [q5], 1)

    op4 = op3.with_qubits(q0, q1, q2, q3)
    assert op4.registers() == ([q0, q1, q2], [q3], 1)
    assert op4.qubits == (q0, q1, q2, q3)


def test_reshape_referencing():

    class Op1(cirq.ArithmeticOperation):

        def apply(self, *register_values: int):
            return register_values[0] + 1

        def registers(self):
            return [cirq.LineQubit.range(2)[::-1]]

        def with_registers(self, *new_registers):
            raise NotImplementedError()

    state = np.ones(4, dtype=np.complex64) / 2
    output = cirq.final_state_vector(cirq.Circuit(Op1()), initial_state=state)
    np.testing.assert_allclose(state, output)
