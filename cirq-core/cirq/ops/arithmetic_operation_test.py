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

from typing import Sequence, Union

import numpy as np
import pytest

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
    # fmt: off
    np.testing.assert_allclose(
        shift_matrix(4, 1),
        np.array(
            [
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]
        ),
        atol=1e-8,
    )
    # fmt: on
    np.testing.assert_allclose(
        shift_matrix(8, -1),
        np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        adder_matrix(4, 2),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ]
        ),
        atol=1e-8,
    )


def test_arithmetic_gate_apply_unitary():
    class Add(cirq.ArithmeticGate):
        def __init__(
            self,
            target_register: Union[int, Sequence[int]],
            input_register: Union[int, Sequence[int]],
        ):
            self.target_register = target_register
            self.input_register = input_register

        def registers(self):
            return self.target_register, self.input_register

        def with_registers(self, *new_registers):
            raise NotImplementedError()

        def apply(self, target_value, input_value):
            return target_value + input_value

    qubits = [cirq.LineQubit.range(i) for i in range(6)]

    inc2 = Add([2, 2], 1)
    np.testing.assert_allclose(cirq.unitary(inc2), shift_matrix(4, 1), atol=1e-8)
    np.testing.assert_allclose(cirq.unitary(inc2.on(*qubits[2])), shift_matrix(4, 1), atol=1e-8)

    dec3 = Add([2, 2, 2], -1)
    np.testing.assert_allclose(cirq.unitary(dec3), shift_matrix(8, -1), atol=1e-8)
    np.testing.assert_allclose(cirq.unitary(dec3.on(*qubits[3])), shift_matrix(8, -1), atol=1e-8)

    add3from2 = Add([2, 2, 2], [2, 2])
    np.testing.assert_allclose(cirq.unitary(add3from2), adder_matrix(8, 4), atol=1e-8)
    np.testing.assert_allclose(
        cirq.unitary(add3from2.on(*qubits[5])), adder_matrix(8, 4), atol=1e-8
    )

    add2from3 = Add([2, 2], [2, 2, 2])
    np.testing.assert_allclose(cirq.unitary(add2from3), adder_matrix(4, 8), atol=1e-8)
    np.testing.assert_allclose(
        cirq.unitary(add2from3.on(*qubits[5])), adder_matrix(4, 8), atol=1e-8
    )

    with pytest.raises(ValueError, match='affected by the gate'):
        _ = cirq.unitary(Add(1, [2, 2]))

    with pytest.raises(ValueError, match='affected by the gate'):
        _ = cirq.unitary(Add(1, [2, 2]).on(*qubits[2]))

    with pytest.raises(ValueError, match='affected by the gate'):
        _ = cirq.unitary(Add(1, 1))

    with pytest.raises(ValueError, match='affected by the gate'):
        _ = cirq.unitary(Add(1, 1).on())

    np.testing.assert_allclose(cirq.unitary(Add(1, 0)), np.eye(1))
    np.testing.assert_allclose(cirq.unitary(Add(1, 0).on()), np.eye(1))

    cirq.testing.assert_has_consistent_apply_unitary(Add([2, 2], [2, 2]))
    cirq.testing.assert_has_consistent_apply_unitary(Add([2, 2], [2, 2]).on(*qubits[4]))

    with pytest.raises(ValueError, match='Wrong number of qubits'):
        _ = Add(1, [2, 2]).on(*qubits[3])

    with pytest.raises(ValueError, match='Wrong shape of qids'):
        _ = Add(1, [2, 3]).on(*qubits[2])
