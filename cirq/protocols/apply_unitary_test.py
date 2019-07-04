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

import numpy as np
import pytest

import cirq


def test_apply_unitary_presence_absence():
    m = np.diag([1, -1])

    class NoUnitaryEffect:
        pass

    class HasUnitary:
        def _unitary_(self) -> np.ndarray:
            return m

    class HasApplyReturnsNotImplemented:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            return NotImplemented

    class HasApplyReturnsNotImplementedButHasUnitary:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            return NotImplemented

        def _unitary_(self) -> np.ndarray:
            return m

    class HasApplyOutputInBuffer:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            zero = args.subspace_index(0)
            one = args.subspace_index(1)
            args.available_buffer[zero] = args.target_tensor[zero]
            args.available_buffer[one] = -args.target_tensor[one]
            return args.available_buffer

    class HasApplyMutateInline:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            one = args.subspace_index(1)
            args.target_tensor[one] *= -1
            return args.target_tensor

    fails = [
        NoUnitaryEffect(),
        HasApplyReturnsNotImplemented(),
    ]
    passes = [
        HasUnitary(),
        HasApplyReturnsNotImplementedButHasUnitary(),
        HasApplyOutputInBuffer(),
        HasApplyMutateInline(),
    ]

    def make_input():
        return np.ones((2, 2))

    def assert_works(val):
        expected_outputs = [
            np.array([1, 1, -1, -1]).reshape((2, 2)),
            np.array([1, -1, 1, -1]).reshape((2, 2)),
        ]
        for axis in range(2):
            result = cirq.apply_unitary(
                val, cirq.ApplyUnitaryArgs(make_input(), buf, [axis]))
            np.testing.assert_allclose(result, expected_outputs[axis])

    buf = np.empty(shape=(2, 2), dtype=np.complex128)

    for f in fails:
        with pytest.raises(TypeError, match='failed to satisfy'):
            _ = cirq.apply_unitary(
                f,
                cirq.ApplyUnitaryArgs(make_input(), buf, [0]))
        assert cirq.apply_unitary(
            f,
            cirq.ApplyUnitaryArgs(make_input(), buf, [0]),
            default=None) is None
        assert cirq.apply_unitary(
            f,
            cirq.ApplyUnitaryArgs(make_input(), buf, [0]),
            default=NotImplemented) is NotImplemented
        assert cirq.apply_unitary(
            f,
            cirq.ApplyUnitaryArgs(make_input(), buf, [0]),
            default=1) == 1

    for s in passes:
        assert_works(s)
        assert cirq.apply_unitary(
            s,
            cirq.ApplyUnitaryArgs(make_input(), buf, [0]),
            default=None) is not None


def test_apply_unitaries():
    a, b, c = cirq.LineQubit.range(3)

    result = cirq.apply_unitaries(
        unitary_values=[cirq.H(a),
                        cirq.CNOT(a, b),
                        cirq.H(c).controlled_by(b)],
        qubits=[a, b, c])
    np.testing.assert_allclose(result.reshape(8), [
        np.sqrt(0.5),
        0,
        0,
        0,
        0,
        0,
        0.5,
        0.5,
    ],
                               atol=1e-8)

    # Different order.
    result = cirq.apply_unitaries(
        unitary_values=[cirq.H(a),
                        cirq.CNOT(a, b),
                        cirq.H(c).controlled_by(b)],
        qubits=[a, c, b])
    np.testing.assert_allclose(result.reshape(8), [
        np.sqrt(0.5),
        0,
        0,
        0,
        0,
        0.5,
        0,
        0.5,
    ],
                               atol=1e-8)

    # Explicit arguments.
    result = cirq.apply_unitaries(
        unitary_values=[cirq.H(a),
                        cirq.CNOT(a, b),
                        cirq.H(c).controlled_by(b)],
        qubits=[a, b, c],
        args=cirq.ApplyUnitaryArgs.default(num_qubits=3))
    np.testing.assert_allclose(result.reshape(8), [
        np.sqrt(0.5),
        0,
        0,
        0,
        0,
        0,
        0.5,
        0.5,
    ],
                               atol=1e-8)

    # Empty.
    result = cirq.apply_unitaries(unitary_values=[], qubits=[])
    np.testing.assert_allclose(result, [1])
    result = cirq.apply_unitaries(unitary_values=[], qubits=[], default=None)
    np.testing.assert_allclose(result, [1])

    # Non-unitary operation.
    with pytest.raises(TypeError, match='non-unitary'):
        _ = cirq.apply_unitaries(unitary_values=[cirq.depolarize(0.5).on(a)],
                                 qubits=[a])
    assert cirq.apply_unitaries(unitary_values=[cirq.depolarize(0.5).on(a)],
                                qubits=[a],
                                default=None) is None
    assert cirq.apply_unitaries(unitary_values=[cirq.depolarize(0.5).on(a)],
                                qubits=[a],
                                default=1) == 1

    # Inconsistent arguments.
    with pytest.raises(ValueError, match='len'):
        _ = cirq.apply_unitaries(unitary_values=[],
                                 qubits=[],
                                 args=cirq.ApplyUnitaryArgs.default(1))


def test_apply_unitaries_mixed_qid_shapes():
    class PlusOneMod3Gate(cirq.SingleQubitGate):
        def _qid_shape_(self):
            return (3,)
        def _unitary_(self):
            return np.array([[0, 0, 1],
                             [1, 0, 0],
                             [0, 1, 0]])

    class PlusOneMod4Gate(cirq.SingleQubitGate):
        def _qid_shape_(self):
            return (4,)
        def _unitary_(self):
            return np.array([[0, 0, 0, 1],
                             [1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0]])

    a, b = cirq.LineQubit.range(2)

    result = cirq.apply_unitaries(unitary_values=[
            PlusOneMod3Gate().on(a),
            cirq.X(a),
            cirq.CNOT(a, b),
            cirq.CNOT(a, b),
            cirq.X(a),
            PlusOneMod3Gate().on(a),
            PlusOneMod3Gate().on(a),
        ],
        qubits=[a, b])
    np.testing.assert_allclose(result.reshape(6), [1]+[0]*5, atol=1e-8)

    result = cirq.apply_unitaries(unitary_values=[
            PlusOneMod3Gate().on(a),
            cirq.X(a),
            cirq.CNOT(a, b),
            cirq.CNOT(a, b),
            cirq.X(a),
            PlusOneMod3Gate().on(a),
            PlusOneMod3Gate().on(a),
        ],
        qubits=[a, b],
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.identity_unitary(qid_shape=(3, 2)),
            available_buffer=cirq.identity_unitary(qid_shape=(3, 2)),
            axes=(0, 1),
            qid_shape=(3, 2)))
    np.testing.assert_allclose(result.reshape(6, 6), np.eye(6), atol=1e-8)

    result = cirq.apply_unitaries(unitary_values=[
            PlusOneMod3Gate().on(a),
            cirq.X(a),
            PlusOneMod4Gate().on(b),
            PlusOneMod4Gate().on(b),
            cirq.X(b),
            PlusOneMod4Gate().on(b),
            PlusOneMod4Gate().on(b),
            cirq.CNOT(a, b),
            PlusOneMod4Gate().on(b),
            cirq.X(b),
            cirq.CNOT(a, b),
            cirq.X(a),
            PlusOneMod3Gate().on(a),
            PlusOneMod3Gate().on(a),
        ],
        qubits=[a, b],
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.identity_unitary(qid_shape=(3, 4)),
            available_buffer=cirq.identity_unitary(qid_shape=(3, 4)),
            axes=(0, 1),
            qid_shape=(3, 4)))
    np.testing.assert_allclose(result.reshape(12, 12),
        np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]),
        atol=1e-8)


def test_default_method_arguments():
    with pytest.raises(TypeError, match='argument'):
        cirq.ApplyUnitaryArgs.default(1, qid_shape=(2,))
