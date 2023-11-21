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
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target


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

    fails = [NoUnitaryEffect(), HasApplyReturnsNotImplemented()]
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
            result = cirq.apply_unitary(val, cirq.ApplyUnitaryArgs(make_input(), buf, [axis]))
            np.testing.assert_allclose(result, expected_outputs[axis])

    buf = np.empty(shape=(2, 2), dtype=np.complex128)

    for f in fails:
        with pytest.raises(TypeError, match='failed to satisfy'):
            _ = cirq.apply_unitary(f, cirq.ApplyUnitaryArgs(make_input(), buf, [0]))
        assert (
            cirq.apply_unitary(f, cirq.ApplyUnitaryArgs(make_input(), buf, [0]), default=None)
            is None
        )
        assert (
            cirq.apply_unitary(
                f, cirq.ApplyUnitaryArgs(make_input(), buf, [0]), default=NotImplemented
            )
            is NotImplemented
        )
        assert cirq.apply_unitary(f, cirq.ApplyUnitaryArgs(make_input(), buf, [0]), default=1) == 1

    for s in passes:
        assert_works(s)
        assert (
            cirq.apply_unitary(s, cirq.ApplyUnitaryArgs(make_input(), buf, [0]), default=None)
            is not None
        )


def test_apply_unitary_args_tensor_manipulation():
    # All below are qubit swap operations with 1j global phase

    class ModifyTargetTensor:
        def _apply_unitary_(self, args):
            zo = args.subspace_index(0b01)
            oz = args.subspace_index(0b10)
            args.available_buffer[zo] = args.target_tensor[zo]
            args.target_tensor[zo] = args.target_tensor[oz]
            args.target_tensor[oz] = args.available_buffer[zo]
            args.target_tensor[...] *= 1j
            args.available_buffer[...] = 99  # Destroy buffer data just in case
            return args.target_tensor

    class TransposeTargetTensor:
        def _apply_unitary_(self, args):
            indices = list(range(len(args.target_tensor.shape)))
            indices[args.axes[0]], indices[args.axes[1]] = (
                indices[args.axes[1]],
                indices[args.axes[0]],
            )
            target = args.target_tensor.transpose(*indices)
            target[...] *= 1j
            args.available_buffer[...] = 99  # Destroy buffer data just in case
            return target

    class ReshapeTargetTensor:
        def _apply_unitary_(self, args):
            zz = args.subspace_index(0b00)
            zo = args.subspace_index(0b01)
            oz = args.subspace_index(0b10)
            oo = args.subspace_index(0b11)
            args.available_buffer[zz] = args.target_tensor[zz]
            args.available_buffer[zo] = args.target_tensor[zo]
            args.available_buffer[oz] = args.target_tensor[oz]
            args.available_buffer[oo] = args.target_tensor[oo]
            # Do a pointless reshape and transpose
            target = args.target_tensor.transpose(
                *range(1, len(args.target_tensor.shape)), 0
            ).reshape(args.target_tensor.shape)
            target[zz] = args.available_buffer[zz]
            target[zo] = args.available_buffer[oz]
            target[oz] = args.available_buffer[zo]
            target[oo] = args.available_buffer[oo]
            target[...] *= 1j
            args.available_buffer[...] = 99  # Destroy buffer data just in case
            return target

    class ModifyAvailableBuffer:
        def _apply_unitary_(self, args):
            zz = args.subspace_index(0b00)
            zo = args.subspace_index(0b01)
            oz = args.subspace_index(0b10)
            oo = args.subspace_index(0b11)
            args.available_buffer[zz] = args.target_tensor[zz]
            args.available_buffer[zo] = args.target_tensor[oz]
            args.available_buffer[oz] = args.target_tensor[zo]
            args.available_buffer[oo] = args.target_tensor[oo]
            args.available_buffer[...] *= 1j
            args.target_tensor[...] = 99  # Destroy buffer data just in case
            return args.available_buffer

    class TransposeAvailableBuffer:
        def _apply_unitary_(self, args):
            indices = list(range(len(args.target_tensor.shape)))
            indices[args.axes[0]], indices[args.axes[1]] = (
                indices[args.axes[1]],
                indices[args.axes[0]],
            )
            output = args.available_buffer.transpose(*indices)
            args.available_buffer[...] = args.target_tensor
            output *= 1j
            args.target_tensor[...] = 99  # Destroy buffer data just in case
            return output

    class ReshapeAvailableBuffer:
        def _apply_unitary_(self, args):
            zz = args.subspace_index(0b00)
            zo = args.subspace_index(0b01)
            oz = args.subspace_index(0b10)
            oo = args.subspace_index(0b11)
            # Do a pointless reshape and transpose
            output = args.available_buffer.transpose(
                *range(1, len(args.available_buffer.shape)), 0
            ).reshape(args.available_buffer.shape)
            output[zz] = args.target_tensor[zz]
            output[zo] = args.target_tensor[oz]
            output[oz] = args.target_tensor[zo]
            output[oo] = args.target_tensor[oo]
            output[...] *= 1j
            args.target_tensor[...] = 99  # Destroy buffer data just in case
            return output

    class CreateNewBuffer:
        def _apply_unitary_(self, args):
            u = (
                np.array(
                    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                    dtype=args.target_tensor.dtype,
                )
                * 1j
            )  # yapf: disable
            # Flatten last two axes and add another fake index to the end of
            # target_tensor so np.matmul treats it like an array of two-qubit
            # column vectors.
            new_shape = args.target_tensor.shape[:-2] + (4, 1)
            ret = np.matmul(u, args.target_tensor.reshape(new_shape)).reshape(
                args.target_tensor.shape
            )
            args.target_tensor[...] = 99  # Destroy buffer data just in case
            args.available_buffer[...] = 98
            return ret

    operations = [
        ModifyTargetTensor(),
        TransposeTargetTensor(),
        ReshapeTargetTensor(),
        ModifyAvailableBuffer(),
        TransposeAvailableBuffer(),
        ReshapeAvailableBuffer(),
        CreateNewBuffer(),
    ]

    def assert_is_swap_simple(val: cirq.SupportsConsistentApplyUnitary) -> None:
        qid_shape = (2, 2)
        op_indices = [0, 1]
        state = np.arange(3 * 3, dtype=np.complex64).reshape((1, 3, 3))
        expected = state.copy()
        buf = expected[..., 0, 1].copy()
        expected[..., 0, 1] = expected[..., 1, 0]
        expected[..., 1, 0] = buf
        expected[..., :2, :2] *= 1j

        args = cirq.ApplyUnitaryArgs(state, np.empty_like(state), [1, 2])
        sub_args = args._for_operation_with_qid_shape(
            op_indices, tuple(qid_shape[i] for i in op_indices)
        )
        sub_result = val._apply_unitary_(sub_args)
        result = _incorporate_result_into_target(args, sub_args, sub_result)
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def assert_is_swap(val: cirq.SupportsConsistentApplyUnitary) -> None:
        qid_shape = (1, 2, 4, 2)
        op_indices = [1, 3]
        state = np.arange(2 * (1 * 3 * 4 * 5), dtype=np.complex64).reshape((1, 2, 1, 5, 3, 1, 4))
        expected = state.copy()
        buf = expected[..., 0, 1, :, :].copy()
        expected[..., 0, 1, :, :] = expected[..., 1, 0, :, :]
        expected[..., 1, 0, :, :] = buf
        expected[..., :2, :2, :, :] *= 1j

        args = cirq.ApplyUnitaryArgs(state, np.empty_like(state), [5, 4, 6, 3])
        sub_args = args._for_operation_with_qid_shape(
            op_indices, tuple(qid_shape[i] for i in op_indices)
        )
        sub_result = val._apply_unitary_(sub_args)
        result = _incorporate_result_into_target(args, sub_args, sub_result)
        np.testing.assert_allclose(result, expected, atol=1e-8, verbose=True)

    for op in operations:
        assert_is_swap_simple(op)
        assert_is_swap(op)


def test_big_endian_subspace_index():
    state = np.zeros(shape=(2, 3, 4, 5, 1, 6, 1, 1))
    args = cirq.ApplyUnitaryArgs(state, np.empty_like(state), [1, 3])
    s = slice(None)
    assert args.subspace_index(little_endian_bits_int=1) == (s, 1, s, 0, s, s, s, s)
    assert args.subspace_index(big_endian_bits_int=1) == (s, 0, s, 1, s, s, s, s)


def test_apply_unitaries():
    a, b, c = cirq.LineQubit.range(3)

    result = cirq.apply_unitaries(
        unitary_values=[cirq.H(a), cirq.CNOT(a, b), cirq.H(c).controlled_by(b)], qubits=[a, b, c]
    )
    np.testing.assert_allclose(
        result.reshape(8), [np.sqrt(0.5), 0, 0, 0, 0, 0, 0.5, 0.5], atol=1e-8
    )

    # Different order.
    result = cirq.apply_unitaries(
        unitary_values=[cirq.H(a), cirq.CNOT(a, b), cirq.H(c).controlled_by(b)], qubits=[a, c, b]
    )
    np.testing.assert_allclose(
        result.reshape(8), [np.sqrt(0.5), 0, 0, 0, 0, 0.5, 0, 0.5], atol=1e-8
    )

    # Explicit arguments.
    result = cirq.apply_unitaries(
        unitary_values=[cirq.H(a), cirq.CNOT(a, b), cirq.H(c).controlled_by(b)],
        qubits=[a, b, c],
        args=cirq.ApplyUnitaryArgs.default(num_qubits=3),
    )
    np.testing.assert_allclose(
        result.reshape(8), [np.sqrt(0.5), 0, 0, 0, 0, 0, 0.5, 0.5], atol=1e-8
    )

    # Empty.
    result = cirq.apply_unitaries(unitary_values=[], qubits=[])
    np.testing.assert_allclose(result, [1])
    result = cirq.apply_unitaries(unitary_values=[], qubits=[], default=None)
    np.testing.assert_allclose(result, [1])

    # Non-unitary operation.
    with pytest.raises(TypeError, match='non-unitary'):
        _ = cirq.apply_unitaries(unitary_values=[cirq.depolarize(0.5).on(a)], qubits=[a])
    assert (
        cirq.apply_unitaries(unitary_values=[cirq.depolarize(0.5).on(a)], qubits=[a], default=None)
        is None
    )
    assert (
        cirq.apply_unitaries(unitary_values=[cirq.depolarize(0.5).on(a)], qubits=[a], default=1)
        == 1
    )

    # Inconsistent arguments.
    with pytest.raises(ValueError, match='len'):
        _ = cirq.apply_unitaries(
            unitary_values=[], qubits=[], args=cirq.ApplyUnitaryArgs.default(1)
        )


def test_apply_unitaries_mixed_qid_shapes():
    class PlusOneMod3Gate(cirq.testing.SingleQubitGate):
        def _qid_shape_(self):
            return (3,)

        def _unitary_(self):
            return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # yapf: disable

    class PlusOneMod4Gate(cirq.testing.SingleQubitGate):
        def _qid_shape_(self):
            return (4,)

        def _unitary_(self):
            return np.array(
                [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
            )  # yapf: disable

    a, b = cirq.LineQid.for_qid_shape((3, 4))

    result = cirq.apply_unitaries(
        unitary_values=[
            PlusOneMod3Gate().on(a.with_dimension(3)),
            cirq.X(a.with_dimension(2)),
            cirq.CNOT(a.with_dimension(2), b.with_dimension(2)),
            cirq.CNOT(a.with_dimension(2), b.with_dimension(2)),
            cirq.X(a.with_dimension(2)),
            PlusOneMod3Gate().on(a.with_dimension(3)),
            PlusOneMod3Gate().on(a.with_dimension(3)),
        ],
        qubits=[a, b],
    )
    np.testing.assert_allclose(result.reshape(12), [1] + [0] * 11, atol=1e-8)

    result = cirq.apply_unitaries(
        unitary_values=[
            PlusOneMod3Gate().on(a.with_dimension(3)),
            cirq.X(a.with_dimension(2)),
            cirq.CNOT(a.with_dimension(2), b.with_dimension(2)),
            cirq.CNOT(a.with_dimension(2), b.with_dimension(2)),
            cirq.X(a.with_dimension(2)),
            PlusOneMod3Gate().on(a.with_dimension(3)),
            PlusOneMod3Gate().on(a.with_dimension(3)),
        ],
        qubits=[a, b],
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((3, 4), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((3, 4), dtype=np.complex64),
            axes=(0, 1),
        ),
    )
    np.testing.assert_allclose(result.reshape(12, 12), np.eye(12), atol=1e-8)

    result = cirq.apply_unitaries(
        unitary_values=[
            PlusOneMod3Gate().on(a.with_dimension(3)),
            cirq.X(a.with_dimension(2)),
            PlusOneMod4Gate().on(b.with_dimension(4)),
            PlusOneMod4Gate().on(b.with_dimension(4)),
            cirq.X(b.with_dimension(2)),
            PlusOneMod4Gate().on(b.with_dimension(4)),
            PlusOneMod4Gate().on(b.with_dimension(4)),
            cirq.CNOT(a.with_dimension(2), b.with_dimension(2)),
            PlusOneMod4Gate().on(b.with_dimension(4)),
            cirq.X(b.with_dimension(2)),
            cirq.CNOT(a.with_dimension(2), b.with_dimension(2)),
            cirq.X(a.with_dimension(2)),
            PlusOneMod3Gate().on(a.with_dimension(3)),
            PlusOneMod3Gate().on(a.with_dimension(3)),
        ],
        qubits=[a, b],
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((3, 4), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((3, 4), dtype=np.complex64),
            axes=(0, 1),
        ),
    )
    np.testing.assert_allclose(
        result.reshape(12, 12),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
        atol=1e-8,
    )


# fmt: off
def test_subspace_size_2():
    result = cirq.apply_unitary(
        unitary_value=cirq.X,
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((3,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((3,), dtype=np.complex64),
            axes=(0,),
            subspaces=[(0, 1)],
        ),
    )
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        ),
        atol=1e-8,
    )

    result = cirq.apply_unitary(
        unitary_value=cirq.X,
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((3,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((3,), dtype=np.complex64),
            axes=(0,),
            subspaces=[(0, 2)],
        ),
    )
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ]
        ),
        atol=1e-8,
    )

    result = cirq.apply_unitary(
        unitary_value=cirq.X,
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((3,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((3,), dtype=np.complex64),
            axes=(0,),
            subspaces=[(1, 2)],
        ),
    )
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ]
        ),
        atol=1e-8,
    )

    result = cirq.apply_unitary(
        unitary_value=cirq.X,
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((4,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((4,), dtype=np.complex64),
            axes=(0,),
            subspaces=[(1, 2)],
        ),
    )
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        ),
        atol=1e-8,
    )


def test_subspaces_size_3():
    plus_one_mod_3_gate = cirq.XPowGate(dimension=3)

    result = cirq.apply_unitary(
        unitary_value=plus_one_mod_3_gate,
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((3,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((3,), dtype=np.complex64),
            axes=(0,),
            subspaces=[(0, 1, 2)],
        ),
    )
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        ),
        atol=1e-8,
    )

    result = cirq.apply_unitary(
        unitary_value=plus_one_mod_3_gate,
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((3,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((3,), dtype=np.complex64),
            axes=(0,),
            subspaces=[(2, 1, 0)],
        ),
    )
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ]
        ),
        atol=1e-8,
    )

    result = cirq.apply_unitary(
        unitary_value=plus_one_mod_3_gate,
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((4,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((4,), dtype=np.complex64),
            axes=(0,),
            subspaces=[(1, 2, 3)],
        ),
    )
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]
        ),
        atol=1e-8,
    )


def test_subspaces_size_1():
    phase_gate = cirq.MatrixGate(np.array([[1j]]))

    result = cirq.apply_unitary(
        unitary_value=phase_gate,
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((2,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((2,), dtype=np.complex64),
            axes=(0,),
            subspaces=[(0,)],
        ),
    )
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [1j, 0],
                [0,  1],
            ]
        ),
        atol=1e-8,
    )

    result = cirq.apply_unitary(
        unitary_value=phase_gate,
        args=cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((2,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((2,), dtype=np.complex64),
            axes=(0,),
            subspaces=[(1,)],
        ),
    )
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [1, 0],
                [0, 1j],
            ]
        ),
        atol=1e-8,
    )

    result = cirq.apply_unitary(
        unitary_value=phase_gate,
        args=cirq.ApplyUnitaryArgs(
            target_tensor=np.array([[0, 1], [1, 0]], dtype=np.complex64),
            available_buffer=np.zeros((2, 2), dtype=np.complex64),
            axes=(0,),
            subspaces=[(1,)],
        ),
    )
    np.testing.assert_allclose(
        result,
        np.array(
            [
                [0,  1],
                [1j, 0],
            ]
        ),
        atol=1e-8,
    )
# fmt: on


def test_invalid_subspaces():
    with pytest.raises(ValueError, match='Subspace specified does not exist in axis'):
        _ = cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((2,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((2,), dtype=np.complex64),
            axes=(0,),
            subspaces=[(1, 2)],
        )
    with pytest.raises(ValueError, match='Subspace count does not match axis count'):
        _ = cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((2,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((2,), dtype=np.complex64),
            axes=(0,),
            subspaces=[(0, 1), (0, 1)],
        )
    with pytest.raises(ValueError, match='has zero dimensions'):
        _ = cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((2,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((2,), dtype=np.complex64),
            axes=(0,),
            subspaces=[()],
        )
    with pytest.raises(ValueError, match='does not have consistent step size'):
        _ = cirq.ApplyUnitaryArgs(
            target_tensor=cirq.eye_tensor((3,), dtype=np.complex64),
            available_buffer=cirq.eye_tensor((3,), dtype=np.complex64),
            axes=(0,),
            subspaces=[(0, 2, 1)],
        )


def test_incorporate_result_not_view():
    tensor = np.zeros((2, 2))
    tensor2 = np.zeros((2, 2))
    buffer = np.empty_like(tensor)
    args = cirq.ApplyUnitaryArgs(tensor, buffer, [0])
    not_sub_args = cirq.ApplyUnitaryArgs(tensor2, buffer, [0])
    with pytest.raises(ValueError, match='view'):
        _incorporate_result_into_target(args, not_sub_args, tensor2)


def test_default_method_arguments():
    with pytest.raises(TypeError, match='exactly one of'):
        cirq.ApplyUnitaryArgs.default(1, qid_shape=(2,))


def test_apply_unitary_args_with_axes_transposed_to_start():
    target = np.zeros((2, 3, 4, 5))
    buffer = np.zeros((2, 3, 4, 5))
    args = cirq.ApplyUnitaryArgs(target, buffer, [1, 3])

    new_args = args.with_axes_transposed_to_start()
    assert new_args.target_tensor.shape == (3, 5, 2, 4)
    assert new_args.available_buffer.shape == (3, 5, 2, 4)
    assert new_args.axes == (0, 1)

    # Confirm aliasing.
    new_args.target_tensor[2, 4, 1, 3] = 1
    assert args.target_tensor[1, 2, 3, 4] == 1
    new_args.available_buffer[2, 4, 1, 3] = 2
    assert args.available_buffer[1, 2, 3, 4] == 2


def test_cast_to_complex():
    y0 = cirq.PauliString({cirq.LineQubit(0): cirq.Y})
    state = 0.5 * np.eye(2)
    args = cirq.ApplyUnitaryArgs(
        target_tensor=state, available_buffer=np.zeros_like(state), axes=(0,)
    )

    with pytest.raises(
        np.ComplexWarning, match='Casting complex values to real discards the imaginary part'
    ):
        cirq.apply_unitary(y0, args)


class NotDecomposableGate(cirq.Gate):
    def num_qubits(self):
        return 1


class DecomposableGate(cirq.Gate):
    def __init__(self, sub_gate: cirq.Gate, allocate_ancilla: bool) -> None:
        super().__init__()
        self._sub_gate = sub_gate
        self._allocate_ancilla = allocate_ancilla

    def num_qubits(self):
        return 1

    def _decompose_(self, qubits):
        if self._allocate_ancilla:
            yield cirq.Z(cirq.NamedQubit('DecomposableGateQubit'))
        yield self._sub_gate(qubits[0])


def test_strat_apply_unitary_from_decompose():
    state = np.eye(2, dtype=np.complex128)
    args = cirq.ApplyUnitaryArgs(
        target_tensor=state, available_buffer=np.zeros_like(state), axes=(0,)
    )
    np.testing.assert_allclose(
        cirq.apply_unitaries(
            [DecomposableGate(cirq.X, False)(cirq.LineQubit(0))], [cirq.LineQubit(0)], args
        ),
        [[0, 1], [1, 0]],
    )

    with pytest.raises(TypeError):
        _ = cirq.apply_unitaries(
            [DecomposableGate(NotDecomposableGate(), True)(cirq.LineQubit(0))],
            [cirq.LineQubit(0)],
            args,
        )


def test_unitary_construction():
    with pytest.raises(TypeError):
        _ = cirq.ApplyUnitaryArgs.for_unitary()

    np.testing.assert_allclose(
        cirq.ApplyUnitaryArgs.for_unitary(num_qubits=3).target_tensor,
        cirq.eye_tensor((2,) * 3, dtype=np.complex128),
    )
