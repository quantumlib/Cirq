# Copyright 2019 The Cirq Developers
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


def make_buffers(shape):
    return (
    np.empty(shape, dtype=np.complex128),
    np.empty(shape, dtype=np.complex128),
    np.empty(shape, dtype=np.complex128))


def test_apply_channel_unitary():
    m = np.diag([1, 1j])
    class HasUnitary:
        def _unitary_(self) -> np.ndarray:
            return m

    shape = (2, 2, 2, 2)
    input = np.ones(shape, dtype=np.complex128)
    buf, ini_buf, sum_buf = make_buffers(shape)

    result = cirq.apply_channel(
        HasUnitary(),
        args=cirq.ApplyChannelArgs(target_tensor=input,
                                   left_axes=[0],
                                   right_axes=[2],
                                   available_buffer=buf,
                                   available_initial_buffer=ini_buf,
                                   available_sum_buffer=sum_buf))
    assert result is buf
    assert result is not input
    assert result is not ini_buf
    assert result is not sum_buf
    np.testing.assert_almost_equal(
        result,
        np.reshape(np.outer([1, 1, 1j, 1j], [1, 1, -1j, -1j]), shape))

    result = cirq.apply_channel(
            HasUnitary(),
            args=cirq.ApplyChannelArgs(target_tensor=input,
                                       left_axes=[1],
                                       right_axes=[3],
                                       available_buffer=buf,
                                       available_initial_buffer=ini_buf,
                                       available_sum_buffer=sum_buf))
    assert result is buf
    np.testing.assert_almost_equal(
            result,
            np.reshape(np.outer([1, 1j, 1, 1j], [1, -1j, 1, -1j]), shape))


def test_apply_channel_apply_unitary():
    class HasApplyOutputInBuffer:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            zero = args.subspace_index(0)
            one = args.subspace_index(1)
            args.available_buffer[zero] = args.target_tensor[zero]
            args.available_buffer[one] = 1j * args.target_tensor[one]
            return args.available_buffer




# def test_apply_channel_presence_absence():
#     pass
#
#     class NoUnitaryEffect:
#         pass
#
#
#     class HasApplyReturnsNotImplemented:
#         def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
#             return NotImplemented
#
#     class HasApplyReturnsNotImplementedButHasUnitary:
#         def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
#             return NotImplemented
#
#         def _unitary_(self) -> np.ndarray:
#             return m
#
#     class HasApplyOutputInBuffer:
#         def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
#             zero = args.subspace_index(0)
#             one = args.subspace_index(1)
#             args.available_buffer[zero] = args.target_tensor[zero]
#             args.available_buffer[one] = -args.target_tensor[one]
#             return args.available_buffer
#
#     class HasApplyMutateInline:
#         def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
#             one = args.subspace_index(1)
#             args.target_tensor[one] *= -1
#             return args.target_tensor
#
#     fails = [
#         NoUnitaryEffect(),
#         HasApplyReturnsNotImplemented(),
#     ]
#     passes = [
#         HasUnitary(),
#         HasApplyReturnsNotImplementedButHasUnitary(),
#         HasApplyOutputInBuffer(),
#         HasApplyMutateInline(),
#     ]
#
#     def make_input():
#         return np.ones(, dtype=np.complex128).reshape((2, 2))
#
#     def assert_works(val):
#         expected_outputs = [
#             np.array([1, 1, -1, -1]).reshape((2, 2)),
#             np.array([1, -1, 1, -1]).reshape((2, 2)),
#         ]
#         for axis in range(2):
#             result = cirq.apply_unitary(
#                     val, cirq.ApplyUnitaryArgs(make_input(), buf, [axis]))
#             np.testing.assert_allclose(result, expected_outputs[axis])
#
#     buf = np.empty(shape=(2, 2), dtype=np.complex128)
#
#     for f in fails:
#         with pytest.raises(TypeError, match='no _apply_unitary_'):
#             _ = cirq.apply_unitary(
#                     f,
#                     cirq.ApplyUnitaryArgs(make_input(), buf, [0]))
#         assert cirq.apply_unitary(
#                 f,
#                 cirq.ApplyUnitaryArgs(make_input(), buf, [0]),
#                 default=None) is None
#         assert cirq.apply_unitary(
#                 f,
#                 cirq.ApplyUnitaryArgs(make_input(), buf, [0]),
#                 default=NotImplemented) is NotImplemented
#         assert cirq.apply_unitary(
#                 f,
#                 cirq.ApplyUnitaryArgs(make_input(), buf, [0]),
#                 default=1) == 1
#
#     for s in passes:
#         assert_works(s)
#         assert cirq.apply_unitary(
#                 s,
#                 cirq.ApplyUnitaryArgs(make_input(), buf, [0]),
#                 default=None) is not None
