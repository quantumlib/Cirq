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


def make_buffers(shape, dtype):
    return (
        np.empty(shape, dtype=dtype),
        np.empty(shape, dtype=dtype),
        np.empty(shape, dtype=dtype))


def apply_channel(val, input, left_axes, right_axes,
        assert_result_is_out_buf=False):
    out_buf, buf0, buf1 = make_buffers(input.shape, input.dtype)
    result = cirq.apply_channel(
            val,
            args=cirq.ApplyChannelArgs(target_tensor=input,
                                       left_axes=left_axes,
                                       right_axes=right_axes,
                                       out_buffer=out_buf,
                                       auxiliary_buffer0=buf0,
                                       auxiliary_buffer1=buf1))
    if assert_result_is_out_buf:
        assert result is out_buf
    return result


def test_apply_channel_simple():
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    class HasChannel():

        def _channel_(self):
            return (
            np.sqrt(0.5) * np.eye(2, dtype=np.complex128), np.sqrt(0.5) * x)

    input = np.copy(x)
    result = apply_channel(HasChannel(), input, [0], [1],
                           assert_result_is_out_buf=True)
    np.testing.assert_almost_equal(result, x)


def test_apply_channel_one_qubit_random():
    rho = cirq.testing.ran

    class HasChannel():

        def _channel_(self):
            return (
                np.sqrt(0.5) * np.eye(2, dtype=np.complex128), np.sqrt(0.5) * x)

    input = np.copy(x)
    result = apply_channel(HasChannel(), input, [0], [1],
                           assert_result_is_out_buf=True)
    np.testing.assert_almost_equal(result, x)


def test_apply_channel_no_protocols_implemented():
    class NoProtocols:
        pass

    input = np.ones((2, 2, 2, 2), dtype=np.complex128)

    with pytest.raises(TypeError):
        apply_channel(NoProtocols(), input, left_axes=[1], right_axes=[1])


def test_apply_channel_unitary():
    m = np.diag([1, 1j])

    shape = (2, 2, 2, 2)
    input = np.ones(shape, dtype=np.complex128)

    class HasUnitary:
        def _unitary_(self) -> np.ndarray:
            return m

    class HasUnitaryButReturnsNotImplemented(HasUnitary):
        def _apply_unitary_(self, args: cirq.ApplyChannelArgs):
            return NotImplemented

    for val in (HasUnitary(), HasUnitaryButReturnsNotImplemented()):
        result = apply_channel(val, input, left_axes=[1], right_axes=[3])
        np.testing.assert_almost_equal(
                result,
                np.reshape(np.outer([1, 1j, 1, 1j], [1, -1j, 1, -1j]), shape))


def test_apply_channel_apply_unitary():
    shape = (2, 2, 2, 2)
    input = np.ones(shape, dtype=np.complex128)

    class HasApplyUnitaryOutputInBuffer:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            zero = args.subspace_index(0)
            one = args.subspace_index(1)
            args.available_buffer[zero] = args.target_tensor[zero]
            args.available_buffer[one] = 1j * args.target_tensor[one]
            return args.available_buffer

    class HasApplyUnitaryMutateInline:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            one = args.subspace_index(1)
            args.target_tensor[one] *= 1j
            return args.target_tensor


    for val in (HasApplyUnitaryOutputInBuffer(), HasApplyUnitaryMutateInline()):
        assert_result_is_out_buf = isinstance(val, HasApplyUnitaryOutputInBuffer)
        result = apply_channel(val, input, left_axes=[1], right_axes=[3],
                               assert_result_is_out_buf=assert_result_is_out_buf)
        np.testing.assert_almost_equal(
                result,
                np.reshape(np.outer([1, 1j, 1, 1j], [1, -1j, 1, -1j]), shape))


def test_apply_channel_apply_unitary_not_implemented():
    class ApplyUnitaryNotImplemeneted:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            return NotImplemented

    input = np.ones((2, 2, 2, 2), dtype=np.complex128)
    out_buf, aux_buf0, aux_buf1 = make_buffers((2, 2, 2, 2), dtype=input.dtype)

    with pytest.raises(TypeError):
         cirq.apply_channel(
            ApplyUnitaryNotImplemeneted(),
            args=cirq.ApplyChannelArgs(target_tensor=input,
                                       left_axes=[1],
                                       right_axes=[3],
                                       out_buffer=out_buf,
                                       auxiliary_buffer0=aux_buf0,
                                       auxiliary_buffer1=aux_buf1))






