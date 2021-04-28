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
from typing import Any, cast, Iterable, Optional, Tuple

import numpy as np
import pytest

import cirq


def make_buffers(shape, dtype):
    return (
        np.empty(shape, dtype=dtype),
        np.empty(shape, dtype=dtype),
        np.empty(shape, dtype=dtype),
    )


def assert_apply_mixture_returns(
    val: Any,
    rho: np.ndarray,
    left_axes: Iterable[int],
    right_axes: Optional[Iterable[int]],
    assert_result_is_out_buf: bool = False,
    expected_result: np.ndarray = None,
):
    out_buf, buf0, buf1 = make_buffers(rho.shape, rho.dtype)
    result = cirq.apply_mixture(
        val,
        args=cirq.ApplyMixtureArgs(
            target_tensor=rho,
            left_axes=left_axes,
            right_axes=right_axes,
            out_buffer=out_buf,
            auxiliary_buffer0=buf0,
            auxiliary_buffer1=buf1,
        ),
    )
    if assert_result_is_out_buf:
        assert result is out_buf
    else:
        assert result is not out_buf

    np.testing.assert_array_almost_equal(result, expected_result)


def test_apply_mixture_bad_args():
    target = np.zeros((3,) + (1, 2, 3) + (3, 1, 2) + (3,))
    with pytest.raises(ValueError, match='Invalid target_tensor shape'):
        cirq.apply_mixture(
            cirq.IdentityGate(3, (1, 2, 3)),
            cirq.ApplyMixtureArgs(
                target,
                np.zeros_like(target),
                np.zeros_like(target),
                np.zeros_like(target),
                (1, 2, 3),
                (4, 5, 6),
            ),
            default=np.array([]),
        )
    target = np.zeros((2, 3, 2, 3))
    with pytest.raises(ValueError, match='Invalid mixture qid shape'):
        cirq.apply_mixture(
            cirq.IdentityGate(2, (2, 9)),
            cirq.ApplyMixtureArgs(
                target,
                np.zeros_like(target),
                np.zeros_like(target),
                np.zeros_like(target),
                (0, 1),
                (2, 3),
            ),
            default=np.array([]),
        )


def test_apply_mixture_simple():
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    class HasApplyMixture:
        def _apply_mixture_(self, args: cirq.ApplyMixtureArgs):
            zero_left = cirq.slice_for_qubits_equal_to(args.left_axes, 0)
            one_left = cirq.slice_for_qubits_equal_to(args.left_axes, 1)
            zero_right = cirq.slice_for_qubits_equal_to(cast(Tuple[int], args.right_axes), 0)
            one_right = cirq.slice_for_qubits_equal_to(cast(Tuple[int], args.right_axes), 1)
            args.out_buffer[:] = 0
            np.copyto(dst=args.auxiliary_buffer0, src=args.target_tensor)
            for kraus_op in [np.sqrt(0.5) * np.eye(2, dtype=np.complex128), np.sqrt(0.5) * x]:
                np.copyto(dst=args.target_tensor, src=args.auxiliary_buffer0)
                cirq.apply_matrix_to_slices(
                    args.target_tensor, kraus_op, [zero_left, one_left], out=args.auxiliary_buffer1
                )

                cirq.apply_matrix_to_slices(
                    args.auxiliary_buffer1,
                    np.conjugate(kraus_op),
                    [zero_right, one_right],
                    out=args.target_tensor,
                )
                args.out_buffer += args.target_tensor
            return args.out_buffer

    rho = np.copy(x)
    assert_apply_mixture_returns(
        HasApplyMixture(), rho, [0], [1], assert_result_is_out_buf=True, expected_result=x
    )


def test_apply_mixture_inline():
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    class HasApplyMixture:
        def _apply_mixture_(self, args: cirq.ApplyMixtureArgs):
            args.target_tensor = 0.5 * args.target_tensor + 0.5 * np.dot(
                np.dot(x, args.target_tensor), x
            )
            return args.target_tensor

    rho = np.copy(x)
    assert_apply_mixture_returns(HasApplyMixture(), rho, [0], [1], expected_result=x)


def test_apply_mixture_returns_aux_buffer():
    rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)

    class ReturnsAuxBuffer0:
        def _apply_mixture_(self, args: cirq.ApplyMixtureArgs):
            return args.auxiliary_buffer0

    with pytest.raises(AssertionError, match='ReturnsAuxBuffer0'):
        assert_apply_mixture_returns(ReturnsAuxBuffer0(), rho, [0], [1])

    class ReturnsAuxBuffer1:
        def _apply_mixture_(self, args: cirq.ApplyMixtureArgs):
            return args.auxiliary_buffer1

    with pytest.raises(AssertionError, match='ReturnsAuxBuffer1'):
        assert_apply_mixture_returns(ReturnsAuxBuffer1(), rho, [0], [1])


def test_apply_mixture_simple_state_vector():
    for _ in range(25):
        state = cirq.testing.random_superposition(2)
        u1 = cirq.testing.random_unitary(2)
        u2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        p1 = np.random.random()
        p2 = 1 - p1

        expected = p1 * np.dot(u1, state) + p2 * np.dot(u2, state)

        class HasMixture:
            def _mixture_(self):
                return ((p1, u1), (p2, cirq.X))

        assert_apply_mixture_returns(
            HasMixture(), state, [0], None, assert_result_is_out_buf=True, expected_result=expected
        )


def test_apply_mixture_simple_split_fallback():
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    class HasMixture:
        def _mixture_(self):
            return ((0.5, np.eye(2, dtype=np.complex128)), (0.5, cirq.X))

    rho = np.copy(x)
    assert_apply_mixture_returns(
        HasMixture(), rho, [0], [1], assert_result_is_out_buf=True, expected_result=x
    )


def test_apply_mixture_fallback_one_qubit_random_on_qubit():
    for _ in range(25):
        state = cirq.testing.random_superposition(2)
        rho = np.outer(np.conjugate(state), state)

        u = cirq.testing.random_unitary(2)

        expected = 0.5 * rho + 0.5 * np.dot(np.dot(u, rho), np.conjugate(np.transpose(u)))

        class HasMixture:
            def _mixture_(self):
                return ((0.5, np.eye(2, dtype=np.complex128)), (0.5, u))

        assert_apply_mixture_returns(
            HasMixture(), rho, [0], [1], assert_result_is_out_buf=True, expected_result=expected
        )


def test_apply_mixture_fallback_two_qubit_random():
    for _ in range(25):
        state = cirq.testing.random_superposition(4)
        rho = np.outer(np.conjugate(state), state)

        u = cirq.testing.random_unitary(4)

        expected = 0.5 * rho + 0.5 * np.dot(np.dot(u, rho), np.conjugate(np.transpose(u)))

        rho.shape = (2, 2, 2, 2)
        expected.shape = (2, 2, 2, 2)

        class HasMixture:
            def _mixture_(self):
                return ((0.5, np.eye(4, dtype=np.complex128)), (0.5, u))

        assert_apply_mixture_returns(
            HasMixture(),
            rho,
            [0, 1],
            [2, 3],
            assert_result_is_out_buf=True,
            expected_result=expected,
        )


def test_apply_mixture_no_protocols_implemented():
    class NoProtocols:
        pass

    rho = np.ones((2, 2, 2, 2), dtype=np.complex128)

    with pytest.raises(TypeError, match='has no'):
        assert_apply_mixture_returns(NoProtocols(), rho, left_axes=[1], right_axes=[1])


def test_apply_mixture_no_protocols_implemented_default():
    class NoProtocols:
        pass

    args = cirq.ApplyMixtureArgs(
        target_tensor=np.eye(2),
        left_axes=[0],
        right_axes=[1],
        out_buffer=None,
        auxiliary_buffer0=None,
        auxiliary_buffer1=None,
    )
    result = cirq.apply_mixture(NoProtocols(), args, default='cirq')
    assert result == 'cirq'


def test_apply_mixture_unitary():
    m = np.diag([1, 1j])

    shape = (2, 2, 2, 2)
    rho = np.ones(shape, dtype=np.complex128)

    class HasUnitary:
        def _unitary_(self) -> np.ndarray:
            return m

    class HasUnitaryButReturnsNotImplemented(HasUnitary):
        def _apply_unitary_(self, args: cirq.ApplyMixtureArgs):
            return NotImplemented

    for val in (HasUnitary(), HasUnitaryButReturnsNotImplemented()):
        assert_apply_mixture_returns(
            val,
            rho,
            left_axes=[1],
            right_axes=[3],
            assert_result_is_out_buf=False,
            expected_result=np.reshape(np.outer([1, 1j, 1, 1j], [1, -1j, 1, -1j]), shape),
        )


def test_apply_mixture_apply_unitary():
    shape = (2, 2, 2, 2)
    rho = np.ones(shape, dtype=np.complex128)

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
        assert_apply_mixture_returns(
            val,
            rho,
            left_axes=[1],
            right_axes=[3],
            assert_result_is_out_buf=False,
            expected_result=np.reshape(np.outer([1, 1j, 1, 1j], [1, -1j, 1, -1j]), shape),
        )


def test_apply_mixture_apply_unitary_not_implemented():
    class ApplyUnitaryNotImplemented:
        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            return NotImplemented

    rho = np.ones((2, 2, 2, 2), dtype=np.complex128)
    out_buf, aux_buf0, aux_buf1 = make_buffers((2, 2, 2, 2), dtype=rho.dtype)

    with pytest.raises(TypeError, match='has no'):
        cirq.apply_mixture(
            ApplyUnitaryNotImplemented(),
            args=cirq.ApplyMixtureArgs(
                target_tensor=rho,
                left_axes=[1],
                right_axes=[3],
                out_buffer=out_buf,
                auxiliary_buffer0=aux_buf0,
                auxiliary_buffer1=aux_buf1,
            ),
        )
