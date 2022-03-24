# Copyright 2021 The Cirq Developers
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


def test_default_parameter():
    qid_shape = (2,)
    tensor = cirq.to_valid_density_matrix(
        0, len(qid_shape), qid_shape=qid_shape, dtype=np.complex64
    )
    args = cirq.ActOnDensityMatrixArgs(
        qubits=cirq.LineQubit.range(1),
        initial_state=0,
    )
    np.testing.assert_almost_equal(args.target_tensor, tensor)
    assert len(args.available_buffer) == 3
    for buffer in args.available_buffer:
        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype
    assert args.qid_shape == qid_shape


def test_shallow_copy_buffers():
    args = cirq.ActOnDensityMatrixArgs(
        qubits=cirq.LineQubit.range(1),
        initial_state=0,
    )
    copy = args.copy(deep_copy_buffers=False)
    assert copy.available_buffer is args.available_buffer


def test_positional_argument():
    qid_shape = (2,)
    tensor = cirq.to_valid_density_matrix(
        0, len(qid_shape), qid_shape=qid_shape, dtype=np.complex64
    )
    with cirq.testing.assert_deprecated(
        'specify all the arguments with keywords', deadline='v0.15'
    ):
        cirq.ActOnDensityMatrixArgs(tensor)


def test_deprecated_warning_and_default_parameter_error():
    tensor = np.ndarray(shape=(2,))
    with cirq.testing.assert_deprecated('Use initial_state instead', deadline='v0.15'):
        with pytest.raises(ValueError, match='dimension of target_tensor is not divisible by 2'):
            cirq.ActOnDensityMatrixArgs(target_tensor=tensor)


def test_decomposed_fallback():
    class Composite(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _decompose_(self, qubits):
            yield cirq.X(*qubits)

    args = cirq.ActOnDensityMatrixArgs(
        qubits=cirq.LineQubit.range(1),
        prng=np.random.RandomState(),
        initial_state=0,
        dtype=np.complex64,
    )

    cirq.act_on(Composite(), args, cirq.LineQubit.range(1))
    np.testing.assert_allclose(
        args.target_tensor, cirq.one_hot(index=(1, 1), shape=(2, 2), dtype=np.complex64)
    )


def test_cannot_act():
    class NoDetails:
        pass

    args = cirq.ActOnDensityMatrixArgs(
        qubits=cirq.LineQubit.range(1),
        prng=np.random.RandomState(),
        initial_state=0,
        dtype=np.complex64,
    )
    with pytest.raises(TypeError, match="Can't simulate operations"):
        cirq.act_on(NoDetails(), args, qubits=())


def test_with_qubits():
    original = cirq.ActOnDensityMatrixArgs(
        qubits=cirq.LineQubit.range(1),
        initial_state=1,
        dtype=np.complex64,
    )
    extened = original.with_qubits(cirq.LineQubit.range(1, 2))
    np.testing.assert_almost_equal(
        extened.target_tensor,
        cirq.density_matrix_kronecker_product(
            np.array([[0, 0], [0, 1]], dtype=np.complex64),
            np.array([[1, 0], [0, 0]], dtype=np.complex64),
        ),
    )


def test_qid_shape_error():
    with pytest.raises(ValueError, match="qid_shape must be provided"):
        cirq.sim.act_on_density_matrix_args._BufferedDensityMatrix.create(initial_state=0)
