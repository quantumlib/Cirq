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

import cirq


def test_one_hot():
    result = cirq.one_hot(shape=4, dtype=np.int32)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, [1, 0, 0, 0])

    np.testing.assert_array_equal(
        cirq.one_hot(shape=[2, 3], dtype=np.complex64), [[1, 0, 0], [0, 0, 0]])

    np.testing.assert_array_equal(
        cirq.one_hot(shape=[2, 3], dtype=np.complex64, index=(0, 2)),
        [[0, 0, 1], [0, 0, 0]])

    np.testing.assert_array_equal(
        cirq.one_hot(shape=5, dtype=np.complex128, index=3), [0, 0, 0, 1, 0])


def test_eye_tensor():
    assert np.all(cirq.eye_tensor((), dtype=int) == np.array(1))
    assert np.all(cirq.eye_tensor((1,), dtype=int) == np.array([[1]]))
    assert np.all(cirq.eye_tensor((2,), dtype=int) == np.array([
        [1, 0],
        [0, 1]]))  # yapf: disable
    assert np.all(cirq.eye_tensor((2, 2), dtype=int) == np.array([
        [[[1, 0], [0, 0]],
         [[0, 1], [0, 0]]],
        [[[0, 0], [1, 0]],
         [[0, 0], [0, 1]]]]))  # yapf: disable
    assert np.all(cirq.eye_tensor((2, 3), dtype=int) == np.array([
        [[[1, 0, 0], [0, 0, 0]],
         [[0, 1, 0], [0, 0, 0]],
         [[0, 0, 1], [0, 0, 0]]],
        [[[0, 0, 0], [1, 0, 0]],
         [[0, 0, 0], [0, 1, 0]],
         [[0, 0, 0], [0, 0, 1]]]]))  # yapf: disable
    assert np.all(cirq.eye_tensor((3, 2), dtype=int) == np.array([
        [[[1, 0], [0, 0], [0, 0]],
         [[0, 1], [0, 0], [0, 0]]],
        [[[0, 0], [1, 0], [0, 0]],
         [[0, 0], [0, 1], [0, 0]]],
        [[[0, 0], [0, 0], [1, 0]],
         [[0, 0], [0, 0], [0, 1]]]]))  # yapf: disable
