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


def test_decomposed_fallback():

    class Composite(cirq.Gate):

        def num_qubits(self) -> int:
            return 1

        def _decompose_(self, qubits):
            yield cirq.X(*qubits)

    args = cirq.ActOnStateVectorArgs(
        target_tensor=cirq.one_hot(shape=(2, 2, 2), dtype=np.complex64),
        available_buffer=np.empty((2, 2, 2), dtype=np.complex64),
        axes=[1],
        prng=np.random.RandomState(),
        log_of_measurement_results={})

    cirq.act_on(Composite(), args)
    np.testing.assert_allclose(
        args.target_tensor,
        cirq.one_hot(index=(0, 1, 0), shape=(2, 2, 2), dtype=np.complex64))


def test_cannot_act():

    class NoDetails:
        pass

    args = cirq.ActOnStateVectorArgs(
        target_tensor=cirq.one_hot(shape=(2, 2, 2), dtype=np.complex64),
        available_buffer=np.empty((2, 2, 2), dtype=np.complex64),
        axes=[1],
        prng=np.random.RandomState(),
        log_of_measurement_results={})

    with pytest.raises(TypeError, match="Failed to act"):
        cirq.act_on(NoDetails(), args)
