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
from unittest import mock
import itertools
import random
import numpy as np
import sympy

import cirq


class PlusGate(cirq.Gate):
    """A qudit gate that increments a qudit state mod its dimension."""

    def __init__(self, dimension, increment=1):
        self.dimension = dimension
        self.increment = increment % dimension

    def _qid_shape_(self):
        return (self.dimension,)

    def _unitary_(self):
        inc = (self.increment - 1) % self.dimension + 1
        u = np.empty((self.dimension, self.dimension))
        u[inc:] = np.eye(self.dimension)[:-inc]
        u[:inc] = np.eye(self.dimension)[-inc:]
        return u


class _TestMixture(cirq.Gate):
    def __init__(self, gate_options):
        self.gate_options = gate_options

    def _qid_shape_(self):
        return cirq.qid_shape(self.gate_options[0], ())

    def _mixture_(self):
        return [(1 / len(self.gate_options), cirq.unitary(g)) for g in self.gate_options]


class _TestDecomposingChannel(cirq.Gate):
    def __init__(self, channels):
        self.channels = channels

    def _qid_shape_(self):
        return tuple(d for chan in self.channels for d in cirq.qid_shape(chan))

    def _decompose_(self, qubits):
        return [chan.on(q) for chan, q in zip(self.channels, qubits)]




dtype = np.complex64
q0, q1 = cirq.LineQubit.range(2)
simulator = cirq.DensityMatrixSimulator(dtype=dtype)
with mock.patch.object(simulator, '_base_iterator', wraps=simulator._base_iterator) as mock_sim:
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit(
                (cirq.X ** b0)(q0),
                (cirq.X ** b1)(q1),
                cirq.measure(q0),
                cirq.measure(q1),
                cirq.H(q0),
                cirq.H(q1),
            )
            result = simulator.run(circuit, repetitions=0)
            np.testing.assert_equal(
                result.measurements, {'0': np.empty([0, 1]), '1': np.empty([0, 1])}
            )
            assert result.repetitions == 0
    assert mock_sim.call_count == 0