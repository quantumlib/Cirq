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

"""Tests for visualize."""

import numpy as np
from matplotlib import pyplot as pl

import cirq
from cirq.devices import GridQubit
from cirq.study import visualize


def test_plot_state_histogram():
    pl.switch_backend('PDF')
    simulator = cirq.Simulator()

    q0 = GridQubit(0, 0)
    q1 = GridQubit(1, 0)
    circuit = cirq.Circuit()
    circuit.append([cirq.X(q0), cirq.X(q1)])
    circuit.append([cirq.measure(q0, key='q0'), cirq.measure(q1, key='q1')])
    result = simulator.run(program=circuit,
                           repetitions=5)

    values_plotted = visualize.plot_state_histogram(result)
    expected_values = [0., 0., 0., 5.]

    np.testing.assert_equal(values_plotted, expected_values)


def test_plot_state_histogram_multi_1():
    pl.switch_backend('PDF')
    qubits = cirq.LineQubit.range(4)
    c = cirq.Circuit(
        cirq.X.on_each(*qubits[1:]),
        cirq.measure(*qubits),  # One multi-qubit measurement
    )
    r = cirq.sample(c, repetitions=5)
    values_plotted = visualize.plot_state_histogram(r)
    expected_values = [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_equal(values_plotted, expected_values)


def test_plot_state_histogram_multi_2():
    pl.switch_backend('PDF')
    qubits = cirq.LineQubit.range(4)
    c = cirq.Circuit(
        cirq.X.on_each(*qubits[1:]),
        cirq.measure(*qubits[:2]),  # One multi-qubit measurement
        cirq.measure_each(*qubits[2:]),  # Multiple single-qubit measurement
    )
    r = cirq.sample(c, repetitions=5)
    values_plotted = visualize.plot_state_histogram(r)
    expected_values = [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_equal(values_plotted, expected_values)
