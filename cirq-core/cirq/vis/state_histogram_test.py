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

"""Tests for state_histogram."""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

import cirq
from cirq.devices import GridQubit
from cirq.vis import state_histogram


def test_get_state_histogram():
    simulator = cirq.Simulator()

    q0 = GridQubit(0, 0)
    q1 = GridQubit(1, 0)
    circuit = cirq.Circuit()
    circuit.append([cirq.X(q0), cirq.X(q1)])
    circuit.append([cirq.measure(q0, key='q0'), cirq.measure(q1, key='q1')])
    result = simulator.run(program=circuit, repetitions=5)

    values_to_plot = state_histogram.get_state_histogram(result)
    expected_values = [0.0, 0.0, 0.0, 5.0]

    np.testing.assert_equal(values_to_plot, expected_values)


def test_get_state_histogram_multi_1():
    qubits = cirq.LineQubit.range(4)
    c = cirq.Circuit(
        cirq.X.on_each(*qubits[1:]),
        cirq.measure(*qubits),  # One multi-qubit measurement
    )
    r = cirq.sample(c, repetitions=5)
    values_to_plot = state_histogram.get_state_histogram(r)
    expected_values = [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_equal(values_to_plot, expected_values)


def test_get_state_histogram_multi_2():
    qubits = cirq.LineQubit.range(4)
    c = cirq.Circuit(
        cirq.X.on_each(*qubits[1:]),
        cirq.measure(*qubits[:2]),  # One multi-qubit measurement
        cirq.measure_each(*qubits[2:]),  # Multiple single-qubit measurement
    )
    r = cirq.sample(c, repetitions=5)
    values_to_plot = state_histogram.get_state_histogram(r)
    expected_values = [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_equal(values_to_plot, expected_values)


def test_plot_state_histogram_result():
    qubits = cirq.LineQubit.range(4)
    c = cirq.Circuit(
        cirq.X.on_each(*qubits[1:]),
        cirq.measure(*qubits),  # One multi-qubit measurement
    )
    r = cirq.sample(c, repetitions=5)
    expected_values = [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    _, (ax1, ax2) = plt.subplots(1, 2)
    state_histogram.plot_state_histogram(r, ax1)
    state_histogram.plot_state_histogram(expected_values, ax2)
    for r1, r2 in zip(ax1.get_children(), ax2.get_children()):
        if isinstance(r1, mpl.patches.Rectangle) and isinstance(r2, mpl.patches.Rectangle):
            assert str(r1) == str(r2)


def test_plot_state_histogram_collection():
    qubits = cirq.LineQubit.range(4)
    c = cirq.Circuit(
        cirq.X.on_each(*qubits[1:]),
        cirq.measure(*qubits),  # One multi-qubit measurement
    )
    r = cirq.sample(c, repetitions=5)
    _, (ax1, ax2) = plt.subplots(1, 2)
    state_histogram.plot_state_histogram(r.histogram(key='0,1,2,3'), ax1)
    expected_values = [5]
    tick_label = ['7']
    state_histogram.plot_state_histogram(expected_values, ax2, tick_label=tick_label, xlabel=None)
    for r1, r2 in zip(ax1.get_children(), ax2.get_children()):
        if isinstance(r1, mpl.patches.Rectangle) and isinstance(r2, mpl.patches.Rectangle):
            assert str(r1) == str(r2)
