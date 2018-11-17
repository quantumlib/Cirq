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
import cirq.google as cg
from cirq.devices import GridQubit
from cirq.study import visualize


def test_plot_state_histogram():
    pl.switch_backend('PDF')
    simulator = cg.XmonSimulator()

    q0 = GridQubit(0, 0)
    q1 = GridQubit(1, 0)
    circuit = cirq.Circuit()
    circuit.append([cirq.X(q0), cirq.X(q1)])
    circuit.append([cirq.MeasurementGate(key='q0')(q0),
                    cirq.MeasurementGate(key='q1')(q1)])
    results = simulator.run_sweep(program=circuit,
                                  repetitions=5)

    values_plotted = visualize.plot_state_histogram(results[0])
    expected_values = [0., 0., 0., 5.]

    np.testing.assert_equal(values_plotted, expected_values)
