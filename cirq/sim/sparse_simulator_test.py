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

import cirq

def test_run_no_measurements():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator()

    circuit = cirq.Circuit.from_ops(cirq.X(q0), cirq.X(q1))
    result = simulator.run(circuit)
    assert len(result.measurements) == 0


def test_run_no_results():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator()

    circuit = cirq.Circuit.from_ops(cirq.X(q0), cirq.X(q1))
    result = simulator.run(circuit)
    assert len(result.measurements) == 0


def test_run_empty_circuit():
    simulator = cirq.Simulator()
    result = simulator.run(cirq.Circuit())
    assert len(result.measurements) == 0


def test_run_bit_flips():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator()
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit.from_ops((cirq.X**b0)(q0),
                                            (cirq.X**b1)(q1),
                                            cirq.measure(q0),
                                            cirq.measure(q1))
            result = simulator.run(circuit)
            np.testing.assert_equal(result.measurements,
                                    {'0': [[b0]], '1': [[b1]]})

def test_run_random_unitary():
    random_circuit = cirq.testing.random_circuit(qubits=8, n_moments=5,
                                                 op_density=0.5)
    [cirq.inverse(cirq.inverse(op) for op in moment.operations) for moment in random_circuit]
