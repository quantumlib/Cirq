# Copyright 2024 The Cirq Developers
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

import cirq
import cirq.transformers.measure_in_random_bases as mrb


def test_append_randomized_measurements_generates_n_circuits():
    # Create a 4-qubit
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)])
    print(circuit.moments)
    num_moments_pre = len(circuit.moments)

    # Append randomized measurements
    circuits = mrb.append_randomized_measurements(circuit)
    print(circuits[0].moments)
    num_moments_post = len(circuits[0].moments)

    assert len(circuits) == 4  # num of qubits
    assert num_moments_post == num_moments_pre + 2  # 1 random gate and 1 measurement gate
