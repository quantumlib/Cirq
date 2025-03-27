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

import numpy as np

import cirq.transformers.noise_adding as na
from cirq import circuits, devices, ops


def test_noise_adding():
    qubits = devices.LineQubit.range(4)
    one_layer = circuits.Circuit(ops.CZ(*qubits[:2]), ops.CZ(*qubits[2:]))
    circuit = one_layer * 10

    # test that p=0 does nothing
    transformed_circuit_p0 = na.DepolarizingNoiseTransformer(0.0)(circuit)
    assert transformed_circuit_p0 == circuit

    # test that p=1 doubles the circuit depth
    transformed_circuit_p1 = na.DepolarizingNoiseTransformer(1.0)(circuit)
    assert len(transformed_circuit_p1) == 20

    # test that we get a deterministic result when using a specific rng
    rng = np.random.default_rng(0)
    transformed_circuit_p0_03 = na.DepolarizingNoiseTransformer(0.03)(circuit, rng=rng)
    expected_circuit = (
        one_layer * 2
        + circuits.Circuit(ops.I(qubits[2]), ops.Z(qubits[3]))
        + one_layer * 4
        + circuits.Circuit(ops.Z(qubits[0]), ops.X(qubits[1]))
        + one_layer * 4
        + circuits.Circuit(ops.I(qubits[2]), ops.X(qubits[3]))
    )
    assert transformed_circuit_p0_03 == expected_circuit

    # test that supplying a dictionary for p works
    transformed_circuit_p_dict = na.DepolarizingNoiseTransformer(
        {tuple(qubits[:2]): 1.0, tuple(qubits[2:]): 0.0}
    )(circuit)
    assert len(transformed_circuit_p_dict) == 20  # depth should be doubled
    assert transformed_circuit_p_dict[1::2].all_qubits() == frozenset(
        qubits[:2]
    )  # no single-qubit gates get added to qubits[2:]
