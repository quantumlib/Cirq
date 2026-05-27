# Copyright 2026 The Cirq Developers
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

import cirq.contrib.ghz.ghz_1d as ghz_1d
import cirq.devices as devices
import cirq.sim as sim


def test_generate_1d_ghz_circuit():
    simulator = sim.Simulator(dtype=np.complex128)
    for nq in [7, 8]:
        qubits = devices.LineQubit.range(nq)
        circuits = [
            ghz_1d.generate_1d_ghz_circuit(qubits, add_dd=False, from_one_end=False),
            ghz_1d.generate_1d_ghz_circuit(qubits, add_dd=False, from_one_end=True),
            ghz_1d.generate_1d_ghz_circuit(qubits, add_dd=True, from_one_end=False),
            ghz_1d.generate_1d_ghz_circuit(qubits, add_dd=True, from_one_end=True),
        ]
        psi0 = simulator.simulate(circuits[0]).final_state_vector
        for circuit in circuits[1:]:
            assert np.isclose(
                np.abs(np.vdot(psi0, simulator.simulate(circuit).final_state_vector)) ** 2, 1.0
            )
        assert len(circuits[-1]) > len(circuits[0])
        assert len(circuits[-2]) == len(circuits[0])
        assert len(circuits[-3]) > len(circuits[0])
