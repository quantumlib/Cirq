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

import cirq.devices as devices
import cirq.experiments.ghz.fidelity as ghz_fidelity
import cirq.experiments.ghz.ghz_1d as ghz_1d
import cirq.sim as sim


def test_measure_ghz_fidelity():
    qubits = devices.LineQubit.range(10)
    sampler = sim.Simulator()
    circuit = ghz_1d.generate_1d_ghz_circuit(qubits)
    rng = np.random.default_rng()
    result = ghz_fidelity.measure_ghz_fidelity(circuit, 20, 20, rng, sampler)
    f, df = result.compute_fidelity(mitigated=False)
    assert f == 1.0
    assert df == 0.0

    qubits = devices.LineQubit.range(4)
    circuit = ghz_1d.generate_1d_ghz_circuit(qubits)
    result = ghz_fidelity.measure_ghz_fidelity(circuit, 2**3 - 1, 2**3, rng, sampler)
    f, df = result.compute_fidelity(mitigated=False)
    assert f == 1.0
    assert df == 0.0
