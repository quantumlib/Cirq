# Copyright 2022 The Cirq Developers
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


class RouteCQC:
    params = [[10, 25, 50, 75, 100], [10, 50, 100, 250, 500, 1000], [0.5], [10]]
    param_names = ["qubits", "depth", "op_density", "grid_device_size"]
    timeout = 300  # Increase timeout to 5 minutes instead of default 60 seconds.

    def setup(self, qubits: int, depth: int, op_density: float, grid_device_size: int):
        gate_domain = {cirq.CNOT: 2, cirq.X: 1}
        self.circuit = cirq.testing.random_circuit(
            qubits, depth, op_density, gate_domain=gate_domain, random_state=12345
        )
        self.device = cirq.testing.construct_grid_device(grid_device_size, grid_device_size)
        self.router = cirq.RouteCQC(self.device.metadata.nx_graph)

    def time_circuit_routing(self, *_):
        self.routed_circuit = self.router(self.circuit)

    def track_routed_circuit_depth_ratio(self, *_) -> float:
        self.routed_circuit = self.router(self.circuit)
        return len(self.routed_circuit) / len(self.circuit)

    def teardown(self, *_):
        self.device.validate_circuit(self.routed_circuit)
