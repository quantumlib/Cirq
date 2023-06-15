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

import random
import numpy as np
import sympy
import cirq


class RabiCalibration:
    params = ([50, 100, 150, 200], [20, 40, 60, 80, 100])
    param_names = ["num_qubits", "num_scan_points"]

    def setup(self, num_qubits: int, _):
        qubits = cirq.GridQubit.rect(1, num_qubits)
        self.symbols = {q: sympy.Symbol(f'a_{q}') for q in qubits}
        self.circuit = cirq.Circuit(
            [cirq.X(q) ** self.symbols[q] for q in qubits], cirq.measure_each(*qubits)
        )
        self.qubit_amps = {q: random.uniform(0.48, 0.52) for q in qubits}

    def time_parameter_resolution(self, _, num_scan_points: int):
        for diff in np.linspace(-0.3, 0.3, num=num_scan_points):
            resolver = {self.symbols[q]: amp + diff for q, amp in self.qubit_amps.items()}
            _ = cirq.resolve_parameters(self.circuit, resolver)
