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

import itertools
import random

import numpy as np
import pytest
import sympy

import cirq


@pytest.mark.parametrize(
    ["num_qubits", "num_scan_points"], itertools.product([50, 100, 150, 200], [20, 40, 60, 80, 100])
)
@pytest.mark.benchmark(group="parameter_resolution")
def test_parameter_resolution(benchmark, num_qubits: int, num_scan_points: int) -> None:
    qubits = cirq.GridQubit.rect(1, num_qubits)
    symbols = {q: sympy.Symbol(f'a_{q}') for q in qubits}
    circuit = cirq.Circuit([cirq.X(q) ** symbols[q] for q in qubits], cirq.measure_each(*qubits))
    qubit_amps = {q: random.uniform(0.48, 0.52) for q in qubits}
    diff_amps = np.linspace(-0.3, 0.3, num=num_scan_points)

    def _f():
        for diff in diff_amps:
            resolver = {symbols[q]: amp + diff for q, amp in qubit_amps.items()}
            _ = cirq.resolve_parameters(circuit, resolver)

    benchmark(_f)
