# Copyright 2025 The Cirq Developers
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
import pytest

import cirq


@pytest.mark.benchmark(group="phased_x_z_gate")
def test_has_stabilizer_effect_for_cliffords(benchmark) -> None:
    phased_xz_cliffords = [
        g.to_phased_xz_gate() for g in cirq.SingleQubitCliffordGate.all_single_qubit_cliffords
    ]
    f = lambda: all(cirq.has_stabilizer_effect(g) for g in phased_xz_cliffords)
    result = benchmark(f)
    assert result


@pytest.mark.benchmark(group="phased_x_z_gate")
def test_has_stabilizer_effect_for_random_gates(benchmark) -> None:
    # benchmark the same number of calls as for Cliffords
    num_calls = len(cirq.SingleQubitCliffordGate.all_single_qubit_cliffords)
    random_phased_xz_gates = [
        cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)
        for x, z, a in np.random.uniform(-1, 1, (num_calls, 3))
    ]
    f = lambda: any(cirq.has_stabilizer_effect(g) for g in random_phased_xz_gates)
    result = benchmark(f)
    assert not result
