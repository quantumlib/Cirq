# Copyright 2019 The Cirq Developers
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

import pytest

import cirq
import cirq.google as cg


@pytest.mark.parametrize('optimizer_type, gateset', [
    ('sqrt_iswap', cg.SQRT_ISWAP_GATESET),
    ('sycamore', cg.SYC_GATESET),
])
def test_optimizer_output_gates_are_supported(optimizer_type, gateset):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CZ(q0, q1),
                           cirq.X(q0)**0.2,
                           cirq.Z(q1)**0.2, cirq.measure(q0, q1, key='m'))
    new_circuit = cg.optimized_for_sycamore(circuit,
                                            optimizer_type=optimizer_type)
    for moment in new_circuit:
        for op in moment:
            assert gateset.is_supported_gate(op.gate)


def test_invalid_input():
    with pytest.raises(ValueError):
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.CZ(q0, q1),
                               cirq.X(q0)**0.2,
                               cirq.Z(q1)**0.2, cirq.measure(q0, q1, key='m'))
        _ = cg.optimized_for_sycamore(circuit, optimizer_type='for_tis_100')
