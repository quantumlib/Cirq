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

import numpy as np
import pytest

import cirq
import cirq_google as cg


_OPTIMIZERS_AND_GATESETS = [
    ('sqrt_iswap', cg.SQRT_ISWAP_GATESET),
    ('sycamore', cg.SYC_GATESET),
    ('xmon', cg.XMON),
    ('xmon_partial_cz', cg.XMON),
]


@pytest.mark.parametrize('optimizer_type, gateset', _OPTIMIZERS_AND_GATESETS)
def test_optimizer_output_gates_are_supported(optimizer_type, gateset):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CZ(q0, q1), cirq.X(q0) ** 0.2, cirq.Z(q1) ** 0.2, cirq.measure(q0, q1, key='m')
    )
    new_circuit = cg.optimized_for_sycamore(circuit, optimizer_type=optimizer_type)
    for moment in new_circuit:
        for op in moment:
            assert gateset.is_supported_operation(op)


@pytest.mark.parametrize('optimizer_type, gateset', _OPTIMIZERS_AND_GATESETS)
def test_optimize_large_measurement_gates(optimizer_type, gateset):
    qubits = cirq.LineQubit.range(53)
    circuit = cirq.Circuit(
        cirq.X.on_each(qubits),
        [cirq.CZ(qubits[i], qubits[i + 1]) for i in range(0, len(qubits) - 1, 2)],
        [cirq.CZ(qubits[i], qubits[i + 1]) for i in range(1, len(qubits) - 1, 2)],
        cirq.measure(*qubits, key='m'),
    )
    new_circuit = cg.optimized_for_sycamore(circuit, optimizer_type=optimizer_type)
    for moment in new_circuit:
        for op in moment:
            assert gateset.is_supported_operation(op)


def test_invalid_input():
    with pytest.raises(ValueError):
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.CZ(q0, q1), cirq.X(q0) ** 0.2, cirq.Z(q1) ** 0.2, cirq.measure(q0, q1, key='m')
        )
        _ = cg.optimized_for_sycamore(circuit, optimizer_type='for_tis_100')


def test_tabulation():
    q0, q1 = cirq.LineQubit.range(2)
    u = cirq.testing.random_special_unitary(4, random_state=np.random.RandomState(52))
    circuit = cirq.Circuit(cirq.MatrixGate(u).on(q0, q1))
    np.testing.assert_allclose(u, cirq.unitary(circuit))

    circuit2 = cg.optimized_for_sycamore(circuit, optimizer_type='sycamore')
    cirq.testing.assert_allclose_up_to_global_phase(u, cirq.unitary(circuit2), atol=1e-5)
    assert len(circuit2) == 13

    # Note this is run on every commit, so it needs to be relatively quick.
    # This requires us to use relatively loose tolerances
    circuit3 = cg.optimized_for_sycamore(
        circuit, optimizer_type='sycamore', tabulation_resolution=0.1
    )
    cirq.testing.assert_allclose_up_to_global_phase(u, cirq.unitary(circuit3), rtol=1e-1, atol=1e-1)
    assert len(circuit3) == 7


def test_no_tabulation():
    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    with pytest.raises(NotImplementedError):
        cg.optimized_for_sycamore(circuit, optimizer_type='sqrt_iswap', tabulation_resolution=0.01)

    with pytest.raises(NotImplementedError):
        cg.optimized_for_sycamore(circuit, optimizer_type='xmon', tabulation_resolution=0.01)

    with pytest.raises(NotImplementedError):
        cg.optimized_for_sycamore(
            circuit, optimizer_type='xmon_partial_cz', tabulation_resolution=0.01
        )


def test_one_q_matrix_gate():
    u = cirq.testing.random_special_unitary(2)
    q = cirq.LineQubit(0)
    circuit0 = cirq.Circuit(cirq.MatrixGate(u).on(q))
    assert len(circuit0) == 1
    circuit_iswap = cg.optimized_for_sycamore(circuit0, optimizer_type='sqrt_iswap')
    assert len(circuit_iswap) == 1
    for moment in circuit_iswap:
        for op in moment:
            assert cg.SQRT_ISWAP_GATESET.is_supported_operation(op)
            # single qubit gates shared between gatesets, so:
            assert cg.SYC_GATESET.is_supported_operation(op)

    circuit_syc = cg.optimized_for_sycamore(circuit0, optimizer_type='sycamore')
    assert len(circuit_syc) == 1
    for moment in circuit_iswap:
        for op in moment:
            assert cg.SYC_GATESET.is_supported_operation(op)
            # single qubit gates shared between gatesets, so:
            assert cg.SQRT_ISWAP_GATESET.is_supported_operation(op)
