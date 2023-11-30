# Copyright 2023 The Cirq Developers
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

import math
from typing import Optional, Tuple

import cirq
import cirq_ft
import numpy as np
import pytest
from attr import frozen
from cirq._compat import cached_property
from cirq_ft.algos.mean_estimation.complex_phase_oracle import ComplexPhaseOracle
from cirq_ft.infra import bit_tools
from cirq_ft.infra import testing as cq_testing
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests


@frozen
class ExampleSelect(cirq_ft.SelectOracle):
    bitsize: int
    control_val: Optional[int] = None

    @cached_property
    def control_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return () if self.control_val is None else (cirq_ft.Register('control', 1),)

    @cached_property
    def selection_registers(self) -> Tuple[cirq_ft.SelectionRegister, ...]:
        return (cirq_ft.SelectionRegister('selection', self.bitsize),)

    @cached_property
    def target_registers(self) -> Tuple[cirq_ft.Register, ...]:
        return (cirq_ft.Register('target', self.bitsize),)

    def decompose_from_registers(self, context, selection, target):
        yield [cirq.CNOT(s, t) for s, t in zip(selection, target)]


@pytest.mark.parametrize('bitsize', [2, 3, 4, 5])
@pytest.mark.parametrize('arctan_bitsize', [5, 6, 7])
@allow_deprecated_cirq_ft_use_in_tests
def test_phase_oracle(bitsize: int, arctan_bitsize: int):
    phase_oracle = ComplexPhaseOracle(ExampleSelect(bitsize), arctan_bitsize)
    g = cq_testing.GateHelper(phase_oracle)

    # Prepare uniform superposition state on selection register and apply phase oracle.
    circuit = cirq.Circuit(cirq.H.on_each(*g.quregs['selection']))
    circuit += cirq.Circuit(cirq.decompose_once(g.operation))

    # Simulate the circut and test output.
    qubit_order = cirq.QubitOrder.explicit(g.quregs['selection'], fallback=cirq.QubitOrder.DEFAULT)
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubit_order)
    state_vector = result.final_state_vector
    state_vector = state_vector.reshape(2**bitsize, len(state_vector) // 2**bitsize)
    prepared_state = state_vector.sum(axis=1)
    for x in range(2**bitsize):
        output_val = -2 * np.arctan(x, dtype=np.double) / np.pi
        output_bits = [*bit_tools.iter_bits_fixed_point(np.abs(output_val), arctan_bitsize)]
        approx_val = np.sign(output_val) * math.fsum(
            [b * (1 / 2 ** (1 + i)) for i, b in enumerate(output_bits)]
        )

        assert math.isclose(output_val, approx_val, abs_tol=1 / 2**bitsize), output_bits

        y = np.exp(1j * approx_val * np.pi) / np.sqrt(2**bitsize)
        assert np.isclose(prepared_state[x], y)


@allow_deprecated_cirq_ft_use_in_tests
def test_phase_oracle_consistent_protocols():
    bitsize, arctan_bitsize = 3, 5
    gate = ComplexPhaseOracle(ExampleSelect(bitsize, 1), arctan_bitsize)
    expected_symbols = ('@',) + ('ROTy',) * bitsize
    assert cirq.circuit_diagram_info(gate).wire_symbols == expected_symbols
