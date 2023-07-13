# Copyright 2021 The Cirq Developers
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
from attr import frozen
import numpy as np
from cirq_ft import infra
from cirq._compat import cached_property

import cirq
import pytest

from cirq_ft.algos import phase_estimation as pe


@frozen
class BasicPrep(infra.GateWithRegisters):
    control_register: infra.Register

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers([self.control_register])

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs
    ) -> cirq.OP_TREE:
        yield cirq.H.on_each(quregs[self.control_register.name])


@frozen
class Lamda(infra.GateWithRegisters):
    control_register: infra.Register
    eigen_vector_register: infra.Register
    theta: int

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers([self.control_register, self.eigen_vector_register])

    @cached_property
    def control_register_name(self):
        return self.control_register.name

    @cached_property
    def eigenvector_register_name(self):
        return self.eigen_vector_register.name

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs
    ) -> cirq.OP_TREE:
        control_register_qubits = quregs[self.control_register_name]
        eigenvector_register_qubits = quregs[self.eigenvector_register_name]
        U = cirq.Z ** (2 * self.theta)
        bits_of_precision = len(control_register_qubits)
        yield [
            cirq.ControlledGate(U).on(bit, *eigenvector_register_qubits)
            ** (2 ** (bits_of_precision - i - 1))
            for i, bit in enumerate(control_register_qubits)
        ]


@frozen
class KitaevPhaseEstimation(pe.PhaseEstimation):
    control_bitsize: int
    eigenvector_bitsize: int
    theta: int

    @cached_property
    def control_register(self) -> infra.Register:
        return infra.Register("control_register", self.control_bitsize)

    @cached_property
    def eigenvector_register(self) -> infra.Register:
        return infra.Register("eigenvector_register", self.eigenvector_bitsize)

    @cached_property
    def control_register_prep(self) -> infra.GateWithRegisters:
        return BasicPrep(self.control_register)

    @cached_property
    def eigen_vector_register_prep(self) -> cirq.Gate:
        return cirq.X

    @cached_property
    def lamda(self) -> infra.GateWithRegisters:
        return Lamda(self.control_register, self.eigenvector_register, self.theta)


@pytest.mark.parametrize('theta', [0.234, 0.78, 0.54])
def test_phase_estimation(theta):
    n_bits = 9
    error_bound = 0.1
    control_register = cirq.NamedQubit.range(n_bits, prefix='c')
    op = KitaevPhaseEstimation(n_bits, 1, theta).on_registers(
        control_register=control_register, eigenvector_register=[cirq.q('ev')]
    )
    cirquit = cirq.Circuit(op)
    cirquit.append(cirq.measure(*control_register, key='m'))
    sim = cirq.Simulator()
    result = sim.run(cirquit, repetitions=10)
    theta_estimates = (
        np.sum(2 ** np.arange(n_bits) * result.measurements['m'], axis=1) / 2**n_bits
    )
    assert abs(np.average(theta_estimates) - theta) < error_bound
    print(theta_estimates)
