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
from typing import Optional, List
from attr import frozen
from cirq._compat import cached_property
import cirq

from cirq_ft import infra


@frozen
class KitaevPhaseEstimation(infra.GateWithRegisters):
    r"""Class representing the Kitaev Phase Estimation algorithm, originally introduced by
    Kitaev in https://arxiv.org/abs/quant-ph/9511026."""

    m: int
    eigenvector_bitsize: int
    U: infra.GateWithRegisters
    eigenvector_prep: Optional[infra.GateWithRegisters] = None

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers.build(
            bits_of_precision_register=self.m, eigenvector_register=self.eigenvector_bitsize
        )

    def qft_inverse(self, qubits):
        """Generator for the inverse QFT on a list of qubits."""
        qreg = list(qubits)
        while len(qreg) > 0:
            q_head = qreg.pop(0)
            yield cirq.H(q_head)
            for i, qubit in enumerate(qreg):
                yield (cirq.CZ ** (-1 / 2 ** (i + 1)))(qubit, q_head)

    def bits_of_precision_prep(self) -> cirq.Gate:
        return cirq.H

    def U_to_the_k_power(self, control_bits, eigen_vector_bit) -> List[cirq.Operation]:
        return [
            cirq.ControlledGate(self.U).on(bit, eigen_vector_bit) ** (2 ** (self.m - i - 1))
            for i, bit in enumerate(control_bits)
        ]

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs
    ) -> cirq.OP_TREE:
        bits_of_precision = quregs["bits_of_precision_register"]
        eigenvector_bits = quregs["eigenvector_register"]

        yield self.bits_of_precision_prep().on_each(*bits_of_precision)
        if self.eigenvector_prep is not None:
            yield self.eigenvector_prep.on(*eigenvector_bits)
        yield [op for op in self.U_to_the_k_power(bits_of_precision, *eigenvector_bits)]
        yield self.qft_inverse([*bits_of_precision])
