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

from typing import Tuple
from numpy.typing import NDArray

import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import select_and_prepare
from cirq_ft.algos.mean_estimation import arctan


@attr.frozen
class ComplexPhaseOracle(infra.GateWithRegisters):
    r"""Applies $ROT_{y}|l>|garbage_{l}> = exp(i * -2arctan{y_{l}})|l>|garbage_{l}>$.

    TODO(#6142): This currently assumes that the random variable `y_{l}` only takes integer
    values. This constraint can be removed by using a standardized floating point to
    binary encoding, like IEEE 754, to encode arbitrary floats in the binary target
    register and use them to compute the more accurate $-2arctan{y_{l}}$ for any arbitrary
    $y_{l}$.
    """

    encoder: select_and_prepare.SelectOracle
    arctan_bitsize: int = 32

    @cached_property
    def control_registers(self) -> Tuple[infra.Register, ...]:
        return self.encoder.control_registers

    @cached_property
    def selection_registers(self) -> Tuple[infra.SelectionRegister, ...]:
        return self.encoder.selection_registers

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature([*self.control_registers, *self.selection_registers])

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        qm = context.qubit_manager
        target_reg = {
            reg.name: qm.qalloc(reg.total_bits()) for reg in self.encoder.target_registers
        }
        target_qubits = infra.merge_qubits(self.encoder.target_registers, **target_reg)
        encoder_op = self.encoder.on_registers(**quregs, **target_reg)

        arctan_sign, arctan_target = qm.qalloc(1), qm.qalloc(self.arctan_bitsize)
        arctan_op = arctan.ArcTan(len(target_qubits), self.arctan_bitsize).on(
            *target_qubits, *arctan_sign, *arctan_target
        )

        yield encoder_op
        yield arctan_op
        for i, q in enumerate(arctan_target):
            yield (cirq.Z(q) ** (1 / 2 ** (1 + i))).controlled_by(*arctan_sign, control_values=[0])
            yield (cirq.Z(q) ** (-1 / 2 ** (1 + i))).controlled_by(*arctan_sign, control_values=[1])

        yield cirq.inverse(arctan_op)
        yield cirq.inverse(encoder_op)

        qm.qfree([*arctan_sign, *arctan_target, *target_qubits])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@'] * infra.total_bits(self.control_registers)
        wire_symbols += ['ROTy'] * infra.total_bits(self.selection_registers)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
