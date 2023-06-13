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

from typing import Sequence

import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import unary_iteration_gate


@attr.frozen
class SelectedMajoranaFermionGate(unary_iteration_gate.UnaryIterationGate):
    """Implements U s.t. U|l>|Psi> -> |l> T_{l} . Z_{l - 1} ... Z_{0} |Psi>

    where T is a single qubit target gate that defaults to pauli Y. The gate is
    implemented using an accumulator bit in the unary iteration circuit as explained
    in the reference below.


    Args:
        selection_regs: Indexing `select` registers of type `SelectionRegisters`. It also contains
            information about the iteration length of each selection register.
        control_regs: Control registers for constructing a controlled version of the gate.
        target_gate: Single qubit gate to be applied to the target qubits.

    References:
        See Fig 9 of https://arxiv.org/abs/1805.03662 for more details.
    """

    selection_regs: infra.SelectionRegisters
    control_regs: infra.Registers = infra.Registers.build(control=1)
    target_gate: cirq.Gate = cirq.Y

    @classmethod
    def make_on(cls, *, target_gate=cirq.Y, **quregs: Sequence[cirq.Qid]) -> cirq.Operation:
        """Helper constructor to automatically deduce selection_regs attribute."""
        return cls(
            selection_regs=infra.SelectionRegisters.build(
                selection=(len(quregs['selection']), len(quregs['target']))
            ),
            target_gate=target_gate,
        ).on_registers(**quregs)

    @cached_property
    def control_registers(self) -> infra.Registers:
        return self.control_regs

    @cached_property
    def selection_registers(self) -> infra.SelectionRegisters:
        return self.selection_regs

    @cached_property
    def target_registers(self) -> infra.Registers:
        return infra.Registers.build(target=self.selection_regs.total_iteration_size)

    @cached_property
    def extra_registers(self) -> infra.Registers:
        return infra.Registers.build(accumulator=1)

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **qubit_regs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        qubit_regs['accumulator'] = context.qubit_manager.qalloc(1)
        control = qubit_regs[self.control_regs[0].name] if self.control_registers.bitsize else []
        yield cirq.X(*qubit_regs['accumulator']).controlled_by(*control)
        yield super().decompose_from_registers(context=context, **qubit_regs)
        context.qubit_manager.qfree(qubit_regs['accumulator'])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self.control_registers.bitsize
        wire_symbols += ["In"] * self.selection_registers.bitsize
        wire_symbols += [f"Z{self.target_gate}"] * self.target_registers.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
        accumulator: Sequence[cirq.Qid],
        **selection_indices: int,
    ) -> cirq.OP_TREE:
        selection_idx = tuple(selection_indices[reg.name] for reg in self.selection_regs)
        target_idx = self.selection_registers.to_flat_idx(*selection_idx)
        yield cirq.CNOT(control, *accumulator)
        yield self.target_gate(target[target_idx]).controlled_by(control)
        yield cirq.CZ(*accumulator, target[target_idx])
