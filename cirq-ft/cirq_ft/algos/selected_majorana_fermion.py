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

from typing import Sequence, Union, Tuple
from numpy.typing import NDArray

import attr
import cirq
import numpy as np

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
        selection_regs: Indexing `select` signature of type `SelectionRegister`. It also contains
            information about the iteration length of each selection register.
        control_regs: Control signature for constructing a controlled version of the gate.
        target_gate: Single qubit gate to be applied to the target qubits.

    References:
        See Fig 9 of https://arxiv.org/abs/1805.03662 for more details.
    """

    selection_regs: Tuple[infra.SelectionRegister, ...] = attr.field(
        converter=lambda v: (v,) if isinstance(v, infra.SelectionRegister) else tuple(v)
    )
    control_regs: Tuple[infra.Register, ...] = attr.field(
        converter=lambda v: (v,) if isinstance(v, infra.Register) else tuple(v)
    )
    target_gate: cirq.Gate = cirq.Y

    @control_regs.default
    def control_regs_default(self):
        return infra.Register('control', 1)

    @classmethod
    def make_on(
        cls,
        *,
        target_gate=cirq.Y,
        **quregs: Union[Sequence[cirq.Qid], NDArray[cirq.Qid]],  # type: ignore[type-var]
    ) -> cirq.Operation:
        """Helper constructor to automatically deduce selection_regs attribute."""
        return SelectedMajoranaFermionGate(
            selection_regs=infra.SelectionRegister(
                'selection', len(quregs['selection']), len(quregs['target'])
            ),
            target_gate=target_gate,
        ).on_registers(**quregs)

    @cached_property
    def control_registers(self) -> Tuple[infra.Register, ...]:
        return self.control_regs

    @cached_property
    def selection_registers(self) -> Tuple[infra.SelectionRegister, ...]:
        return self.selection_regs

    @cached_property
    def target_registers(self) -> Tuple[infra.Register, ...]:
        total_iteration_size = np.prod(
            tuple(reg.iteration_length for reg in self.selection_registers)
        )
        return (infra.Register('target', int(total_iteration_size)),)

    @cached_property
    def extra_registers(self) -> Tuple[infra.Register, ...]:
        return (infra.Register('accumulator', 1),)

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        quregs['accumulator'] = np.array(context.qubit_manager.qalloc(1))
        control = (
            quregs[self.control_regs[0].name] if infra.total_bits(self.control_registers) else []
        )
        yield cirq.X(*quregs['accumulator']).controlled_by(*control)
        yield super(SelectedMajoranaFermionGate, self).decompose_from_registers(
            context=context, **quregs
        )
        context.qubit_manager.qfree(quregs['accumulator'])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * infra.total_bits(self.control_registers)
        wire_symbols += ["In"] * infra.total_bits(self.selection_registers)
        wire_symbols += [f"Z{self.target_gate}"] * infra.total_bits(self.target_registers)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
        accumulator: Sequence[cirq.Qid],
        **selection_indices: int,
    ) -> cirq.OP_TREE:
        selection_shape = tuple(reg.iteration_length for reg in self.selection_regs)
        selection_idx = tuple(selection_indices[reg.name] for reg in self.selection_regs)
        target_idx = int(np.ravel_multi_index(selection_idx, selection_shape))
        yield cirq.CNOT(control, *accumulator)
        yield self.target_gate(target[target_idx]).controlled_by(control)
        yield cirq.CZ(*accumulator, target[target_idx])
