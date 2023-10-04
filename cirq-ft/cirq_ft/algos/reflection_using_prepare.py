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

from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray

import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import multi_control_multi_target_pauli as mcmt
from cirq_ft.algos import select_and_prepare


@attr.frozen(cache_hash=True)
class ReflectionUsingPrepare(infra.GateWithRegisters):
    """Applies reflection around a state prepared by `prepare_gate`

    Applies $R_{s} = I - 2|s><s|$ using $R_{s} = P^†(I - 2|0><0|)P$ s.t. $P|0> = |s>$.
    Here
        $|s>$: The state along which we want to reflect.
        $P$: Unitary that prepares that state $|s>$ from the zero state $|0>$
        $R_{s}$: Reflection operator that adds a `-1` phase to all states in the subspace
            spanned by $|s>$.

    The composite gate corresponds to implementing the following circuit:

    |control> ------------------ Z -------------------
                                 |
    |L>       ---- PREPARE^† --- o --- PREPARE -------


    Args:
        prepare_gate: An instance of `cq.StatePreparationAliasSampling` gate the corresponds to
            `PREPARE`.
        control_val: If 0/1, a controlled version of the reflection operator is constructed.
            Defaults to None, in which case the resulting reflection operator is not controlled.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
        (https://arxiv.org/abs/1805.03662).
            Babbush et. al. (2018). Figure 1.
    """

    prepare_gate: select_and_prepare.PrepareOracle
    control_val: Optional[int] = None

    @cached_property
    def control_registers(self) -> Tuple[infra.Register, ...]:
        return () if self.control_val is None else (infra.Register('control', 1),)

    @cached_property
    def selection_registers(self) -> Tuple[infra.SelectionRegister, ...]:
        return self.prepare_gate.selection_registers

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature([*self.control_registers, *self.selection_registers])

    def decompose_from_registers(
        self,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        qm = context.qubit_manager
        # 0. Allocate new ancillas, if needed.
        phase_target = qm.qalloc(1)[0] if self.control_val is None else quregs.pop('control')[0]
        state_prep_ancilla = {
            reg.name: qm.qalloc(reg.total_bits()) for reg in self.prepare_gate.junk_registers
        }
        state_prep_selection_regs = quregs
        prepare_op = self.prepare_gate.on_registers(
            **state_prep_selection_regs, **state_prep_ancilla
        )
        # 1. PREPARE†
        yield cirq.inverse(prepare_op)
        # 2. MultiControlled Z, controlled on |000..00> state.
        phase_control = infra.merge_qubits(self.selection_registers, **state_prep_selection_regs)
        yield cirq.X(phase_target) if not self.control_val else []
        yield mcmt.MultiControlPauli([0] * len(phase_control), target_gate=cirq.Z).on_registers(
            controls=phase_control, target=phase_target
        )
        yield cirq.X(phase_target) if not self.control_val else []
        # 3. PREPARE
        yield prepare_op

        # 4. Deallocate ancilla.
        qm.qfree([q for anc in state_prep_ancilla.values() for q in anc])
        if self.control_val is None:
            qm.qfree([phase_target])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@' if self.control_val else '@(0)'] * infra.total_bits(
            self.control_registers
        )
        wire_symbols += ['R_L'] * infra.total_bits(self.selection_registers)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __repr__(self):
        return f'cirq_ft.ReflectionUsingPrepare({self.prepare_gate}, {self.control_val})'

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cirq.ops.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'ReflectionUsingPrepare':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if (
            isinstance(control_values, Sequence)
            and isinstance(control_values[0], int)
            and len(control_values) == 1
            and self.control_val is None
        ):
            return ReflectionUsingPrepare(self.prepare_gate, control_val=control_values[0])
        raise NotImplementedError(
            f'Cannot create a controlled version of {self} with control_values={control_values}.'
        )
