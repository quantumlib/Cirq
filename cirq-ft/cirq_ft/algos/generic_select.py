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

"""Gates for applying generic selected unitaries."""

from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray

import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import select_and_prepare, unary_iteration_gate


def _to_tuple(dps: Sequence[cirq.DensePauliString]) -> Tuple[cirq.DensePauliString, ...]:
    return tuple(dps)


@attr.frozen
class GenericSelect(select_and_prepare.SelectOracle, unary_iteration_gate.UnaryIterationGate):
    r"""A SELECT gate for selecting and applying operators from an array of `PauliString`s.

    $$
    \mathrm{SELECT} = \sum_{l}|l \rangle \langle l| \otimes U_l
    $$

    Where $U_l$ is a member of the Pauli group.

    This gate uses the unary iteration scheme to apply `select_unitaries[selection]` to `target`
    controlled on the single-bit `control` register.

    Args:
        selection_bitsize: The size of the indexing `select` register. This should be at least
            `log2(len(select_unitaries))`
        target_bitsize: The size of the `target` register.
        select_unitaries: List of `DensePauliString`s to apply to the `target` register. Each
            dense pauli string must contain `target_bitsize` terms.
        control_val: Optional control value. If specified, a singly controlled gate is constructed.
    """
    selection_bitsize: int
    target_bitsize: int
    select_unitaries: Tuple[cirq.DensePauliString, ...] = attr.field(converter=_to_tuple)
    control_val: Optional[int] = None

    def __attrs_post_init__(self):
        if any(len(dps) != self.target_bitsize for dps in self.select_unitaries):
            raise ValueError(
                f"Each dense pauli string in {self.select_unitaries} should contain "
                f"{self.target_bitsize} terms."
            )
        min_bitsize = (len(self.select_unitaries) - 1).bit_length()
        if self.selection_bitsize < min_bitsize:
            raise ValueError(
                f"selection_bitsize={self.selection_bitsize} should be at-least {min_bitsize}"
            )

    @cached_property
    def control_registers(self) -> infra.Registers:
        registers = [] if self.control_val is None else [infra.Register('control', 1)]
        return infra.Registers(registers)

    @cached_property
    def selection_registers(self) -> infra.SelectionRegisters:
        return infra.SelectionRegisters(
            [
                infra.SelectionRegister(
                    'selection', self.selection_bitsize, len(self.select_unitaries)
                )
            ]
        )

    @cached_property
    def target_registers(self) -> infra.Registers:
        return infra.Registers.build(target=self.target_bitsize)

    def decompose_from_registers(
        self, context, **quregs: NDArray[cirq.Qid]  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        if self.control_val == 0:
            yield cirq.X(*quregs['control'])
        yield super(GenericSelect, self).decompose_from_registers(context=context, **quregs)
        if self.control_val == 0:
            yield cirq.X(*quregs['control'])

    def nth_operation(  # type: ignore[override]
        self,
        context: cirq.DecompositionContext,
        selection: int,
        control: cirq.Qid,
        target: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        """Applies `self.select_unitaries[selection]`.

        Args:
             context: `cirq.DecompositionContext` stores options for decomposing gates (eg:
                cirq.QubitManager).
             selection: takes on values [0, self.iteration_lengths[0])
             control: Qid that is the control qubit or qubits
             target: Target register qubits
        """
        ps = self.select_unitaries[selection].on(*target)
        return ps.with_coefficient(np.sign(complex(ps.coefficient).real)).controlled_by(control)

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cirq.ops.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'GenericSelect':
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
            return GenericSelect(
                self.selection_bitsize,
                self.target_bitsize,
                self.select_unitaries,
                control_val=control_values[0],
            )
        raise NotImplementedError(
            f'Cannot create a controlled version of {self} with control_values={control_values}.'
        )

    def __repr__(self) -> str:
        return (
            f'cirq_ft.GenericSelect('
            f'{self.selection_bitsize},'
            f'{self.target_bitsize}, '
            f'{self.select_unitaries}, '
            f'{self.control_val})'
        )
