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
from cirq_ft.algos import reflection_using_prepare, select_and_prepare


@attr.frozen(cache_hash=True)
class QubitizationWalkOperator(infra.GateWithRegisters):
    r"""Constructs a Szegedy Quantum Walk operator using LCU oracles SELECT and PREPARE.

    Constructs a Szegedy quantum walk operator $W = R_{L} . SELECT$, which is a product of
    two reflections $R_{L} = (2|L><L| - I)$ and $SELECT=\sum_{l}|l><l|H_{l}$.

    The action of $W$ partitions the Hilbert space into a direct sum of two-dimensional irreducible
    vector spaces. For an arbitrary eigenstate $|k>$ of $H$ with eigenvalue $E_k$, $|\ell>|k>$ and
    an orthogonal state $\phi_{k}$ span the irreducible two-dimensional space that $|\ell>|k>$ is
    in under the action of $W$. In this space, $W$ implements a Pauli-Y rotation by an angle of
    $-2arccos(E_{k} / \lambda)$ s.t. $W = e^{i arccos(E_k / \lambda) Y}$.

    Thus, the walk operator $W$ encodes the spectrum of $H$ as a function of eigenphases of $W$
    s.t. $spectrum(H) = \lambda cos(arg(spectrum(W)))$ where $arg(e^{i\phi}) = \phi$.

    Args:
        select: The SELECT lcu gate implementing $SELECT=\sum_{l}|l><l|H_{l}$.
        prepare: Then PREPARE lcu gate implementing
            $PREPARE|00...00> = \sum_{l=0}^{L - 1}\sqrt{\frac{w_{l}}{\lambda}} |l> = |\ell>$
        control_val: If 0/1, a controlled version of the walk operator is constructed. Defaults to
            None, in which case the resulting walk operator is not controlled.
        power: Constructs $W^{power}$ by repeatedly decomposing into `power` copies of $W$.
            Defaults to 1.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
        (https://arxiv.org/abs/1805.03662).
            Babbush et. al. (2018). Figure 1.
    """
    select: select_and_prepare.SelectOracle
    prepare: select_and_prepare.PrepareOracle
    control_val: Optional[int] = None
    power: int = 1

    def __attrs_post_init__(self):
        assert self.select.control_registers == self.reflect.control_registers

    @cached_property
    def control_registers(self) -> infra.Registers:
        return self.select.control_registers

    @cached_property
    def selection_registers(self) -> infra.SelectionRegisters:
        return self.prepare.selection_registers

    @cached_property
    def target_registers(self) -> infra.Registers:
        return self.select.target_registers

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    @cached_property
    def reflect(self) -> reflection_using_prepare.ReflectionUsingPrepare:
        return reflection_using_prepare.ReflectionUsingPrepare(
            self.prepare, control_val=self.control_val
        )

    def decompose_from_registers(
        self,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        select_reg = {reg.name: quregs[reg.name] for reg in self.select.registers}
        select_op = self.select.on_registers(**select_reg)

        reflect_reg = {reg.name: quregs[reg.name] for reg in self.reflect.registers}
        reflect_op = self.reflect.on_registers(**reflect_reg)
        for _ in range(self.power):
            yield select_op
            yield reflect_op

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@' if self.control_val else '@(0)'] * self.control_registers.total_bits()
        wire_symbols += ['W'] * (self.registers.total_bits() - self.control_registers.total_bits())
        wire_symbols[-1] = f'W^{self.power}' if self.power != 1 else 'W'
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cirq.ops.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'QubitizationWalkOperator':
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
            c_select = self.select.controlled(control_values=control_values)
            assert isinstance(c_select, select_and_prepare.SelectOracle)
            return QubitizationWalkOperator(
                c_select, self.prepare, control_val=control_values[0], power=self.power
            )
        raise NotImplementedError(
            f'Cannot create a controlled version of {self} with control_values={control_values}.'
        )

    def with_power(self, new_power: int) -> 'QubitizationWalkOperator':
        return QubitizationWalkOperator(
            self.select, self.prepare, control_val=self.control_val, power=new_power
        )

    def __repr__(self) -> str:
        return (
            f'cirq_ft.QubitizationWalkOperator('
            f'{self.select}, '
            f'{self.prepare}, '
            f'{self.control_val}, '
            f'{self.power})'
        )

    def __pow__(self, power: int):
        return self.with_power(self.power * power)

    def _t_complexity_(self):
        if self.power > 1:
            return self.power * infra.t_complexity(self.with_power(1))
        return NotImplemented
