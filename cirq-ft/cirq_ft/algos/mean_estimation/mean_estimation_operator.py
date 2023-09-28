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
from cirq_ft.algos import reflection_using_prepare as rup
from cirq_ft.algos import select_and_prepare as sp
from cirq_ft.algos.mean_estimation import complex_phase_oracle


@attr.frozen
class CodeForRandomVariable:
    r"""A collection of `encoder` and `synthesizer` for a random variable y.

    We say we have "the code" for a random variable $y$ defined on a probability space
    $(W, p)$ if we have both, a synthesizer and an encoder defined as follows:

    The synthesizer is responsible to "prepare" the state
    $\sum_{w \in W} \sqrt{p(w)} |w> |garbage_{w}>$ on the "selection register" $w$ and potentially
    using a "junk register" corresponding to $|garbage_{w}>$. Thus, for convenience, the synthesizer
    follows the LCU PREPARE Oracle API.
    $$
    synthesizer|0> = \sum_{w \in W} \sqrt{p(w)} |w> |garbage_{w}>
    $$


    The encoder is responsible to encode the value of random variable $y(w)$ in a "target register"
    when the corresponding "selection register" stores integer $w$. Thus, for convenience, the
    encoder follows the LCU SELECT Oracle API.
    $$
    encoder|w>|0^b> = |w>|y(w)>
    $$
    where b is the number of bits required to encode the real range of random variable y.

    References:
        https://arxiv.org/abs/2208.07544, Definition 2.2 for synthesizer (P) and
        Definition 2.10 for encoder (Y).
    """

    synthesizer: sp.PrepareOracle
    encoder: sp.SelectOracle

    def __attrs_post_init__(self):
        assert self.synthesizer.selection_registers == self.encoder.selection_registers


@attr.frozen
class MeanEstimationOperator(infra.GateWithRegisters):
    r"""Mean estimation operator $U=REFL_{p} ROT_{y}$ as per Sec 3.1 of arxiv.org:2208.07544.

    The MeanEstimationOperator (aka KO Operator) expects `CodeForRandomVariable` to specify the
    synthesizer and encoder, that follows LCU SELECT/PREPARE API for convenience. It is composed
    of two unitaries:

        - REFL_{p}: Reflection around the state prepared by synthesizer $P$. It applies the unitary
            $P^{\dagger}(2|0><0| - I)P$.
        - ROT_{y}: Applies a complex phase $\exp(i * -2\arctan{y_{w}})$ when the selection register
            stores $w$. This is achieved by using the encoder to encode $y(w)$ in a temporary target
            register.

    Note that both $REFL_{p}$ and $ROT_{y}$ only act upon a selection register, thus mean estimation
    operator expects only a selection register (and a control register, for a controlled version for
    phase estimation).
    """

    code: CodeForRandomVariable
    cv: Tuple[int, ...] = attr.field(
        converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=()
    )
    power: int = 1
    arctan_bitsize: int = 32

    @cv.validator
    def _validate_cv(self, attribute, value):
        assert value in [(), (0,), (1,)]

    @cached_property
    def reflect(self) -> rup.ReflectionUsingPrepare:
        return rup.ReflectionUsingPrepare(
            self.code.synthesizer, control_val=None if self.cv == () else self.cv[0]
        )

    @cached_property
    def select(self) -> complex_phase_oracle.ComplexPhaseOracle:
        return complex_phase_oracle.ComplexPhaseOracle(self.code.encoder, self.arctan_bitsize)

    @cached_property
    def control_registers(self) -> Tuple[infra.Register, ...]:
        return self.code.encoder.control_registers

    @cached_property
    def selection_registers(self) -> Tuple[infra.SelectionRegister, ...]:
        return self.code.encoder.selection_registers

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature([*self.control_registers, *self.selection_registers])

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        select_reg = {reg.name: quregs[reg.name] for reg in self.select.signature}
        reflect_reg = {reg.name: quregs[reg.name] for reg in self.reflect.signature}
        select_op = self.select.on_registers(**select_reg)
        reflect_op = self.reflect.on_registers(**reflect_reg)
        for _ in range(self.power):
            yield select_op
            # Add a -1 global phase since `ReflectUsingPrepare` applies $R_{s} = I - 2|s><s|$
            # but we want to apply $R_{s} = 2|s><s| - I$ and this algorithm is sensitive to global
            # phase.
            yield [reflect_op, cirq.global_phase_operation(-1)]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = [] if self.cv == () else [["@(0)", "@"][self.cv[0]]]
        wire_symbols += ['U_ko'] * (
            infra.total_bits(self.signature) - infra.total_bits(self.control_registers)
        )
        if self.power != 1:
            wire_symbols[-1] = f'U_ko^{self.power}'
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cirq.ops.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'MeanEstimationOperator':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if (
            isinstance(control_values, Sequence)
            and len(control_values) == 1
            and isinstance(control_values[0], int)
            and not self.cv
        ):
            c_select = self.code.encoder.controlled(control_values=control_values)
            assert isinstance(c_select, sp.SelectOracle)
            return MeanEstimationOperator(
                CodeForRandomVariable(encoder=c_select, synthesizer=self.code.synthesizer),
                cv=self.cv + (control_values[0],),
                power=self.power,
                arctan_bitsize=self.arctan_bitsize,
            )
        raise NotImplementedError(
            f'Cannot create a controlled version of {self} with control_values={control_values}.'
        )

    def with_power(self, new_power: int) -> 'MeanEstimationOperator':
        return MeanEstimationOperator(
            self.code, cv=self.cv, power=new_power, arctan_bitsize=self.arctan_bitsize
        )

    def __pow__(self, power: int):
        return self.with_power(self.power * power)
