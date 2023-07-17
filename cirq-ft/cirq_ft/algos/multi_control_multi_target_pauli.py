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
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate


class MultiTargetCNOT(infra.GateWithRegisters):
    """Implements single control, multi-target CNOT_{n} gate in 2*log(n) + 1 CNOT depth.

    Implements CNOT_{n} = |0><0| I + |1><1| X^{n} using a circuit of depth 2*log(n) + 1
    containing only CNOT gates. See Appendix B.1 of https://arxiv.org/abs/1812.00954 for
    reference.
    """

    def __init__(self, num_targets: int):
        self._num_targets = num_targets

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers.build(control=1, targets=self._num_targets)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ):
        control, targets = quregs['control'], quregs['targets']

        def cnots_for_depth_i(i: int, q: NDArray[cirq.Qid]) -> cirq.OP_TREE:
            for c, t in zip(q[: 2**i], q[2**i : min(len(q), 2 ** (i + 1))]):
                yield cirq.CNOT(c, t)

        depth = len(targets).bit_length()
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(depth - i - 1, targets))
        yield cirq.CNOT(*control, targets[0])
        for i in range(depth):
            yield cirq.Moment(cnots_for_depth_i(i, targets))

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=["@"] + ["X"] * self._num_targets)


@attr.frozen
class MultiControlPauli(infra.GateWithRegisters):
    """Implements multi-control, single-target C^{n}P gate.

    Implements $C^{n}P = (1 - |1^{n}><1^{n}|) I + |1^{n}><1^{n}| P^{n}$ using $n-1$
    clean ancillas using a multi-controlled `AND` gate.

    References:
        [Constructing Large Controlled Nots]
        (https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html)
    """

    cvs: Tuple[int, ...] = attr.field(converter=infra.to_tuple)
    target_gate: cirq.Pauli = cirq.X

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers.build(controls=len(self.cvs), target=1)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray['cirq.Qid']
    ) -> cirq.OP_TREE:
        controls, target = quregs['controls'], quregs['target']
        qm = context.qubit_manager
        and_ancilla, and_target = qm.qalloc(len(self.cvs) - 2), qm.qalloc(1)
        yield and_gate.And(self.cvs).on_registers(
            control=controls, ancilla=and_ancilla, target=and_target
        )
        yield self.target_gate.on(*target).controlled_by(*and_target)
        yield and_gate.And(self.cvs, adjoint=True).on_registers(
            control=controls, ancilla=and_ancilla, target=and_target
        )
        qm.qfree(and_ancilla + and_target)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@" if b else "@(0)" for b in self.cvs]
        wire_symbols += [str(self.target_gate)]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_(self) -> infra.TComplexity:
        and_cost = infra.t_complexity(and_gate.And(self.cvs))
        controlled_pauli_cost = infra.t_complexity(self.target_gate.controlled(1))
        and_inv_cost = infra.t_complexity(and_gate.And(self.cvs, adjoint=True))
        return and_cost + controlled_pauli_cost + and_inv_cost

    def _apply_unitary_(self, args: 'cirq.ApplyUnitaryArgs') -> np.ndarray:
        return cirq.apply_unitary(self.target_gate.controlled(control_values=self.cvs), args)

    def _has_unitary_(self) -> bool:
        return True
