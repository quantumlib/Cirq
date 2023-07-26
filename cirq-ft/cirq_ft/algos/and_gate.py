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

from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra


@attr.frozen
class And(infra.GateWithRegisters):
    r"""And gate optimized for T-count.

    Assumptions:
        * And(cv).on(c1, c2, target) assumes that target is initially in the |0> state.
        * And(cv, adjoint=True).on(c1, c2, target) will always leave the target in |0> state.

    Multi-controlled AND version decomposes into an AND ladder with `#controls - 2` ancillas.

    Args:
        cv: A tuple of integers representing 1 control bit per control qubit.
        adjoint: If True, the $And^\dagger$ is implemented using measurement based un-computation.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
        (https://arxiv.org/abs/1805.03662)
            Babbush et. al. 2018. Section III.A. and Fig. 4.

        [Verifying Measurement Based Uncomputation](https://algassert.com/post/1903).
            Gidney, C. 2019.

    Raises:
        ValueError: If number of control values (i.e. `len(self.cv)`) is less than 2.
    """

    cv: Tuple[int, ...] = attr.field(default=(1, 1), converter=infra.to_tuple)
    adjoint: bool = False

    @cv.validator
    def _validate_cv(self, attribute, value):
        if len(value) < 2:
            raise ValueError(f"And gate needs at-least 2 control values, supplied {value} instead.")

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers.build(control=len(self.cv), ancilla=len(self.cv) - 2, target=1)

    def __pow__(self, power: int) -> "And":
        if power == 1:
            return self
        if power == -1:
            return And(self.cv, adjoint=self.adjoint ^ True)
        return NotImplemented  # coverage: ignore

    def __str__(self) -> str:
        suffix = "" if self.cv == (1,) * len(self.cv) else str(self.cv)
        return f"And†{suffix}" if self.adjoint else f"And{suffix}"

    def __repr__(self) -> str:
        return f"cirq_ft.And({self.cv}, adjoint={self.adjoint})"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        controls = ["(0)", "@"]
        target = "And†" if self.adjoint else "And"
        wire_symbols = [controls[c] for c in self.cv]
        wire_symbols += ["Anc"] * (len(self.cv) - 2)
        wire_symbols += [target]
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _has_unitary_(self) -> bool:
        return not self.adjoint

    def _decompose_single_and(
        self, cv1: int, cv2: int, c1: cirq.Qid, c2: cirq.Qid, target: cirq.Qid
    ) -> cirq.ops.op_tree.OpTree:
        """Decomposes a single `And` gate on 2 controls and 1 target in terms of Clifford+T gates.

        * And(cv).on(c1, c2, target) uses 4 T-gates and assumes target is in |0> state.
        * And(cv, adjoint=True).on(c1, c2, target) uses measurement based un-computation
            (0 T-gates) and will always leave the target in |0> state.
        """
        pre_post_ops = [cirq.X(q) for (q, v) in zip([c1, c2], [cv1, cv2]) if v == 0]
        yield pre_post_ops
        if self.adjoint:
            yield cirq.H(target)
            yield cirq.measure(target, key=f"{target}")
            yield cirq.CZ(c1, c2).with_classical_controls(f"{target}")
            yield cirq.reset(target)
        else:
            yield [cirq.H(target), cirq.T(target)]
            yield [cirq.CNOT(c1, target), cirq.CNOT(c2, target)]
            yield [cirq.CNOT(target, c1), cirq.CNOT(target, c2)]
            yield [cirq.T(c1) ** -1, cirq.T(c2) ** -1, cirq.T(target)]
            yield [cirq.CNOT(target, c1), cirq.CNOT(target, c2)]
            yield [cirq.H(target), cirq.S(target)]
        yield pre_post_ops

    def _decompose_via_tree(
        self,
        controls: NDArray[cirq.Qid],  # type:ignore[type-var]
        control_values: Sequence[int],
        ancillas: NDArray[cirq.Qid],
        target: cirq.Qid,
    ) -> cirq.ops.op_tree.OpTree:
        """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls- 2."""
        if len(controls) == 2:
            yield And(control_values, adjoint=self.adjoint).on(*controls, target)
            return
        new_controls = np.concatenate([ancillas[0:1], controls[2:]])
        new_control_values = (1, *control_values[2:])
        and_op = And(control_values[:2], adjoint=self.adjoint).on(*controls[:2], ancillas[0])
        if self.adjoint:
            yield from self._decompose_via_tree(
                new_controls, new_control_values, ancillas[1:], target
            )
            yield and_op
        else:
            yield and_op
            yield from self._decompose_via_tree(
                new_controls, new_control_values, ancillas[1:], target
            )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        control, ancilla, target = quregs['control'], quregs['ancilla'], quregs['target']
        if len(self.cv) == 2:
            yield self._decompose_single_and(
                self.cv[0], self.cv[1], control[0], control[1], *target
            )
        else:
            yield self._decompose_via_tree(control, self.cv, ancilla, *target)

    def _t_complexity_(self) -> infra.TComplexity:
        pre_post_cliffords = len(self.cv) - sum(self.cv)  # number of zeros in self.cv
        num_single_and = len(self.cv) - 1
        if self.adjoint:
            return infra.TComplexity(clifford=4 * num_single_and + 2 * pre_post_cliffords)
        else:
            return infra.TComplexity(
                t=4 * num_single_and, clifford=9 * num_single_and + 2 * pre_post_cliffords
            )
