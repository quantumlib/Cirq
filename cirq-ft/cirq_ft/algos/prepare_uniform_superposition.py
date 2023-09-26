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
from cirq_ft.algos import and_gate, arithmetic_gates


@attr.frozen
class PrepareUniformSuperposition(infra.GateWithRegisters):
    r"""Prepares a uniform superposition over first $n$ basis states using $O(log(n))$ T-gates.

    Performs a single round of amplitude amplification and prepares a uniform superposition over
    the first $n$ basis states $|0>, |1>, ..., |n - 1>$. The expected T-complexity should be
    $10 * log(L) + 2 * K$ T-gates and $2$ single qubit rotation gates, where $n = L * 2^K$.

    However, the current T-complexity is $12 * log(L)$ T-gates and $2 + 2 * (K + log(L))$ rotations
    because of two open issues:
        - https://github.com/quantumlib/cirq-qubitization/issues/233 and
        - https://github.com/quantumlib/cirq-qubitization/issues/235

    Args:
        n: The gate prepares a uniform superposition over first $n$ basis states.
        cv: Control values for each control qubit. If specified, a controlled version
            of the gate is constructed.

    References:
        See Fig 12 of https://arxiv.org/abs/1805.03662 for more details.
    """

    n: int
    cv: Tuple[int, ...] = attr.field(
        converter=lambda v: (v,) if isinstance(v, int) else tuple(v), default=()
    )

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature.build(controls=len(self.cv), target=(self.n - 1).bit_length())

    def __repr__(self) -> str:
        return f"cirq_ft.PrepareUniformSuperposition({self.n}, cv={self.cv})"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        control_symbols = ["@" if cv else "@(0)" for cv in self.cv]
        target_symbols = ['target'] * self.signature.get_left('target').total_bits()
        target_symbols[0] = f"UNIFORM({self.n})"
        return cirq.CircuitDiagramInfo(wire_symbols=control_symbols + target_symbols)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        controls, target = quregs.get('controls', ()), quregs['target']
        # Find K and L as per https://arxiv.org/abs/1805.03662 Fig 12.
        n, k = self.n, 0
        while n > 1 and n % 2 == 0:
            k += 1
            n = n // 2
        l, logL = int(n), self.signature.get_left('target').total_bits() - k
        logL_qubits = target[:logL]

        yield [
            op.controlled_by(*controls, control_values=self.cv) for op in cirq.H.on_each(*target)
        ]
        if not len(logL_qubits):
            return

        ancilla = context.qubit_manager.qalloc(1)
        theta = np.arccos(1 - (2 ** np.floor(np.log2(l))) / l)
        yield arithmetic_gates.LessThanGate(logL, l).on(*logL_qubits, *ancilla)
        yield cirq.Rz(rads=theta)(*ancilla)
        yield arithmetic_gates.LessThanGate(logL, l).on(*logL_qubits, *ancilla)

        yield cirq.H.on_each(*logL_qubits)

        and_ancilla = context.qubit_manager.qalloc(len(self.cv) + logL - 2)
        and_op = and_gate.And((0,) * logL + self.cv).on_registers(
            ctrl=np.asarray([*logL_qubits, *controls])[:, np.newaxis],
            junk=np.asarray(and_ancilla)[:, np.newaxis],
            target=ancilla,
        )
        yield and_op
        yield cirq.Rz(rads=theta)(*ancilla)
        yield cirq.inverse(and_op)

        yield cirq.H.on_each(*logL_qubits)
        context.qubit_manager.qfree([*ancilla, *and_ancilla])
