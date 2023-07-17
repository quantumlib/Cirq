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

from typing import Sequence, Union
from numpy.typing import NDArray

import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import multi_control_multi_target_pauli as mcmtp


@attr.frozen
class MultiTargetCSwap(infra.GateWithRegisters):
    """Implements a multi-target controlled swap unitary $CSWAP_n = |0><0| I + |1><1| SWAP_n$.

    This decomposes into a qubitwise SWAP on the two target registers, and takes 14*n T-gates.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
        Low et. al. 2018. See Appendix B.2.c.
    """

    bitsize: int

    @classmethod
    def make_on(
        cls, **quregs: Union[Sequence[cirq.Qid], NDArray[cirq.Qid]]  # type: ignore[type-var]
    ) -> cirq.Operation:
        """Helper constructor to automatically deduce bitsize attributes."""
        return cls(bitsize=len(quregs['target_x'])).on_registers(**quregs)

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers.build(control=1, target_x=self.bitsize, target_y=self.bitsize)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        control, target_x, target_y = quregs['control'], quregs['target_x'], quregs['target_y']
        yield [cirq.CSWAP(*control, t_x, t_y) for t_x, t_y in zip(target_x, target_y)]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return cirq.CircuitDiagramInfo(
                ("@",) + ("swap_x",) * self.bitsize + ("swap_y",) * self.bitsize
            )
        return cirq.CircuitDiagramInfo(("@",) + ("×(x)",) * self.bitsize + ("×(y)",) * self.bitsize)

    def __repr__(self) -> str:
        return f"cirq_ft.MultiTargetCSwap({self.bitsize})"

    def _t_complexity_(self) -> infra.TComplexity:
        return infra.TComplexity(t=7 * self.bitsize, clifford=10 * self.bitsize)


@attr.frozen
class MultiTargetCSwapApprox(MultiTargetCSwap):
    """Approximately implements a multi-target controlled swap unitary using only 4 * n T-gates.

    Implements the unitary $CSWAP_n = |0><0| I + |1><1| SWAP_n$ such that the output state is
    correct up to a global phase factor of +1 / -1.

    This is useful when the incorrect phase can be absorbed in a garbage state of an algorithm; and
    thus ignored, see the reference for more details.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
        Low et. al. 2018. See Appendix B.2.c.
    """

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        control, target_x, target_y = quregs['control'], quregs['target_x'], quregs['target_y']

        def g(q: cirq.Qid, adjoint=False) -> cirq.ops.op_tree.OpTree:
            yield [cirq.S(q), cirq.H(q)]
            yield cirq.T(q) ** (1 - 2 * adjoint)
            yield [cirq.H(q), cirq.S(q) ** -1]

        cnot_x_to_y = [cirq.CNOT(x, y) for x, y in zip(target_x, target_y)]
        cnot_y_to_x = [cirq.CNOT(y, x) for x, y in zip(target_x, target_y)]
        g_inv_on_y = [list(g(q, True)) for q in target_y]  # Uses len(target_y) T-gates
        g_on_y = [list(g(q)) for q in target_y]  # Uses len(target_y) T-gates

        yield [cnot_y_to_x, g_inv_on_y, cnot_x_to_y, g_inv_on_y]
        yield mcmtp.MultiTargetCNOT(len(target_y)).on(*control, *target_y)
        yield [g_on_y, cnot_x_to_y, g_on_y, cnot_y_to_x]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return cirq.CircuitDiagramInfo(
                ("@(approx)",) + ("swap_x",) * self.bitsize + ("swap_y",) * self.bitsize
            )
        return cirq.CircuitDiagramInfo(
            ("@(approx)",) + ("×(x)",) * self.bitsize + ("×(y)",) * self.bitsize
        )

    def __repr__(self) -> str:
        return f"cirq_ft.MultiTargetCSwapApprox({self.bitsize})"

    def _t_complexity_(self) -> infra.TComplexity:
        """TComplexity as explained in Appendix B.2.c of https://arxiv.org/abs/1812.00954"""
        n = self.bitsize
        # 4 * n: G gates, each wth 1 T and 4 cliffords
        # 4 * n: CNOTs
        # 2 * n - 1: CNOTs from 1 MultiTargetCNOT
        return infra.TComplexity(t=4 * n, clifford=22 * n - 1)


@attr.frozen
class SwapWithZeroGate(infra.GateWithRegisters):
    """Swaps |Psi_0> with |Psi_x> if selection register stores index `x`.

    Implements the unitary U |x> |Psi_0> |Psi_1> ... |Psi_{n-1}> --> |x> |Psi_x> |Rest of Psi>.
    Note that the state of `|Rest of Psi>` is allowed to be anything and should not be depended
    upon.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    selection_bitsize: int
    target_bitsize: int
    n_target_registers: int

    def __attrs_post_init__(self):
        assert self.n_target_registers <= 2**self.selection_bitsize

    @cached_property
    def selection_registers(self) -> infra.SelectionRegisters:
        return infra.SelectionRegisters(
            [infra.SelectionRegister('selection', self.selection_bitsize, self.n_target_registers)]
        )

    @cached_property
    def target_registers(self) -> infra.Registers:
        return infra.Registers.build(target=(self.n_target_registers, self.target_bitsize))

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers([*self.selection_registers, *self.target_registers])

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        selection, target = quregs['selection'], quregs['target']
        assert target.shape == (self.n_target_registers, self.target_bitsize)
        cswap_n = MultiTargetCSwapApprox(self.target_bitsize)
        # Imagine a complete binary tree of depth `logN` with `N` leaves, each denoting a target
        # register. If the selection register stores index `r`, we want to bring the value stored
        # in leaf indexed `r` to the leaf indexed `0`. At each node of the binary tree, the left
        # subtree contains node with current bit 0 and right subtree contains nodes with current
        # bit 1. Thus, leaf indexed `0` is the leftmost node in the tree.
        # Start iterating from the root of the tree. If the j'th bit is set in the selection
        # register (i.e. the control would be activated); we know that the value we are searching
        # for is in the right subtree. In order to (eventually) bring the desired value to node
        # 0; we swap all values in the right subtree with all values in the left subtree. This
        # takes (N / (2 ** (j + 1)) swaps at level `j`.
        # Therefore, in total, we need $\sum_{j=0}^{logN-1} \frac{N}{2 ^ {j + 1}}$ controlled swaps.
        for j in range(len(selection)):
            for i in range(0, self.n_target_registers - 2**j, 2 ** (j + 1)):
                # The inner loop is executed at-most `N - 1` times, where `N:= len(target_regs)`.
                yield cswap_n.on_registers(
                    control=selection[len(selection) - j - 1],
                    target_x=target[i],
                    target_y=target[i + 2**j],
                )

    def __repr__(self) -> str:
        return (
            "cirq_ft.SwapWithZeroGate("
            f"{self.selection_bitsize},"
            f"{self.target_bitsize},"
            f"{self.n_target_registers}"
            f")"
        )

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@(r⇋0)"] * self.selection_bitsize
        for i in range(self.n_target_registers):
            wire_symbols += [f"swap_{i}"] * self.target_bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
