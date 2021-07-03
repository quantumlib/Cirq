# Copyright 2018 The Cirq Developers
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

"""An optimization pass that combines adjacent series of gates on two qubits and
outputs a circuit with SQRT_ISWAP or SQRT_ISWAP_INV gates."""

from typing import Callable, Sequence, TYPE_CHECKING

import numpy as np

from cirq import ops
from cirq.optimizers import two_qubit_to_sqrt_iswap, merge_interactions

if TYPE_CHECKING:
    import cirq


class MergeInteractionsToSqrtIswap(merge_interactions.MergeInteractionsAbc):
    """Combines series of adjacent one and two-qubit gates operating on a pair
    of qubits and replaces each series with the minimum number of SQRT_ISWAP
    gates."""

    def __init__(
        self,
        tolerance: float = 1e-8,
        require_three_sqrt_iswap: bool = False,
        use_sqrt_iswap_inv: bool = False,
        post_clean_up: Callable[[Sequence[ops.Operation]], ops.OP_TREE] = lambda op_list: op_list,
    ) -> None:
        super().__init__(tolerance=tolerance, post_clean_up=post_clean_up)
        self.require_three_sqrt_iswap = require_three_sqrt_iswap
        self.use_sqrt_iswap_inv = use_sqrt_iswap_inv

    def _may_keep_old_op(self, old_op: 'cirq.Operation') -> bool:
        """Returns True if the old two-qubit operation may be left unchanged
        without decomposition."""
        return isinstance(old_op.gate, ops.ISwapPowGate) and old_op.gate.exponent == 0.5

    def _two_qubit_matrix_to_operations(
        self,
        q0: 'cirq.Qid',
        q1: 'cirq.Qid',
        mat: np.ndarray,
    ) -> Sequence['cirq.Operation']:
        """Decomposes the merged two-qubit gate unitary into the minimum number
        of SQRT_ISWAP gates.

        Args:
            q0: The first qubit being operated on.
            q1: The other qubit being operated on.
            mat: Defines the operation to apply to the pair of qubits.

        Returns:
            A list of operations implementing the matrix.
        """
        return two_qubit_to_sqrt_iswap.two_qubit_matrix_to_sqrt_iswap_operations(
            q0,
            q1,
            mat,
            required_sqrt_iswap_count=3 if self.require_three_sqrt_iswap else None,
            use_sqrt_iswap_inv=self.use_sqrt_iswap_inv,
            atol=self.tolerance,
            check_preconditions=False,
        )
