# Copyright 2021 The Cirq Developers
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

from typing import Callable, Optional, Sequence, TYPE_CHECKING

import numpy as np

from cirq import ops
from cirq.optimizers import two_qubit_to_sqrt_iswap, merge_interactions

if TYPE_CHECKING:
    import cirq


class MergeInteractionsToSqrtIswap(merge_interactions.MergeInteractionsAbc):
    """Combines series of adjacent one- and two-qubit, non-parametrized gates
    operating on a pair of qubits and replaces each series with the minimum
    number of SQRT_ISWAP gates.

    See also: ``two_qubit_matrix_to_sqrt_iswap_operations``
    """

    def __init__(
        self,
        tolerance: float = 1e-8,
        *,
        required_sqrt_iswap_count: Optional[int] = None,
        use_sqrt_iswap_inv: bool = False,
        post_clean_up: Callable[[Sequence[ops.Operation]], ops.OP_TREE] = lambda op_list: op_list,
    ) -> None:
        """
        Args:
            tolerance: A limit on the amount of absolute error introduced by the
                construction.
            required_sqrt_iswap_count: When specified, each merged group of
                two-qubit gates will be decomposed into exactly this many
                sqrt-iSWAP gates even if fewer is possible (maximum 3).  Circuit
                optimization will raise a ``ValueError`` if this number is 2 or
                lower and synthesis of any set of merged interactions requires
                more.
            use_sqrt_iswap_inv: If True, optimizes circuits using
                ``SQRT_ISWAP_INV`` gates instead of ``SQRT_ISWAP``.
            post_clean_up: This function is called on each set of optimized
                operations before they are put into the circuit to replace the
                old operations.

        Raises:
            ValueError:
                If ``required_sqrt_iswap_count`` is not one of the supported
                values 0, 1, 2, or 3.
        """
        if required_sqrt_iswap_count is not None and not 0 <= required_sqrt_iswap_count <= 3:
            raise ValueError('the argument `required_sqrt_iswap_count` must be 0, 1, 2, or 3.')
        super().__init__(tolerance=tolerance, post_clean_up=post_clean_up)
        self.required_sqrt_iswap_count = required_sqrt_iswap_count
        self.use_sqrt_iswap_inv = use_sqrt_iswap_inv

    def _may_keep_old_op(self, old_op: 'cirq.Operation') -> bool:
        """Returns True if the old two-qubit operation may be left unchanged
        without decomposition."""
        if self.use_sqrt_iswap_inv:
            return isinstance(old_op.gate, ops.ISwapPowGate) and old_op.gate.exponent == -0.5
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
            required_sqrt_iswap_count=self.required_sqrt_iswap_count,
            use_sqrt_iswap_inv=self.use_sqrt_iswap_inv,
            atol=self.tolerance,
            check_preconditions=False,
        )
