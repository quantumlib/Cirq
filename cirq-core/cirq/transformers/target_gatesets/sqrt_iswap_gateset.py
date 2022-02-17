# Copyright 2022 The Cirq Developers
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

"""√iSWAP + single qubit rotations target gateset."""

from typing import TYPE_CHECKING, Optional

from cirq import ops
from cirq.transformers.analytical_decompositions import two_qubit_to_sqrt_iswap
from cirq.transformers.target_gatesets import compilation_target_gateset

if TYPE_CHECKING:
    import numpy as np
    import cirq


class SqrtIswapTargetGateset(compilation_target_gateset.TwoQubitAnalyticalCompilationTarget):
    """Target gateset containing √iSWAP + single qubit rotations + Measurement gates."""

    def __init__(
        self,
        *,
        atol: float = 1e-8,
        required_sqrt_iswap_count: Optional[int] = None,
        use_sqrt_iswap_inv: bool = False,
    ):
        """Initializes SqrtIswapTargetGateset

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
            required_sqrt_iswap_count: When specified, the `decompose_to_target_gateset` will
                decompose each operation into exactly this many sqrt-iSWAP gates even if fewer is
                possible (maximum 3). A ValueError will be raised if this number is 2 or lower and
                synthesis of the operation requires more.
            use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used as part of the gateset
                instead of `cirq.SQRT_ISWAP`.

        Raises:
            ValueError: If `required_sqrt_iswap_count` is specified and is not 0, 1, 2, or 3.
        """
        if required_sqrt_iswap_count is not None and not 0 <= required_sqrt_iswap_count <= 3:
            raise ValueError('the argument `required_sqrt_iswap_count` must be 0, 1, 2, or 3.')
        super().__init__(
            ops.SQRT_ISWAP_INV if use_sqrt_iswap_inv else ops.SQRT_ISWAP,
            ops.MeasurementGate,
            ops.AnyUnitaryGateFamily(1),
            name='SqrtIswapInvTargetGateset' if use_sqrt_iswap_inv else 'SqrtIswapTargetGateset',
        )
        self.atol = atol
        self.required_sqrt_iswap_count = required_sqrt_iswap_count
        self.use_sqrt_iswap_inv = use_sqrt_iswap_inv

    def _decompose_two_qubit_matrix_to_operations(
        self, q0: 'cirq.Qid', q1: 'cirq.Qid', mat: 'np.ndarray'
    ) -> 'cirq.OP_TREE':
        return two_qubit_to_sqrt_iswap.two_qubit_matrix_to_sqrt_iswap_operations(
            q0,
            q1,
            mat,
            required_sqrt_iswap_count=self.required_sqrt_iswap_count,
            use_sqrt_iswap_inv=self.use_sqrt_iswap_inv,
            atol=self.atol,
            check_preconditions=False,
            clean_operations=True,
        )
