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

"""CZ + single rotations target gateset."""

from typing import TYPE_CHECKING

from cirq import ops
from cirq.transformers.analytical_decompositions import two_qubit_to_cz
from cirq.transformers.target_gatesets import compilation_target_gateset

if TYPE_CHECKING:
    import numpy as np
    import cirq


class CZTargetGateset(compilation_target_gateset.TwoQubitAnalyticalCompilationTarget):
    """Target gateset containing CZ + single qubit rotations + Measurement gates."""

    def __init__(self, *, atol: float = 1e-8, allow_partial_czs: bool = False) -> None:
        """Initializes CZTargetGateset

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
            allow_partial_czs: If set, all powers of the form `cirq.CZ**t`, and not just
             `cirq.CZ`, are part of this gateset.
        """
        super().__init__(
            ops.CZPowGate if allow_partial_czs else ops.CZ,
            ops.MeasurementGate,
            ops.AnyUnitaryGateFamily(1),
            name='CZPowTargetGateset' if allow_partial_czs else 'CZTargetGateset',
        )
        self.atol = atol
        self.allow_partial_czs = allow_partial_czs

    def _decompose_two_qubit_matrix_to_operations(
        self, q0: 'cirq.Qid', q1: 'cirq.Qid', mat: 'np.ndarray'
    ) -> 'cirq.OP_TREE':
        return two_qubit_to_cz.two_qubit_matrix_to_operations(
            q0, q1, mat, allow_partial_czs=self.allow_partial_czs, atol=self.atol
        )


CIRQ_DEFAULT_TARGET_GATESET = CZTargetGateset(allow_partial_czs=True)
