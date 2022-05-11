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

"""Target gateset used for compiling circuits to CZ + 1-q rotations + measurement gates."""

from typing import Any, Dict, TYPE_CHECKING

from cirq import ops, protocols
from cirq.transformers.analytical_decompositions import two_qubit_to_cz
from cirq.transformers.target_gatesets import compilation_target_gateset

if TYPE_CHECKING:
    import cirq


class CZTargetGateset(compilation_target_gateset.TwoQubitCompilationTargetGateset):
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
            ops.GlobalPhaseGate,
            name='CZPowTargetGateset' if allow_partial_czs else 'CZTargetGateset',
        )
        self.atol = atol
        self.allow_partial_czs = allow_partial_czs

    def _decompose_two_qubit_operation(self, op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
        if not protocols.has_unitary(op):
            return NotImplemented
        return two_qubit_to_cz.two_qubit_matrix_to_cz_operations(
            op.qubits[0],
            op.qubits[1],
            protocols.unitary(op),
            allow_partial_czs=self.allow_partial_czs,
            atol=self.atol,
        )

    def __repr__(self) -> str:
        return f'cirq.CZTargetGateset(atol={self.atol}, allow_partial_czs={self.allow_partial_czs})'

    def _value_equality_values_(self) -> Any:
        return self.atol, self.allow_partial_czs

    def _json_dict_(self) -> Dict[str, Any]:
        return {'atol': self.atol, 'allow_partial_czs': self.allow_partial_czs}

    @classmethod
    def _from_json_dict_(cls, atol, allow_partial_czs, **kwargs):
        return cls(atol=atol, allow_partial_czs=allow_partial_czs)
