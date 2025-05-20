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

from __future__ import annotations

from typing import Any, Dict, Sequence, Type, TYPE_CHECKING, Union

from cirq import ops, protocols
from cirq.transformers.analytical_decompositions import two_qubit_to_cz
from cirq.transformers.target_gatesets import compilation_target_gateset

if TYPE_CHECKING:
    import cirq


class CZTargetGateset(compilation_target_gateset.TwoQubitCompilationTargetGateset):
    """Target gateset accepting CZ + single qubit rotations + measurement gates.

    By default, `cirq.CZTargetGateset` will accept and compile unknown gates to
    the following universal target gateset:
    - `cirq.CZ` / `cirq.CZPowGate`: The two qubit entangling gate.
    - `cirq.PhasedXZGate`: Single qubit rotations.
    - `cirq.MeasurementGate`: Measurements.
    - `cirq.GlobalPhaseGate`: Global phase.

    Optionally, users can also specify additional gates / gate families which should
    be accepted by this gateset via the `additional_gates` argument.

    When compiling a circuit, any unknown gate, i.e. a gate which is not accepted by
    this gateset, will be compiled to the default gateset (i.e. `cirq.CZ`/`cirq.CZPowGate`,
    `cirq.PhasedXZGate`, `cirq.MeasurementGate`).
    """

    def __init__(
        self,
        *,
        atol: float = 1e-8,
        allow_partial_czs: bool = False,
        additional_gates: Sequence[Union[Type[cirq.Gate], cirq.Gate, cirq.GateFamily]] = (),
        preserve_moment_structure: bool = True,
        reorder_operations: bool = False,
    ) -> None:
        """Initializes CZTargetGateset

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
            allow_partial_czs: If set, all powers of the form `cirq.CZ**t`, and not just
             `cirq.CZ`, are part of this gateset.
            additional_gates: Sequence of additional gates / gate families which should also
              be "accepted" by this gateset. This is empty by default.
            preserve_moment_structure: Whether to preserve the moment structure of the
                circuit during compilation or not.
            reorder_operations: Whether to attempt to reorder the operations in order to reduce
                circuit depth or not (can be True only if preserve_moment_structure=False).
        """
        super().__init__(
            ops.CZPowGate if allow_partial_czs else ops.CZ,
            ops.MeasurementGate,
            ops.PhasedXZGate,
            ops.GlobalPhaseGate,
            *additional_gates,
            name='CZPowTargetGateset' if allow_partial_czs else 'CZTargetGateset',
            preserve_moment_structure=preserve_moment_structure,
            reorder_operations=reorder_operations,
        )
        self.additional_gates = tuple(
            g if isinstance(g, ops.GateFamily) else ops.GateFamily(gate=g) for g in additional_gates
        )
        self._additional_gates_repr_str = ", ".join(
            [ops.gateset._gate_str(g, repr) for g in additional_gates]
        )
        self.atol = atol
        self.allow_partial_czs = allow_partial_czs

    def _decompose_two_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
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
        return (
            f'cirq.CZTargetGateset('
            f'atol={self.atol}, '
            f'allow_partial_czs={self.allow_partial_czs}, '
            f'additional_gates=[{self._additional_gates_repr_str}]'
            f')'
        )

    def _value_equality_values_(self) -> Any:
        return self.atol, self.allow_partial_czs, frozenset(self.additional_gates)

    def _json_dict_(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {'atol': self.atol, 'allow_partial_czs': self.allow_partial_czs}
        if self.additional_gates:
            d['additional_gates'] = list(self.additional_gates)
        return d

    @classmethod
    def _from_json_dict_(cls, atol, allow_partial_czs, additional_gates=(), **kwargs):
        return cls(
            atol=atol, allow_partial_czs=allow_partial_czs, additional_gates=additional_gates
        )
