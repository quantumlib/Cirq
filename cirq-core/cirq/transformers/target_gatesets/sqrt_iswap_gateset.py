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

"""Target gateset used for compiling circuits to √iSWAP + 1-q rotations + measurement gates."""

from typing import Any, Dict, Optional, Sequence, Type, Union, TYPE_CHECKING

from cirq import ops, protocols
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers.analytical_decompositions import two_qubit_to_sqrt_iswap
from cirq.transformers.target_gatesets import compilation_target_gateset

if TYPE_CHECKING:
    import cirq


class SqrtIswapTargetGateset(compilation_target_gateset.TwoQubitCompilationTargetGateset):
    """Target gateset accepting √iSWAP + single qubit rotations + measurement gates.

    By default, `cirq.SqrtIswapTargetGateset` will accept and compile unknown gates to
    the following universal target gateset:
    - `cirq.SQRT_ISWAP` / `cirq.SQRT_ISWAP_INV`: The two qubit entangling gate.
    - `cirq.PhasedXZGate`: Single qubit rotations.
    - `cirq.MeasurementGate`: Measurements.
    - `cirq.GlobalPhaseGate`: Global phase.

    Optionally, users can also specify additional gates / gate families which should
    be accepted by this gateset via the `additional_gates` argument.

    When compiling a circuit, any unknown gate, i.e. a gate which is not accepted by
    this gateset, will be compiled to the default gateset (i.e. `cirq.SQRT_ISWAP`/
    `cirq.cirq.SQRT_ISWAP_INV`, `cirq.PhasedXZGate`, `cirq.MeasurementGate`).
    """

    def __init__(
        self,
        *,
        atol: float = 1e-8,
        required_sqrt_iswap_count: Optional[int] = None,
        use_sqrt_iswap_inv: bool = False,
        additional_gates: Sequence[Union[Type['cirq.Gate'], 'cirq.Gate', 'cirq.GateFamily']] = (),
    ):
        """Initializes `cirq.SqrtIswapTargetGateset`

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
            required_sqrt_iswap_count: When specified, the `decompose_to_target_gateset` will
                decompose each operation into exactly this many sqrt-iSWAP gates even if fewer is
                possible (maximum 3). A ValueError will be raised if this number is 2 or lower and
                synthesis of the operation requires more.
            use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used as part of the gateset,
                instead of `cirq.SQRT_ISWAP`.
            additional_gates: Sequence of additional gates / gate families which should also
              be "accepted" by this gateset. This is empty by default.

        Raises:
            ValueError: If `required_sqrt_iswap_count` is specified and is not 0, 1, 2, or 3.
        """
        if required_sqrt_iswap_count is not None and not 0 <= required_sqrt_iswap_count <= 3:
            raise ValueError('the argument `required_sqrt_iswap_count` must be 0, 1, 2, or 3.')
        super().__init__(
            ops.SQRT_ISWAP_INV if use_sqrt_iswap_inv else ops.SQRT_ISWAP,
            ops.MeasurementGate,
            ops.PhasedXZGate,
            ops.GlobalPhaseGate,
            *additional_gates,
            name='SqrtIswapInvTargetGateset' if use_sqrt_iswap_inv else 'SqrtIswapTargetGateset',
        )
        self.additional_gates = tuple(
            g if isinstance(g, ops.GateFamily) else ops.GateFamily(gate=g) for g in additional_gates
        )
        self._additional_gates_repr_str = ", ".join(
            [ops.gateset._gate_str(g, repr) for g in additional_gates]
        )
        self.atol = atol
        self.required_sqrt_iswap_count = required_sqrt_iswap_count
        self.use_sqrt_iswap_inv = use_sqrt_iswap_inv

    def _decompose_two_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
        if protocols.has_unitary(op):
            return two_qubit_to_sqrt_iswap.two_qubit_matrix_to_sqrt_iswap_operations(
                op.qubits[0],
                op.qubits[1],
                protocols.unitary(op),
                required_sqrt_iswap_count=self.required_sqrt_iswap_count,
                use_sqrt_iswap_inv=self.use_sqrt_iswap_inv,
                atol=self.atol,
                check_preconditions=False,
                clean_operations=True,
            )
        if protocols.is_parameterized(op):
            return two_qubit_to_sqrt_iswap.parameterized_2q_op_to_sqrt_iswap_operations(
                op, use_sqrt_iswap_inv=self.use_sqrt_iswap_inv
            )
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f'cirq.SqrtIswapTargetGateset('
            f'atol={self.atol}, '
            f'required_sqrt_iswap_count={self.required_sqrt_iswap_count}, '
            f'use_sqrt_iswap_inv={self.use_sqrt_iswap_inv}, '
            f'additional_gates=[{self._additional_gates_repr_str}]'
            f')'
        )

    def _value_equality_values_(self) -> Any:
        return (
            self.atol,
            self.required_sqrt_iswap_count,
            self.use_sqrt_iswap_inv,
            frozenset(self.additional_gates),
        )

    def _json_dict_(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            'atol': self.atol,
            'required_sqrt_iswap_count': self.required_sqrt_iswap_count,
            'use_sqrt_iswap_inv': self.use_sqrt_iswap_inv,
        }
        if self.additional_gates:
            d['additional_gates'] = list(self.additional_gates)
        return d

    @classmethod
    def _from_json_dict_(
        cls, atol, required_sqrt_iswap_count, use_sqrt_iswap_inv, additional_gates=(), **kwargs
    ):
        return cls(
            atol=atol,
            required_sqrt_iswap_count=required_sqrt_iswap_count,
            use_sqrt_iswap_inv=use_sqrt_iswap_inv,
            additional_gates=additional_gates,
        )
