# Copyright 2019 The Cirq Developers
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
"""A no-qubit global phase operation."""
from typing import Any, Dict, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from cirq import value, protocols
from cirq._compat import deprecated_class
from cirq.ops import gate_operation, raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
@deprecated_class(deadline='v0.16', fix='Use cirq.global_phase_operation')
class GlobalPhaseOperation(gate_operation.GateOperation):
    def __init__(self, coefficient: value.Scalar, atol: float = 1e-8) -> None:
        gate = GlobalPhaseGate(coefficient, atol)
        super().__init__(gate, [])

    def with_qubits(self, *new_qubits) -> 'GlobalPhaseOperation':
        if new_qubits:
            raise ValueError(f'{self!r} applies to 0 qubits but new_qubits={new_qubits!r}.')
        return self

    @property
    def coefficient(self) -> value.Scalar:
        return self.gate.coefficient  # type: ignore

    @coefficient.setter
    def coefficient(self, coefficient: value.Scalar):
        # coverage: ignore
        self.gate._coefficient = coefficient  # type: ignore

    def __str__(self) -> str:
        return str(self.coefficient)

    def __repr__(self) -> str:
        return f'cirq.GlobalPhaseOperation({self.coefficient!r})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['coefficient'])


@value.value_equality(approximate=True)
class GlobalPhaseGate(raw_types.Gate):
    def __init__(self, coefficient: value.Scalar, atol: float = 1e-8) -> None:
        if abs(1 - abs(coefficient)) > atol:
            raise ValueError(f'Coefficient is not unitary: {coefficient!r}')
        self._coefficient = coefficient

    @property
    def coefficient(self) -> value.Scalar:
        return self._coefficient

    def _value_equality_values_(self) -> Any:
        return self.coefficient

    def _has_unitary_(self) -> bool:
        return True

    def __pow__(self, power) -> 'cirq.GlobalPhaseGate':
        if isinstance(power, (int, float)):
            return GlobalPhaseGate(self.coefficient ** power)
        return NotImplemented

    def _unitary_(self) -> np.ndarray:
        return np.array([[self.coefficient]])

    def _apply_unitary_(self, args) -> np.ndarray:
        args.target_tensor *= self.coefficient
        return args.target_tensor

    def _has_stabilizer_effect_(self) -> bool:
        return True

    def _act_on_(self, args: 'cirq.ActOnArgs', qubits):
        from cirq.sim import clifford

        if isinstance(args, clifford.ActOnCliffordTableauArgs):
            # Since CliffordTableau does not keep track of the global phase,
            # it's safe to just ignore it here.
            return True

        if isinstance(args, clifford.ActOnStabilizerCHFormArgs):
            args.state.omega *= self.coefficient
            return True

        return NotImplemented

    def __str__(self) -> str:
        return str(self.coefficient)

    def __repr__(self) -> str:
        return f'cirq.GlobalPhaseGate({self.coefficient!r})'

    def _op_repr_(self, qubits: Sequence['cirq.Qid']) -> str:
        return f'cirq.global_phase_operation({self.coefficient!r})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['coefficient'])

    def _qid_shape_(self) -> Tuple[int, ...]:
        return tuple()


def global_phase_operation(coefficient: value.Scalar, atol: float = 1e-8) -> 'cirq.GateOperation':
    return GlobalPhaseGate(coefficient, atol)()
