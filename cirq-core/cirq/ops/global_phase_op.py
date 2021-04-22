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
from typing import Any, Dict, Tuple, TYPE_CHECKING

import numpy as np

from cirq import value, protocols
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class GlobalPhaseOperation(raw_types.Operation):
    def __init__(self, coefficient: value.Scalar, atol: float = 1e-8) -> None:
        if abs(1 - abs(coefficient)) > atol:
            raise ValueError(f'Coefficient is not unitary: {coefficient!r}')
        self.coefficient = coefficient

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        return ()

    def with_qubits(self, *new_qubits) -> 'GlobalPhaseOperation':
        if new_qubits:
            raise ValueError(f'{self!r} applies to 0 qubits but new_qubits={new_qubits!r}.')
        return self

    def _value_equality_values_(self) -> Any:
        return self.coefficient

    def _has_unitary_(self) -> bool:
        return True

    def __pow__(self, power):
        if isinstance(power, (int, float)):
            return GlobalPhaseOperation(self.coefficient ** power)
        return NotImplemented

    def _unitary_(self) -> np.ndarray:
        return np.array([[self.coefficient]])

    def _apply_unitary_(self, args) -> np.ndarray:
        args.target_tensor *= self.coefficient
        return args.target_tensor

    def _has_stabilizer_effect_(self) -> bool:
        return True

    def _act_on_(self, args: Any):
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
        return f'cirq.GlobalPhaseOperation({self.coefficient!r})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['coefficient'])
