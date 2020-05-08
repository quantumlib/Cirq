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
from typing import Any, Dict, Tuple, TYPE_CHECKING

from cirq import value, protocols
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class WaitGate(raw_types.Gate):
    """A single-qubit idle gate that represents waiting.

    In non-noisy simulators, this gate is just an identity gate. But noisy
    simulators and noise models may insert more error for longer waits.
    """

    def __init__(self, duration: 'cirq.DURATION_LIKE') -> None:
        """Initialize a wait gate with the given duration.

        Args:
            duration: A constant or parameterized wait duration. This can be
                an instance of `datetime.timedelta` or `cirq.Duration`.
        """
        self.duration = value.Duration(duration)
        if not protocols.is_parameterized(self.duration) and self.duration < 0:
            raise ValueError('duration < 0')

    def _is_parameterized_(self):
        return protocols.is_parameterized(self.duration)

    def _resolve_parameters_(self, param_resolver):
        return WaitGate(
            protocols.resolve_parameters(self.duration, param_resolver))

    def _num_qubits_(self) -> int:
        return 1

    def _has_unitary_(self) -> bool:
        return True

    def _apply_unitary_(self, args):
        return args.target_tensor  # Identity.

    def _decompose_(self, qubits):
        return []

    def _trace_distance_bound_(self):
        return 0

    def __pow__(self, power):
        if power == 1 or power == -1:
            # The inverse of a wait is still a wait.
            return self
        # Other scalar exponents could scale the wait... but ultimately it is
        # ambiguous whether the user wanted to scale the duration or just wanted
        # to affect the unitary. Play it safe and fail.
        return NotImplemented

    def __str__(self) -> str:
        return f'WaitGate({self.duration})'

    def __repr__(self) -> str:
        return f'cirq.WaitGate({repr(self.duration)})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['duration'])

    def _value_equality_values_(self) -> Any:
        return self.duration

    def _quil_(self, qubits: Tuple['cirq.Qid', ...],
               formatter: 'cirq.QuilFormatter'):
        return 'WAIT\n'
