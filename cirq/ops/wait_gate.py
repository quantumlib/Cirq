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
from typing import AbstractSet, Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

import sympy

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

    def __init__(
        self,
        duration: 'cirq.DURATION_LIKE',
        num_qubits: Optional[int] = None,
        qid_shape: Tuple[int, ...] = None,
    ) -> None:
        """Initialize a wait gate with the given duration.

        Args:
            duration: A constant or parameterized wait duration. This can be
                an instance of `datetime.timedelta` or `cirq.Duration`.
        """
        self.duration = value.Duration(duration)
        if not protocols.is_parameterized(self.duration) and self.duration < 0:
            raise ValueError('duration < 0')
        if qid_shape is None:
            if num_qubits is None:
                # Assume one qubit for backwards compatibility
                qid_shape = (2,)
            else:
                qid_shape = (2,) * num_qubits
        if num_qubits is None:
            num_qubits = len(qid_shape)
        if not qid_shape:
            raise ValueError('Waiting on an empty set of qubits.')
        if num_qubits != len(qid_shape):
            raise ValueError('len(qid_shape) != num_qubits')
        self._qid_shape = qid_shape

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.duration)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.duration)

    def _resolve_parameters_(self, param_resolver, recursive):
        return WaitGate(protocols.resolve_parameters(self.duration, param_resolver, recursive))

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

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
        d = protocols.obj_to_dict_helper(self, ['duration'])
        if len(self._qid_shape) != 1:
            d['num_qubits'] = len(self._qid_shape)
        if any(d != 2 for d in self._qid_shape):
            d['qid_shape'] = self._qid_shape
        return d

    @classmethod
    def _from_json_dict_(cls, duration, num_qubits=None, qid_shape=None, **kwargs):
        return cls(
            duration=duration,
            num_qubits=num_qubits,
            qid_shape=None if qid_shape is None else tuple(qid_shape),
        )

    def _value_equality_values_(self) -> Any:
        return self.duration

    def _quil_(self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter'):
        return 'WAIT\n'


def wait(
    *target: 'cirq.Qid',
    duration: 'cirq.DURATION_LIKE' = None,
    picos: Union[int, float, sympy.Basic] = 0,
    nanos: Union[int, float, sympy.Basic] = 0,
    micros: Union[int, float, sympy.Basic] = 0,
    millis: Union[int, float, sympy.Basic] = 0,
) -> raw_types.Operation:
    """Creates a WaitGate applied to all the given qubits.

    The duration can be specified as a DURATION_LIKE or using keyword args with
    numbers in the appropriate units. See Duration for details.

    Args:
        *target: The qubits that should wait.
        value: Wait duration (see Duration).
        picos: Picoseconds to wait (see Duration).
        nanos: Picoseconds to wait (see Duration).
        micros: Picoseconds to wait (see Duration).
        millis: Picoseconds to wait (see Duration).
    """
    return WaitGate(
        duration=value.Duration(
            duration,
            picos=picos,
            nanos=nanos,
            micros=micros,
            millis=millis,
        ),
        qid_shape=protocols.qid_shape(target),
    ).on(*target)
