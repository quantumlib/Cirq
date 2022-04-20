# Copyright 2018 The Cirq Developers
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

"""Marker classes for indicating which additional features gates support.

For example: some gates are reversible, some have known matrices, etc.
"""

import abc
import warnings

from cirq import value
from cirq._compat import deprecated_class
from cirq.ops import raw_types


class InterchangeableQubitsGate(metaclass=abc.ABCMeta):
    """Indicates operations should be equal under some qubit permutations."""

    def qubit_index_to_equivalence_group_key(self, index: int) -> int:
        """Returns a key that differs between non-interchangeable qubits."""
        return 0


class _SingleQubitGateMeta(value.ABCMetaImplementAnyOneOf):
    def __instancecheck__(cls, instance):
        warnings.warn(
            'isinstance(gate, SingleQubitGate) is deprecated. '
            'Use cirq.num_qubits(gate) == 1 instead',
            DeprecationWarning,
        )
        return isinstance(instance, raw_types.Gate) and instance._num_qubits_() == 1


@deprecated_class(deadline='v1.0', fix='Define _num_qubits_ manually.')
class SingleQubitGate(raw_types.Gate, metaclass=_SingleQubitGateMeta):
    """A gate that must be applied to exactly one qubit."""

    def _num_qubits_(self) -> int:
        return 1
