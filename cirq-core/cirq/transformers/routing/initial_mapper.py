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

import abc
from typing import Dict, TYPE_CHECKING

from cirq import value

if TYPE_CHECKING:
    import cirq


class AbstractInitialMapper(metaclass=abc.ABCMeta):
    """Base class for creating custom initial mapping strategies.

    An initial mapping strategy is a placement strategy that places logical qubit variables in an
    input circuit onto physical qubits that correspond to a specified device. This placment can be
    thought of as a mapping k -> m[k] where k is a logical qubit and m[k] is the physical qubit it
    is mapped to. Any initial mapping strategy must satisfy two constraints:
        1. all logical qubits must be placed on the device if the number of logical qubits is <=
            than the number of physical qubits.
        2. if two logical qubits interact (i.e. there exists a 2-qubit operation on them) at any
            point in the input circuit, then they must lie in the same connected components of the
            device graph induced on the physical qubits in the initial mapping.

    """

    @abc.abstractmethod
    def initial_mapping(self, circuit: 'cirq.AbstractCircuit') -> Dict['cirq.Qid', 'cirq.Qid']:
        """Maps the logical qubits of a circuit onto physical qubits on a device.

        Args:
            circuit: the input circuit with logical qubits.

        Returns:
          qubit_map: the initial mapping of logical qubits to physical qubits.
        """


@value.value_equality
class HardCodedInitialMapper(AbstractInitialMapper):
    """Initial Mapper class takes a hard-coded mapping and returns it."""

    def __init__(self, _map: Dict['cirq.Qid', 'cirq.Qid']) -> None:
        self._map = _map

    def initial_mapping(self, circuit: 'cirq.AbstractCircuit') -> Dict['cirq.Qid', 'cirq.Qid']:
        """Returns the hard-coded initial mapping.

        Args:
            circuit: the input circuit with logical qubits.

        Returns:
            the hard-codded initial mapping.

        Raises:
            ValueError: if the qubits in circuit are not a subset of the qubit keys in the mapping.
        """
        if not circuit.all_qubits().issubset(set(self._map.keys())):
            raise ValueError("The qubits in circuit must be a subset of the keys in the mapping")
        return self._map

    def _value_equality_values_(self):
        return tuple(sorted(self._map.items()))

    def __repr__(self) -> str:
        return f'cirq.HardCodedInitialMapper({self._map})'
