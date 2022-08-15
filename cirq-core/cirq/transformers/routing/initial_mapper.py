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

from typing import TYPE_CHECKING, Dict
import abc

from cirq.value import value_equality

if TYPE_CHECKING:
    import cirq


@value_equality
class AbstractInitialMapper(metaclass=abc.ABCMeta):
    """Base class for creating custom initial mapping strategies."""

    @abc.abstractmethod
    def initial_mapping(self) -> Dict['cirq.Qid', 'cirq.Qid']:
        """Maps the logical qubits of a circuit onto physical qubits on a device.
        Returns:
          qubit_map: the initial mapping of logical qubits to physical qubits.
        """

    def __str__(self) -> str:
        return str(repr(self.initial_mapping()))

    def _value_equality_values_(self):
        return self.initial_mapping()


class IdentityInitialMapper(AbstractInitialMapper):
    """Initial Mapper that takes a hard-coded mapping and returns it."""

    def __init__(self, map: Dict['cirq.Qid', 'cirq.Qid']) -> None:
        self._map = map

    def initial_mapping(self) -> Dict['cirq.Qid', 'cirq.Qid']:
        return self._map

    def __repr__(self) -> str:
        return f'cirq.IdentityInitialMapper({self.__str__()})'
