# Copyright 2021 The Cirq Developers
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
"""An interface for quantum states as targets for operations."""
import abc
from typing import TypeVar, TYPE_CHECKING, Generic, Set, Dict, Any, Sequence

if TYPE_CHECKING:
    import cirq


TSelfTarget = TypeVar('TSelfTarget', bound='OperationTarget')
TActOnArgs = TypeVar('TActOnArgs', bound='cirq.ActOnArgs')


class OperationTarget(Generic[TActOnArgs]):
    """An interface for quantum states as targets for operations."""

    @abc.abstractmethod
    def create_merged_state(self) -> TActOnArgs:
        """Creates a final merged state."""

    @abc.abstractmethod
    def apply_operation(self, op: 'cirq.Operation'):
        """Applies the operation to the state."""

    @abc.abstractmethod
    def copy(self: TSelfTarget) -> TSelfTarget:
        """Copies the object."""

    @abc.abstractmethod
    def values_set(self) -> Set[TActOnArgs]:
        """Gets the distinct set of values."""

    @property
    @abc.abstractmethod
    def qubits(self) -> Sequence['cirq.Qid']:
        """Gets the qubit order maintained by this target."""

    @property
    @abc.abstractmethod
    def log_of_measurement_results(self) -> Dict[str, Any]:
        """Gets the log of measurement results."""
